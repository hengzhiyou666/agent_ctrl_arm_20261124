"""Run a grounded tool-calling episode on an official LIBERO task."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBERO = ROOT.parent / "benchmarks" / "LIBERO"
CAPTAIN = ROOT / "CaPtain-Harness" / "src"
TASK_INDEX = 0
HORIZON = 280
CONTROL_HZ = 20
TARGETS = (
    "alphabet_soup", "cream_cheese", "salad_dressing", "bbq_sauce", "ketchup",
    "tomato_sauce", "butter", "milk", "chocolate_pudding", "orange_juice",
)

TOOL_PARAMETERS = {
    "observe": {},
    "plan_pick_place": {
        "frame_id": {"type": "string"},
        "target": {"type": "string", "enum": list(TARGETS),
                   "description": "The object named in the user instruction, expressed as a listed identifier."},
        "destination": {"type": "string", "enum": ["basket"]},
    },
    "execute": {"plan_id": {"type": "string"}},
    "verify_place": {"plan_id": {"type": "string"}},
}
TOOL_DESCRIPTIONS = {
    "observe": "Read synchronized fixed and wrist RGB images. Start here.",
    "plan_pick_place": "Find the selected object and the basket in the latest RGB frame and plan without moving the arm.",
    "execute": "Execute once. Required: plan_id with the exact string returned by plan_pick_place.",
    "verify_place": "Read fresh RGB images. Required: the same plan_id used by execute.",
}
TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": TOOL_DESCRIPTIONS[name],
     "parameters": {"type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False}}
    for name, properties in TOOL_PARAMETERS.items()
]


def configure_environment(libero: Path) -> None:
    config = libero / ".libero-local"
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config))
    os.environ.setdefault("MPLCONFIGDIR", str(config / "mpl"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
    os.environ.setdefault("CAPTAIN_LAZY_RENDER", "1")
    mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    if os.environ["MUJOCO_GL"] == "egl" and mesa.is_file():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(mesa))
    sys.path.insert(0, str(libero))
    sys.path.insert(0, str(CAPTAIN))


@lru_cache(maxsize=16)
def load_initial_states(suite, task_index: int):
    import numpy as np
    from numpy._core.multiarray import _reconstruct
    import torch

    # Official LIBERO initial-state archives predate the NumPy module rename.
    numpy_types = [np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
                   (_reconstruct, "numpy.core.multiarray._reconstruct")]
    with torch.serialization.safe_globals(numpy_types):
        return suite.get_task_init_states(task_index)


class LiberoTools:
    def __init__(self, backend, output: Path, floor_z: float):
        self.backend = backend
        self.output = output
        self.floor_z = floor_z
        self.frame = None
        self.plan = None
        self.plan_number = 0
        self.started = False
        self.motion_complete = False

    @staticmethod
    def _value(result):
        if not result.success:
            raise RuntimeError(result.message or "LIBERO controller rejected the operation")
        return result.value

    def _capture(self):
        data = self._value(self.backend.get_observation_data())
        cameras = {camera.camera_name: Path(camera.rgb_path) for camera in data.cameras}
        names = {name: str(path.relative_to(self.output)) for name, path in cameras.items()}
        frame_id = cameras["agentview"].stem.removesuffix("_agentview")
        return data, {"frame_id": frame_id, "images": names}, cameras

    def observe(self):
        data, receipt, cameras = self._capture()
        self.frame = (receipt["frame_id"], data)
        if not self.started:
            self.plan = None
        return receipt, cameras

    def plan_pick_place(self, frame_id: str, target: str, destination: str):
        from libero_adapter.rgb import plan_from_rgb

        if self.started:
            raise RuntimeError("this evaluation episode permits one execution")
        if self.frame is None or self.frame[0] != frame_id:
            raise RuntimeError("use the latest frame_id from observe")
        if target not in TARGETS or destination != "basket":
            raise RuntimeError("use a listed target object and destination basket")
        self.plan = plan_from_rgb(f"pick_place_{self.plan_number + 1}", self.frame[1], self.floor_z, target=target)
        self.plan_number += 1
        self.frame = None
        return {"plan_id": self.plan.plan_id, "frame_id": frame_id, "target": target,
                "source": "RGB and camera calibration",
                "pick_xy_m": [round(value, 4) for value in self.plan.pick_xy],
                "place_xy_m": [round(value, 4) for value in self.plan.place_xy],
                "motion": ["approach", "descend", "close", "lift", "transfer", "release"]}

    def execute(self, plan_id: str):
        if self.plan is None or plan_id != self.plan.plan_id or self.started:
            raise RuntimeError("use the unused plan_id returned by plan_pick_place")
        self.started = True
        for name, position, quaternion in self.plan.waypoints:
            if name == "close":
                self._value(self.backend.close_gripper())
            elif name == "open":
                self._value(self.backend.open_gripper())
            else:
                result = self.backend.move_to_pose(position, quaternion)
                if not result.success:
                    raise RuntimeError(f"{name}: {result.message}")
        self.motion_complete = True
        return {"plan_id": plan_id, "motion_complete": True}

    def verify_place(self, plan_id: str):
        if self.plan is None or plan_id != self.plan.plan_id or not self.motion_complete:
            raise RuntimeError("execute the plan before verification")
        _, receipt, cameras = self._capture()
        return {"plan_id": plan_id, "visual_observation": "inspect the final RGB views", **receipt}, cameras

    def call(self, name: str, args: object):
        images = {}
        try:
            properties = TOOL_PARAMETERS.get(name)
            if properties is None:
                raise RuntimeError("tool is outside the allowed schema")
            if not isinstance(args, dict) or set(args) != set(properties):
                required = ", ".join(properties) or "none"
                raise RuntimeError(f"{name} requires exactly these arguments: {required}")
            if not all(isinstance(value, str) for value in args.values()):
                raise RuntimeError("all tool arguments must be strings")
            value = getattr(self, name)(**args)
            if isinstance(value, tuple):
                value, images = value
            receipt = {"ok": True, **value}
        except RuntimeError as error:
            receipt = {"ok": False, "error": str(error)}
        encoded = json.dumps(receipt, ensure_ascii=False)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps({"tool": name, "arguments": args, "receipt": receipt}, ensure_ascii=False) + "\n")
        print(f"{name} → {encoded}", flush=True)
        return receipt, images


def run_script(tools: LiberoTools, target: str) -> None:
    previous, _ = tools.call("observe", {})
    if not previous["ok"]:
        raise RuntimeError(previous["error"])
    previous, _ = tools.call("plan_pick_place",
                             {"frame_id": previous["frame_id"], "target": target, "destination": "basket"})
    if not previous["ok"]:
        raise RuntimeError(previous["error"])
    plan_id = previous["plan_id"]
    for name in ("execute", "verify_place"):
        previous, _ = tools.call(name, {"plan_id": plan_id})
        if not previous["ok"]:
            raise RuntimeError(previous["error"])


def run_agent(tools: LiberoTools, instruction: str, endpoint: str, model: str, *, describe_result: bool) -> str | None:
    from captain_harness.react.chat_llm import ChatCompletionsClient

    def record_model_event(event: dict) -> None:
        with (tools.output / "model.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(event, ensure_ascii=False) + "\n")

    client = ChatCompletionsClient(endpoint=endpoint, max_output_tokens=256, temperature=0,
                                  timeout_s=85, connect_timeout_s=4, stream_max_retries=2,
                                  all_tools=True, hoist_tool_images=True,
                                  extra_body={"reasoning_effort": "none"},
                                  event_sink=record_model_event)
    system = ("Control the single Franka arm by calling one defined tool per turn. Start with observe. "
              "Then use plan_pick_place with the exact observed frame_id; select the target named in the user's instruction "
              "using the listed identifier and set destination to basket. "
              "The execute and verify_place tools each REQUIRE a plan_id argument. "
              "Copy the exact plan_id value from the plan_pick_place receipt. Execute once, then verify_place. "
              "If a tool receipt reports an invalid argument, correct the argument and call it again. "
              "Call verify_place once after execution. "
              "If asked to describe the outcome, inspect the fresh verification images and answer in text.")
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": instruction}]}]
    verified = False
    for _ in range(10):
        reply = client.respond({"model": model, "instructions": system, "input": items,
                                "tools": TOOL_SCHEMAS, "parallel_tool_calls": False})
        calls = [item for item in reply.items if item.get("type") == "function_call"]
        if verified:
            parts = [part["text"] for item in reply.items if item.get("type") == "message"
                     for part in item.get("content", []) if part.get("type") == "output_text"]
            return "\n".join(parts) or None
        if len(calls) != 1:
            items.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Continue with exactly one listed tool call and all required arguments."}]})
            continue
        call = calls[0]
        try:
            args = json.loads(call.get("arguments", "{}"))
        except (ValueError, TypeError):
            args = None
        name = call["name"]
        receipt, images = tools.call(name, args)
        if name == "execute" and tools.started and not receipt["ok"]:
            raise RuntimeError(receipt["error"])
        if name == "verify_place" and receipt["ok"] and not describe_result:
            return None
        if name == "plan_pick_place" and receipt["ok"]:
            for item in items:
                if item.get("type") == "function_call_output":
                    item["output"] = [part for part in item["output"] if part["type"] == "input_text"]
        content = [{"type": "input_text", "text": json.dumps(receipt, ensure_ascii=False)}]
        for camera, image in images.items():
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.extend([{"type": "input_text", "text": f"{camera} RGB"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}])
        items.extend((call, {"type": "function_call_output", "call_id": call["call_id"], "output": content}))
        verified |= name == "verify_place" and receipt["ok"]
    if not verified:
        raise RuntimeError("the model did not reach verification within the tool budget")
    return None


@lru_cache(maxsize=1)
def environment_context(libero_root: Path):
    if not (libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO source not found at {libero_root}")
    configure_environment(libero_root)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    from simulation.replay import overlay_wrist_view

    class ReplayEnv(OffScreenRenderEnv):
        def __init__(self, **kwargs):
            self.replay_frames = []
            super().__init__(**kwargs)

        def record_frame(self):
            global_rgb = self.sim.render(width=400, height=300, camera_name="agentview", depth=False)
            wrist_rgb = self.sim.render(width=160, height=120, camera_name="robot0_eye_in_hand", depth=False)
            overview = Image.fromarray(global_rgb[::-1])
            overlay_wrist_view(overview, Image.fromarray(wrist_rgb[::-1]))
            self.replay_frames.append(overview)

        def step(self, action):
            observation = super().step(action)
            if self.replay_frames:
                self.record_frame()
            return observation

        def save_replay(self, path: Path):
            if len(self.replay_frames) < 2:
                return
            interval_ms = 1000 // CONTROL_HZ
            durations = [interval_ms] * (len(self.replay_frames) - 1) + [650]
            self.replay_frames[0].save(path, save_all=True, append_images=self.replay_frames[1:],
                                       duration=durations, loop=0)

    return benchmark.get_benchmark("libero_object")(), ReplayEnv


def evaluate_case(*, task_id: int, seed: int, output: Path, mode: str, endpoint: str, model: str,
                  record_replay: bool = False, libero_root: Path = DEFAULT_LIBERO) -> dict:
    suite, replay_env = environment_context(libero_root.resolve())
    if not 0 <= task_id < suite.get_num_tasks():
        raise ValueError(f"Invalid task index: {task_id}")
    task = suite.get_task(task_id)
    states = load_initial_states(suite, task_id)
    if not 0 <= seed < len(states):
        raise ValueError(f"Invalid official initial-state index: {seed}")

    from captain_harness.adapters.libero.backend import LiberoActionBackend, reset_to_seeded_init_state
    from captain_harness.adapters.libero.scoring import HostOnlyFinalScorer
    from captain_harness.react.chat_llm import ModelRequestError

    target = task.name.removeprefix("pick_up_the_").removesuffix("_and_place_it_in_the_basket")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = replay_env(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                    camera_heights=360, camera_widths=480, camera_depths=False,
                    control_freq=CONTROL_HZ, ignore_done=True, horizon=HORIZON + 5)
    outcome = {"suite": "libero_object", "task_index": task_id, "instruction": task.language,
               "seed": seed, "mode": mode, "model": model if mode == "agent" else None}
    try:
        initial = reset_to_seeded_init_state(env, states, seed=seed)
        backend = LiberoActionBackend(env, observation_dir=output / "observations",
                                      initial_observation=initial.observation, horizon=HORIZON,
                                      control_frequency=CONTROL_HZ, image_size=(480, 360),
                                      controller="OSC_POSE", gripper_steps=20,
                                      osc_position_tolerance_m=0.0025)
        # The fixed support surface is calibration, analogous to a marked floor plane in a workcell.
        sim_model, data = env.sim.model, env.sim.data
        floor_z = (data.geom_xpos[sim_model.geom_name2id("floor")][2]
                   - data.xpos[sim_model.body_name2id("robot0_base")][2])
        scorer = HostOnlyFinalScorer(env.check_success)
        tools = LiberoTools(backend, output, float(floor_z))
        if record_replay:
            env.record_frame()
        try:
            if mode == "agent":
                outcome["agent_visual_report"] = run_agent(tools, task.language, endpoint, model,
                                                            describe_result=record_replay)
            else:
                run_script(tools, target)
        except ModelRequestError:
            raise
        except RuntimeError as error:
            outcome["error"] = str(error)
        outcome["official_success"] = scorer.score_after_loop(loop_ended=True)
        outcome["episode_completed"] = "error" not in outcome
        outcome["control_steps"] = backend.simulation_steps
        if record_replay:
            env.save_replay(output / "pick-place.gif")
    finally:
        env.close()
    (output / "result.json").write_text(json.dumps(outcome, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scripted", "agent"))
    parser.add_argument("--libero", type=Path, default=DEFAULT_LIBERO)
    parser.add_argument("--task", type=int, default=TASK_INDEX, choices=range(len(TARGETS)))
    parser.add_argument("--seed", type=int, default=0, help="official initial-state index")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--no-replay", action="store_true", help="skip recording the continuous GIF")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or ROOT / "runs" / f"libero-{args.mode}-{stamp}"
    outcome = evaluate_case(task_id=args.task, seed=args.seed, output=output, mode=args.mode,
                            endpoint=args.endpoint, model=args.model, record_replay=not args.no_replay,
                            libero_root=args.libero)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return 0 if outcome.get("official_success") and outcome.get("episode_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

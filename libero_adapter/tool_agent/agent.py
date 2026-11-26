"""Run a Qwen-controlled LIBERO-Object episode with independently callable tools."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

from PIL import Image

from libero_adapter.run import CONTROL_HZ, DEFAULT_LIBERO, HORIZON, ROOT, environment_context, load_initial_states
from libero_adapter.skills.observation import observe
from .live_monitor import LiveMonitor
from .session import TOOL_SCHEMAS, ToolSession


TOOL_BUDGET = 44
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
SYSTEM = (
    "Control the simulated Franka using one listed tool per response. Choose each next action from the instructions "
    "and actual receipts. Keep the requested target. Use the initial frame_id while the scene has not moved. "
    "You may use find_ridge for a carton or change wrist view and reconstruct its visible shape if needed. "
    "If you have a usable shape_id, pass it when proposing grasps. Prefer checked grasps with clearance. "
    "Include a destination_id for a basket task. Check grip when retention is in doubt. "
    "A stalled descent can move partway and obscure the target. Use retreat_arm when needed, refresh RGB, "
    "then change the grasp height or angle before retrying. After release, look at fresh inspect_place RGB "
    "and verify the target inside the basket. If the view is unclear, change the view before repeating. "
    "Use finish_task with your visual evidence to claim placed or explain what blocks you."
)


def image_parts(images: dict[str, Path]) -> list[dict]:
    parts = []
    for label, path in images.items():
        with Image.open(path) as original:
            image = original.convert("RGB")
        image.thumbnail((384, 288))
        with BytesIO() as encoded:
            image.save(encoded, format="JPEG", quality=85)
            payload = base64.b64encode(encoded.getvalue()).decode("ascii")
        parts.extend([{"type": "input_text", "text": label},
                      {"type": "input_image", "image_url": f"data:image/jpeg;base64,{payload}"}])
    return parts


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_policy(session: ToolSession, instruction: str, endpoint: str, model: str) -> tuple[dict, int]:
    from captain_harness.react.chat_llm import ChatCompletionsClient

    def record_model(event: dict) -> None:
        with (session.output / "model.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    client = ChatCompletionsClient(endpoint=endpoint, max_output_tokens=160, temperature=0,
                                  timeout_s=85, connect_timeout_s=4, stream_max_retries=2,
                                  all_tools=True, hoist_tool_images=True, max_images=2,
                                  extra_body={"reasoning_effort": "none"}, event_sink=record_model)
    initial, images = observe(session, "overview")
    write_json(session.output / "initial_observation.json", initial)
    items = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": f"{instruction}\nInitial observation: {json.dumps(initial)}"},
        *image_parts(images)]}]
    last_failure = None
    rejected_call = None
    repeat_rejections = 0
    for turn in range(1, TOOL_BUDGET + 1):
        reply = client.respond({"model": model, "instructions": SYSTEM, "input": items,
                                "tools": TOOL_SCHEMAS, "parallel_tool_calls": False})
        calls = [item for item in reply.items if item.get("type") == "function_call"]
        if len(calls) != 1:
            items.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Choose exactly one listed tool. To end, call finish_task."}]})
            continue
        call = calls[0]
        try:
            arguments = json.loads(call.get("arguments", "{}"))
        except (ValueError, TypeError):
            arguments = None
        receipt, images = session.call(call["name"], arguments, turn=turn,
                                       model_call={"response_id": reply.response_id, "call_id": call["call_id"]})
        if call["name"] == "finish_task" and receipt["ok"]:
            return receipt, turn
        if not receipt["ok"]:
            last_failure = receipt.get("reason")
        content = [{"type": "input_text", "text": json.dumps(receipt, ensure_ascii=False)}, *image_parts(images)]
        items.extend((call, {"type": "function_call_output", "call_id": call["call_id"], "output": content}))
        current_rejection = (call["name"], arguments) if not receipt["ok"] else None
        if current_rejection is not None and current_rejection == rejected_call:
            repeat_rejections += 1
            if repeat_rejections >= 4:
                reason = f"Repeated rejected tool request: {call['name']}. {last_failure}"
                return {"status": "repeated_tool_rejection", "outcome": "blocked", "reason": reason}, turn
            items.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "The same tool request was rejected twice. Address the reason in its receipt and choose a different request."}]})
        else:
            repeat_rejections = 0
        rejected_call = current_rejection
    reason = f"The agent reached the {TOOL_BUDGET}-turn budget."
    if last_failure:
        reason += f" Last rejected action: {last_failure}"
    return {"status": "turn_budget", "outcome": "blocked", "reason": reason}, TOOL_BUDGET


def evaluate(*, task_id: int, seed: int, output: Path, model: str, endpoint: str,
             libero_root: Path, record_replay: bool, instruction: str | None = None,
             live_monitor: bool = False) -> dict:
    start = monotonic()
    suite, replay_env = environment_context(libero_root.resolve())
    task = suite.get_task(task_id)
    states = load_initial_states(suite, task_id)
    if not 0 <= seed < len(states):
        raise ValueError(f"Invalid official LIBERO initial-state index: {seed}")

    from captain_harness.adapters.libero.backend import LiberoActionBackend, reset_to_seeded_init_state
    from captain_harness.adapters.libero.scoring import HostOnlyFinalScorer

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "tools.json", TOOL_SCHEMAS)
    env = replay_env(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                     camera_heights=360, camera_widths=480, camera_depths=False,
                     control_freq=CONTROL_HZ, ignore_done=True, horizon=HORIZON + 5)
    outcome = {"suite": "libero_object", "task_index": task_id, "seed": seed, "mode": "tool_agent",
               "instruction": task.language, "model": model, "control_budget": HORIZON,
               "tool_turn_budget": TOOL_BUDGET, "live_monitor": live_monitor}
    if instruction is not None:
        outcome["custom_instruction"] = instruction
    monitor = None
    try:
        initial = reset_to_seeded_init_state(env, states, seed=seed)
        backend = LiberoActionBackend(env, observation_dir=output / "observations",
                                      initial_observation=initial.observation, horizon=HORIZON,
                                      control_frequency=CONTROL_HZ, image_size=(480, 360),
                                      controller="OSC_POSE", gripper_steps=20,
                                      osc_position_tolerance_m=0.0025)
        sim_model, data = env.sim.model, env.sim.data
        floor_z = (data.geom_xpos[sim_model.geom_name2id("floor")][2]
                   - data.xpos[sim_model.body_name2id("robot0_base")][2])
        target = task.name.removeprefix("pick_up_the_").removesuffix("_and_place_it_in_the_basket")
        if live_monitor:
            monitor = LiveMonitor(model=model, endpoint=endpoint, output=output,
                                  instruction=instruction or task.language)
        session = ToolSession(backend, output, float(floor_z), target, monitor=monitor)
        scorer = HostOnlyFinalScorer(env.check_success)
        if record_replay:
            env.record_frame()
        agent_report, turns = run_policy(session, instruction or task.language, endpoint, model)
        outcome.update(agent_report=agent_report, model_turns=turns, control_steps=backend.simulation_steps,
                       episode_completed=agent_report.get("status") == "report_accepted",
                       agent_visual_success=agent_report.get("outcome") == "placed",
                       official_success=scorer.score_after_loop(loop_ended=True))
        if not outcome["official_success"]:
            outcome["failure_reason"] = agent_report.get("reason")
        if record_replay:
            replay_path = output / "pick-place.gif"
            env.save_replay(replay_path)
            if not replay_path.exists():
                env.replay_frames[0].save(replay_path, duration=650)
            outcome["replay_frames"] = len(env.replay_frames)
    finally:
        if monitor is not None:
            monitor.close()
        env.close()
    outcome["elapsed_seconds"] = round(monotonic() - start, 3)
    write_json(output / "result.json", outcome)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0, choices=range(10))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--libero", type=Path, default=DEFAULT_LIBERO)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--instruction", help="Optional instruction for a separate targeted demonstration.")
    parser.add_argument("--live-monitor", action="store_true", help="Run Qwen's dual RGB motion monitor at the control clock.")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or ROOT / "runs" / f"libero-tool-agent-{stamp}"
    result = evaluate(task_id=args.task, seed=args.seed, output=output, model=args.model, endpoint=args.endpoint,
                      libero_root=args.libero, record_replay=not args.no_replay, instruction=args.instruction,
                      live_monitor=args.live_monitor)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["official_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

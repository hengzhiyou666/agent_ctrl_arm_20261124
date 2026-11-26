"""Run a Qwen-directed LIBERO episode using independently callable skills."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path

from libero_adapter.run import CONTROL_HZ, DEFAULT_LIBERO, ROOT, environment_context, load_initial_states
from .session import SkillSession, TOOL_SCHEMAS


CONTROL_BUDGET = 400
TOOL_BUDGET = 22
SYSTEM = (
    "Use the registered robot skills to achieve the user's goal in the simulated scene. "
    "On each turn decide which one skill is useful based on the current evidence and the last tool receipt. "
    "Copy exact frame and plan IDs from receipts. Pay attention to each tool's prerequisites. "
    "An unsuccessful motion can stop partway. Use its receipt to decide whether to look again, "
    "try a changed plan, or stop. Repeat a failed action only when you have changed its preconditions. "
    "Use finish_task to report the result. A placement claim needs a successful release command over the basket "
    "and supporting visual evidence from inspect_place. If you cannot proceed, report blocked."
)


def run_policy(session: SkillSession, instruction: str, endpoint: str, model: str) -> dict:
    from captain_harness.react.chat_llm import ChatCompletionsClient

    def record_model(event: dict) -> None:
        with (session.output / "model.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    client = ChatCompletionsClient(endpoint=endpoint, max_output_tokens=180, temperature=0,
                                  timeout_s=85, connect_timeout_s=4, stream_max_retries=2,
                                  all_tools=True, hoist_tool_images=True, max_images=3,
                                  extra_body={"reasoning_effort": "none"}, event_sink=record_model)
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": instruction}]}]
    for _ in range(TOOL_BUDGET):
        reply = client.respond({"model": model, "instructions": SYSTEM, "input": items,
                                "tools": TOOL_SCHEMAS, "parallel_tool_calls": False})
        calls = [item for item in reply.items if item.get("type") == "function_call"]
        if len(calls) != 1:
            items.append({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Choose one registered skill. To report a result, call finish_task."}]})
            continue
        call = calls[0]
        try:
            arguments = json.loads(call.get("arguments", "{}"))
        except (ValueError, TypeError):
            arguments = None
        receipt, images = session.call(call["name"], arguments)
        if call["name"] == "finish_task" and receipt["ok"]:
            return receipt
        content = [{"type": "input_text", "text": json.dumps(receipt, ensure_ascii=False)}]
        for label, path in images.items():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.extend([{"type": "input_text", "text": f"{label} RGB"},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}])
        items.extend((call, {"type": "function_call_output", "call_id": call["call_id"], "output": content}))
    raise RuntimeError(f"The agent reached the {TOOL_BUDGET}-turn budget.")


def evaluate(*, task_id: int, seed: int, output: Path, model: str, endpoint: str,
             libero_root: Path, record_replay: bool) -> dict:
    suite, replay_env = environment_context(libero_root.resolve())
    task = suite.get_task(task_id)
    states = load_initial_states(suite, task_id)
    if not 0 <= seed < len(states):
        raise ValueError(f"Invalid LIBERO initial-state index: {seed}")

    from captain_harness.adapters.libero.backend import LiberoActionBackend, reset_to_seeded_init_state
    from captain_harness.adapters.libero.scoring import HostOnlyFinalScorer

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = replay_env(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                     camera_heights=360, camera_widths=480, camera_depths=False,
                     control_freq=CONTROL_HZ, ignore_done=True, horizon=CONTROL_BUDGET + 5)
    outcome = {"suite": "libero_object", "task_index": task_id, "instruction": task.language,
               "seed": seed, "mode": "skill_agent", "model": model, "control_budget": CONTROL_BUDGET,
               "tool_turn_budget": TOOL_BUDGET}
    try:
        initial = reset_to_seeded_init_state(env, states, seed=seed)
        backend = LiberoActionBackend(env, observation_dir=output / "observations",
                                      initial_observation=initial.observation, horizon=CONTROL_BUDGET,
                                      control_frequency=CONTROL_HZ, image_size=(480, 360),
                                      controller="OSC_POSE", gripper_steps=20,
                                      osc_position_tolerance_m=0.0025)
        sim_model, data = env.sim.model, env.sim.data
        floor_z = (data.geom_xpos[sim_model.geom_name2id("floor")][2]
                   - data.xpos[sim_model.body_name2id("robot0_base")][2])
        session = SkillSession(backend, output, float(floor_z))
        scorer = HostOnlyFinalScorer(env.check_success)
        if record_replay:
            env.record_frame()
        try:
            outcome["agent_report"] = run_policy(session, task.language, endpoint, model)
        except RuntimeError as error:
            outcome["error"] = str(error)
        outcome["official_success"] = scorer.score_after_loop(loop_ended=True)
        outcome["control_steps"] = backend.simulation_steps
        if record_replay:
            env.save_replay(output / "pick-place.gif")
    finally:
        env.close()
    (output / "result.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, default=0, choices=range(10))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--libero", type=Path, default=DEFAULT_LIBERO)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or ROOT / "runs" / f"libero-skill-agent-{stamp}"
    result = evaluate(task_id=args.task, seed=args.seed, output=output, model=args.model, endpoint=args.endpoint,
                      libero_root=args.libero, record_replay=not args.no_replay)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["official_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

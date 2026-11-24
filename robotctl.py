"""Run one local Franka ball grasp and save actual simulator camera frames."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CAPTAIN = ROOT / "CaPtain-Harness" / "src"


def run_script(tools):
    steps = (
        ("observe", lambda value: {}),
        ("plan_grasp", lambda value: {"frame_id": value["frame_id"], "target": "orange_ball"}),
        ("execute", lambda value: {"plan_id": value["plan_id"]}),
        ("verify_grasp", lambda value: {"plan_id": value["plan_id"]}),
    )
    receipt = {}
    for tool, args in steps:
        receipt, _ = tools.call(tool, args(receipt))
        if not receipt["ok"]:
            raise RuntimeError(receipt["error"])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="运行一局 Franka 仿真抓球并保存回放")
    parser.add_argument("mode", choices=("agent", "scripted"), help="让 Qwen 选工具，或按固定顺序验收机械臂")
    parser.add_argument("--output", type=Path, help="保存回放和结果的目录")
    parser.add_argument("--ball-xy", type=float, nargs=2, default=(0.49, 0.08), metavar=("X", "Y"),
                        help="在工作台上换一个小球起点，单位是米")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1", help="本地模型接口")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama 模型名")
    args = parser.parse_args()

    import math
    x, y = args.ball_xy
    if not (math.isfinite(x) and math.isfinite(y) and 0.35 <= x <= 0.63 and -0.20 <= y <= 0.20):
        parser.error("小球起点应位于本实验的工作台范围内")

    from harness.grasp_tools import GraspTools
    from simulation.franka import FrankaBall, GraspScorer, VIDEO_MS

    if args.mode == "agent":
        if not (CAPTAIN / "captain_harness" / "react" / "chat_llm.py").is_file():
            parser.error("缺少 CaPtain-Harness，请按 README 取回它的模型接口代码")
        sys.path.insert(0, str(CAPTAIN))
        from harness.agent import run_agent

    out = (args.output or ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.mode)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"保存到 {out}", flush=True)
    sim = FrankaBall(tuple(args.ball_xy))
    scorer = GraspScorer(sim)
    tools = GraspTools(sim, out)
    outcome = {"mode": args.mode, "model": args.model if args.mode == "agent" else None}
    try:
        sim.rgb().save(out / "start.png")
        receipt = run_agent(tools, args.endpoint, args.model) if args.mode == "agent" else run_script(tools)
        outcome["visible_verification"] = receipt
    except RuntimeError as error:
        outcome["error"] = str(error)
        print(f"本局结束：{error}", file=sys.stderr, flush=True)
    finally:
        outcome["physical_score"] = scorer.evaluate()
        outcome["success"] = bool(outcome["physical_score"]["success"] and
                                  outcome.get("visible_verification", {}).get("held") is True)
        sim.rgb().save(out / "result.png")
        sim.rgb("detail").save(out / "detail.png")
        if sim.frames:
            sim.frames[0].save(out / "grasp.gif", save_all=True, append_images=sim.frames[1:],
                               duration=VIDEO_MS, loop=0)
        (out / "result.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        sim.close()
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    return 0 if outcome["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

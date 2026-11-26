"""Run one local Franka ball grasp and save actual simulator camera frames."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CAPTAIN = ROOT / "CaPtain-Harness" / "src"


def run_script(tools, target: str, lift_cm: float):
    steps = (
        ("observe", lambda value: {}),
        ("plan_grasp", lambda value: {"frame_id": value["frame_id"], "target": target, "lift_cm": lift_cm}),
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
                        help="设置橙球起点，单位是米")
    parser.add_argument("--ball-radius", type=float, metavar="METERS",
                        help="测试用，改变仿真中的橙球半径")
    parser.add_argument("--green-xy", type=float, nargs=2, default=(0.46, -0.11), metavar=("X", "Y"),
                        help="设置绿球起点，单位是米")
    parser.add_argument("--green-radius", type=float, metavar="METERS", help="测试用，改变仿真中的绿球半径")
    parser.add_argument("--single-ball", action="store_true", help="运行单个橙球的旧场景")
    parser.add_argument("--target", choices=("green_ball", "orange_ball"), help="设置任务目标和独立评分的球")
    parser.add_argument("--lift-cm", type=float, help="设置任务和独立评分的目标抬升高度")
    parser.add_argument("--instruction", help="改写给模型的任务句子，评分目标由 --target 和 --lift-cm 指定")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1", help="本地模型接口")
    parser.add_argument("--model", default="qwen3.5:4b", help="Ollama 模型名")
    args = parser.parse_args()

    from harness.grasp_tools import GraspTools
    from simulation.franka import DEFAULT_LIFT, MAX_LIFT, MIN_LIFT, FrankaBall, GraspScorer, VIDEO_MS

    target = args.target or ("orange_ball" if args.single_ball else "green_ball")
    lift_cm = args.lift_cm if args.lift_cm is not None else (DEFAULT_LIFT * 100 if args.single_ball else 20.0)
    lift_m = lift_cm / 100
    if args.single_ball and target == "green_ball":
        parser.error("单球场景只有橙球")
    if not MIN_LIFT <= lift_m <= MAX_LIFT:
        parser.error(f"抬升高度应在 {MIN_LIFT * 100:g} 至 {MAX_LIFT * 100:g} 厘米内")
    for x, y in (args.ball_xy,) if args.single_ball else (args.ball_xy, args.green_xy):
        if not (0.35 <= x <= 0.63 and -0.20 <= y <= 0.20):
            parser.error("小球起点应位于本实验的工作台范围内")
    if not args.single_ball and math.dist(args.ball_xy, args.green_xy) < 0.10:
        parser.error("两个球的起点至少相距 10 厘米")
    for radius in (args.ball_radius, args.green_radius):
        if radius is not None and not 0.018 <= radius <= 0.035:
            parser.error("仿真球半径应在 0.018 至 0.035 米内")
    color = target.split("_")[0]
    label = {"green": "绿色", "orange": "橙色"}[color]
    instruction = args.instruction or f"用爪子将{label}的小球垂直抬高{lift_cm:g}cm"

    if args.mode == "agent":
        if not (CAPTAIN / "captain_harness" / "react" / "chat_llm.py").is_file():
            parser.error("缺少 CaPtain-Harness，请按 README 取回它的模型接口代码")
        sys.path.insert(0, str(CAPTAIN))
        from harness.agent import run_agent

    out = (args.output or ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.mode)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"保存到 {out}", flush=True)
    green_xy = None if args.single_ball else tuple(args.green_xy)
    sim = FrankaBall(tuple(args.ball_xy), ball_radius=args.ball_radius, green_xy=green_xy,
                     green_radius=args.green_radius)
    scorer = GraspScorer(sim, color=color, lift_m=lift_m)
    tools = GraspTools(sim, out)
    outcome = {"mode": args.mode, "model": args.model if args.mode == "agent" else None,
               "instruction": instruction, "expected_target": target, "expected_lift_cm": lift_cm}
    try:
        sim.rgb().save(out / "start.png")
        receipt = (run_agent(tools, args.endpoint, args.model, instruction) if args.mode == "agent"
                   else run_script(tools, target, lift_cm))
        outcome["visible_verification"] = receipt
    except RuntimeError as error:
        outcome["error"] = str(error)
        print(f"本局结束：{error}", file=sys.stderr, flush=True)
    finally:
        outcome["physical_score"] = scorer.evaluate()
        visible = outcome.get("visible_verification", {})
        outcome["success"] = bool(outcome["physical_score"]["success"] and visible.get("target") == target
                                  and visible.get("requested_lift_cm") == lift_cm
                                  and all(visible.get(key) is True for key in ("held", "height_reached", "vertical")))
        sim.rgb().save(out / "result.png")
        sim.rgb("detail").save(out / "detail.png")
        sim.rgb("wrist").save(out / "wrist.png")
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

"""The only robot operations exposed to the policy model."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from simulation.franka import FrankaBall, GraspError, GraspTrajectory

TOOL_PARAMETERS = {
    "observe": {},
    "plan_grasp": {"frame_id": {"type": "string"},
                   "target": {"type": "string", "enum": ["orange_ball"]}},
    "execute": {"plan_id": {"type": "string"}},
    "verify_grasp": {"plan_id": {"type": "string"}},
}
TOOL_DESCRIPTIONS = {
    "observe": "Read the current RGB camera image and get its frame_id. Call this first.",
    "plan_grasp": "Plan a grasp for orange_ball using the latest observed frame_id. The robot stays still.",
    "execute": "Run a prepared grasp plan by plan_id. The robot reaches, closes, lifts, and holds.",
    "verify_grasp": "Check the current camera and gripper after executing plan_id.",
}
TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": TOOL_DESCRIPTIONS[name],
     "parameters": {"type": "object", "properties": parameters,
                    "required": list(parameters), "additionalProperties": False}}
    for name, parameters in TOOL_PARAMETERS.items()
]


class GraspTools:
    def __init__(self, sim: FrankaBall, out: Path):
        self.sim = sim
        self.out = out
        self.frame_number = 0
        self.latest_frame = None
        self.plans: dict[str, GraspTrajectory] = {}
        self.executed_plan: str | None = None

    def observe(self) -> tuple[dict, Path]:
        frame = self.sim.observe("detail")
        self.frame_number += 1
        frame_id = f"f{self.frame_number}"
        image = self.out / f"{frame_id}.jpg"
        frame.image.save(image, quality=89)
        self.latest_frame = (frame_id, frame, self.sim.steps)
        return {"frame_id": frame_id, "camera": frame.camera, "image": image.name}, image

    def plan_grasp(self, frame_id: str, target: str) -> dict:
        if target != "orange_ball":
            raise GraspError("这个场景的可抓取对象是 orange_ball")
        if self.latest_frame is None or frame_id != self.latest_frame[0] or self.latest_frame[2] != self.sim.steps:
            raise GraspError("观测已过期，请先调用 observe 获取新的 frame_id")
        plan = self.sim.plan_grasp(self.latest_frame[1].ball_center())
        plan_id = f"p{len(self.plans) + 1}"
        self.plans[plan_id] = plan
        return {"plan_id": plan_id, "frame_id": frame_id, "motion": "接近、下降、闭合、抬升、保持"}

    def execute(self, plan_id: str) -> dict:
        if plan_id not in self.plans or plan_id != f"p{len(self.plans)}":
            raise GraspError("使用最近一次 plan_grasp 返回的 plan_id")
        if self.executed_plan == plan_id:
            raise GraspError("该计划已经执行，请直接调用 verify_grasp")
        self.executed_plan = plan_id
        self.sim.execute_grasp(self.plans[plan_id])
        return {"plan_id": plan_id, "motion_complete": True, "next": "verify_grasp"}

    def verify_grasp(self, plan_id: str) -> tuple[dict, Path]:
        if self.executed_plan != plan_id:
            raise GraspError("请先执行这个 plan_id")
        result = self.sim.verify_grasp(self.plans[plan_id])
        image = self.out / "verification.jpg"
        self.sim.rgb(result.get("camera", "detail")).save(image, quality=92)
        return {**result, "plan_id": plan_id, "image": image.name}, image

    def call(self, name: str, args: object) -> tuple[dict, Path | None]:
        image = None
        try:
            parameters = TOOL_PARAMETERS.get(name)
            if parameters is None or not isinstance(args, dict) or set(args) != set(parameters):
                raise GraspError("工具名或参数不在本实验的白名单里")
            if any(not isinstance(value, str) for value in args.values()):
                raise GraspError("工具参数必须是字符串")
            value = getattr(self, name)(**args)
            if isinstance(value, tuple):
                value, image = value
            result = {"ok": True, **value}
        except GraspError as error:
            result = {"ok": False, "error": str(error)}
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tool": name, "arguments": args, "receipt": result,
                                     "simulation_step": self.sim.steps}, ensure_ascii=False) + "\n")
        print(f"{name} → {json.dumps(result, ensure_ascii=False)}", flush=True)
        return result, image


def receipt_content(receipt: dict, image: Path | None) -> list[dict]:
    content = [{"type": "input_text", "text": json.dumps(receipt, ensure_ascii=False)}]
    if image is not None:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded})
    return content

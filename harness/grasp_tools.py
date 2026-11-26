"""The only robot operations exposed to the policy model."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path

from simulation.ball_vision import CameraFrame, estimate_ball
from simulation.franka import FrankaBall, GraspError, GraspTrajectory

TARGET_COLORS = {"orange_ball": "orange", "green_ball": "green"}
MIN_LIFT_CM = 8
MAX_LIFT_CM = 25

TOOL_PARAMETERS = {
    "observe": {},
    "plan_grasp": {
        "frame_id": {"type": "string"},
        "target": {"type": "string", "enum": list(TARGET_COLORS)},
        "lift_cm": {"type": "number", "minimum": MIN_LIFT_CM, "maximum": MAX_LIFT_CM,
                    "description": "Requested vertical rise from the ball's starting position, in centimeters."},
    },
    "execute": {"plan_id": {"type": "string"}},
    "verify_grasp": {"plan_id": {"type": "string"}},
}
TOOL_DESCRIPTIONS = {
    "observe": "Read synchronized overview and wrist RGB images with one shared frame_id. Call this first.",
    "plan_grasp": "Estimate the selected ball from both RGB views of the latest frame_id, and plan a grasp and vertical lift without moving the arm.",
    "execute": "Run a prepared grasp plan by plan_id. The robot reaches, closes, lifts, and holds.",
    "verify_grasp": "Check whether the target ball is held and has reached the requested lift using fresh RGB images and gripper state.",
}
TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": TOOL_DESCRIPTIONS[name],
     "parameters": {"type": "object", "properties": parameters,
                    "required": list(parameters), "additionalProperties": False}}
    for name, parameters in TOOL_PARAMETERS.items()
]


@dataclass(frozen=True)
class PlannedGrasp:
    trajectory: GraspTrajectory
    target: str
    lift_cm: float


class GraspTools:
    def __init__(self, sim: FrankaBall, out: Path):
        self.sim = sim
        self.out = out
        self.frame_number = 0
        self.latest_frame: tuple[str, tuple[CameraFrame, CameraFrame], int] | None = None
        self.plans: dict[str, PlannedGrasp] = {}
        self.executed_plan: str | None = None

    def _capture(self, suffix: str = "") -> tuple[dict, tuple[CameraFrame, CameraFrame], dict[str, Path]]:
        frames = self.sim.observe_pair()
        self.frame_number += 1
        frame_id = f"f{self.frame_number}"
        images = {}
        for frame in frames:
            path = self.out / f"{frame_id}{suffix}-{frame.camera}.jpg"
            frame.image.save(path, quality=92)
            images[frame.camera] = path
        return {"frame_id": frame_id, "images": {camera: path.name for camera, path in images.items()}}, frames, images

    def observe(self) -> tuple[dict, dict[str, Path]]:
        receipt, frames, images = self._capture()
        self.latest_frame = (receipt["frame_id"], frames, self.sim.steps)
        return receipt, images

    def plan_grasp(self, frame_id: str, target: str, lift_cm: float) -> dict:
        if target not in TARGET_COLORS:
            raise GraspError("请选择可抓取对象 orange_ball 或 green_ball")
        if type(lift_cm) not in (int, float) or not MIN_LIFT_CM <= lift_cm <= MAX_LIFT_CM:
            raise GraspError(f"lift_cm 必须是 {MIN_LIFT_CM} 到 {MAX_LIFT_CM} 之间的数字，单位为厘米")
        if self.latest_frame is None or frame_id != self.latest_frame[0] or self.latest_frame[2] != self.sim.steps:
            raise GraspError("观测已过期，请先调用 observe 获取新的 frame_id")
        color = TARGET_COLORS[target]
        estimate = estimate_ball(self.latest_frame[1], color=color)
        trajectory = self.sim.plan_grasp(estimate, color=color, lift_m=lift_cm / 100)
        plan_id = f"p{len(self.plans) + 1}"
        self.plans[plan_id] = PlannedGrasp(trajectory, target, lift_cm)
        return {"plan_id": plan_id, "frame_id": frame_id, "target": target,
                "requested_lift_cm": lift_cm, "motion": "接近、下降、闭合、垂直抬升、保持",
                "ball_estimate": {"source": "two_rgb_views", "coordinate_frame": "world",
                                  "color": color,
                                  "center_m": [round(float(value), 4) for value in estimate.center],
                                  "radius_m": round(float(estimate.radius), 4)}}

    def execute(self, plan_id: str) -> dict:
        if plan_id not in self.plans or plan_id != f"p{len(self.plans)}":
            raise GraspError("使用最近一次 plan_grasp 返回的 plan_id")
        if self.executed_plan == plan_id:
            raise GraspError("该计划已经执行，请直接调用 verify_grasp")
        self.executed_plan = plan_id
        plan = self.plans[plan_id]
        self.sim.execute_grasp(plan.trajectory)
        return {"plan_id": plan_id, "target": plan.target, "requested_lift_cm": plan.lift_cm,
                "motion_complete": True, "next": "verify_grasp"}

    def verify_grasp(self, plan_id: str) -> tuple[dict, dict[str, Path]]:
        if self.executed_plan != plan_id:
            raise GraspError("请先执行这个 plan_id")
        receipt, frames, images = self._capture("-verification")
        plan = self.plans[plan_id]
        result = self.sim.verify_grasp(plan.trajectory, frames)
        return {**result, **receipt, "plan_id": plan_id, "target": plan.target,
                "requested_lift_cm": plan.lift_cm}, images

    def call(self, name: str, args: object) -> tuple[dict, dict[str, Path]]:
        images = {}
        try:
            parameters = TOOL_PARAMETERS.get(name)
            if parameters is None or not isinstance(args, dict) or set(args) != set(parameters):
                raise GraspError("工具名或参数不在本实验的白名单里")
            for key, spec in parameters.items():
                value = args[key]
                if spec["type"] == "string" and not isinstance(value, str):
                    raise GraspError(f"{key} 必须是字符串")
            value = getattr(self, name)(**args)
            if isinstance(value, tuple):
                value, images = value
            result = {"ok": True, **value}
        except GraspError as error:
            result = {"ok": False, "error": str(error)}
        with (self.out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tool": name, "arguments": args, "receipt": result,
                                     "simulation_step": self.sim.steps}, ensure_ascii=False) + "\n")
        print(f"{name} → {json.dumps(result, ensure_ascii=False)}", flush=True)
        return result, images


def receipt_content(receipt: dict, images: dict[str, Path]) -> list[dict]:
    content = [{"type": "input_text", "text": json.dumps(receipt, ensure_ascii=False)}]
    for camera, image in images.items():
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        content.append({"type": "input_text", "text": f"{camera} RGB image ({image.name})"})
        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded})
    return content

"""Expose individual robot skills as validated model tools."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from . import feedback, gripper, movement, observation, planning


TOOLS = {**observation.TOOLS, **planning.TOOLS, **movement.TOOLS, **gripper.TOOLS, **feedback.TOOLS}
TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": description,
     "parameters": {"type": "object", "properties": parameters,
                    "required": list(parameters), "additionalProperties": False}}
    for name, (_, description, parameters) in TOOLS.items()
]


@dataclass
class SkillSession:
    backend: Any
    output: Path
    floor_z: float
    frames: dict = field(default_factory=dict)
    frame_steps: dict[str, int] = field(default_factory=dict)
    placement_frames: dict[str, str] = field(default_factory=dict)
    plans: dict[str, planning.GraspPlan] = field(default_factory=dict)
    arm_plan_id: str | None = None
    arm_waypoint: str | None = None
    grasp_command_plan_id: str | None = None
    basket_release_plan_id: str | None = None

    def call(self, name: str, arguments: object) -> tuple[dict, dict[str, Path]]:
        images = {}
        registered = TOOLS.get(name)
        if registered is None:
            receipt = {"ok": False, "status": "invalid_tool", "reason": "Choose a registered skill."}
        else:
            handler, _, parameters = registered
            if not isinstance(arguments, dict) or set(arguments) != set(parameters):
                receipt = {"ok": False, "status": "invalid_arguments",
                           "reason": f"Required arguments: {', '.join(parameters) or 'none'}"}
            else:
                try:
                    for key, specification in parameters.items():
                        value = arguments[key]
                        allowed = specification.get("enum")
                        if not isinstance(value, str) or (allowed is not None and value not in allowed):
                            raise RuntimeError(f"Invalid value for {key}. Use the tool schema and exact IDs from receipts.")
                    answer = handler(self, **arguments)
                    if isinstance(answer, tuple):
                        answer, images = answer
                    receipt = {"ok": True, **answer}
                except RuntimeError as error:
                    receipt = {"ok": False, "status": "skill_rejected", "reason": str(error)}
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"tool": name, "arguments": arguments, "receipt": receipt},
                                    ensure_ascii=False) + "\n")
        print(f"{name} {arguments!r} → {json.dumps(receipt, ensure_ascii=False)}", flush=True)
        return receipt, images

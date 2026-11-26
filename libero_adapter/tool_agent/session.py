"""Tool schemas, evidence stores and the only tool-dispatch entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import dist
from pathlib import Path
from typing import Any

from libero_adapter.skills import feedback, gripper, movement, observation
from . import depth, grasp, load, memory, retention, ridge, shape, verify, vision
from .live_motion import watch_motion


def finish_task(session, outcome: str, frame_id: str, reason: str) -> dict:
    receipt = feedback.finish_task(session, outcome, frame_id, reason)
    if outcome == "placed":
        evidence = session.verifications.get(frame_id)
        if evidence is None:
            raise RuntimeError("Check this inspection frame with verify_place first.")
        if evidence["status"] != "inside":
            raise RuntimeError(f"Placement evidence is {evidence['status']}. Look again, recover, or report blocked.")
        if evidence["target"] != session.target:
            raise RuntimeError("The verified item differs from the item in the task instruction.")
    return receipt


def move_arm(session, plan_id: str, waypoint: str) -> dict:
    plan = session.plans.get(plan_id)
    if plan is None:
        raise RuntimeError("Use a plan_id returned by plan_motion.")
    if plan.target != session.target:
        raise RuntimeError("This plan targets a different object. Keep the object named in the instruction.")
    if waypoint == "pick_hover":
        if session.grasp_command_plan_id is not None:
            raise RuntimeError("The gripper still has a close command. Open it before attempting a new grasp.")
        if session.plan_start_steps.get(plan_id) != session.backend.simulation_steps:
            raise RuntimeError("This motion plan is stale. Observe again and update the grasp before reapproaching.")
    receipt = movement.move_arm(session, plan_id, waypoint)
    if receipt["status"] != "motion_failed" or receipt.get("reason") == "motion_guard_rejected":
        return receipt
    requested, _ = plan.waypoints[waypoint]
    measured = session.backend.get_robot_state()
    actual = measured.value.end_effector_position_m if measured.success else None
    evidence = {"target": plan.target, "waypoint": waypoint, "attempted_z_m": float(requested[2]),
                "requested_xyz_m": [round(float(value), 4) for value in requested],
                "tcp_z_m": float(actual[2]) if actual is not None else None,
                "position_residual_m": round(dist(requested, actual), 5) if actual is not None else None,
                "simulation_step": session.backend.simulation_steps}
    session.motion_failures.append(evidence)
    receipt["motion_evidence"] = evidence
    if waypoint == "grasp":
        receipt["next_options"] = ("Lift the open gripper with retreat_arm if it obscures the target. "
                                   "Refresh the observation and choose a different height or jaw angle.")
    return receipt


def retreat_arm(session) -> dict:
    if session.grasp_command_plan_id is not None:
        raise RuntimeError("A grasp may still be held. Use its lift waypoint or deliberately release before retreating.")
    result = session.backend.get_robot_state()
    if not result.success or result.value is None:
        raise RuntimeError(result.message or "Unable to read the current arm pose.")
    state = result.value
    x, y, z = state.end_effector_position_m
    lifted = min(max(z + 0.07, session.floor_z + 0.25), session.floor_z + 0.33)
    if lifted <= z + 0.005:
        raise RuntimeError("The gripper is already high. Capture a new view.")
    session.arm_plan_id = session.arm_waypoint = None
    movement_result = session.backend.move_to_pose((x, y, lifted), state.end_effector_quaternion_wxyz)
    return {"ok": movement_result.success, "status": "retreated" if movement_result.success else "motion_failed",
            "reason": movement_result.message or "Capture RGB again from the cleared view.",
            "requested_z_m": round(lifted, 4), "simulation_steps": session.backend.simulation_steps}


def test_lift(session, plan_id: str) -> dict:
    if session.grasp_command_plan_id != plan_id or session.arm_plan_id != plan_id or session.arm_waypoint != "grasp":
        raise RuntimeError("Reach the grasp waypoint and close on this plan first. A test lift can only start there.")
    state = session.backend.get_robot_state()
    if not state.success or state.value is None:
        raise RuntimeError(state.message or "Unable to read the end-effector pose.")
    x, y, z = state.value.end_effector_position_m
    target = (x, y, z + 0.04)
    result = session.backend.move_to_pose(target, state.value.end_effector_quaternion_wxyz)
    session.arm_waypoint = "test_lift" if result.success else None
    return {"ok": result.success, "status": "test_lift_reached" if result.success else "motion_failed",
            "reason": result.message or "Check whether the target moved up with the fingers before continuing.",
            "plan_id": plan_id, "commanded_lift_m": 0.04, "simulation_steps": session.backend.simulation_steps}


SCAN_OFFSETS = {"left": (0.0, -0.04), "right": (0.0, 0.04), "center": (0.0, 0.0)}


def move_wrist_view(session, view: str, object_id: str | None = None) -> dict:
    if object_id is None:
        object_id = next((key for key, obj in reversed(session.objects.items())
                          if obj["target"] == session.target and session.object_epochs.get(key) == session.contact_epoch),
                         None)
    obj = session.objects.get(object_id)
    if obj is None or session.object_epochs.get(object_id) != session.contact_epoch:
        raise RuntimeError("There is no current target location. Call detect_object on a fresh frame, then locate_object for its 3D ID.")
    if obj["target"] != session.target:
        raise RuntimeError("Scan the target object named in the task.")
    measured = session.backend.get_robot_state()
    if not measured.success or measured.value is None:
        raise RuntimeError(measured.message or "Unable to read wrist pose.")
    state = measured.value
    if session.grasp_command_plan_id is not None or state.gripper_open_fraction < 0.75:
        raise RuntimeError("Wrist scanning requires an open gripper without a held object.")
    if state.end_effector_position_m[2] < session.floor_z + 0.22:
        raise RuntimeError("Raise the gripper with retreat_arm before scanning from another angle.")
    dx, dy = SCAN_OFFSETS[view]
    x, y, _ = obj["position"]
    position = (float(x + dx), float(y + dy), session.floor_z + 0.29)
    if not (0.30 <= position[0] <= 0.80 and abs(position[1]) <= 0.38):
        raise RuntimeError("This viewpoint is outside the camera motion workspace. Choose another view.")
    session.pending_scan_view = None
    session.arm_plan_id = session.arm_waypoint = None
    moved = session.backend.move_to_pose(position, (0.0, 1.0, 0.0, 0.0))
    if moved.success:
        session.current_scan_object_id = object_id
        session.pending_scan_view = {"object_id": object_id, "view": view, "contact_epoch": session.contact_epoch}
    return {"ok": moved.success, "status": "viewpoint_reached" if moved.success else "motion_failed",
            "reason": moved.message or "Call observe with wrist or both to capture this angle.",
            "view": view, "object_id": object_id, "requested_xyz_m": [round(value, 4) for value in position],
            "simulation_steps": session.backend.simulation_steps}


def observe(session, camera: str) -> tuple[dict, dict[str, Path]]:
    receipt, images = observation.observe(session, camera)
    if session.pending_scan_view is not None and "wrist" in images:
        scan = dict(session.pending_scan_view)
        session.shape_frame_views[receipt["frame_id"]] = scan
        receipt["wrist_view"] = scan["view"]
        receipt["scan_object_id"] = scan["object_id"]
    return receipt, images


def reconstruct_shape(session, object_id: str) -> dict:
    latest_by_view = {}
    for frame_id, data in session.shape_frame_views.items():
        if data["object_id"] == object_id and data["contact_epoch"] == session.contact_epoch:
            latest_by_view[data["view"]] = frame_id
    if len(latest_by_view) < 2:
        visited = ", ".join(latest_by_view) or "none"
        raise RuntimeError(f"Captured wrist views for {object_id}: {visited}. Move to a different wrist view and observe it before reconstructing.")
    return shape.reconstruct_shape(session, object_id, *latest_by_view.values())


TOOLS = {
    **observation.TOOLS,
    "observe": (observe, *observation.TOOLS["observe"][1:]),
    **vision.TOOLS,
    **depth.TOOLS,
    "move_wrist_view": (move_wrist_view,
                        "Move the wrist camera to a selected angle above a localized, stationary object. "
                        "Choose left and right for different angles and call observe with wrist after each move. "
                        "Use when a single view leaves the shape or grasp uncertain.",
                         {"object_id": {"type": "string", "optional": True,
                                        "description": "Current object ID from locate_object. Defaults to the latest current localization of the task target."},
                         "view": {"type": "string", "enum": list(SCAN_OFFSETS)}}),
    "reconstruct_shape": (
        reconstruct_shape,
        "Fuse the latest wrist RGB from each distinct view of the selected stationary object. "
        "Use after choosing at least two views and capturing wrist RGB at each. Returns a shape ID, "
        "visible 3D points and measurement quality. No movement occurs.",
        {"object_id": {"type": "string", "description": "Target ID returned by locate_object before the wrist views."}},
    ),
    **load.TOOLS,
    **ridge.TOOLS,
    **grasp.TOOLS,
    "move_arm": (move_arm, *movement.TOOLS["move_arm"][1:]),
    "retreat_arm": (retreat_arm,
                    "Raise the open gripper vertically to clear the view after a blocked descent or after release. "
                    "Use when the arm obscures the scene, then take a new picture.", {}),
    "test_lift": (test_lift,
                  "After closing the fingers at a grasp, raise the hand by 4 cm at fixed orientation. "
                  "Use to test a new or uncertain grip. Capture the wrist and check retention afterward. "
                  "Reaching the lift position alone is not proof that the object moved.", {"plan_id": {"type": "string"}}),
    **gripper.TOOLS,
    "get_gripper_state": feedback.TOOLS["get_gripper_state"],
    **retention.TOOLS,
    **verify.TOOLS,
    **memory.TOOLS,
    "finish_task": (finish_task,
                    "Report placed when verify_place says inside on a new inspection frame. Report blocked with a reason if unable to proceed.",
                    feedback.TOOLS["finish_task"][2]),
}

TOOL_SCHEMAS = [
    {"type": "function", "name": name, "description": description,
     "parameters": {"type": "object", "additionalProperties": False,
                    "properties": {key: {k: v for k, v in spec.items() if k != "optional"}
                                   for key, spec in properties.items()},
                    "required": [key for key, spec in properties.items() if not spec.get("optional")]}}
    for name, (_, description, properties) in TOOLS.items()
]


@dataclass
class ToolSession:
    backend: Any
    output: Path
    floor_z: float
    target: str
    monitor: Any = None
    frames: dict = field(default_factory=dict)
    frame_steps: dict = field(default_factory=dict)
    detections: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)
    depths: dict = field(default_factory=dict)
    shapes: dict = field(default_factory=dict)
    shape_frame_views: dict = field(default_factory=dict)
    object_epochs: dict = field(default_factory=dict)
    pending_scan_view: dict | None = None
    current_scan_object_id: str | None = None
    contact_epoch: int = 0
    objects: dict = field(default_factory=dict)
    destinations: dict = field(default_factory=dict)
    ridges: dict = field(default_factory=dict)
    grasps: dict = field(default_factory=dict)
    grasp_checks: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)
    plan_start_steps: dict = field(default_factory=dict)
    placement_frames: dict = field(default_factory=dict)
    verifications: dict = field(default_factory=dict)
    motion_failures: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    grip_checks: list = field(default_factory=list)
    drop_diagnoses: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    grip_force_estimates: dict = field(default_factory=dict)
    attempt_memory: dict = field(default_factory=dict)
    arm_plan_id: str | None = None
    arm_waypoint: str | None = None
    grasp_command_plan_id: str | None = None
    basket_release_plan_id: str | None = None

    def call(self, name: str, arguments: object, *, turn: int, model_call: dict | None = None) -> tuple[dict, dict[str, Path]]:
        images = {}
        start_step = self.backend.simulation_steps
        action_tools = {"move_arm", "set_gripper", "retreat_arm", "move_wrist_view", "test_lift"}
        action = name in action_tools
        before = self._opening() if action else None
        try:
            tool = TOOLS.get(name)
            if tool is None:
                raise RuntimeError("Choose a tool from the supplied list.")
            handler, _, properties = tool
            required = {key for key, spec in properties.items() if not spec.get("optional")}
            if not isinstance(arguments, dict) or not required <= set(arguments) <= set(properties):
                raise RuntimeError(f"Required arguments: {', '.join(sorted(required)) or 'none'}. "
                                   f"Optional arguments: {', '.join(sorted(set(properties) - required)) or 'none'}.")
            for key, value in arguments.items():
                spec = properties[key]
                allowed = spec.get("enum")
                if spec.get("type") == "string" and not isinstance(value, str):
                    raise RuntimeError(f"Invalid {key}. Use the specified value or copy its ID from a tool receipt.")
                if allowed is not None and value not in allowed:
                    raise RuntimeError(f"Invalid {key}. Use the specified value or copy its ID from a tool receipt.")
            if action and self.monitor is not None:
                allow_interrupt = (name in {"retreat_arm", "move_wrist_view"}
                                   or (name == "move_arm" and arguments["waypoint"] in {"pick_hover", "grasp"}))
                allow_held = name == "test_lift" or (name == "move_arm" and arguments["waypoint"] in {"lift", "basket"})
                with watch_motion(self.backend, self.monitor, f"action_{turn}", name, arguments,
                                  allow_interrupt=allow_interrupt, allow_held_interrupt=allow_held, pace_hz=20.0) as watch:
                    result = handler(self, **arguments)
                    if watch.interruption is not None:
                        result.update(ok=False, status="motion_interrupted", interruption=watch.interruption,
                                      reason="Qwen requested a stop during motion. Observe and choose the next tool.")
                        interruption_frame, images = observe(self, "both")
                        result["interruption_frame_id"] = interruption_frame["frame_id"]
                    watch.outcome = result
            else:
                result = handler(self, **arguments)
            if isinstance(result, tuple):
                result, images = result
            receipt = {"ok": True, **result}
            if receipt["ok"] and name == "locate_object":
                self.object_epochs[receipt["object_id"]] = self.contact_epoch
            if receipt["ok"] and name == "plan_motion":
                self.plan_start_steps[receipt["plan_id"]] = start_step
        except RuntimeError as error:
            receipt = {"ok": False, "status": "tool_rejected", "reason": str(error)}
        end_step = self.backend.simulation_steps
        if action:
            receipt.update(gripper_open_fraction_before=before, gripper_open_fraction_after=self._opening())
            self.action_history.append({"tool": name, "arguments": arguments, "receipt": receipt,
                                        "start_step": start_step, "end_step": end_step})
        if name in action_tools - {"move_wrist_view"} and end_step != start_step:
            self.contact_epoch += 1
            self.pending_scan_view = None
            self.current_scan_object_id = None

        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            event = {"turn": turn, "tool": name, "arguments": arguments, "receipt": receipt,
                     "images": {key: str(path.relative_to(self.output)) for key, path in images.items()}}
            if model_call is not None:
                event["model_call"] = model_call
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"{name} {arguments!r} → {json.dumps(receipt, ensure_ascii=False)}", flush=True)
        return receipt, images

    def _opening(self) -> float | None:
        state = self.backend.get_robot_state()
        return round(float(state.value.gripper_open_fraction), 4) if state.success and state.value is not None else None

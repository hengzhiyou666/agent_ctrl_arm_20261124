"""Build grasp waypoints for the agent to select individually."""

from dataclasses import dataclass, replace

from libero_adapter.motion import PROFILES
from libero_adapter.rgb import plan_from_rgb


RAISE_M = 0.025
TALL_TARGETS = frozenset({"salad_dressing", "ketchup", "milk", "orange_juice"})
WAYPOINT_NAMES = {
    "approach": "pick_hover",
    "descend": "grasp",
    "lift": "lift",
    "transfer": "basket",
}
Position = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class GraspPlan:
    plan_id: str
    target: str
    waypoints: dict[str, tuple[Position, Quaternion]]
    source_frame_id: str
    grasp_height_m: float
    raised_from: str | None = None


def _save_plan(session, plan: GraspPlan) -> dict:
    session.plans[plan.plan_id] = plan
    return {
        "plan_id": plan.plan_id,
        "target": plan.target,
        "source_frame_id": plan.source_frame_id,
        "raised_from": plan.raised_from,
        "grasp_height_m": round(plan.grasp_height_m, 4),
        "waypoints": list(plan.waypoints),
    }


def plan_grasp(session, frame_id: str, target: str) -> dict:
    if target not in PROFILES:
        raise RuntimeError("unsupported target")
    if frame_id not in session.frames:
        raise RuntimeError("unknown frame_id; use a frame returned by observe")
    if session.frame_steps.get(frame_id) != session.backend.simulation_steps:
        raise RuntimeError("stale frame_id; observe again before planning")

    plan_id = f"grasp_{len(session.plans) + 1}"
    plan = plan_from_rgb(plan_id, session.frames[frame_id], session.floor_z, target=target)
    waypoints = {
        WAYPOINT_NAMES[stage]: (position, quaternion)
        for stage, position, quaternion in plan.waypoints
        if stage in WAYPOINT_NAMES
    }
    return _save_plan(session, GraspPlan(
        plan_id, target, waypoints, frame_id, PROFILES[target].grasp_height_m,
    ))


def raise_grasp(session, plan_id: str) -> dict:
    plan = session.plans.get(plan_id)
    if plan is None:
        raise RuntimeError("unknown plan_id")
    if plan.target not in TALL_TARGETS:
        raise RuntimeError("higher grasps are available for salad_dressing, ketchup, milk and orange_juice")
    if plan.raised_from is not None:
        raise RuntimeError("this plan already uses the higher grasp")

    height = plan.grasp_height_m + RAISE_M
    grasp_z = session.floor_z + height
    clearance_z = grasp_z + 0.17
    waypoints = {
        name: ((*position[:2], grasp_z if name == "grasp" else max(position[2], clearance_z)), quaternion)
        for name, (position, quaternion) in plan.waypoints.items()
    }
    return _save_plan(session, replace(
        plan, plan_id=f"grasp_{len(session.plans) + 1}", waypoints=waypoints,
        grasp_height_m=height, raised_from=plan_id,
    ))


TOOLS = {
    "plan_grasp": (
        plan_grasp,
        "Plan a grasp for the requested target from a fresh camera observation. Returns named motion waypoints.",
        {"frame_id": {"type": "string"}, "target": {"type": "string", "enum": sorted(PROFILES)}},
    ),
    "raise_grasp": (
        raise_grasp,
        "Return a new plan with a grasp 25 mm higher for a tall target if the original descent is blocked. "
        "Apply once. Use the returned plan ID for further moves.",
        {"plan_id": {"type": "string"}},
    ),
}

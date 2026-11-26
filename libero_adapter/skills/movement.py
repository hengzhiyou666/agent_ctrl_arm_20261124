"""Move the arm to one named waypoint from a saved grasp plan."""

WAYPOINTS = ("pick_hover", "grasp", "lift", "basket")


def move_arm(session, plan_id: str, waypoint: str) -> dict:
    def receipt(ok: bool, status: str, reason: str, executed: bool = False) -> dict:
        return {
            "ok": ok,
            "status": status,
            "reason": reason,
            "executed": executed,
            "plan_id": plan_id,
            "waypoint": waypoint,
            "simulation_steps": session.backend.simulation_steps,
            "grasp_command_plan_id": session.grasp_command_plan_id,
            "basket_release_plan_id": session.basket_release_plan_id,
        }

    plan = session.plans.get(plan_id)
    if plan is None:
        return receipt(False, "unknown_plan", "Use a plan ID returned by plan_grasp or raise_grasp.")
    if waypoint not in WAYPOINTS or waypoint not in plan.waypoints:
        return receipt(False, "unknown_waypoint", "Use an available named waypoint from this plan.")
    if waypoint == "grasp" and (session.arm_plan_id, session.arm_waypoint) != (plan_id, "pick_hover"):
        return receipt(False, "approach_required", "Reach pick_hover for this plan before descending to grasp.")
    if waypoint in ("lift", "basket") and session.grasp_command_plan_id != plan_id:
        return receipt(False, "close_required", "This plan has no successful gripper close command. If the descent was blocked, revise the grasp or inspect the scene before continuing.")
    if waypoint == "basket" and (session.arm_plan_id, session.arm_waypoint) != (plan_id, "lift"):
        return receipt(False, "lift_required", "Reach the lift waypoint for this plan before transferring to the basket.")

    position, quaternion = plan.waypoints[waypoint]
    session.arm_plan_id = session.arm_waypoint = None
    result = session.backend.move_to_pose(position, quaternion)
    if not result.success:
        return receipt(False, "motion_failed", result.message or "Controller failed to reach the waypoint.", result.executed)

    session.arm_plan_id, session.arm_waypoint = plan_id, waypoint
    reason = ("Reached above the basket. Gripper release is a separate command."
              if waypoint == "basket" else "Controller reached the requested waypoint.")
    return receipt(True, "waypoint_reached", reason, result.executed)


TOOLS = {
    "move_arm": (
        move_arm,
        "Move to one waypoint: pick_hover approaches from above; grasp descends from that hover; "
        "lift raises after a close command; basket transfers after lift. Each call moves only the arm.",
        {"plan_id": {"type": "string"}, "waypoint": {"type": "string", "enum": list(WAYPOINTS)}},
    ),
}

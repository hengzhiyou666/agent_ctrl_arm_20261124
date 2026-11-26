"""Open or close the gripper at the arm's current position."""


def set_gripper(session, state: str) -> dict:
    def receipt(ok: bool, status: str, reason: str, executed: bool = False) -> dict:
        return {
            "ok": ok,
            "status": status,
            "reason": reason,
            "executed": executed,
            "state": state,
            "simulation_steps": session.backend.simulation_steps,
            "grasp_command_plan_id": session.grasp_command_plan_id,
            "basket_release_plan_id": session.basket_release_plan_id,
        }

    if state not in ("open", "close"):
        return receipt(False, "invalid_state", "Choose open or close.")
    if state == "close" and (session.arm_plan_id is None or session.arm_waypoint != "grasp"):
        return receipt(False, "grasp_required", "Reach a planned grasp waypoint before closing the gripper.")

    operation = session.backend.open_gripper if state == "open" else session.backend.close_gripper
    result = operation()
    if not result.success:
        if result.executed:
            session.grasp_command_plan_id = None
        return receipt(False, "gripper_command_failed", result.message or "Controller could not complete the gripper command.", result.executed)

    if state == "close":
        session.grasp_command_plan_id = session.arm_plan_id
        session.basket_release_plan_id = None
    else:
        if session.arm_waypoint == "basket" and session.grasp_command_plan_id == session.arm_plan_id:
            session.basket_release_plan_id = session.arm_plan_id
        session.grasp_command_plan_id = None
    return receipt(True, "gripper_command_completed",
                   "Gripper command completed. Use a camera observation to check the object.", result.executed)


TOOLS = {
    "set_gripper": (
        set_gripper,
        "Close at a reached grasp waypoint, or open at the current position. "
        "Open at the basket waypoint to release an object into the basket. Use camera evidence to check the object.",
        {"state": {"type": "string", "enum": ["open", "close"]}},
    ),
}

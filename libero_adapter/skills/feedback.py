"""Return measured robot state and validate the evidence for an agent report."""


def get_gripper_state(session) -> dict:
    result = session.backend.get_robot_state()
    if not result.success:
        raise RuntimeError(result.message or "Unable to read the gripper state.")
    state = result.value
    return {
        "gripper_open_fraction": round(float(state.gripper_open_fraction), 4),
        "grasp_command_plan_id": session.grasp_command_plan_id,
        "basket_release_plan_id": session.basket_release_plan_id,
        "simulation_step": session.backend.simulation_steps,
    }


def finish_task(session, outcome: str, frame_id: str, reason: str) -> dict:
    if not reason.strip():
        raise RuntimeError("Describe the result briefly in reason.")
    if outcome == "placed":
        release_plan = session.basket_release_plan_id
        if release_plan is None:
            raise RuntimeError("No basket release command has completed. Continue with a skill, or report blocked.")
        if (session.placement_frames.get(frame_id) != release_plan
                or session.frame_steps.get(frame_id) != session.backend.simulation_steps):
            raise RuntimeError("Use the frame_id from a fresh inspect_place call after the basket release.")
    return {"status": "report_accepted", "outcome": outcome, "frame_id": frame_id, "reason": reason}


TOOLS = {
    "get_gripper_state": (
        get_gripper_state,
        "Read the measured gripper opening as a fraction from 0 (closed) to 1 (open), and recent command receipts.",
        {},
    ),
    "finish_task": (
        finish_task,
        "Report the outcome. Use placed only with visual support from a fresh inspect_place frame after the basket release. "
        "Use blocked when you cannot proceed, with frame_id set to none if you have no inspection frame.",
        {"outcome": {"type": "string", "enum": ["placed", "blocked"]},
         "frame_id": {"type": "string"}, "reason": {"type": "string"}},
    ),
}

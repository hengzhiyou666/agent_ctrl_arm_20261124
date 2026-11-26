"""Independent camera observation and visual inspection skills."""

from pathlib import Path


CAMERA_NAMES = {
    "overview": "agentview",
    "wrist": "robot0_eye_in_hand",
}


def observe(session, camera: str) -> tuple[dict, dict[str, Path]]:
    if camera not in (*CAMERA_NAMES, "both"):
        raise RuntimeError("choose camera overview, wrist, or both")

    result = session.backend.get_observation_data()
    if not result.success:
        raise RuntimeError(result.message or "camera capture failed")

    observation = result.value
    available = {view.camera_name: Path(view.rgb_path) for view in observation.cameras}
    requested = tuple(CAMERA_NAMES) if camera == "both" else (camera,)
    missing = [name for name in requested if CAMERA_NAMES[name] not in available]
    if missing:
        raise RuntimeError(f"unavailable camera: {', '.join(missing)}")
    images = {name: available[CAMERA_NAMES[name]] for name in requested}
    image_paths = {name: str(path.relative_to(session.output)) for name, path in images.items()}

    frame_id = f"f{len(session.frames) + 1}"
    step = session.backend.simulation_steps
    session.frames[frame_id] = observation
    session.frame_steps[frame_id] = step
    return {"frame_id": frame_id, "simulation_step": step, "images": image_paths}, images


def inspect_place(session, camera: str) -> tuple[dict, dict[str, Path]]:
    receipt, images = observe(session, camera)
    frame_id = receipt["frame_id"]
    release_plan = session.basket_release_plan_id
    receipt["basket_release_plan_id"] = release_plan
    if release_plan is not None:
        session.placement_frames[frame_id] = release_plan
    robot_state = session.frames[frame_id].robot_state
    if robot_state is not None:
        receipt["gripper_open_fraction"] = round(float(robot_state.gripper_open_fraction), 4)
    receipt["visual_observation"] = ("Inspect these fresh RGB images for evidence of placement."
                                     if release_plan else "No successful release at the basket has been recorded.")
    return receipt, images


CAMERA_PARAMETER = {"camera": {"type": "string", "enum": [*CAMERA_NAMES, "both"]}}

TOOLS = {
    "observe": (
        observe,
        "Capture fresh RGB from the overview, wrist, or both cameras. Returns a frame_id for grasp planning. "
        "The frame is valid for planning while the robot and scene are unchanged.",
        CAMERA_PARAMETER,
    ),
    "inspect_place": (
        inspect_place,
        "Get fresh visual evidence for checking the placement. Choose the overview, wrist, or both cameras.",
        CAMERA_PARAMETER,
    ),
}

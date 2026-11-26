"""Read-only grip and drop checks from encoders, RGB and action receipts."""

from math import isfinite

import numpy as np
from scipy.ndimage import binary_opening, find_objects, label

from libero_adapter.rgb import _read_rgb


# At the evaluator's 20-tick OSC close, a free LIBERO Panda finger settles at
# about 0.023 of its travel. Extra margin accounts for transient deflection.
# Fractions refer to the measured first finger and do not measure jaw force.
_EMPTY_MAX = 0.04
_BLOCKED_MIN = 0.075
_UNSETTLED_MIN = 0.90
_HELD_PHASES = ("test_lift", "lift", "basket")


def _plan(session, plan_id):
    plan = session.plans.get(plan_id)
    if plan is None or plan.target != session.target:
        raise RuntimeError("Use a plan_id for the object named in the task.")
    return plan


def _fraction(value):
    if isinstance(value, (float, int)) and isfinite(value) and 0 <= value <= 1:
        return round(float(value), 4)
    return None


def _band(fraction):
    if fraction is None:
        return "unavailable"
    if fraction <= _EMPTY_MAX:
        return "near_empty_close"
    if fraction < _BLOCKED_MIN:
        return "ambiguous_narrow_gap"
    if fraction < _UNSETTLED_MIN:
        return "gap_above_empty_range"
    return "wide_or_unsettled"


def _observation(session, frame_id):
    now = session.backend.simulation_steps
    if frame_id is None:
        frame_id = next((name for name, step in reversed(session.frame_steps.items()) if step == now), None)
    if frame_id is None:
        return None, None
    if session.frame_steps.get(frame_id) != now or frame_id not in session.frames:
        raise RuntimeError("This frame is missing or stale. Use a new observe frame or omit frame_id.")
    return frame_id, session.frames[frame_id]


def _wrist_foreground(observation):
    result = {"status": "unavailable"}
    if observation is None:
        return result
    wrist = next((view for view in observation.cameras if view.camera_name == "robot0_eye_in_hand"), None)
    if wrist is None:
        return result
    try:
        rgb = _read_rgb(wrist)
    except OSError:
        return result
    h, w = rgb.shape[:2]
    x0, x1, y0, y1 = round(.34 * w), round(.66 * w), round(.55 * h), round(.74 * h)
    crop = rgb[y0:y1, x0:x1]
    sides = np.concatenate((rgb[y0:y1, round(.13 * w):round(.27 * w)],
                            rgb[y0:y1, round(.73 * w):round(.87 * w)]), axis=1)
    if min(crop.shape[:2]) < 8 or sides.shape[1] == 0:
        return result

    # Local contrast against the sides can expose a body coupled to the
    # gripper. A shadow, the floor or another item can also give a cue.
    background = np.median(sides, axis=1)[:, None, :]
    contrast = background - crop
    mask = (np.abs(contrast).mean(axis=2) > 27) & (contrast.mean(axis=2) > 21)
    labels, count = label(binary_opening(mask))
    support = 0.0
    if count:
        for index, (rows, columns) in enumerate(find_objects(labels), start=1):
            center_x = (columns.start + columns.stop) / 2
            if (0.29 * crop.shape[1] <= center_x <= 0.71 * crop.shape[1]
                    and rows.stop - rows.start >= 0.17 * crop.shape[0]):
                support = max(support, float(np.count_nonzero(labels[rows, columns] == index) / mask.size))
    return {"status": "central_foreground" if support >= 0.15 else "no_clear_foreground",
            "support_fraction": round(support, 3), "window_xyxy": [x0, y0, x1, y1]}


def _check(session, plan_id, frame_id):
    frame_id, observation = _observation(session, frame_id)
    measured = session.backend.get_robot_state()
    state = measured.value if measured.success else None
    fraction = _fraction(state.gripper_open_fraction) if state is not None else None
    phase = session.arm_waypoint if session.arm_plan_id == plan_id else None
    active = session.grasp_command_plan_id == plan_id
    wrist = _wrist_foreground(observation)
    aperture = _band(fraction)
    seen = wrist["status"] == "central_foreground"

    status = "uncertain"
    reason = "The measured gap or visibility is ambiguous."
    next_option = "Call observe with wrist after a short lift, then run check_grip before transfer."
    if not active:
        status = "no_active_grasp"
        reason = "No completed close command is active for this plan."
        next_option = "Check the latest approach and gripper receipts, or verify_place if the object was released."
    elif aperture == "near_empty_close" and not seen:
        status = "empty"
        reason = "The finger closed into its free-travel range. There is no clear retained-object evidence."
        next_option = "Open the gripper, clear the view, observe the target again and choose a different grasp."
    elif aperture == "gap_above_empty_range" and seen and phase in _HELD_PHASES:
        status = "gripped"
        reason = "The closed gripper maintains a gap above empty travel and wrist RGB shows a body below the jaws."
        next_option = {
            "test_lift": "Continue to the full lift waypoint. Check the grip again whenever it is uncertain.",
            "lift": "Transfer to the basket and check the grip again before opening.",
            "basket": "If above the destination, open and verify placement.",
        }[phase]
    elif aperture == "gap_above_empty_range":
        status = "possible_grip"
        reason = "A measured gap is consistent with material between the fingers. Retention remains uncertain."
        if phase in _HELD_PHASES:
            next_option = "Get a fresh wrist view, check for scene occlusion, and compare with the previous grip reading."
    elif phase in _HELD_PHASES:
        next_option = "Get a fresh wrist view and use diagnose_drop before opening or transferring again."

    return {
        "status": status,
        "verified": status == "gripped",
        "plan_id": plan_id,
        "frame_id": frame_id,
        "simulation_step": session.backend.simulation_steps,
        "phase": phase,
        "close_active": active,
        "reason": reason,
        "evidence": {"gripper_open_fraction": fraction, "aperture_band": aperture,
                     "wrist": wrist, "target_identity_confirmed": False},
        "next_option": next_option,
    }


def check_grip(session, plan_id: str, frame_id: str | None = None) -> dict:
    """Inspect the current gripper without commanding any motion."""
    plan = _plan(session, plan_id)
    result = _check(session, plan_id, frame_id)
    result["target"] = plan.target
    session.grip_checks.append(result)
    return result


def _history(session, plan_id):
    active = []
    closed = False
    for action in getattr(session, "action_history", ()):
        receipt = action.get("receipt", {})
        arguments = action.get("arguments") or {}
        if action.get("end_step", 0) <= action.get("start_step", 0):
            continue
        same_plan = arguments.get("plan_id") == plan_id
        if action["tool"] == "move_arm" and arguments.get("waypoint") == "pick_hover" and same_plan:
            active = [action]
            closed = False
        elif action["tool"] in ("move_arm", "test_lift") and same_plan:
            active.append(action)
        elif action["tool"] == "set_gripper":
            if arguments.get("state") == "close":
                closed = receipt.get("grasp_command_plan_id") == plan_id
                if closed:
                    active.append(action)
            elif arguments.get("state") == "open" and closed:
                active.append(action)
                closed = False
    return active


def _action_samples(actions, checks, current):
    samples = {}
    close_step = None
    opening_step = None
    phases = []
    previous_phase = None
    for action in actions:
        arguments, receipt = action["arguments"], action["receipt"]
        step = int(action["end_step"])
        if action["tool"] == "set_gripper" and receipt.get("ok"):
            if arguments["state"] == "close":
                close_step = step
                opening_step = None
                samples.clear()
                samples[step] = ("grasp", _fraction(receipt.get("gripper_open_fraction_after")))
                previous_phase = "grasp"
            else:
                opening_step = step
        if action["tool"] in ("move_arm", "test_lift"):
            phase = arguments.get("waypoint") if action["tool"] == "move_arm" else "test_lift"
            phases.append({"phase": phase, "status": receipt.get("status"),
                           "start_step": action["start_step"], "end_step": step})
            if close_step is not None and opening_step is None and phase in _HELD_PHASES and receipt.get("ok"):
                before = _fraction(receipt.get("gripper_open_fraction_before"))
                if before is not None:
                    samples[action["start_step"]] = (previous_phase, before)
                samples[step] = (phase, _fraction(receipt.get("gripper_open_fraction_after")))
                previous_phase = phase
    for check in [*checks, current]:
        if check["plan_id"] != current["plan_id"] or not check["close_active"]:
            continue
        step = check["simulation_step"]
        if (close_step is not None and step < close_step) or (opening_step is not None and step >= opening_step):
            continue
        fraction = check["evidence"]["gripper_open_fraction"]
        if fraction is not None:
            samples[step] = (check["phase"], fraction)
    collected = [{"phase": phase, "step": step, "open_fraction": value, "aperture": _band(value)}
                 for step, (phase, value) in sorted(samples.items()) if value is not None]
    return collected, phases, opening_step


def _hypothesis(cause, confidence, basis):
    return {"cause": cause, "confidence": confidence, "basis": basis}


def _diagnosis(current, samples, phases, opened, placement):
    if placement and placement["status"] == "inside":
        return "placed", True, [], "Report placement using the same fresh inspection frame."

    prior = None
    for sample in samples:
        if sample["aperture"] == "gap_above_empty_range":
            prior = sample
        elif sample["aperture"] == "near_empty_close" and prior and sample["step"] > prior["step"]:
            if sample["phase"] == "basket" and prior["phase"] == "lift":
                name = "lost_during_transfer"
            elif sample["phase"] in ("test_lift", "lift") and prior["phase"] in ("grasp", "test_lift"):
                name = "lost_during_lift"
            else:
                name = "loss_phase_uncertain"
            return name, True, [
                _hypothesis("loss_of_retention", "likely", "The measured gap fell from above the empty range to near-empty while close remained active."),
                _hypothesis("sliding_or_rotating_at_the_pad_contact", "possible", "The disappearing gap is consistent with an object sliding or rotating free."),
            ], "Observe where the object landed. Replan with a stable pinch across opposing surfaces and recheck after lifting."

    latest = phases[-1] if phases else None
    if latest and latest["status"] in ("motion_failed", "motion_interrupted"):
        if latest["status"] == "motion_interrupted":
            return "motion_interrupted", True, [
                _hypothesis("action_interrupted", "high", f"The {latest['phase']} motion was interrupted."),
            ], "Observe the current scene and arm pose before choosing the next action."
        return "motion_failed", True, [
            _hypothesis("waypoint_not_reached", "high", f"The controller did not reach {latest['phase']}."),
            _hypothesis("obstruction_or_controller_stall", "possible", "A pose failure alone cannot identify the obstacle."),
        ], "Inspect the latest view and pose error. Change grasp height or direction if the descent stalled."

    if placement and placement["status"] == "outside":
        return "outside_after_release", True, [
            _hypothesis("target_remained_outside", "high", "The independent placement check found the target outside."),
        ], "Observe its current location before planning a new grasp."
    if samples and samples[0]["phase"] == "grasp" and all(s["aperture"] == "near_empty_close" for s in samples):
        return "ungrasped", True, [
            _hypothesis("missed_object_during_close", "likely", "The closed jaw stayed in the empty range from initial closure."),
            _hypothesis("grasp_height_or_alignment_error", "possible", "The fingers may have passed below, above, or beside the target."),
        ], "Open, clear the view and re-localize. Choose a different contact height or jaw angle."
    if current["status"] == "gripped":
        return "still_gripped", True, [], current["next_option"]
    if opened is not None:
        return "uncertain", False, [
            _hypothesis("release_or_earlier_loss", "possible", "The fingers opened but verified pre-release retention and final placement are incomplete."),
        ], "Use inspect_place and verify_place. Look outside the basket if the scene is unclear."
    return "uncertain", False, [], current["next_option"]


def diagnose_drop(session, plan_id: str, frame_id: str | None = None) -> dict:
    """Separate grasp, lift, transfer and release evidence for a failed attempt."""
    plan = _plan(session, plan_id)
    current = _check(session, plan_id, frame_id)
    actions = _history(session, plan_id)
    samples, phases, opened = _action_samples(actions, session.grip_checks, current)
    placement = session.verifications.get(current["frame_id"])
    if placement is not None and placement.get("release_plan_id") != plan_id:
        placement = None
    status, verified, hypotheses, next_option = _diagnosis(current, samples, phases, opened, placement)
    diagnosis_id = f"drop_{len(session.drop_diagnoses) + 1}"
    result = {
        "diagnosis_id": diagnosis_id,
        "plan_id": plan_id,
        "target": plan.target,
        "simulation_step": session.backend.simulation_steps,
        "status": status,
        "verified": verified,
        "verification_scope": "The observed event or encoder pattern is supported. The physical cause remains a hypothesis.",
        "hypotheses": hypotheses,
        "evidence": {"current": current["evidence"], "aperture_samples": samples,
                     "motion_phases": phases[-5:], "release_step": opened,
                     "placement_status": placement.get("status") if placement else None},
        "next_options": [next_option],
        "finger_normal_force_n": None,
        "force_note": "Finger normal force is unavailable. These readings cannot establish that force was too low.",
    }
    session.drop_diagnoses[diagnosis_id] = result
    return result


_ARGUMENTS = {
    "plan_id": {"type": "string", "description": "The grasp plan being checked."},
    "frame_id": {"type": "string", "optional": True, "description": "A current observe or inspect_place frame ID."},
}

TOOLS = {
    "check_grip": (
        check_grip,
        "Read the measured finger gap and latest wrist image. Use after closing, after lifting, or before release. "
        "Returns gripped, possible_grip, empty, uncertain, or no_active_grasp. Checks do not move the arm.",
        _ARGUMENTS,
    ),
    "diagnose_drop": (
        diagnose_drop,
        "Compare measured closure, lift, transfer and release evidence when pickup fails or the target drops. "
        "Returns an evidence-supported phase, possible causes and a retry option. Checks do not move the arm.",
        _ARGUMENTS,
    ),
}

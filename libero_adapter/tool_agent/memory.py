"""Store and recall executed grasp attempts within the current episode."""

from __future__ import annotations

import json


RECALL_LIMIT = 5
OBSERVED_PLACEMENT = {"inside", "outside", "uncertain"}


def _events(session):
    path = session.output / "events.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _compact_event(event):
    receipt = event["receipt"]
    compact = {"turn": event["turn"], "status": receipt["status"],
               "step": receipt.get("simulation_steps")}
    if not receipt.get("ok"):
        compact["reason"] = receipt.get("reason")
        if "motion_evidence" in receipt:
            motion = receipt["motion_evidence"]
            compact["position_residual_m"] = motion.get("position_residual_m")
            compact["tcp_z_m"] = motion.get("tcp_z_m")
    return compact


def _attempt_receipts(session, plan_id):
    evidence = {"waypoint_receipts": {}, "gripper_receipts": {}}
    active_grasp_plan = None
    last_grasp_waypoint_plan = None
    chosen_grasp = None
    for event in _events(session):
        tool, receipt = event["tool"], event["receipt"]
        if tool == "plan_motion" and receipt.get("ok") and receipt.get("plan_id") == plan_id:
            chosen_grasp = receipt.get("grasp_id")
        if tool == "move_arm" and receipt.get("executed"):
            this_plan = receipt.get("plan_id")
            waypoint = receipt.get("waypoint")
            last_grasp_waypoint_plan = this_plan if receipt.get("ok") and waypoint == "grasp" else None
            if this_plan == plan_id:
                evidence["waypoint_receipts"][waypoint] = _compact_event(event)
        if tool != "set_gripper" or not receipt.get("executed"):
            continue
        state = receipt.get("state")
        if state == "close":
            this_plan = receipt.get("grasp_command_plan_id") or last_grasp_waypoint_plan
            if this_plan == plan_id:
                evidence["gripper_receipts"]["close"] = _compact_event(event)
            active_grasp_plan = this_plan if receipt.get("ok") else None
        elif state == "open":
            if active_grasp_plan == plan_id:
                evidence["gripper_receipts"]["open"] = {
                    **_compact_event(event),
                    "at_basket": bool(receipt.get("ok") and receipt.get("basket_release_plan_id") == plan_id),
                }
            active_grasp_plan = None
    return chosen_grasp, evidence


def _placement(session, plan_id, frame_id):
    if frame_id is None:
        candidates = [(fid, result) for fid, result in session.verifications.items()
                      if result.get("release_plan_id") == plan_id]
        if not candidates:
            return None
        frame_id, verification = candidates[-1]
    else:
        verification = session.verifications.get(frame_id)
    if (verification is None or verification.get("release_plan_id") != plan_id
            or verification.get("status") not in OBSERVED_PLACEMENT):
        raise RuntimeError("Use a verify_place frame that belongs to this plan's release.")
    return {"frame_id": frame_id, "status": verification["status"], "source": "verify_place RGB"}


def _diagnosis(session, plan_id, diagnosis_id):
    if diagnosis_id is None:
        candidates = [(did, result) for did, result in session.drop_diagnoses.items()
                      if result.get("plan_id") == plan_id]
        if not candidates:
            return None
        diagnosis_id, diagnosis = candidates[-1]
    else:
        diagnosis = session.drop_diagnoses.get(diagnosis_id)
    if diagnosis is None or diagnosis.get("plan_id") != plan_id:
        raise RuntimeError("Use a diagnose_drop ID returned for this plan.")
    return {"diagnosis_id": diagnosis_id, "status": diagnosis["status"],
            "verified": diagnosis.get("verified", False), "hypotheses": diagnosis.get("hypotheses", []),
            "next_options": diagnosis.get("next_options", [])}


def _phase(evidence, placement, diagnosis, last_action):
    if placement is not None:
        return "visual_placement_" + placement["status"]
    if diagnosis is not None:
        return "grasp_failure_assessed"
    commands = evidence["gripper_receipts"]
    if last_action["status"] in {"motion_failed", "gripper_command_failed"}:
        return last_action["status"]
    if "open" in commands:
        return "opened_at_basket_unverified" if commands["open"]["at_basket"] else "opened_elsewhere"
    if "close" in commands:
        return "closed_unverified"
    return "motion_only"


def remember_attempt(session, plan_id: str, diagnosis_id: str | None = None,
                     verification_frame_id: str | None = None) -> dict:
    """Record what physically happened when the selected grasp plan was tried."""
    plan = session.plans.get(plan_id)
    if plan is None:
        raise RuntimeError("Use a plan_id returned by plan_motion.")
    grasp_id, evidence = _attempt_receipts(session, plan_id)
    if not any(evidence.values()):
        raise RuntimeError("This plan has no executed move or gripper receipt to remember.")
    grasp = session.grasps.get(grasp_id, {})
    placement = _placement(session, plan_id, verification_frame_id)
    diagnosis = _diagnosis(session, plan_id, diagnosis_id)
    previous = session.attempt_memory.get(plan_id)
    position, orientation = plan.waypoints["grasp"]
    last_action = max((*evidence["waypoint_receipts"].values(), *evidence["gripper_receipts"].values()),
                      key=lambda event: event["turn"])
    snapshot = {"plan_id": plan_id, "target": plan.target, "grasp_id": grasp_id,
                "strategy": grasp.get("strategy"), "pose": {"xyz_m": list(position), "wxyz": list(orientation)},
                "phase": _phase(evidence, placement, diagnosis, last_action), "evidence": evidence,
                "last_action_turn": last_action["turn"],
                "placement": placement, "diagnosis": diagnosis}
    unchanged = previous is not None and all(previous[key] == value for key, value in snapshot.items())
    if unchanged:
        return {"status": "already_remembered", "memory_id": previous["memory_id"], "phase": previous["phase"]}

    memory_id = previous["memory_id"] if previous is not None else f"attempt_{len(session.attempt_memory) + 1}"
    revision = previous["revision"] + 1 if previous is not None else 1
    record = {**snapshot, "memory_id": memory_id, "revision": revision,
              "recorded_at_step": session.backend.simulation_steps}
    with (session.output / "attempt_memory.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    session.attempt_memory[plan_id] = record
    return {"status": "remembered", "memory_id": memory_id, "revision": revision, "phase": snapshot["phase"]}


def recall_attempts(session, target: str) -> dict:
    """Return recent remembered evidence for comparing alternative grasps."""
    if target != session.target:
        raise RuntimeError("Use the target named in the current task.")
    records = [record for record in session.attempt_memory.values() if record["target"] == target]
    results = sorted(records, key=lambda record: record["last_action_turn"], reverse=True)
    return {"target": target, "scope": "current_episode", "remembered_count": len(records),
            "attempts": results[:RECALL_LIMIT]}


TOOLS = {
    "remember_attempt": (
        remember_attempt,
        "Save the executed evidence for one grasp plan in this episode. Use after a failed approach, "
        "loss of grip, or checked placement to compare retry choices. Optionally pass the diagnose_drop "
        "ID or verify_place frame ID. The saved record cites actual move, gripper and inspection receipts.",
        {"plan_id": {"type": "string", "description": "Executed plan ID returned by plan_motion."},
         "diagnosis_id": {"type": "string", "optional": True, "description": "Matching ID from diagnose_drop."},
         "verification_frame_id": {"type": "string", "optional": True,
                                   "description": "Matching frame evaluated by verify_place."}},
    ),
    "recall_attempts": (
        recall_attempts,
        "Recall up to five recently saved attempts for the target in this episode. Use before retrying "
        "to compare poses, motion failures, release evidence, or diagnosed grip loss and choose a backup.",
        {"target": {"type": "string", "description": "Target object ID named in the task, for example milk."}},
    ),
}

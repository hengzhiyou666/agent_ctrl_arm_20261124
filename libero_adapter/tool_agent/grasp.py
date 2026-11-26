"""Propose and screen grasp poses independently of commanding the arm."""

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from libero_adapter.motion import PROFILES
from libero_adapter.skills.planning import GraspPlan, RAISE_M, TALL_TARGETS


_SPEC = json.loads(Path(__file__).with_name("gripper_spec.json").read_text(encoding="utf-8"))
_FLOOR_MARGIN_M = 0.003
_PREFERENCES = {
    "observed_shape_pinch": (-1, "Opposed surface patches agree with the jaw direction inside the Panda pad footprint."),
    "higher_top_down": (0, "Start here on this tall object. The pads aim 25 mm higher, shortening the descent."),
    "top_down": (1, "Nominal height and narrower estimated closure axis. Use for ordinary top-down grasps."),
    "across_roof_ridge": (2, "Pinch across the estimated roof ridge. The RGB edge is a hypothesis and the span uses nominal dimensions."),
    "ridge_end_pinch": (3, "Close along the ridge toward its two ends. The visible line gives a lower bound on the needed opening."),
    "inferred_shape_pinch": (3, "The visible shape suggests this pinch. Unseen contact surfaces and width remain uncertain."),
    "rotated_top_down": (3, "Try a different jaw direction if access is blocked. Check the wider estimated closure span."),
}


def get_gripper_spec(session) -> dict:
    """Return static simulated Panda finger and contact pad dimensions."""
    return dict(_SPEC)


def _current_object(session, object_id: str) -> dict:
    entry = session.objects.get(object_id)
    if entry is None:
        raise RuntimeError("Unknown object_id. Use locate_object to obtain an object ID.")
    current = session.frame_steps.get(entry["frame_id"]) == session.backend.simulation_steps
    during_scan = (getattr(session, "current_scan_object_id", None) == object_id
                   and getattr(session, "object_epochs", {}).get(object_id) == getattr(session, "contact_epoch", 0))
    if not (current or during_scan):
        raise RuntimeError("Object estimate is stale. Observe and localize it again before planning.")
    return entry


def _dimensions(entry: dict) -> tuple[float, float, float]:
    raw = entry.get("dimensions_m")
    if isinstance(raw, Mapping):
        raw = [raw.get(axis) for axis in ("width", "depth", "height")]
    try:
        width, depth, height = (float(value) for value in raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError("The object estimate needs positive width, depth and height in meters.") from error
    if not all(math.isfinite(value) and value > 0 for value in (width, depth, height)):
        raise RuntimeError("The object estimate needs positive width, depth and height in meters.")
    return width, depth, height


def _xy(entry: dict, name: str, default=None) -> tuple[float, float]:
    raw = entry.get(name, default)
    try:
        x, y = (float(value) for value in raw[:2])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"A valid {name} point is required.") from error
    if not all(math.isfinite(value) for value in (x, y)):
        raise RuntimeError(f"The {name} point must be finite.")
    return x, y


def _pose(xy, height, floor, yaw):
    yaw = (yaw + 90) % 180 - 90 if not -90 <= yaw <= 90 else yaw
    angle = math.radians(yaw)
    # For this EEF frame the pad closure axis is local X up to sign.
    return ((*xy, floor + height), (0.0, math.cos(angle / 2), math.sin(angle / 2), 0.0),
            (math.cos(angle), math.sin(angle)), yaw)


def _new_candidate(session, object_id, xy, height, yaw, span, strategy, basis, evidence=None):
    position, quaternion, axis, yaw = _pose(xy, height, session.floor_z, yaw)
    pad_z = position[2] - sum(_SPEC["pad_axial_bounds_from_tcp_m"]) / 2
    estimated_contacts = [
        (position[0] + sign * axis[0] * span / 2, position[1] + sign * axis[1] * span / 2, pad_z)
        for sign in (-1, 1)
    ]
    grasp_id = f"candidate_{len(session.grasps) + 1}"
    candidate = {
        "grasp_id": grasp_id,
        "object_id": object_id,
        "strategy": strategy,
        "basis": basis,
        "grasp_height_m": height,
        "position_m": position,
        "quaternion_wxyz": quaternion,
        "jaw_axis_xy": axis,
        "approach_axis": (0.0, 0.0, -1.0),
        "jaw_yaw_deg": yaw,
        "estimated_grip_span_m": span,
        "estimated_pad_contact_points_m": estimated_contacts,
    }
    if evidence is not None and "shape_id" in evidence:
        candidate["shape_evidence"] = evidence
    elif evidence is not None:
        candidate["ridge_evidence"] = evidence
    session.grasps[grasp_id] = candidate
    return candidate


def _latest_descent_stall(session, entry):
    x, y = _xy(entry, "pick_xy", entry.get("position"))
    now = session.backend.simulation_steps
    for failure in reversed(getattr(session, "motion_failures", ())):
        requested = failure.get("requested_xyz_m")
        attempted = failure.get("attempted_z_m")
        measured = failure.get("tcp_z_m")
        if (failure.get("target") == entry["target"] and failure.get("waypoint") == "grasp"
                and 0 <= now - failure.get("simulation_step", now) <= 100
                and requested is not None and len(requested) >= 2 and attempted is not None and measured is not None
                and math.hypot(requested[0] - x, requested[1] - y) <= 0.045
                and measured - attempted >= 0.007):
            return failure
    return None


def _preference(candidate, stall):
    order, reason = _PREFERENCES[candidate["strategy"]]
    if stall is None:
        return (0, order), reason

    pose_z = candidate["position_m"][2]
    attempted = stall["attempted_z_m"]
    measured = stall["tcp_z_m"]
    # Prioritize an alternative above the observed stopping height. A higher
    # pose near it may still help. The stop does not identify what blocked motion.
    if pose_z >= measured - 0.003:
        return (0, order), f"{reason} The TCP stopped at {measured:.3f} m on the previous descent. This candidate avoids descending farther."
    if pose_z >= attempted + 0.008:
        return (1, order), f"{reason} This gains {(pose_z - attempted) * 1000:.0f} mm over the stalled request but may still descend below the stopping height."
    return (2, order), f"{reason} A descent to this height already stalled near this object. Prefer a higher candidate before retrying."


def _annotate_preferences(session, entry, candidates):
    stall = _latest_descent_stall(session, entry)
    return [(_preference(candidate, stall), candidate) for candidate in candidates]


def _current_shape(session, object_id, shape_id):
    shape = getattr(session, "shapes", {}).get(shape_id)
    if shape is None or shape.get("object_id") != object_id:
        raise RuntimeError("Use a shape_id reconstructed for this object_id.")
    if (shape.get("simulation_step") != session.backend.simulation_steps
            or shape.get("contact_epoch", 0) != getattr(session, "contact_epoch", 0)):
        raise RuntimeError("The shape estimate is stale. Reconstruct the shape again using current wrist views.")
    return shape


def _shape_points(session, object_id, shape_id):
    shape = _current_shape(session, object_id, shape_id)
    if not shape.get("usable_for_grasp", shape.get("quality") == "supported"):
        return shape, None, None
    try:
        with np.load(shape["point_cloud_path"], allow_pickle=False) as cloud:
            points = np.asarray(cloud["points_xyz_m"], dtype=float)
            errors = np.asarray(cloud["reprojection_error_px"], dtype=float)
            pairs = cloud["source_pair"]
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError("Cannot read this shape's registered point cloud. Reconstruct it again.") from error
    if points.ndim != 2 or points.shape[1] != 3 or errors.shape != (len(points),) or pairs.shape != (len(points), 2):
        raise RuntimeError("The shape's point cloud or per-point evidence has invalid dimensions.")
    valid = np.isfinite(points).all(axis=1) & np.isfinite(errors) & (errors <= 2.5)
    return shape, points[valid], pairs[valid]


def _observed_patch(points, axis):
    if len(points) < 4:
        return None
    centered = points - points.mean(axis=0)
    variance, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    if variance[1] < 0.0015 ** 2 or variance[0] > 0.40 * variance[1]:
        return None
    normal = vectors[:, 0]
    alignment = abs(float(normal @ axis))
    return alignment if alignment >= 0.75 else None


def _shape_axes(points, shape, jaw_yaw):
    axes = []
    visible = shape.get("axes_xy") or {}
    for value in (visible.get("short"), visible.get("long")):
        if value is not None:
            axis = np.asarray(value, dtype=float)
            norm = np.linalg.norm(axis)
            if axis.shape == (2,) and norm > 1e-6 and np.isfinite(norm):
                axes.append(axis / norm)
    covariance = np.cov(points[:, :2], rowvar=False)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues[1] > 1.25 * max(eigenvalues[0], 1e-8):
        axes.extend(vectors.T)
    angle = math.radians(jaw_yaw)
    axes.extend((np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])))
    distinct = []
    for axis in axes:
        if all(abs(axis @ other) < math.cos(math.radians(12)) for other in distinct):
            distinct.append(axis)
    return distinct


def _section_candidate(points, pairs, axis, pad_z, floor_z, reference_yaw, stock_spans, shape_id, evidence_frames):
    within_pad_depth = abs(points[:, 2] - pad_z) <= _SPEC["pad_face_depth_m"] / 2
    section = points[within_pad_depth]
    if len(section) < 6:
        return None

    side_axis = np.array([-axis[1], axis[0]])
    pad_center_tangent = float(np.median(section[:, :2] @ side_axis))
    inside_pad_width = abs(section[:, :2] @ side_axis - pad_center_tangent) <= _SPEC["pad_face_width_m"] / 2
    section = section[inside_pad_width]
    section_pairs = pairs[within_pad_depth][inside_pad_width]
    if len(section) < 6:
        return None

    closure_projection = section[:, :2] @ axis
    low, high = np.quantile(closure_projection, [0.1, 0.9])
    visible_span = float(high - low)
    if not 0.012 <= visible_span <= _SPEC["max_pad_gap_m"]:
        return None

    face_windows = [abs(closure_projection - edge) <= max(0.003, 0.1 * visible_span) for edge in (low, high)]
    normal_axis = np.array([*axis, 0.0])
    normal_alignment = [_observed_patch(section[near_face], normal_axis) for near_face in face_windows]
    observed_faces = all(alignment is not None for alignment in normal_alignment)
    yaw = math.degrees(math.atan2(axis[1], axis[0]))
    delta = math.radians(yaw - reference_yaw)
    nominal_projection = abs(math.cos(delta)) * stock_spans[0] + abs(math.sin(delta)) * stock_spans[1]
    # The visible section span is a lower bound without opposed surface evidence.
    estimate = visible_span if observed_faces else max(visible_span + 0.004, nominal_projection)
    pose_xy = axis * ((low + high) / 2) + side_axis * pad_center_tangent
    tcp_z = pad_z + sum(_SPEC["pad_axial_bounds_from_tcp_m"]) / 2
    if tcp_z - floor_z < _SPEC["pad_axial_bounds_from_tcp_m"][1] + _FLOOR_MARGIN_M:
        return None

    within_finger_width = abs(points[:, :2] @ side_axis - pad_center_tangent) <= _SPEC["pad_face_width_m"] / 2
    past_open_pad = abs((points[:, :2] - pose_xy) @ axis) >= _SPEC["max_pad_gap_m"] / 2
    outside_gap = int(np.count_nonzero(within_finger_width & past_open_pad & (points[:, 2] >= pad_z)))
    between_fingers = points[within_finger_width & ~past_open_pad]
    observed_top = float(np.quantile(between_fingers[:, 2], 0.9)) if len(between_fingers) else pad_z
    evidence = {
        "shape_id": shape_id,
        "geometry": "observed_opposing_patches" if observed_faces else "inferred_from_partial_shape",
        "visible_span_m": round(visible_span, 4),
        "points_near_each_contact": [int(np.count_nonzero(mask)) for mask in face_windows],
        "normal_alignment": [round(value, 3) if value is not None else None for value in normal_alignment],
        "support_view_pairs": np.unique(section_pairs, axis=0).astype(int).tolist(),
        "source_frame_ids": evidence_frames,
        "observed_points_outside_open_pad_corridor": outside_gap,
        "required_insertion_m": max(0.0, round(observed_top - pad_z, 4)),
        "insertion_source": "near-pad visible surface above estimated pad center",
    }
    strategy = "observed_shape_pinch" if observed_faces else "inferred_shape_pinch"
    basis = ("triangulated opposed surface patches and calibrated Panda pad dimensions" if observed_faces
             else "triangulated partial cross-section, nominal closure bound and Panda pad dimensions")
    args = (tuple(float(value) for value in pose_xy), tcp_z - floor_z, yaw, estimate, strategy, basis)
    return args, evidence


def _shape_candidates(session, entry, object_id, shape_id, xy, yaw, short_span, long_span):
    shape, points, pairs = _shape_points(session, object_id, shape_id)
    if points is None or len(points) < 8:
        return [], "Too few reliable visible points to estimate a contact cross-section."
    frame_ids = list(shape["frame_ids"])
    candidates = []
    for axis in _shape_axes(points, shape, yaw):
        for pad_z in np.unique(np.quantile(points[:, 2], [0.30, 0.55, 0.80])):
            candidate = _section_candidate(points, pairs, axis, pad_z, session.floor_z, yaw,
                                           (short_span, long_span), shape_id, frame_ids)
            if candidate is not None:
                args, evidence = candidate
                if np.linalg.norm(np.asarray(args[0]) - xy) <= max(long_span * 0.75, 0.035):
                    candidates.append((args, evidence))
    candidates.sort(key=lambda item: (
        item[1]["observed_points_outside_open_pad_corridor"] >= 3,
        item[0][3] > _SPEC["max_pad_gap_m"],
        item[1]["geometry"] != "observed_opposing_patches",
        -min(item[1]["points_near_each_contact"]),
    ))
    return candidates[:3], "Partial shape has contact candidates." if candidates else "The visible shape has no supported section within the Panda pad dimensions."


def _ridge_candidates(ridge, floor_z, reference_yaw, short_span, long_span):
    xy = _xy(ridge, "midpoint_m")
    cross = np.asarray(_xy(ridge, "jaw_closing_axis"))
    length_axis = np.asarray(ridge.get("ridge_axis", [-cross[1], cross[0]])[:2], dtype=float)
    if np.linalg.norm(cross) < 1e-6 or np.linalg.norm(length_axis) < 1e-6:
        return []
    roof_height = float(ridge["midpoint_m"][2]) - floor_z
    visible_length = ridge.get("visible_length_m")
    endpoints = ridge.get("endpoints_m")
    if visible_length is None and endpoints is not None:
        visible_length = float(np.linalg.norm(np.asarray(endpoints[1]) - endpoints[0]))
    specs = []
    for strategy, axis, rim_inset, width_floor in (
        ("across_roof_ridge", cross, 0.012, ridge.get("grip_span_m")),
        ("ridge_end_pinch", length_axis, 0.006, visible_length),
    ):
        if strategy == "ridge_end_pinch" and width_floor is None:
            continue
        yaw = math.degrees(math.atan2(axis[1], axis[0]))
        difference = math.radians(yaw - reference_yaw)
        nominal_span = abs(math.cos(difference)) * short_span + abs(math.sin(difference)) * long_span
        span = max(nominal_span, float(width_floor)) if width_floor is not None else nominal_span
        height = roof_height - rim_inset
        pad_z_offset = -sum(_SPEC["pad_axial_bounds_from_tcp_m"]) / 2
        evidence = {"ridge_id": ridge.get("ridge_id"), "geometry": "inferred_contact_from_roi_edge",
                    "visible_ridge_length_m": visible_length,
                    "minimum_opening_m": width_floor,
                    "required_insertion_m": round(max(0.0, rim_inset - pad_z_offset), 4),
                    "insertion_source": "estimated roof height and calibrated pad center"}
        args = (xy, height, yaw, span, strategy, "RGB roof-edge hypothesis and Panda pad dimensions")
        specs.append((args, evidence))
    return specs


def propose_grasps(session, object_id: str, shape_id: str | None = None) -> dict:
    """Return selectable top-down poses at different heights and jaw angles."""
    entry = _current_object(session, object_id)
    width, depth, object_height = _dimensions(entry)
    xy = _xy(entry, "pick_xy", entry.get("position"))
    profile = PROFILES.get(entry["target"])
    nominal_yaw = profile.jaw_yaw_deg if profile else 0.0
    yaw = float(entry.get("yaw_deg", nominal_yaw))
    if not math.isfinite(yaw):
        raise RuntimeError("The estimated jaw yaw must be finite.")
    nominal_height = profile.grasp_height_m if profile else object_height * 0.5
    short_span, long_span = sorted((width, depth))
    basis = "LIBERO landmark and nominal grasp calibration" if profile else "localized object bounds"
    args = [(xy, nominal_height, yaw, short_span, "top_down", basis),
            (xy, nominal_height, yaw + 90, long_span, "rotated_top_down", basis)]
    if entry["target"] in TALL_TARGETS:
        args.insert(0, (xy, nominal_height + RAISE_M, yaw, short_span, "higher_top_down",
                        "25 mm higher than the LIBERO nominal pad height"))

    candidate_specs = [(candidate_args, None) for candidate_args in args]
    if entry.get("ridge"):
        candidate_specs.extend(_ridge_candidates(entry["ridge"], session.floor_z, yaw, short_span, long_span))
    shape_status = None
    if shape_id is not None:
        shaped, shape_status = _shape_candidates(session, entry, object_id, shape_id, xy, yaw, short_span, long_span)
        candidate_specs.extend(shaped)

    # Repeating proposals or adding independent shape evidence keeps earlier IDs stable.
    known = {candidate["source_key"]: candidate for candidate in session.grasps.values()
             if candidate["object_id"] == object_id}
    candidates = []
    for candidate_args, evidence in candidate_specs:
        evidence_id = (evidence.get("shape_id") or evidence.get("ridge_id")) if evidence else None
        key = (object_id, candidate_args, evidence_id)
        candidate = known.get(key)
        if candidate is None:
            candidate = _new_candidate(session, object_id, *candidate_args, evidence=evidence)
            candidate["source_key"] = key
        candidates.append(candidate)
    ranked = sorted(_annotate_preferences(session, entry, candidates), key=lambda item: item[0][0])
    recommended = next(((reason, candidate) for (_, reason), candidate in ranked
                        if 0 < candidate["estimated_grip_span_m"] <= _SPEC["max_pad_gap_m"]
                        and candidate.get("shape_evidence", {}).get("observed_points_outside_open_pad_corridor", 0) < 3
                        and (candidate.get("shape_evidence") or candidate.get("ridge_evidence") or {}).get("required_insertion_m", 0)
                        <= _SPEC["finger_base_from_pad_center_m"]), None)
    return {"object_id": object_id, "frame_id": entry["frame_id"],
            "shape_id": shape_id, "shape_status": shape_status,
            "recommended_grasp_id": recommended[1]["grasp_id"] if recommended else None,
            "recommendation_reason": recommended[0] if recommended else "No proposed span fits the pad gap.",
            "candidates": [{**{name: value for name, value in candidate.items() if name != "source_key"},
                            "preference_reason": reason} for (_, reason), candidate in ranked]}


def _inside_workspace(x: float, y: float) -> bool:
    return 0.30 <= x <= 0.80 and abs(y) <= 0.40


def _detached_kinematics(backend):
    build_model = getattr(backend, "planning_model", None)
    read_state = getattr(backend, "get_robot_state", None)
    if build_model is None or read_state is None:
        return None
    try:
        state = read_state()
        if state.success and state.value is not None:
            return build_model(), state.value.joint_positions_rad
    except (RuntimeError, ValueError, ImportError):
        # A backend can still provide the gripper-width and workspace screen.
        pass
    return None


def _check_endpoint(model, seed, candidate: dict, floor_z: float) -> dict:
    desired = np.eye(4)
    desired[:3, 3] = candidate["position_m"]
    desired[:3, :3] = Rotation.from_quat(np.roll(candidate["quaternion_wxyz"], -1)).as_matrix()
    empty_scene = np.empty((0, 3))

    def no_self_collision(joints):
        return not model.collision(joints, 1.0, empty_scene, first_hit_only=True)["collision"]

    joints = model.solve_ik(desired, seed, aperture=1.0, max_iterations=75,
                            position_tolerance=0.003, orientation_tolerance=0.03,
                            valid=no_self_collision)
    model.set_state(seed if joints is None else joints, 1.0)
    clearance = model.gripper_support_distance(desired, np.array([0.0, 0.0, 1.0]), -floor_z)
    diagnostic = model.last_ik
    return {"ik": "solved" if joints is not None else "no_solution_found",
            "self_collision": False if joints is not None else None,
            "position_residual_m": round(float(diagnostic["position_residual_m"]), 4),
            "gripper_floor_clearance_m": round(float(clearance), 4)}


def check_grasps(session, object_id: str) -> dict:
    """Check candidate width, floor margin and detached endpoint IK before planning."""
    entry = _current_object(session, object_id)
    candidates = [candidate for candidate in session.grasps.values() if candidate["object_id"] == object_id]
    if not candidates:
        raise RuntimeError("Call propose_grasps for this object first.")
    _, _, object_height = _dimensions(entry)
    kinematics = _detached_kinematics(session.backend)
    checks = []
    for (preference, preference_reason), candidate in _annotate_preferences(session, entry, candidates):
        grip_span = candidate["estimated_grip_span_m"]
        height = candidate["grasp_height_m"]
        floor_clearance = height - _SPEC["pad_axial_bounds_from_tcp_m"][1]
        x, y, _ = candidate["position_m"]
        reasons = []
        warnings = []
        evidence = candidate.get("shape_evidence")
        insertion_evidence = evidence or candidate.get("ridge_evidence")
        minimum_opening = (insertion_evidence or {}).get("minimum_opening_m")
        if not math.isfinite(grip_span) or not 0 < grip_span <= _SPEC["max_pad_gap_m"]:
            reasons.append("Observed ridge length already exceeds the maximum pad gap." if minimum_opening is not None
                           and minimum_opening > _SPEC["max_pad_gap_m"]
                           else "Estimated span exceeds usable pad opening or has no valid measurement.")
        elif grip_span > _SPEC["recommended_grip_span_m"]:
            warnings.append("Little extra opening remains. Approach carefully and inspect the result.")
        if not math.isfinite(height) or floor_clearance < _FLOOR_MARGIN_M:
            reasons.append("The pad would approach too close to the calibrated floor.")
        if height > object_height + _SPEC["pad_axial_bounds_from_tcp_m"][1]:
            reasons.append("The pads would be above the estimated object surface.")
        if not _inside_workspace(x, y):
            reasons.append("This pose is outside the calibrated Panda workspace.")
        if insertion_evidence is not None and insertion_evidence["required_insertion_m"] > _SPEC["finger_base_from_pad_center_m"]:
            reasons.append("Required insertion would extend beyond the pad-to-finger-base allowance.")
        if evidence is not None:
            try:
                _current_shape(session, object_id, evidence["shape_id"])
            except RuntimeError as error:
                reasons.append(str(error))
            if evidence["geometry"] == "inferred_from_partial_shape":
                warnings.append("No reliable opposed patches. The estimated contacts include unseen surfaces.")
            if evidence["observed_points_outside_open_pad_corridor"] >= 3:
                reasons.append("Several observed target points overhang the open pads along this approach.")
        ik = {"ik": "skipped_geometry_rejected" if reasons else "unavailable"}
        if not reasons and kinematics is not None:
            ik = _check_endpoint(*kinematics, candidate, session.floor_z)
            if ik["ik"] == "no_solution_found":
                warnings.append("Endpoint IK found no collision-free joint solution within its search budget.")
            if ik["gripper_floor_clearance_m"] < -0.001:
                reasons.append("The open gripper envelope intersects the calibrated floor.")
            elif ik["gripper_floor_clearance_m"] < 0:
                warnings.append("The open gripper is within 1 mm of the calibrated floor; move with care.")
        grasp_id = candidate["grasp_id"]
        result = {"grasp_id": grasp_id, "strategy": candidate["strategy"], "passed": not reasons,
                  "reasons": reasons, "warnings": warnings,
                  "preference_reason": preference_reason,
                  "pad_gap_margin_m": round(_SPEC["max_pad_gap_m"] - grip_span, 4),
                  "pad_floor_clearance_m": round(floor_clearance, 4), "endpoint": ik}
        if evidence is not None:
            result["shape_id"] = evidence["shape_id"]
            result["contact_geometry"] = evidence["geometry"]
        if insertion_evidence is not None:
            result["required_insertion_m"] = insertion_evidence["required_insertion_m"]
            result["screened_insertion_limit_m"] = _SPEC["finger_base_from_pad_center_m"]
        session.grasp_checks[grasp_id] = result
        checks.append(((bool(reasons), ik["ik"] == "no_solution_found", preference), result))
    ranked = [result for _, result in sorted(checks, key=lambda item: item[0])]
    recommended = ranked[0] if ranked[0]["passed"] else None
    return {"object_id": object_id, "frame_id": entry["frame_id"], "checks": ranked,
            "recommended_grasp_id": recommended["grasp_id"] if recommended else None,
            "recommendation_reason": recommended["preference_reason"] if recommended else "No candidate passed the geometry screen.",
            "scope": "Geometry and optional detached open-gripper endpoint IK. The scene, full path and grasp retention require separate verification."}


def plan_motion(session, grasp_id: str, destination_id: str | None = None) -> dict:
    """Create individually callable waypoints for one checked grasp pose."""
    candidate = session.grasps.get(grasp_id)
    if candidate is None:
        raise RuntimeError("Unknown grasp_id. Select a candidate from propose_grasps.")
    entry = _current_object(session, candidate["object_id"])
    source_frame_id = entry["frame_id"]
    evidence = candidate.get("shape_evidence")
    if evidence is not None:
        _current_shape(session, candidate["object_id"], evidence["shape_id"])
        source_frame_id = evidence["source_frame_ids"][-1]
    check = session.grasp_checks.get(grasp_id)
    if check is None:
        raise RuntimeError("Call check_grasps for this object before planning.")
    if not check["passed"]:
        raise RuntimeError("This candidate did not pass check_grasps. Select a passing candidate or localize again.")

    grasp = candidate["position_m"]
    quaternion = candidate["quaternion_wxyz"]
    height = candidate["grasp_height_m"]
    hover = (*grasp[:2], session.floor_z + max(0.25, height + 0.17))
    waypoints = {"pick_hover": (hover, quaternion), "grasp": (grasp, quaternion),
                 "lift": (hover, quaternion)}
    if destination_id is not None:
        destination = session.destinations.get(destination_id)
        if destination is None:
            raise RuntimeError("Unknown destination_id. Use locate_basket to obtain the ID.")
        if session.frame_steps.get(destination["frame_id"]) != session.backend.simulation_steps:
            raise RuntimeError("Basket estimate is stale. Observe and locate it again before planning.")
        place = _xy(destination, "place_xy", destination.get("position"))
        if not _inside_workspace(*place):
            raise RuntimeError("Basket estimate is outside the calibrated Panda workspace. Observe again.")
        waypoints["basket"] = ((*place, session.floor_z + max(0.235, height + 0.17)), quaternion)

    plan_id = f"grasp_{len(session.plans) + 1}"
    session.plans[plan_id] = GraspPlan(plan_id, entry["target"], waypoints, source_frame_id, height)
    return {"plan_id": plan_id, "grasp_id": grasp_id, "target": entry["target"],
            "strategy": candidate["strategy"], "source_frame_id": source_frame_id, "destination_id": destination_id,
            "shape_id": evidence["shape_id"] if evidence else None,
            "grasp_height_m": round(height, 4), "waypoints": list(waypoints)}


TOOLS = {
    "get_gripper_spec": (
        get_gripper_spec,
        "Read the Panda gripper's maximum pad gap, finger dimensions and usable grasping clearance. "
        "Use when deciding how wide an object or its narrow section may be grabbed.",
        {},
    ),
    "propose_grasps": (
        propose_grasps,
        "Propose top-down grasp positions and jaw angles for a localized object. Pass shape_id from reconstruct_shape "
        "to use measured cross-sections and distinguish observed from inferred contacts. A roof edge from find_ridge "
        "adds perpendicular and along-ridge pinches. Returns candidate IDs and "
        "recommended_grasp_id with the reason.",
        {"object_id": {"type": "string"}, "shape_id": {"type": "string", "optional": True}},
    ),
    "check_grasps": (
        check_grasps,
        "Screen proposed grasps for pad opening, floor clearance, workspace bounds and endpoint IK when supported. "
        "Read the ranked candidates and recommended_grasp_id before selecting a grasp for plan_motion. "
        "Recommendations include evidence from any recent stalled descent near this object.",
        {"object_id": {"type": "string"}},
    ),
    "plan_motion": (
        plan_motion,
        "Build waypoints for one grasp candidate you selected after checking it. The available waypoints are "
        "pick_hover, grasp, lift. Pass a destination_id from locate_basket to add the basket waypoint. "
        "Returns a plan ID. The arm stays still.",
        {"grasp_id": {"type": "string"}, "destination_id": {"type": "string", "optional": True}},
    ),
}

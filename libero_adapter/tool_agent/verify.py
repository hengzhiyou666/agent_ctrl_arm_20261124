"""Check basket placement from fresh RGB and the matching release receipt."""

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, find_objects, label, uniform_filter

from libero_adapter.rgb import PICK_LANDMARKS, _read_rgb, _target_mask, _target_pixels, largest_component


def _camera(observation, name):
    return next((view for view in observation.cameras if view.camera_name == name), None)


def _change(before, after):
    difference = np.abs(before - after)
    return (difference.max(axis=2) >= 29) & (difference.mean(axis=2) >= 16)


def _component(mask):
    labels, count = label(binary_opening(mask))
    if not count:
        return np.zeros_like(mask), None
    region = int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    return labels == region, find_objects(labels)[region - 1]


def _bbox(rows, columns):
    return [columns.start, rows.start, columns.stop, rows.stop]


def _basket_opening(rgb, wrist=False):
    height, width = rgb.shape[:2]
    rows, columns = np.ogrid[:height, :width]
    if wrist:
        horizontal, vertical, brightness = (0.25, 0.86), (0.35, 0.83), 200
    else:
        horizontal, vertical, brightness = (0.57, 0.83), (0.34, 0.66), 170
    liner = (rgb.min(axis=2) > brightness) & (np.ptp(rgb, axis=2) < 25)
    liner &= ((horizontal[0] * width < columns) & (columns < horizontal[1] * width)
              & (vertical[0] * height < rows) & (rows < vertical[1] * height))
    rim = largest_component(binary_closing(liner, iterations=2), max(30, round(height * width * 0.0006)))
    r, c = rim
    rim_height, rim_width = r.stop - r.start, c.stop - c.start
    if rim_height < 0.07 * height or rim_width < 0.12 * width:
        raise RuntimeError("Too little of the basket liner is visible.")
    x_inset = round(rim_width * (0.16 if wrist else 0.10))
    y_inset = round(rim_height * (0.15 if wrist else 0.12))
    opening = (slice(r.start + y_inset, r.stop - round(rim_height * 0.09)),
               slice(c.start + x_inset, c.stop - x_inset))
    return liner, rim, opening


def _source_change(before, changed, target, detection):
    height, width = before.shape[:2]
    minimum = max(12, round(height * width * 0.00012))
    if detection:
        x0, y0, x1, y1 = detection["bbox_xyxy"]
        patch = np.s_[y0:y1, x0:x1]
        color = _target_mask(before[patch], target)
        if color.sum() >= minimum:
            return float(changed[patch][color].mean()), int(color.sum())
    if target == "chocolate_pudding":
        # The initial tray is generally seen by the wrist. A fixed-camera
        # chromatic match in this case can be a different object.
        return None, 0
    try:
        x, y = _target_pixels(before, target)
    except RuntimeError:
        return None, 0
    dx, dy = max(5, round(0.027 * width)), max(6, round(0.045 * height))
    patch = np.s_[max(0, round(y) - dy):round(y) + dy + 1,
                  max(0, round(x) - dx):round(x) + dx + 1]
    color = _target_mask(before[patch], target)
    pixels = int(color.sum())
    if pixels < minimum:
        return None, pixels
    return float(changed[patch][color].mean()), pixels


def _wrist_opening(rgb):
    height, width = rgb.shape[:2]
    rows, columns = np.ogrid[:height, :width]
    bright = (rgb.min(axis=2) > 195) & (np.ptp(rgb, axis=2) < 25)
    bright &= ((0.12 * width < columns) & (columns < 0.91 * width)
               & (0.32 * height < rows) & (rows < 0.74 * height))
    # The liner has a wide, thick top rim. Bright floor seams are thinner.
    structure = np.ones((max(3, round(0.016 * height)), max(3, round(0.19 * width))), dtype=bool)
    regions, count = label(binary_opening(bright, structure=structure))
    if not count:
        return _basket_opening(rgb, wrist=True)[2], False
    r, c = max(find_objects(regions), key=lambda box: box[1].stop - box[1].start)
    rim_width = c.stop - c.start
    margin = round(0.02 * width)
    return (slice(r.stop, min(round(0.82 * height), r.stop + round(0.90 * rim_width))),
            slice(c.start - margin, c.stop + margin)), True


def _can_lid(interior):
    # A silver lid has a restrained, fairly even gray face. The basket's
    # similarly gray creases tend to be thin, dimmer and less compact.
    gray = interior.mean(axis=2)
    variance = uniform_filter(gray * gray, 5) - uniform_filter(gray, 5) ** 2
    neutral = ((np.ptp(interior, axis=2) <= 20) & (gray >= 128) & (gray <= 208)
               & (variance < 130))
    labels, _ = label(binary_opening(neutral, iterations=2))
    height, width = gray.shape
    for index, (rows, columns) in enumerate(find_objects(labels), start=1):
        lid_height, lid_width = rows.stop - rows.start, columns.stop - columns.start
        center_x = (columns.start + columns.stop) / 2
        candidate = labels[rows, columns] == index
        if (0.18 * width <= center_x <= 0.86 * width and 0.65 <= lid_width / lid_height <= 1.5
                and candidate.sum() >= max(18, 0.027 * gray.size) and candidate.mean() >= 0.50
                and np.median(gray[rows, columns][candidate]) >= 158
                and lid_height >= 0.16 * height and lid_width >= 0.12 * width):
            return True
    return False


def _wrist_evidence(view, target):
    if view is None:
        return {"usable": False}
    try:
        rgb = _read_rgb(view)
        opening, rim_found = _wrist_opening(rgb)
    except (OSError, RuntimeError):
        return {"usable": False}

    interior = rgb[opening]
    red, green, blue = np.moveaxis(interior, 2, 0)
    bright = interior.max(axis=2) >= 65
    color = _target_mask(interior, target) & bright
    if target == "chocolate_pudding":
        # The tray's brown side is dark when it rests below the front rim.
        color = (red > green + 7) & (red > blue + 9) & (red > 29)
    elif target == "ketchup":
        color &= red > green + 14
    foreground, bounds = _component(color)
    color_fraction = float(foreground.mean())
    color_centered = False
    if bounds is not None:
        rows, columns = bounds
        height, width = interior.shape[:2]
        color_centered = (0.12 * width <= (columns.start + columns.stop) / 2 <= 0.88 * width
                          and rows.start < 0.88 * height)
        if target == "chocolate_pudding":
            color_centered = (0.22 * width <= (columns.start + columns.stop) / 2 <= 0.78 * width
                              and rows.start < 0.75 * height
                              and 0.55 <= (columns.stop - columns.start) / (rows.stop - rows.start) <= 2.4
                              and foreground[rows, columns].mean() > 0.42)
    lid_found = rim_found and target in ("alphabet_soup", "tomato_sauce") and _can_lid(interior)
    color_threshold = 0.007 if rim_found and target == "cream_cheese" else 0.012

    return {
        "usable": True,
        "rim_detected": rim_found,
        "opening_xyxy": _bbox(*opening),
        "target_color_fraction": round(color_fraction, 4),
        "can_lid_visible": bool(lid_found),
        "target_cue": bool((color_fraction >= color_threshold and color_centered) or lid_found),
    }


def _overview_evidence(before, after, target, detection):
    liner, rim, opening = _basket_opening(before)
    changed = _change(before, after)
    unchanged_liner = float((liner[rim] & ~changed[rim]).sum() / liner[rim].sum())
    interior = after[opening]
    foreground, _ = _component(changed[opening])
    appearance = _target_mask(interior, target)
    red, green, blue = np.moveaxis(interior, 2, 0)
    robot_blue = (blue > red + 14) & (blue > green + 12)
    robot_green = (green > red + 10) & (green > blue + 10)
    arm_color = robot_blue if target in ("salad_dressing", "tomato_sauce") else robot_blue | robot_green
    source_change, source_pixels = _source_change(before, changed, target, detection)
    return {
        "opening_xyxy": _bbox(*opening),
        "liner_still_visible": round(unchanged_liner, 4),
        "new_basket_foreground_fraction": round(float((foreground & ~arm_color).mean()), 4),
        "new_basket_target_color_fraction": round(float((foreground & appearance & ~arm_color).mean()), 4),
        "source_change_fraction": round(source_change, 4) if source_change is not None else None,
        "source_color_pixels": source_pixels,
    }


def verify_place(session, frame_id: str, target: str) -> dict:
    """Classify one post-release observation using only visible evidence."""
    release_plan_id = session.basket_release_plan_id
    plan = session.plans.get(release_plan_id)
    if plan is None or session.placement_frames.get(frame_id) != release_plan_id:
        raise RuntimeError("Use inspect_place after completing a release at the basket.")
    if session.frame_steps.get(frame_id) != session.backend.simulation_steps:
        raise RuntimeError("This inspection frame is stale. Call inspect_place again.")
    if target != plan.target:
        raise RuntimeError(f"Use the released plan's target, {plan.target!r}.")

    source_frame_id = plan.source_frame_id
    current = session.frames[frame_id]
    source = session.frames.get(source_frame_id)
    state = current.robot_state
    open_fraction = float(state.gripper_open_fraction) if state is not None else None
    result = {
        "status": "uncertain",
        "frame_id": frame_id,
        "source_frame_id": source_frame_id,
        "target": target,
        "release_plan_id": release_plan_id,
        "evidence": {"gripper_open_fraction": round(open_fraction, 4) if open_fraction is not None else None},
        "reason": "The post-release frame does not clearly show an open gripper.",
    }
    if open_fraction is not None and open_fraction >= 0.75:
        _classify(session, result, source, current)
    session.verifications[frame_id] = result
    return result


def _classify(session, result, source, current):
    target = result["target"]
    evidence = result["evidence"]
    evidence["wrist"] = _wrist_evidence(_camera(current, "robot0_eye_in_hand"), target)
    before_camera = _camera(source, "agentview") if source is not None else None
    after_camera = _camera(current, "agentview")
    if before_camera is None or after_camera is None:
        result["reason"] = "A matching pre-grasp and post-release overview is unavailable."
        return
    try:
        before, after = _read_rgb(before_camera), _read_rgb(after_camera)
        if before.shape != after.shape:
            raise RuntimeError("Overview image sizes differ.")
        if not np.allclose(before_camera.extrinsics_4x4, after_camera.extrinsics_4x4, atol=0.002):
            raise RuntimeError("The overview camera moved between observations.")
        detection = next((d for d in session.detections.values()
                          if d["frame_id"] == result["source_frame_id"]
                          and d["target"] == target and d["camera"] == "overview"), None)
        overview = _overview_evidence(before, after, target, detection)
    except (OSError, RuntimeError):
        result["reason"] = "The fixed-camera views could not be aligned or the basket liner is hard to see."
        return
    evidence["overview"] = overview
    wrist_cue = evidence["wrist"].get("target_cue", False)
    source_change = overview["source_change_fraction"]
    changed_basket = overview["new_basket_foreground_fraction"]
    colored_basket = overview["new_basket_target_color_fraction"]
    visible_rim = overview["liner_still_visible"] >= 0.55
    if source_change is not None and source_change <= 0.20 and changed_basket < 0.015 and not wrist_cue:
        result.update(status="outside", reason="The original target is still visible at its starting location.")
    elif (visible_rim and source_change is not None and source_change >= 0.65
          and (wrist_cue or colored_basket >= 0.013 or changed_basket >= 0.09)):
        result.update(status="inside", reason="The original target moved and fresh RGB shows foreground within the basket.")
    elif (visible_rim and target == "chocolate_pudding" and source_change is None and wrist_cue
          and evidence["wrist"].get("rim_detected")
          and evidence["wrist"]["target_color_fraction"] >= 0.024):
        result.update(status="inside", reason="The wrist camera sees a pudding-colored component inside the detected basket rim.")
    else:
        result["reason"] = "The basket contents or target identity are unclear. Inspect both views and check for a dropped object."


TOOLS = {
    "verify_place": (
        verify_place,
        "Check a fresh frame from inspect_place after opening at the basket. Match the target in the release plan. "
        "Returns inside when RGB supports placement, outside when the target remains at its source, "
        "or uncertain when evidence is unclear. Use this result before finish_task.",
        {"frame_id": {"type": "string", "description": "A fresh frame ID returned by inspect_place."},
         "target": {"type": "string", "enum": list(PICK_LANDMARKS),
                    "description": "Target name in the released grasp plan."}},
    ),
}

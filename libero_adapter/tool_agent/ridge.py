"""Estimate a carton roof line from RGB edges and calibrated projection."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from libero_adapter.rgb import point_on_plane


# The camera ray needs a metric surface to meet. These heights are reference
# carton geometry measured from the ground plane, not a depth sensor reading.
NOMINAL_ROOF_HEIGHT_M = {"milk": 0.143, "orange_juice": 0.144}


def _roof_edges(rgb, box, landmark):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width < 10 or height < 12:
        return []
    u, v = landmark
    # The detection landmark can be on the label just below the roof. Sample
    # above it as well, without considering the many printed edges below it.
    x_low = max(0, left - round(width * 0.08))
    x_high = min(rgb.shape[1], right + round(width * 0.08))
    y_low = max(0, min(top, round(v)) - max(4, round(height * 0.16)))
    y_high = min(rgb.shape[0], bottom, round(v + max(4, height * 0.15)))
    crop = rgb[y_low:y_high, x_low:x_high]
    if crop.shape[0] < 5 or crop.shape[1] < 10:
        return []

    scale = 3
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(gray, 12, 30)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12,
                            minLineLength=scale * max(8, round(width * 0.24)), maxLineGap=4)
    if lines is None:
        return []

    candidates = []
    for a, b, c, d in np.asarray(lines).reshape(-1, 4):
        p = np.array([x_low + a / scale, y_low + b / scale])
        q = np.array([x_low + c / scale, y_low + d / scale])
        delta = q - p
        length = np.linalg.norm(delta)
        center = (p + q) / 2
        if abs(delta[0]) < 0.72 * length or abs(center[0] - u) > width * 0.42:
            continue
        if not v - max(8, height * 0.16) <= center[1] <= v + max(2, height * 0.05):
            continue
        # An upper edge with enough support is a better roof hypothesis than
        # a long printed line on the front panel just below it.
        support = length / width - 0.45 * abs(center[0] - u) / width
        score = support + 0.075 * (v - center[1])
        if p[0] > q[0]:
            p, q = q, p
        candidates.append((float(score), float(support), p, q))
    return sorted(candidates, key=lambda candidate: candidate[0], reverse=True)


def find_ridge(session, object_id: str):
    """Find a visible top edge and return a provisional 3D ridge hypothesis."""
    obj = session.objects.get(object_id)
    if obj is None:
        raise RuntimeError("Use an object_id from locate_object.")
    target = obj["target"]
    if target not in NOMINAL_ROOF_HEIGHT_M:
        return {"status": "unsupported", "object_id": object_id,
                "reason": "This roof-edge method is calibrated for the milk and orange juice cartons."}
    frame_id = obj["frame_id"]
    if session.frame_steps.get(frame_id) != session.backend.simulation_steps:
        raise RuntimeError("Observe and locate the object again to use a fresh roof image.")
    detection = session.detections[obj["detection_id"]]
    camera = next((item for item in session.frames[frame_id].cameras
                   if item.camera_name == detection["camera_name"]), None)
    if camera is None:
        raise RuntimeError("The detection's calibrated camera is unavailable.")
    if camera.camera_name != "agentview":
        return {"status": "uncertain", "object_id": object_id,
                "reason": "The roof-edge crop expects the oblique overview. Try that view or another grasp."}

    with Image.open(Path(detection["image_path"])) as file:
        image = file.convert("RGB")
    rgb = np.asarray(image)
    height = NOMINAL_ROOF_HEIGHT_M[target]
    center_xy = np.asarray(obj["position"][:2])
    dims = obj["dimensions_m"]
    footprint = max(float(dims["width"]), float(dims["depth"]))
    candidates = _roof_edges(rgb, detection["bbox_xyxy"], detection["landmark_uv"])
    for _, support, p_uv, q_uv in candidates:
        p = point_on_plane(camera, p_uv, session.floor_z + height)
        q = point_on_plane(camera, q_uv, session.floor_z + height)
        midpoint = (p + q) / 2
        length = np.linalg.norm(q - p)
        if not max(0.018, footprint * 0.35) <= length <= footprint * 1.35:
            continue
        if np.linalg.norm(midpoint[:2] - center_xy) > footprint * 0.85:
            continue

        axis = (q - p) / length
        closing_axis = [-float(axis[1]), float(axis[0]), 0.0]
        ridge_id = f"r{len(session.ridges) + 1}"
        path = session.output / f"{ridge_id}-roof-edge.png"
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(detection["bbox_xyxy"]), outline="#eaca51", width=2)
        draw.line((tuple(p_uv), tuple(q_uv)), fill="#52f4ff", width=2)
        u, v = (p_uv + q_uv) / 2
        draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill="#52f4ff")
        image.save(path)
        receipt = {
            "status": "hypothesis", "ridge_id": ridge_id, "object_id": object_id, "frame_id": frame_id,
            "camera": detection["camera"], "coordinate_frame": "robot_base", "unit": "m",
            "method": "OpenCV roof-edge detection and calibrated ray projection onto a nominal height plane",
            "roof_height_assumed_m": height, "roof_height_source": "nominal_libero_carton_height",
            "endpoints_uv": [p_uv.round(2).tolist(), q_uv.round(2).tolist()],
            "endpoints_m": [p.round(4).tolist(), q.round(4).tolist()],
            "midpoint_m": midpoint.round(4).tolist(), "ridge_axis": axis.round(4).tolist(),
            "jaw_closing_axis": closing_axis, "approach_axis": [0.0, 0.0, -1.0],
            "visible_length_m": round(float(length), 4),
            "confidence": round(min(0.75, max(0.0, support)), 3), "confidence_type": "heuristic_edge_support",
            "uncertainty": "A roof silhouette edge can be offset from the fold. Use as a geometry candidate and check the grasp before moving.",
            "annotation": path.name,
        }
        session.ridges[ridge_id] = receipt
        obj["ridge"] = receipt
        return receipt, {"roof_edge": path}

    return {"status": "uncertain", "object_id": object_id, "frame_id": frame_id,
            "reason": "No supported roof edge near the detected carton. Try another view or evaluate another grasp candidate."}


TOOLS = {
    "find_ridge": (
        find_ridge,
        "Find a visible roof edge on a located milk or orange-juice carton. Returns a provisional 3D "
        "midpoint and jaw closing direction, or uncertainty. Use before proposing a roof pinch.",
        {"object_id": {"type": "string"}},
    ),
}

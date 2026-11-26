"""RGB detection, segmentation, and metric grounding for LIBERO Object."""

from itertools import product
from math import cos, radians, sin
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label

from libero_adapter.motion import PROFILES
from libero_adapter.rgb import (PICK_LANDMARKS, _basket_pixels, _read_rgb,
                                _target_mask, _target_pixels, point_on_plane)
from libero_adapter.skills.observation import CAMERA_NAMES


# Bounds in metres of the scaled reference visual meshes from LIBERO's HOPE
# assets, in their usual upright or flat placement. Useful as priors on this
# task family when only a partial RGB view is available.
OBJECT_GEOMETRY = {
    "alphabet_soup": (0.066, 0.071, 0.084, "cylinder"),
    "cream_cheese": (0.083, 0.043, 0.019, "flat_box"),
    "salad_dressing": (0.044, 0.064, 0.147, "bottle"),
    "bbq_sauce": (0.033, 0.050, 0.114, "bottle"),
    "ketchup": (0.043, 0.065, 0.149, "bottle"),
    "tomato_sauce": (0.066, 0.070, 0.083, "cylinder"),
    "butter": (0.077, 0.040, 0.018, "flat_box"),
    "milk": (0.054, 0.055, 0.143, "ridged_carton"),
    "chocolate_pudding": (0.083, 0.049, 0.030, "flat_box"),
    "orange_juice": (0.054, 0.055, 0.144, "ridged_carton"),
}


def _frame(session, frame_id):
    if frame_id not in session.frames:
        raise RuntimeError("unknown frame_id; capture a frame with observe")
    if session.frame_steps[frame_id] != session.backend.simulation_steps:
        raise RuntimeError("frame is stale after robot motion; observe and detect again")
    return {view.camera_name: view for view in session.frames[frame_id].cameras}


def _camera(views, camera):
    view = views.get(CAMERA_NAMES[camera])
    if view is None:
        raise RuntimeError(f"camera {camera} is unavailable in this frame")
    return view


def _artifact(session, name):
    folder = session.output / "vision"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def _bbox(camera, image_shape, uv, xyz, floor_z, dimensions):
    """Bound the expected silhouette by projecting size around the RGB landmark."""
    image_height, image_width = image_shape[:2]
    width, depth, height, _ = dimensions
    x, y = xyz[:2]
    corners = np.array(list(product((x - width / 2, x + width / 2),
                                    (y - depth / 2, y + depth / 2),
                                    (floor_z, floor_z + height))))
    pose = np.asarray(camera.extrinsics_4x4).reshape(4, 4)
    intrinsics = np.asarray(camera.intrinsics_3x3).reshape(3, 3)
    optical = (corners - pose[:3, 3]) @ pose[:3, :3]
    if np.any(optical[:, 2] <= 1e-3):
        raise RuntimeError("the RGB landmark projects too close to the camera")
    uv_h = optical @ intrinsics.T
    projected = np.vstack((uv_h[:, :2] / uv_h[:, 2:3], uv))
    minimum = np.floor(projected.min(axis=0)).astype(int) - 5
    maximum = np.ceil(projected.max(axis=0)).astype(int) + 5
    x0, y0 = np.maximum((0, 0), minimum)
    x1, y1 = np.minimum((image_width, image_height), maximum)
    if x1 - x0 < 5 or y1 - y0 < 5:
        raise RuntimeError("the projected target is mostly outside the image; observe again")
    return tuple(map(int, (x0, y0, x1, y1)))


def detect_object(session, frame_id: str, target: str):
    """Find a known object on an RGB frame and save an annotated image."""
    if target not in OBJECT_GEOMETRY:
        raise RuntimeError(f"target {target!r} has no RGB detector profile")
    views = _frame(session, frame_id)
    # The tall neighbouring ketchup bottle obscures the pudding in the overview.
    camera_name = "wrist" if target == "chocolate_pudding" else "overview"
    camera = _camera(views, camera_name)
    rgb = _read_rgb(camera)
    uv = _target_pixels(rgb, target)
    landmark_z = session.floor_z + PICK_LANDMARKS[target][0]
    point = point_on_plane(camera, uv, landmark_z)
    bbox = _bbox(camera, rgb.shape, uv, point, session.floor_z, OBJECT_GEOMETRY[target])
    x0, y0, x1, y1 = bbox
    components, count = label(binary_closing(_target_mask(rgb[y0:y1, x0:x1], target)))
    largest = int(np.bincount(components.ravel())[1:].max()) if count else 0
    confidence = round(min(0.95, 0.5 + 0.45 * largest / max(40.0, 0.22 * (x1 - x0) * (y1 - y0))), 2)
    detection_id = f"d{len(session.detections) + 1}"
    boxed = _artifact(session, f"{detection_id}_box.png")
    image = Image.fromarray(rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#00e0a6", width=2)
    draw.text((x0, max(0, y0 - 12)), target, fill="#f5fff9", stroke_width=1, stroke_fill="#0b3028")
    image.save(boxed)
    record = {
        "frame_id": frame_id, "target": target, "camera": camera_name,
        "camera_name": camera.camera_name, "bbox_xyxy": bbox,
        "image_path": Path(camera.rgb_path), "boxed_path": boxed, "landmark_uv": uv,
    }
    session.detections[detection_id] = record
    return {
        "detection_id": detection_id, "frame_id": frame_id, "target": target,
        "camera": camera_name, "bbox_xyxy": list(bbox),
        "confidence": confidence, "confidence_type": "heuristic_rgb_support",
        "boxed_image": str(boxed.relative_to(session.output)),
    }, {"boxed_target": boxed}


def _detection(session, detection_id):
    detection = session.detections.get(detection_id)
    if detection is None:
        raise RuntimeError("unknown detection_id; call detect_object first")
    views = _frame(session, detection["frame_id"])
    return detection, views


def _seeded_component(mask, uv):
    components, count = label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(components.ravel())
    ys, xs = np.nonzero(mask & (counts[components] >= 5))
    if not xs.size:
        return np.zeros_like(mask, dtype=bool)
    index = np.argmin((xs - uv[0]) ** 2 + (ys - uv[1]) ** 2)
    return components == components[ys[index], xs[index]]


def _refine_mask(crop, seed, bbox_slice, uv):
    try:
        import cv2
    except ImportError:
        return seed, "visible_rgb_component"

    previous_threads = cv2.getNumThreads()
    # Keep GrabCut predictable when many evaluation workers share one machine.
    cv2.setNumThreads(1)
    try:
        labels = np.full(seed.shape, cv2.GC_BGD, dtype=np.uint8)
        labels[bbox_slice] = cv2.GC_PR_FGD
        labels[seed] = cv2.GC_FGD
        backgrounds = np.zeros((1, 65), dtype=np.float64)
        foregrounds = np.zeros_like(backgrounds)
        cv2.grabCut(crop, labels, None, backgrounds, foregrounds, 1, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed, "visible_rgb_component"
    finally:
        cv2.setNumThreads(previous_threads)

    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    foreground = binary_fill_holes(_seeded_component(foreground, uv))
    box_area = int(np.prod(labels[bbox_slice].shape))
    if foreground.sum() <= seed.sum() or foreground.sum() > 0.9 * box_area:
        return seed, "visible_rgb_component"
    return foreground, "rgb_seeded_grabcut"


def segment_object(session, detection_id: str):
    """Use the detected visual landmark to seed an RGB foreground mask."""
    detection, views = _detection(session, detection_id)
    rgb = _read_rgb(_camera(views, detection["camera"]))
    x0, y0, x1, y1 = detection["bbox_xyxy"]
    image_height, image_width = rgb.shape[:2]
    left, top = max(0, x0 - 4), max(0, y0 - 4)
    right, bottom = min(image_width, x1 + 4), min(image_height, y1 + 4)
    region = rgb[top:bottom, left:right]
    bbox_slice = np.s_[y0 - top:y1 - top, x0 - left:x1 - left]
    uv = np.asarray(detection["landmark_uv"]) - [left, top]
    color = np.zeros(region.shape[:2], dtype=bool)
    color[bbox_slice] = _target_mask(region[bbox_slice], detection["target"])
    seed = _seeded_component(binary_closing(color, iterations=2), uv)
    if not seed.any():
        raise RuntimeError("insufficient matching RGB pixels to segment this detection; observe again")
    mask, method = _refine_mask(region.astype(np.uint8), seed, bbox_slice, uv)
    full_mask = np.zeros(rgb.shape[:2], dtype=bool)
    full_mask[top:bottom, left:right] = mask
    mask_id = f"m{len(session.masks) + 1}"
    path = _artifact(session, f"{mask_id}_mask.png")
    Image.fromarray(full_mask.astype(np.uint8) * 255).save(path)
    overlay = _artifact(session, f"{mask_id}_overlay.png")
    edge = binary_dilation(full_mask) & ~full_mask
    display = rgb.astype(np.uint8)
    display[full_mask] = (0.55 * display[full_mask] + 0.45 * np.array([0, 230, 151])).astype(np.uint8)
    display[edge] = [0, 230, 151]
    Image.fromarray(display).save(overlay)
    record = {"detection_id": detection_id, "frame_id": detection["frame_id"],
              "camera": detection["camera"], "target": detection["target"],
              "mask_path": path, "overlay_path": overlay, "method": method}
    session.masks[mask_id] = record
    return {
        "mask_id": mask_id, "detection_id": detection_id, "camera": detection["camera"],
        "pixel_count": int(full_mask.sum()), "method": method,
        "mask_image": str(path.relative_to(session.output)),
        "overlay_image": str(overlay.relative_to(session.output)),
    }, {"segmentation_overlay": overlay}


def _mask(session, mask_id, detection_id):
    record = session.masks.get(mask_id)
    if record is None or record["detection_id"] != detection_id:
        raise RuntimeError("mask_id must come from segment_object for the same detection_id")
    with Image.open(record["mask_path"]) as image:
        return np.asarray(image.convert("L")) > 0


def _backproject(camera, xs, ys):
    pixels = np.stack((xs, ys, np.ones_like(xs)))
    intrinsics = np.asarray(camera.intrinsics_3x3).reshape(3, 3)
    pose = np.asarray(camera.extrinsics_4x4).reshape(4, 4)
    return (pose[:3, :3] @ np.linalg.solve(intrinsics, pixels)).T, pose[:3, 3]


def _flat_axis(camera, mask, top_z, dimensions):
    ys, xs = np.nonzero(mask)
    if xs.size < 30:
        return None
    rays, origin = _backproject(camera, xs, ys)
    if np.any(rays[:, 2] >= -1e-4):
        return None
    xy = origin[:2] + rays[:, :2] * ((top_z - origin[2]) / rays[:, 2])[:, None]
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    if values[1] / max(values[0], 1e-8) < 1.6:
        return None
    spans = np.ptp(xy @ vectors, axis=0)
    if not 0.4 * dimensions[0] <= spans[1] <= 1.65 * dimensions[0]:
        return None
    return vectors[:, 1].round(3).tolist()


def _depth_xy(session, detection, camera, mask, depth_id, reference_xy, dimensions):
    record = session.depths.get(depth_id)
    if record is None or any(record.get(field) != detection[field] for field in ("frame_id", "camera")):
        raise RuntimeError("depth_id must come from the same RGB frame and camera as the detection")
    with np.load(record["depth_path"], allow_pickle=False) as archive:
        depth = archive["object_depth_m"]
    if depth.ndim != 2 or (mask is not None and depth.shape != mask.shape):
        raise RuntimeError("depth map must be aligned to the detection camera's RGB frame")
    x0, y0, x1, y1 = detection["bbox_xyxy"]
    if y1 > depth.shape[0] or x1 > depth.shape[1]:
        raise RuntimeError("depth map does not cover the detection's image coordinates")
    support = np.isfinite(depth) & (depth > 0.0)
    roi = np.zeros_like(support)
    roi[y0:y1, x0:x1] = True
    support &= roi if mask is None else roi & mask
    ys, xs = np.nonzero(support)
    if xs.size < 8:
        return None, int(xs.size), "sparse_object_depth"

    rays, origin = _backproject(camera, xs, ys)
    points = origin + rays * depth[ys, xs, None]
    width, dep, height, _ = dimensions
    plausible = (np.linalg.norm(points[:, :2] - reference_xy, axis=1) < max(width, dep))
    plausible &= (points[:, 2] >= session.floor_z - 0.01)
    plausible &= (points[:, 2] <= session.floor_z + height + 0.035)
    points = points[plausible]
    if len(points) < 8:
        return None, len(points), "depth_points_outside_expected_object_region"

    xy = np.median(points[:, :2], axis=0)
    if np.linalg.norm(xy - reference_xy) > 0.025:
        return None, len(points), "depth_surface_disagrees_with_rgb_landmark"
    # Depth samples visible surfaces. Limit the correction to preserve the
    # centre estimate in the presence of view-dependent surface bias.
    return (xy + reference_xy) / 2, len(points), "used_visible_surface_points"


def locate_basket(session, frame_id: str):
    """Independently locate the basket rim using the current overview image."""
    camera = _camera(_frame(session, frame_id), "overview")
    rim_uv = _basket_pixels(_read_rgb(camera))
    position = point_on_plane(camera, rim_uv, session.floor_z + 0.155).tolist()
    if not 0.30 <= position[0] <= 0.80 or abs(position[1]) > 0.40:
        raise RuntimeError("basket estimate is outside the calibrated Franka workspace; observe again")
    destination_id = f"b{len(session.destinations) + 1}"
    record = {"frame_id": frame_id, "kind": "basket", "camera": "overview", "position": position,
              "place_xy": position[:2], "coordinate_frame": "robot_base",
              "position_source": "rgb_liner_projected_at_nominal_rim_height"}
    session.destinations[destination_id] = record
    return {"destination_id": destination_id, **record}


def locate_object(session, detection_id: str, mask_id: str | None = None, depth_id: str | None = None):
    """Estimate the supported object's base-frame location from its RGB landmark."""
    detection, views = _detection(session, detection_id)
    camera = _camera(views, detection["camera"])
    target = detection["target"]
    dimensions = OBJECT_GEOMETRY[target]
    width, depth, height, shape = dimensions
    uv = detection["landmark_uv"]
    landmark_z = session.floor_z + PICK_LANDMARKS[target][0]
    reference_xy = point_on_plane(camera, uv, landmark_z)[:2]

    mask = _mask(session, mask_id, detection_id) if mask_id is not None else None
    mask_landmark_distance = None
    uncertainty_xy = 0.025
    if mask is not None:
        ys, xs = np.nonzero(mask)
        mask_landmark_distance = round(float(np.sqrt(np.min((xs - uv[0]) ** 2 + (ys - uv[1]) ** 2))), 1)
        if mask_landmark_distance <= 4:
            uncertainty_xy = 0.020
    correction, depth_support, depth_status = None, 0, None
    if depth_id is not None:
        correction, depth_support, depth_status = _depth_xy(session, detection, camera, mask, depth_id,
                                                            reference_xy, dimensions)
    xy = reference_xy if correction is None else correction
    if not 0.30 <= xy[0] <= 0.80 or abs(xy[1]) > 0.40:
        raise RuntimeError("RGB target estimate is outside the calibrated Franka workspace; observe again")

    yaw = PROFILES[target].jaw_yaw_deg
    angle = radians(yaw)
    axes = {"jaw_axis_xy": [round(cos(angle), 3), round(sin(angle), 3)], "up_xyz": [0, 0, 1]}
    axes_source = {"jaw_axis_xy": "nominal_grasp_profile", "up_xyz": "calibrated_floor_normal"}
    if shape == "flat_box" and mask is not None:
        flat_axis = _flat_axis(camera, mask, session.floor_z + height, dimensions)
        if flat_axis is not None:
            axes["visible_long_axis_xy"] = flat_axis
            axes_source["visible_long_axis_xy"] = "mask_pca_at_nominal_top_height"
    object_id = f"o{len(session.objects) + 1}"
    record = {
        "target": target, "frame_id": detection["frame_id"], "detection_id": detection_id,
        "camera": detection["camera"], "mask_id": mask_id, "depth_id": depth_id,
        "position": [float(xy[0]), float(xy[1]), float(session.floor_z + height / 2)],
        "pick_xy": list(map(float, xy)), "yaw_deg": yaw,
        "dimensions_m": {"width": width, "depth": depth, "height": height}, "shape": shape,
        "axes": axes,
        "coordinate_frame": "robot_base", "dimensions_source": "nominal_libero_asset",
        "axes_source": axes_source,
    }
    session.objects[object_id] = record
    return {
        "object_id": object_id, **record,
        "position_source": "rgb_landmark_plus_visible_surface_depth" if correction is not None
                           else "rgb_landmark_projected_at_nominal_height",
        "uncertainty_m": {"xy": uncertainty_xy, "z": 0.015},
        "uncertainty_type": "heuristic_tolerance",
        "mask_landmark_distance_px": mask_landmark_distance,
        "depth_support_pixels": depth_support, "depth_status": depth_status,
    }


TOOLS = {
    "detect_object": (
        detect_object,
        "Find the named object in the current RGB frame. Use after observe. "
        "Returns detection_id, the camera name, and an image with a box on the target.",
        {"frame_id": {"type": "string"}, "target": {"type": "string", "enum": list(OBJECT_GEOMETRY)}},
    ),
    "segment_object": (
        segment_object,
        "Mark the target's visible foreground pixels inside its detection. Use to isolate it "
        "from neighbours or to select its depth pixels. Returns mask_id and a mask overlay image.",
        {"detection_id": {"type": "string"}},
    ),
    "locate_object": (
        locate_object,
        "Estimate the detected object's 3D centre in robot base coordinates. Use before proposing a grasp. "
        "Optionally pass mask_id and depth_id from the same frame to refine the estimate. Returns object_id.",
        {"detection_id": {"type": "string"}, "mask_id": {"type": "string", "optional": True},
         "depth_id": {"type": "string", "optional": True}},
    ),
    "locate_basket": (
        locate_basket,
        "Find the basket in the current overview RGB frame and estimate its rim centre. "
        "Use before planning a transfer to the basket. Returns destination_id for plan_motion.",
        {"frame_id": {"type": "string"}},
    ),
}

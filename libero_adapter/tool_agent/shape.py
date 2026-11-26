"""Reconstruct visible surfaces from calibrated RGB at stationary wrist views."""

from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

from .depth import CAMERAS, _calibration, _stereo_samples


MIN_BASELINE_M = 0.025
CROP_RADIUS_M = 0.055
VOXEL_M = 0.003


def _load_view(session, frame_id, object_id):
    observation = session.frames.get(frame_id)
    metadata = session.shape_frame_views.get(frame_id)
    if observation is None or metadata is None:
        raise RuntimeError("Capture a wrist frame with observe after moving to a view for this object.")
    if metadata["object_id"] != object_id or metadata["contact_epoch"] != session.contact_epoch:
        raise RuntimeError("This wrist frame belongs to a different object or predates contact. Capture new views.")
    wrist = next((view for view in observation.cameras if view.camera_name == CAMERAS["wrist"]), None)
    if wrist is None:
        raise RuntimeError("The frame has no wrist RGB capture.")
    with Image.open(wrist.rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb, _calibration(wrist), metadata["view"]


def _points_from_rgb(views, floor_z, center_xy):
    point_sets, error_sets, pair_sets = [], [], []
    baselines = []
    for left, right in combinations(range(len(views)), 2):
        rgb_a, first, _ = views[left]
        rgb_b, second, _ = views[right]
        baseline = float(np.linalg.norm(first[1][:3, 3] - second[1][:3, 3]))
        if baseline < MIN_BASELINE_M:
            continue
        baselines.append(baseline)

        depth, error = _stereo_samples(rgb_a, rgb_b, first, second, floor_z)
        rows, cols = np.nonzero(np.isfinite(depth))
        pixels = np.stack((cols, rows, np.ones_like(cols)))
        rays = (first[1][:3, :3] @ np.linalg.solve(first[0], pixels)).T
        xyz = first[1][:3, 3] + rays * depth[rows, cols, None]
        near_target = np.linalg.norm(xyz[:, :2] - center_xy, axis=1) <= CROP_RADIUS_M
        point_sets.append(xyz[near_target])
        error_sets.append(error[rows[near_target], cols[near_target]])
        pair_sets.append(np.tile([left, right], (np.count_nonzero(near_target), 1)))
    if not baselines:
        raise RuntimeError("The actual wrist camera positions are too close. Capture a different view.")

    points = np.concatenate(point_sets)
    errors = np.concatenate(error_sets)
    pairs = np.concatenate(pair_sets)
    best_first = np.argsort(errors)
    voxels = np.rint(points[best_first] / VOXEL_M).astype(np.int32)
    _, unique = np.unique(voxels, axis=0, return_index=True)
    keep = best_first[np.sort(unique)]
    return points[keep], errors[keep], pairs[keep], max(baselines)


def _summarize(points):
    if len(points) < 2:
        return {"quality": "sparse", "usable_for_grasp": False, "axes_xy": {},
                "center_xyz_m": None, "visible_extent_m": {}, "visible_bounds_m": {}}

    lower, upper = np.quantile(points, [0.10, 0.90], axis=0)
    cover = upper - lower
    center = np.median(points, axis=0)
    supported = len(points) >= 10 and float(cover.max()) >= 0.015
    extents = dict(zip("xyz", cover.round(4).tolist()))
    axes = {}
    if supported:
        xy = points[:, :2] - center[:2]
        values, vectors = np.linalg.eigh(xy.T @ xy)
        long_axis = vectors[:, -1]
        if long_axis[np.argmax(np.abs(long_axis))] < 0:
            long_axis = -long_axis
        short_axis = np.array([-long_axis[1], long_axis[0]])
        along = np.quantile(np.column_stack((xy @ long_axis, xy @ short_axis)), [0.10, 0.90], axis=0)
        extents.update(zip(("long", "short"), (along[1] - along[0]).round(4).tolist()))
        if values[-1] >= 1.65 * max(values[0], 1e-8) and extents["long"] >= 0.015:
            axes = {"long": long_axis.round(5).tolist(), "short": short_axis.round(5).tolist()}

    return {
        "quality": "supported" if supported else "sparse",
        "usable_for_grasp": supported,
        "axes_xy": axes,
        "center_xyz_m": center.round(4).tolist(),
        "visible_extent_m": extents,
        "visible_bounds_m": {"lower_xyz": lower.round(4).tolist(), "upper_xyz": upper.round(4).tolist()},
    }


def _shape_preview(views, points, pairs, output):
    thumbs = []
    for index, (rgb, calibration, label) in enumerate(views):
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        seen = points[np.any(pairs == index, axis=1)]
        projected = np.column_stack((seen, np.ones(len(seen)))) @ calibration[3].T
        for x_h, y_h, z in projected:
            if z <= 0:
                continue
            x, y = x_h / z, y_h / z
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#e8ff35", outline="#132033")
        draw.rectangle((0, 0, image.width, 19), fill="#101820")
        draw.text((5, 4), f"Wrist {label}: {len(seen)} triangulated surface samples", fill="white")
        image.thumbnail((400, 300))
        thumbs.append(image)
    preview = Image.new("RGB", (sum(im.width for im in thumbs), max(im.height for im in thumbs)), "#101820")
    x = 0
    for image in thumbs:
        preview.paste(image, (x, 0))
        x += image.width
    preview.save(output)


def reconstruct_shape(session, object_id: str, first_frame_id: str, second_frame_id: str,
                      third_frame_id: str | None = None):
    """Triangulate a named target's visible surface from two or three wrist views."""
    obj = session.objects.get(object_id)
    if obj is None:
        raise RuntimeError("Use an object_id from locate_object.")
    frame_ids = [first_frame_id, second_frame_id]
    if third_frame_id is not None:
        frame_ids.append(third_frame_id)
    if len(set(frame_ids)) != len(frame_ids):
        raise RuntimeError("Use distinct wrist frames from different viewpoints of the same object.")

    views = [_load_view(session, frame_id, object_id) for frame_id in frame_ids]
    if len({label for _, _, label in views}) != len(views):
        raise RuntimeError("Use wrist frames from different named views.")
    center_xy = np.asarray(obj["position"][:2], dtype=np.float64)
    points, errors, pairs, baseline = _points_from_rgb(views, session.floor_z, center_xy)
    stats = _summarize(points)
    shape_id = f"shape_{len(session.shapes) + 1}"
    folder = session.output / "shapes"
    folder.mkdir(parents=True, exist_ok=True)
    cloud = folder / f"{shape_id}.npz"
    preview = folder / f"{shape_id}.png"
    np.savez_compressed(cloud, points_xyz_m=points, reprojection_error_px=errors, source_pair=pairs)
    _shape_preview(views, points, pairs, preview)
    result = {
        "shape_id": shape_id,
        "object_id": object_id,
        "frame_ids": frame_ids,
        "simulation_step": session.backend.simulation_steps,
        "contact_epoch": session.contact_epoch,
        "source": "wrist_rgb_triangulation",
        "coordinate_frame": "robot_base",
        "sample_count": len(points),
        "baseline_m": round(baseline, 4),
        "target_crop_radius_m": CROP_RADIUS_M,
        "median_reprojection_error_px": round(float(np.median(errors)), 3) if len(errors) else None,
        "preview": str(preview.relative_to(session.output)),
        **stats,
        "note": ("Bounds, median and axes describe only visible matches near the target estimate. Hidden surfaces and full object size are unknown."
                 if stats["usable_for_grasp"] else "Too few target matches for grasp geometry. Capture another view or use the approximate object geometry."),
    }
    session.shapes[shape_id] = {**result, "point_cloud_path": cloud, "preview_path": preview}
    return result, {"shape_preview": preview}


TOOLS = {
    "reconstruct_shape": (
        reconstruct_shape,
        "Reconstruct the visible 3D surface near a located object after two distinct move_wrist_view and observe calls. "
        "Use before touching the object if its grasp shape is unclear. Returns a shape_id for propose_grasps.",
        {"object_id": {"type": "string"}, "first_frame_id": {"type": "string"},
         "second_frame_id": {"type": "string"}, "third_frame_id": {"type": "string", "optional": True}},
    ),
}

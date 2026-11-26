"""Estimate depth from synchronized, calibrated RGB views."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CAMERAS = {"overview": "agentview", "wrist": "robot0_eye_in_hand"}


def _calibration(camera):
    intrinsic = np.asarray(camera.intrinsics_3x3, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(camera.extrinsics_4x4, dtype=np.float64).reshape(4, 4)
    inverse = np.linalg.inv(pose)
    return intrinsic, pose, inverse, intrinsic @ inverse[:3, :]


def _plane_depth(shape, intrinsic, pose, floor_z):
    v, u = np.indices(shape, dtype=np.float64)
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    rays = pixels @ (pose[:3, :3] @ np.linalg.inv(intrinsic)).T
    vertical = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (floor_z - pose[2, 3]) / vertical
    valid = (vertical < -1e-5) & (z > 0.03) & (z < 5.0)
    return np.where(valid, z, np.nan).astype(np.float32)


def _feature_matches(rgb, other_rgb, primary, secondary):
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "SIFT_create"):
        return None

    sift = cv2.SIFT_create(nfeatures=1400, contrastThreshold=0.012)
    key_a, desc_a = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
    key_b, desc_b = sift.detectAndCompute(cv2.cvtColor(other_rgb, cv2.COLOR_RGB2GRAY), None)
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 2:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def unambiguous(pairs):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.77 * pair[1].distance}

    forward = unambiguous(matcher.knnMatch(desc_a, desc_b, k=2))
    backward = unambiguous(matcher.knnMatch(desc_b, desc_a, k=2))
    pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    if not pairs:
        return None

    pixels_a = np.asarray([key_a[a].pt for a, _ in pairs], dtype=np.float64)
    pixels_b = np.asarray([key_b[b].pt for _, b in pairs], dtype=np.float64)
    intrinsic_a, pose_a, _, proj_a = primary
    intrinsic_b, _, inverse_b, proj_b = secondary
    relative = inverse_b @ pose_a
    tx, ty, tz = relative[:3, 3]
    cross_t = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    fundamental = np.linalg.inv(intrinsic_b).T @ cross_t @ relative[:3, :3] @ np.linalg.inv(intrinsic_a)
    hom_a = np.column_stack((pixels_a, np.ones(len(pixels_a))))
    hom_b = np.column_stack((pixels_b, np.ones(len(pixels_b))))
    line_b = hom_a @ fundamental.T
    line_a = hom_b @ fundamental
    numerator = np.abs(np.einsum("ij,ij->i", hom_b, line_b))
    denominator = np.linalg.norm(np.column_stack((line_a[:, :2], line_b[:, :2])), axis=1)
    good = (denominator > 1e-10) & (numerator < 1.5 * denominator)
    if not np.any(good):
        return None

    pixels_a, pixels_b = pixels_a[good], pixels_b[good]
    points = cv2.triangulatePoints(proj_a, proj_b, pixels_a.T, pixels_b.T).T
    usable = np.abs(points[:, 3]) > 1e-10
    return pixels_a[usable], pixels_b[usable], points[usable] / points[usable, 3:4]


def _stereo_samples(rgb, other_rgb, primary, secondary, floor_z):
    surface = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    reprojection = np.full_like(surface, np.nan)
    matches = _feature_matches(rgb, other_rgb, primary, secondary)
    if matches is None:
        return surface, reprojection

    pixels_a, pixels_b, points = matches
    projected_a = points @ primary[3].T
    projected_b = points @ secondary[3].T
    z_a, z_b = projected_a[:, 2], projected_b[:, 2]
    valid = (z_a > 0.06) & (z_a < 2.5) & (z_b > 0.06) & (z_b < 2.5)
    valid &= (points[:, 2] > floor_z + 0.004) & (points[:, 2] < floor_z + 0.75)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_a = np.linalg.norm(projected_a[:, :2] / z_a[:, None] - pixels_a, axis=1)
        error_b = np.linalg.norm(projected_b[:, :2] / z_b[:, None] - pixels_b, axis=1)
    error = np.maximum(error_a, error_b)
    valid &= error <= 2.0
    ray_a = points[:, :3] - primary[1][:3, 3]
    ray_b = points[:, :3] - secondary[1][:3, 3]
    triangle = np.linalg.norm(np.cross(ray_a, ray_b), axis=1)
    valid &= triangle > np.sin(np.deg2rad(2.0)) * np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)

    best_first = sorted(np.flatnonzero(valid), key=lambda index: error[index])
    height, width = surface.shape
    for index in best_first:
        x, y = np.rint(pixels_a[index]).astype(int)
        if 0 <= x < width and 0 <= y < height and not np.isfinite(surface[y, x]):
            surface[y, x] = z_a[index]
            reprojection[y, x] = error[index]
    return surface, reprojection


def _preview(rgb, surface, plane, path):
    rows, cols = np.nonzero(np.isfinite(surface))
    if len(rows):
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        for y, x in zip(rows, cols):
            scaled = float(np.clip(surface[y, x] / 1.6, 0.0, 1.0))
            color = (round(255 * (1 - scaled)), 205, round(255 * scaled))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")
        label = f"RGB stereo: {len(rows)} surface samples | warm near, cool far"
    else:
        scaled = np.clip(np.nan_to_num(plane, nan=0.0) / 1.6, 0.0, 1.0)
        preview = np.stack((255 * (1 - scaled), np.full_like(scaled, 205), 255 * scaled), axis=-1)
        preview[~np.isfinite(plane)] = (16, 24, 32)
        im = Image.fromarray(preview.astype(np.uint8))
        draw = ImageDraw.Draw(im)
        label = "Projected distance to floor, ignoring objects | warm near, cool far"
    draw.rectangle((0, 0, im.width, 18), fill="#101820")
    draw.text((5, 3), label, fill="white")
    im.save(path)


def estimate_depth(session, frame_id: str, camera: str):
    """Return sparse metric depth where RGB stereo matches, plus a ground prior."""
    if frame_id not in session.frames:
        raise RuntimeError("Unknown frame_id. Observe the scene first.")
    if session.frame_steps.get(frame_id) != session.backend.simulation_steps:
        raise RuntimeError("Stale frame_id. Observe the scene again after motion.")
    if camera not in CAMERAS:
        raise RuntimeError("Choose overview or wrist for the depth map.")

    observation = session.frames[frame_id]
    views = {view.camera_name: view for view in observation.cameras}
    primary = views.get(CAMERAS[camera])
    if primary is None:
        raise RuntimeError(f"The {camera} camera is unavailable for {frame_id}.")

    with Image.open(primary.rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    calibration = _calibration(primary)
    plane = _plane_depth(rgb.shape[:2], calibration[0], calibration[1], session.floor_z)
    other = next((views[name] for name in CAMERAS.values() if name != CAMERAS[camera] and name in views), None)
    if other is not None:
        with Image.open(other.rgb_path) as image:
            other_rgb = np.asarray(image.convert("RGB"))
        surface, errors = _stereo_samples(rgb, other_rgb, calibration, _calibration(other), session.floor_z)
    else:
        surface = np.full_like(plane, np.nan)
        errors = np.full_like(plane, np.nan)

    depth_id = f"depth_{len(session.depths) + 1}"
    folder = Path(session.output) / "depth"
    folder.mkdir(parents=True, exist_ok=True)
    npz_path = folder / f"{depth_id}.npz"
    preview_path = folder / f"{depth_id}.png"
    np.savez_compressed(npz_path, object_depth_m=surface, support_plane_depth_m=plane, reprojection_error_px=errors)
    _preview(rgb, surface, plane, preview_path)

    count = int(np.count_nonzero(np.isfinite(surface)))
    receipt = {
        "depth_id": depth_id,
        "frame_id": frame_id,
        "camera": camera,
        "method": "calibrated_rgb_stereo" if count else "support_plane_only",
        "valid_surface_samples": count,
        "units": "m along camera optical axis",
        "preview": str(preview_path.relative_to(session.output)),
        "note": ("Only colored dots have measured stereo depth. The separate ground-plane estimate does not measure objects."
                 if count else "No verified object depth. The available ground-plane estimate does not measure objects."),
    }
    session.depths[depth_id] = {**receipt, "depth_path": npz_path, "preview_path": preview_path}
    return receipt, {"depth": preview_path}


TOOLS = {
    "estimate_depth": (
        estimate_depth,
        "Estimate sparse metric depth from the two RGB views in a frame. Call when object distance is uncertain. "
        "The depth ID belongs to the selected camera, and missing stereo points remain unknown.",
        {"frame_id": {"type": "string"}, "camera": {"type": "string", "enum": ["overview", "wrist"]}},
    ),
}

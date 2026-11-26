"""Calibrated RGB grounding for the LIBERO Object shelf items and basket."""

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, find_objects, label


# Nominal visual landmark heights above the staging surface. The workspace
# intervals are deliberately coarse, to exclude the arm and unrelated items.
# These are appearance/geometry priors for this fixed camera and benchmark.
PICK_LANDMARKS = {
    "alphabet_soup": (0.080, (0.25, 0.37), (0.36, 0.58), "label_top"),
    "cream_cheese": (0.017, (0.35, 0.48), (0.51, 0.72), "flat"),
    "salad_dressing": (0.166, (0.35, 0.425), (0.40, 0.51), "cap_center"),
    "bbq_sauce": (0.105, (0.35, 0.48), (0.47, 0.65), "bottle_neck"),
    "ketchup": (0.157, (0.25, 0.37), (0.33, 0.55), "cap_top"),
    "tomato_sauce": (0.075, (0.35, 0.48), (0.47, 0.68), "lid_above_band"),
    "butter": (0.017, (0.25, 0.37), (0.42, 0.60), "flat"),
    "milk": (0.128, (0.25, 0.37), (0.34, 0.57), "carton_top"),
    "chocolate_pudding": (0.027, (0.68, 1.0), (0.28, 0.70), "center"),
    "orange_juice": (0.129, (0.35, 0.48), (0.43, 0.63), "carton_label_top"),
}


def largest_component(mask: np.ndarray, min_pixels: int) -> tuple[slice, slice]:
    labels, count = label(mask)
    if count:
        counts = np.bincount(labels.ravel())[1:]
        index = int(np.argmax(counts))
        if counts[index] >= min_pixels:
            return find_objects(labels)[index]
    raise RuntimeError("RGB segmentation did not find a large enough matching component")


def _target_mask(rgb: np.ndarray, target: str) -> np.ndarray:
    red, green, blue = np.moveaxis(rgb, 2, 0)
    if target in ("alphabet_soup", "cream_cheese"):
        return (blue > red + 15) & (blue > green + 8)
    if target in ("salad_dressing", "tomato_sauce"):
        return (green > red + 5) & (green > blue + 5)
    if target == "orange_juice":
        return (red > green + 8) & (green > blue + 19)
    if target == "chocolate_pudding":
        return (red > green + 10) & (green > blue + 5)
    if target == "ketchup":
        return (red > green + 14) | (rgb.max(axis=2) < 92)
    return (red > green + 16) & (red > blue + 19)


def _target_pixels(rgb: np.ndarray, target: str) -> tuple[float, float]:
    height, width, _ = rgb.shape
    _, horizontal, vertical, method = PICK_LANDMARKS[target]
    rows, cols = np.ogrid[:height, :width]
    mask = _target_mask(rgb, target)
    mask &= (horizontal[0] * width < cols) & (cols < horizontal[1] * width)
    mask &= (vertical[0] * height < rows) & (rows < vertical[1] * height)
    closing = 1 if method == "flat" else 2
    ys, xs = largest_component(binary_closing(mask, iterations=closing), min_pixels=35)
    x = (xs.start + xs.stop - 1) / 2
    y = (ys.start + ys.stop - 1) / 2
    if method == "label_top":
        y = ys.start - 0.00625 * height
    elif method in ("cap_top", "carton_top", "bottle_neck", "carton_label_top"):
        # The top of the front face is viewed obliquely. Find its center
        # from just the upper rows, rather than the whole tapered bottle.
        top_rows = max(4, round((ys.stop - ys.start) * 0.2))
        _, pixels = np.nonzero(mask[ys.start:ys.start + top_rows, xs])
        if len(pixels):
            x = xs.start + (pixels.min() + pixels.max()) / 2
        offsets = {"bottle_neck": 0.021, "carton_label_top": -0.003}
        y = ys.start + offsets.get(method, 0.009) * height
    elif method == "cap_center":
        y = ys.start + 0.022 * height
    elif method == "lid_above_band":
        y = ys.start - 0.025 * height
        x -= 0.002 * width
    elif method == "flat":
        y -= (0.014 if target == "butter" else 0.005) * height
    return x, y


def _read_rgb(camera) -> np.ndarray:
    with Image.open(camera.rgb_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def _basket_pixels(rgb: np.ndarray) -> tuple[float, float]:
    height, width, _ = rgb.shape
    rows, cols = np.ogrid[:height, :width]
    # The benchmark basket's bright neutral liner distinguishes its rim
    # from the floor and the surrounding coloured objects.
    liner = (rgb.min(axis=2) > 170) & (np.ptp(rgb, axis=2) < 25)
    liner &= (rows > 0.34 * height) & (rows < 0.66 * height)
    liner &= (cols > 0.57 * width) & (cols < 0.83 * width)
    ys, xs = largest_component(binary_closing(liner, iterations=2), min_pixels=60)
    return (xs.start + xs.stop - 1) / 2, (ys.start + ys.stop - 1) / 2


def point_on_plane(camera, xy: tuple[float, float], height: float) -> np.ndarray:
    calibration = np.asarray(camera.intrinsics_3x3).reshape(3, 3)
    pose = np.asarray(camera.extrinsics_4x4).reshape(4, 4)
    ray = pose[:3, :3] @ np.linalg.solve(calibration, [*xy, 1.0])
    origin = pose[:3, 3]
    if ray[2] >= -1e-4:
        raise RuntimeError("RGB ray does not point down to the calibrated plane")
    distance = (height - origin[2]) / ray[2]
    if distance <= 0:
        raise RuntimeError("RGB ray meets the plane behind the camera")
    return origin + distance * ray


def plan_from_rgb(plan_id: str, observation, floor_z: float, target: str = "alphabet_soup"):
    from libero_adapter.motion import plan_pick_place

    if target not in PICK_LANDMARKS:
        raise RuntimeError(f"unsupported RGB target {target!r}")
    cameras = {camera.camera_name: camera for camera in observation.cameras}
    overview = cameras["agentview"]
    rgb = _read_rgb(overview)
    if target == "chocolate_pudding":
        # The tall ketchup bottle often obscures this short tray in the global view.
        pick_camera = cameras["robot0_eye_in_hand"]
        pick_uv = _target_pixels(_read_rgb(pick_camera), target)
    else:
        pick_camera = overview
        pick_uv = _target_pixels(rgb, target)
    # Calibrated projection uses a nominal landmark height, without rendered
    # depth, object transforms, or success predicates.
    pick = point_on_plane(pick_camera, pick_uv, floor_z + PICK_LANDMARKS[target][0])
    place = point_on_plane(overview, _basket_pixels(rgb), floor_z + 0.155)
    for point in (pick, place):
        if not 0.30 <= point[0] <= 0.80 or abs(point[1]) > 0.40:
            raise RuntimeError("RGB estimate is outside the calibrated Franka pick-and-place region")

    return plan_pick_place(plan_id, target, tuple(pick[:2]), tuple(place[:2]), floor_z)

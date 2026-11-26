"""Estimate a colored sphere from calibrated RGB images."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares


class GraspError(RuntimeError):
    """A public, actionable error from observation or grasp execution."""


@dataclass(frozen=True)
class CameraFrame:
    camera: str
    image: Image.Image
    focal: float
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class BallEstimate:
    center: np.ndarray
    radius: float
    triangulation_error: float


@dataclass(frozen=True)
class _Silhouette:
    direction: np.ndarray
    angle: float
    coverage: float


def _color_mask(frame: CameraFrame, color: str) -> np.ndarray:
    red, green, blue = np.asarray(frame.image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    if color == "orange":
        target = (red > 105) & (red > 1.35 * green) & (green > 34) & (green > 1.35 * blue)
        name = "橙色"
    elif color == "green":
        target = (green > 55) & (green > 1.35 * red) & (green > 1.35 * blue)
        name = "绿色"
    else:
        raise GraspError(f"暂不支持颜色 {color!r}，请选择橙色或绿色")

    labels, count = ndimage.label(target)
    if count == 0:
        raise GraspError(f"{frame.camera} 相机没有看到{name}小球")
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.argmax())
    if sizes[largest] < 24:
        raise GraspError(f"{frame.camera} 相机里的{name}区域太小")
    if np.count_nonzero(sizes >= 0.8 * sizes[largest]) > 1:
        raise GraspError(f"{frame.camera} 相机看到多个相近的{name}区域，无法区分目标")
    return ndimage.binary_fill_holes(labels == largest + 1)


def _silhouette(frame: CameraFrame, color: str) -> _Silhouette:
    mask = _color_mask(frame, color)
    # Sample where the mask crosses between adjacent pixel centers. A contour cut
    # off by the image edge contributes no fabricated boundary points.
    side_row, side_col = np.nonzero(mask[:, 1:] != mask[:, :-1])
    top_row, top_col = np.nonzero(mask[1:, :] != mask[:-1, :])
    height, width = mask.shape
    x = np.r_[side_col + 1.0, top_col + 0.5] - width / 2
    y = np.r_[side_row + 0.5, top_row + 1.0] - height / 2
    if len(x) < 18:
        raise GraspError(f"{frame.camera} 相机里的小球轮廓不足")
    rays = np.column_stack((x / frame.focal, y / frame.focal, np.ones(len(x))))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    def axis(params: np.ndarray) -> np.ndarray:
        vector = np.r_[params[:2], 1.0]
        return vector / np.linalg.norm(vector)

    # The exact outline of a sphere lies on a cone around the center ray.
    # Fit chord distances on the unit sphere, including off-axis projections.
    def residual(params: np.ndarray, samples: np.ndarray) -> np.ndarray:
        return np.linalg.norm(samples - axis(params), axis=1) - params[2]

    mean = rays.mean(axis=0)
    seed = np.r_[mean[:2] / mean[2], 0.0]
    seed[2] = np.median(residual(seed, rays))
    error_scale = 0.8 / frame.focal
    fit = least_squares(residual, seed, args=(rays,), loss="soft_l1",
                        f_scale=error_scale, max_nfev=40)
    inliers = np.abs(fit.fun) < 2.0 * error_scale
    if not fit.success or inliers.sum() < max(18, 0.6 * len(rays)):
        raise GraspError(f"{frame.camera} 相机里的目标轮廓与球形不一致")
    fit = least_squares(residual, fit.x, args=(rays[inliers],), max_nfev=20)
    direction = axis(fit.x)
    chord = fit.x[2]
    if not fit.success or not 2.0 / frame.focal < chord < math.sqrt(2.0):
        raise GraspError(f"{frame.camera} 相机的小球轮廓拟合不可靠")

    horizontal = np.cross(np.array([0.0, 1.0, 0.0]), direction)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(direction, horizontal)
    edges = rays[inliers]
    angles = np.sort(np.arctan2(edges @ vertical, edges @ horizontal))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    coverage = 2 * np.pi - gaps.max()
    if coverage < math.radians(190):
        raise GraspError(f"{frame.camera} 相机里的小球遮挡过多")
    return _Silhouette(frame.rotation @ direction, 2 * math.asin(chord / 2), float(coverage))


def estimate_ball(frames: tuple[CameraFrame, CameraFrame], color: str = "orange") -> BallEstimate:
    """Triangulate a sphere of the selected color from two simultaneous RGB views."""
    first, second = frames
    first_view, second_view = _silhouette(first, color), _silhouette(second, color)
    if np.linalg.norm(np.cross(first_view.direction, second_view.direction)) < 0.08:
        raise GraspError("两个相机的有效视角太接近，无法可靠估算深度")
    rays = np.column_stack((first_view.direction, -second_view.direction))
    distances, *_ = np.linalg.lstsq(rays, second.position - first.position, rcond=None)
    if np.any(distances <= 0.04):
        raise GraspError("两张图像的小球没有可用的前方深度")
    first_center = first.position + distances[0] * first_view.direction
    second_center = second.position + distances[1] * second_view.direction
    error = float(np.linalg.norm(first_center - second_center))
    radii = distances * np.sin([first_view.angle, second_view.angle])
    radius = float(radii.mean())
    if error > 0.01 or abs(radii[0] - radii[1]) > 0.25 * radius:
        raise GraspError("两张图像估算的小球位置或大小不一致")
    return BallEstimate((first_center + second_center) / 2, radius, error)


def estimate_ball_from_radius(frame: CameraFrame, radius: float, color: str = "orange") -> np.ndarray:
    """Estimate the center from one view and a radius measured earlier in the episode."""
    silhouette = _silhouette(frame, color)
    if silhouette.coverage < math.radians(240):
        raise GraspError(f"{frame.camera} 相机的小球轮廓不足，无法单目核验")
    distance = radius / math.sin(silhouette.angle)
    return frame.position + distance * silhouette.direction

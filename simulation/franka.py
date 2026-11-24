"""Single-ball Franka Panda scene with a bounded, position-controlled grasp."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

# Surfaceless Mesa also works when the terminal has no GPU render device.
os.environ.setdefault("MUJOCO_GL", "egl")
mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
if os.environ["MUJOCO_GL"] == "egl" and mesa.exists():
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(mesa))
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
BALL_RADIUS = 0.024
MIN_LIFT = 0.08
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0
VIDEO_MS = 50


class GraspError(RuntimeError):
    """A public, actionable error from the simulation or the planner."""


@dataclass(frozen=True)
class CameraFrame:
    camera: str
    image: Image.Image
    depth: np.ndarray
    focal: float
    position: np.ndarray
    rotation: np.ndarray

    def ball_center(self) -> np.ndarray:
        """Fit the known-radius orange ball to RGB-selected depth pixels."""
        red, green, blue = np.asarray(self.image, dtype=np.float32).transpose(2, 0, 1)
        mask = (red > 105) & (red > 1.35 * green) & (green > 34) & (green > 1.35 * blue)
        labels, count = ndimage.label(mask)
        if not count:
            raise GraspError("相机没有看到橙色小球")
        sizes = np.bincount(labels.ravel())[1:]
        mask = labels == int(sizes.argmax() + 1)
        if sizes.max() < 24:
            raise GraspError("看见的橙色区域太小，无法可靠定位")
        interior = ndimage.binary_erosion(mask)
        if interior.sum() >= 16:
            mask = interior
        height, width = mask.shape
        row, col = np.nonzero(mask & np.isfinite(self.depth) & (self.depth > 0.04))
        if len(row) < 14:
            raise GraspError("小球的有效深度点不足")
        depth = self.depth[row, col].astype(float)
        camera_points = np.column_stack(((col + 0.5 - width / 2) * depth / self.focal,
                                        (row + 0.5 - height / 2) * depth / self.focal,
                                        depth))
        points = camera_points @ self.rotation.T + self.position
        visible_surface = np.median(points, axis=0)
        ray = visible_surface - self.position
        initial = visible_surface + BALL_RADIUS * ray / np.linalg.norm(ray)

        def sphere_error(center):
            return np.linalg.norm(points - center, axis=1) - BALL_RADIUS

        fit = least_squares(sphere_error, initial, bounds=(initial - 0.035, initial + 0.035),
                            loss="soft_l1", f_scale=0.0015, max_nfev=35)
        if not fit.success or np.median(np.abs(fit.fun)) > 0.004:
            raise GraspError("橙色区域和小球的深度形状不一致")
        return fit.x


@dataclass(frozen=True)
class GraspTrajectory:
    start_joints: np.ndarray
    center: np.ndarray
    approach: tuple[np.ndarray, ...]
    lower: tuple[np.ndarray, ...]
    lift: tuple[np.ndarray, ...]


class FrankaBall:
    """Simulate the Franka and expose camera frames and grasp execution."""

    def __init__(self, ball_xy: tuple[float, float] = (0.49, 0.08)):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("franka_ball.xml")))
        self.data = mujoco.MjData(self.model)
        self.tcp = self.model.site("grasp_tcp").id
        self.gripper_bodies = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.data.qpos[:9] = [*HOME, 0.04, 0.04]
        self.data.joint("ball_free").qpos[:2] = ball_xy
        self.data.ctrl[:] = [*HOME, GRIP_OPEN]
        mujoco.mj_forward(self.model, self.data)
        self.down = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.frames: list[Image.Image] = []
        self.steps = 0
        self.record_every = round((VIDEO_MS / 1000) / self.model.opt.timestep)
        self._step_for(0.20, record=False)

    def close(self) -> None:
        self.renderer.close()

    def rgb(self, camera: str = "overview") -> Image.Image:
        self.renderer.update_scene(self.data, camera=camera)
        return Image.fromarray(self.renderer.render())

    def observe(self, camera: str) -> CameraFrame:
        image = self.rgb(camera)
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        camera_id = self.model.camera(camera).id
        focal = image.height / (2 * math.tan(math.radians(self.model.cam_fovy[camera_id]) / 2))
        cam_rotation = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        return CameraFrame(camera, image, depth, focal, self.data.cam_xpos[camera_id].copy(), cam_rotation)

    def _record(self) -> None:
        self.frames.append(self.rgb())

    def _step(self, *, record: bool = True) -> None:
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if record and self.steps % self.record_every == 0:
            self._record()

    def _step_for(self, seconds: float, *, record: bool = True) -> None:
        for _ in range(round(seconds / self.model.opt.timestep)):
            self._step(record=record)

    def _solve_pose(self, scratch: mujoco.MjData, start: np.ndarray, position: np.ndarray) -> np.ndarray:
        joints = start.copy()
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros_like(jac_pos)
        for _ in range(100):
            scratch.qpos[:7] = joints
            mujoco.mj_forward(self.model, scratch)
            offset = position - scratch.site_xpos[self.tcp]
            rotation = scratch.site_xmat[self.tcp].reshape(3, 3)
            turn = Rotation.from_matrix(self.down @ rotation.T).as_rotvec()
            if np.linalg.norm(offset) < 0.0005 and np.linalg.norm(turn) < 0.007:
                return joints
            mujoco.mj_jacSite(self.model, scratch, jac_pos, jac_rot, self.tcp)
            jacobian = np.vstack((jac_pos[:, :7], 0.35 * jac_rot[:, :7]))
            error = np.r_[offset, 0.35 * turn]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.014**2 * np.eye(6), error)
            step *= min(1.0, 0.08 / max(np.max(np.abs(step)), 0.08))
            joints = np.clip(joints + step, self.model.jnt_range[:7, 0] + 0.001,
                             self.model.jnt_range[:7, 1] - 0.001)
        raise GraspError("逆运动学没有找到可行的抓取位姿")

    def _clear_table(self, scratch: mujoco.MjData, first: np.ndarray, second: np.ndarray) -> None:
        table = self.model.geom("table").id
        arm_base = self.model.body("link0").id
        ball_body = self.model.body("orange_ball").id
        samples = max(2, math.ceil(np.max(np.abs(first - second)) / 0.04))
        for part in np.linspace(0, 1, samples + 1):
            scratch.qpos[:7] = first + part * (second - first)
            mujoco.mj_forward(self.model, scratch)
            for contact in scratch.contact:
                if contact.geom1 != table and contact.geom2 != table:
                    continue
                other = contact.geom2 if contact.geom1 == table else contact.geom1
                if self.model.geom_bodyid[other] not in (arm_base, ball_body, 0):
                    raise GraspError("规划会使机械臂碰到桌面")

    def plan_grasp(self, center: np.ndarray) -> GraspTrajectory:
        x, y, z = center
        if not (0.34 <= x <= 0.65 and -0.23 <= y <= 0.23 and 0.018 <= z <= 0.041):
            raise GraspError("小球不在本工作台的可抓取区域")
        scratch = mujoco.MjData(self.model)
        scratch.qpos[7:9] = 0.04
        # No ball state from the live simulator is used by the planner.
        scratch.joint("ball_free").qpos[:3] = center
        first_joints = self.data.qpos[:7].copy()
        first_position = self.data.site_xpos[self.tcp].copy()
        above = center + np.array([0, 0, 0.135])
        lift_to = center + np.array([0, 0, 0.155])
        targets = ((first_position, above), (above, center), (center, lift_to))
        joints = first_joints
        paths = []
        for a, b in targets:
            waypoints = []
            count = max(2, math.ceil(np.linalg.norm(a - b) / 0.028))
            for pose in np.linspace(a, b, count + 1)[1:]:
                goal = self._solve_pose(scratch, joints, pose)
                self._clear_table(scratch, joints, goal)
                waypoints.append(goal)
                joints = goal
            paths.append(tuple(waypoints))
        return GraspTrajectory(first_joints, center, *paths)

    def _follow(self, waypoints: tuple[np.ndarray, ...], seconds: float) -> None:
        start = self.data.ctrl[:7].copy()
        ticks = max(1, round(seconds / (len(waypoints) * self.model.opt.timestep)))
        for goal in waypoints:
            for step in range(1, ticks + 1):
                fraction = step / ticks
                self.data.ctrl[:7] = start + fraction * (goal - start)
                self._step()
            start = goal
        self._step_for(0.26)

    def execute_grasp(self, plan: GraspTrajectory) -> None:
        if not np.allclose(self.data.qpos[:7], plan.start_joints, atol=0.008, rtol=0):
            raise GraspError("计划已过期，机械臂姿态发生变化，请重新观察")
        self.frames.clear()
        self._record()
        self.data.ctrl[7] = GRIP_OPEN
        self._follow(plan.approach, 1.7)
        self._follow(plan.lower, 0.8)
        if np.linalg.norm(self.data.site_xpos[self.tcp] - plan.center) > 0.008:
            raise GraspError("夹爪未到达抓取点，已停止执行")
        close_ticks = round(0.48 / self.model.opt.timestep)
        for tick in range(1, close_ticks + 1):
            self.data.ctrl[7] = GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * tick / close_ticks
            self._step()
        self._step_for(0.22)
        self._follow(plan.lift, 0.9)
        self._step_for(0.75)
        self._record()

    def verify_grasp(self, plan: GraspTrajectory) -> dict:
        aperture = float(self.data.qpos[7:9].sum())
        for camera in ("detail", "overview"):
            try:
                center = self.observe(camera).ball_center()
            except GraspError:
                continue
            lift = center[2] - plan.center[2]
            distance = np.linalg.norm(center - self.data.site_xpos[self.tcp])
            return {"held": bool(lift >= MIN_LIFT and distance < 0.045 and aperture > 0.018),
                    "lift_cm": round(float(lift) * 100, 2), "camera": camera}
        return {"held": None, "reason": "夹爪遮住了球，无法从相机确认"}


class GraspScorer:
    """Independent simulation-only verdict, never placed in a model receipt."""

    def __init__(self, sim: FrankaBall):
        self.sim = sim
        self.initial_z = float(sim.data.body("orange_ball").xpos[2])

    def evaluate(self) -> dict:
        sim = self.sim
        ball = sim.model.geom("ball").id
        contact_with = set()
        for contact in sim.data.contact:
            if contact.geom1 == ball or contact.geom2 == ball:
                other = contact.geom2 if contact.geom1 == ball else contact.geom1
                contact_with.add(sim.model.geom_bodyid[other])
        lift = float(sim.data.body("orange_ball").xpos[2] - self.initial_z)
        both_fingers = sim.gripper_bodies.issubset(contact_with)
        speed = float(np.linalg.norm(sim.data.joint("ball_free").qvel[:3]))
        return {"success": bool(both_fingers and lift >= MIN_LIFT and speed < 0.08),
                "lift_cm": round(lift * 100, 2), "both_fingers_in_contact": both_fingers,
                "ball_speed_m_s": round(speed, 4)}

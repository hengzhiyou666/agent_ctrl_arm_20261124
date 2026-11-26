"""Franka Panda scene with calibrated RGB cameras and a bounded grasp."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
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
from scipy.spatial.transform import Rotation

from simulation.ball_vision import BallEstimate, CameraFrame, GraspError, estimate_ball, estimate_ball_from_radius
from simulation.replay import overlay_wrist_view

HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
MIN_LIFT = 0.08
MAX_LIFT = 0.25
DEFAULT_LIFT = 0.155
HEIGHT_TOLERANCE = 0.01
VERTICAL_TOLERANCE = 0.01
LIFT_CORRECTION_THRESHOLD = 0.0025
MAX_LIFT_CORRECTION = 0.015
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0
VIDEO_MS = 50
BALL_PARTS = {"orange": ("ball_free", "ball"), "green": ("green_ball_free", "green_ball_geom")}


@dataclass(frozen=True)
class GraspTrajectory:
    start_joints: np.ndarray
    center: np.ndarray
    radius: float
    color: str
    lift_m: float
    approach: tuple[np.ndarray, ...]
    lower: tuple[np.ndarray, ...]
    lift: tuple[np.ndarray, ...]


class FrankaBall:
    """Simulate the Franka and expose camera frames and grasp execution."""

    def __init__(self, ball_xy: tuple[float, float] = (0.49, 0.08), *, ball_radius: float | None = None,
                 green_xy: tuple[float, float] | None = None, green_radius: float | None = None):
        scene = str(Path(__file__).with_name("franka_ball.xml"))
        spec = mujoco.MjSpec.from_file(scene)
        self.ball_colors = ("orange",) if green_xy is None else ("orange", "green")
        if green_xy is None:
            spec.delete(spec.body("green_ball"))
        # The physical scene needs a size. The RGB estimator receives only camera frames.
        for color, radius in zip(self.ball_colors, (ball_radius, green_radius)):
            if radius is not None:
                spec.geom(BALL_PARTS[color][1]).size[0] = radius
                spec.body(f"{color}_ball").pos[2] = radius + 0.0005
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.tcp = self.model.site("grasp_tcp").id
        self.gripper_bodies = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.data.qpos[:9] = [*HOME, 0.04, 0.04]
        self.data.joint("ball_free").qpos[:2] = ball_xy
        if green_xy is not None:
            self.data.joint("green_ball_free").qpos[:2] = green_xy
        self.data.ctrl[:] = [*HOME, GRIP_OPEN]
        mujoco.mj_forward(self.model, self.data)
        self.down = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.frames: list[Image.Image] = []
        self.steps = 0
        self.step_observer: Callable[[], None] | None = None
        self.record_every = round((VIDEO_MS / 1000) / self.model.opt.timestep)
        self._step_for(0.20, record=False)

    def close(self) -> None:
        self.renderer.close()

    def rgb(self, camera: str = "overview") -> Image.Image:
        self.renderer.update_scene(self.data, camera=camera)
        return Image.fromarray(self.renderer.render())

    def observe(self, camera: str) -> CameraFrame:
        image = self.rgb(camera)
        camera_id = self.model.camera(camera).id
        focal = image.height / (2 * math.tan(math.radians(self.model.cam_fovy[camera_id]) / 2))
        cam_rotation = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        return CameraFrame(camera, image, focal, self.data.cam_xpos[camera_id].copy(), cam_rotation)

    def observe_pair(self) -> tuple[CameraFrame, CameraFrame]:
        # Rendering does not advance the simulator. Both images sample the same state.
        return self.observe("overview"), self.observe("wrist")

    def _record(self) -> None:
        overview = self.rgb()
        overlay_wrist_view(overview, self.rgb("wrist"))
        self.frames.append(overview)

    def _step(self, *, record: bool = True) -> None:
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.step_observer is not None:
            self.step_observer()
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
        samples = max(2, math.ceil(np.max(np.abs(first - second)) / 0.04))
        for part in np.linspace(0, 1, samples + 1):
            scratch.qpos[:7] = first + part * (second - first)
            mujoco.mj_forward(self.model, scratch)
            for contact in scratch.contact:
                if contact.geom1 != table and contact.geom2 != table:
                    continue
                other = contact.geom2 if contact.geom1 == table else contact.geom1
                if self.model.geom_bodyid[other] not in (arm_base, 0):
                    raise GraspError("规划会使机械臂碰到桌面")

    def plan_grasp(self, estimate: BallEstimate, *, color: str = "orange", lift_m: float = DEFAULT_LIFT) -> GraspTrajectory:
        center, radius = estimate.center, estimate.radius
        if color not in self.ball_colors or not MIN_LIFT <= lift_m <= MAX_LIFT:
            raise GraspError("目标或抬升高度不在当前场景支持范围内")
        x, y, z = center
        if not (0.34 <= x <= 0.65 and -0.23 <= y <= 0.23 and 0.014 <= z <= 0.046):
            raise GraspError("小球不在本工作台的可抓取区域")
        if not 0.012 <= radius <= 0.037 or abs(z - radius) > 0.012:
            raise GraspError("估计尺寸超出夹爪能力，或球心高度与台面不符")
        scratch = self._planning_state()
        first_joints = self.data.qpos[:7].copy()
        first_position = self.data.site_xpos[self.tcp].copy()
        above = center + np.array([0, 0, 0.135])
        lift_to = center + np.array([0, 0, lift_m])
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
        return GraspTrajectory(first_joints, center, radius, color, lift_m, *paths)

    def _planning_state(self) -> mujoco.MjData:
        scratch = mujoco.MjData(self.model)
        scratch.qpos[7:9] = 0.04
        # Arm/table clearance uses known geometry. Goal objects are outside this check.
        for index, ball_color in enumerate(self.ball_colors):
            scratch.joint(BALL_PARTS[ball_color][0]).qpos[:3] = [2.0 + index, 0.0, 1.0]
        return scratch

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
        self._refine_lift(plan)
        self._record()

    def _refine_lift(self, plan: GraspTrajectory) -> None:
        measurement = self.verify_grasp(plan, self.observe_pair())
        if measurement.get("held") is not True or measurement.get("vertical") is not True:
            return
        error = plan.lift_m - measurement["lift_cm"] / 100
        if not LIFT_CORRECTION_THRESHOLD < abs(error) <= MAX_LIFT_CORRECTION:
            return
        start = plan.lift[-1]
        target = plan.center + np.array([0, 0, plan.lift_m + error])
        scratch = self._planning_state()
        goal = self._solve_pose(scratch, start, target)
        self._clear_table(scratch, start, goal)
        self._follow((goal,), 0.35)

    def verify_grasp(self, plan: GraspTrajectory, frames: tuple[CameraFrame, CameraFrame]) -> dict:
        aperture = float(self.data.qpos[7:9].sum())
        gripper = self.data.site_xpos[self.tcp]

        def measure(center: np.ndarray, source: str) -> dict:
            lift = center[2] - plan.center[2]
            horizontal = np.linalg.norm(center[:2] - plan.center[:2])
            distance = np.linalg.norm(center - gripper)
            held = bool(lift >= 0.035 and distance < 0.045 and aperture > plan.radius)
            return {"held": held, "lift_cm": round(float(lift) * 100, 2),
                    "height_reached": bool(abs(lift - plan.lift_m) <= HEIGHT_TOLERANCE),
                    "vertical": bool(horizontal <= VERTICAL_TOLERANCE),
                    "horizontal_cm": round(float(horizontal) * 100, 2), "measurement_source": source}

        try:
            estimate = estimate_ball(frames, plan.color)
        except GraspError:
            estimate = None
        if estimate is not None and abs(estimate.radius - plan.radius) <= 0.25 * plan.radius:
            return measure(estimate.center, "dual_rgb")

        # A finger can cut the silhouette in one view. A sufficiently visible
        # other view can reuse the radius measured from the initial RGB pair.
        for frame in reversed(frames):
            try:
                center = estimate_ball_from_radius(frame, plan.radius, plan.color)
            except GraspError:
                continue
            receipt = measure(center, f"{frame.camera}_rgb_with_measured_radius")
            if receipt["held"]:
                return receipt
        if aperture < plan.radius:
            return {"held": False, "reason": "夹爪开口小于估计的球半径"}
        return {"held": None, "reason": "相机里可见的球体轮廓不足，无法确认抓持"}


class GraspScorer:
    """Independent simulation-only verdict, never placed in a model receipt."""

    def __init__(self, sim: FrankaBall, *, color: str = "orange", lift_m: float | None = None):
        self.sim = sim
        self.color = color
        self.lift_m = lift_m
        self.initial_positions = {name: sim.data.body(f"{name}_ball").xpos.copy() for name in sim.ball_colors}
        self.max_horizontal = 0.0
        self.max_other_displacement = 0.0
        sim.step_observer = self.record_motion

    def record_motion(self) -> None:
        for color, initial in self.initial_positions.items():
            delta = self.sim.data.body(f"{color}_ball").xpos - initial
            if color == self.color:
                self.max_horizontal = max(self.max_horizontal, math.hypot(delta[0], delta[1]))
            else:
                self.max_other_displacement = max(self.max_other_displacement, math.hypot(*delta))

    def evaluate(self) -> dict:
        sim = self.sim
        joint, geom = BALL_PARTS[self.color]
        ball = sim.model.geom(geom).id
        contact_with = set()
        for contact in sim.data.contact:
            if contact.geom1 == ball or contact.geom2 == ball:
                other = contact.geom2 if contact.geom1 == ball else contact.geom1
                contact_with.add(sim.model.geom_bodyid[other])
        lift = float(sim.data.body(f"{self.color}_ball").xpos[2] - self.initial_positions[self.color][2])
        both_fingers = sim.gripper_bodies.issubset(contact_with)
        speed = float(np.linalg.norm(sim.data.joint(joint).qvel[:3]))
        height_reached = lift >= MIN_LIFT if self.lift_m is None else abs(lift - self.lift_m) <= HEIGHT_TOLERANCE
        vertical = self.max_horizontal <= VERTICAL_TOLERANCE
        other_unchanged = self.max_other_displacement <= VERTICAL_TOLERANCE
        return {"success": both_fingers and height_reached and vertical and other_unchanged and speed < 0.08,
                "target": f"{self.color}_ball", "lift_cm": round(lift * 100, 2), "height_reached": height_reached,
                "max_horizontal_cm": round(self.max_horizontal * 100, 2), "vertical": vertical,
                "other_ball_max_displacement_cm": round(self.max_other_displacement * 100, 2),
                "both_fingers_in_contact": both_fingers, "ball_speed_m_s": round(speed, 4)}

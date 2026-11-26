"""Sample live RGB and apply cooperative stops between LIBERO control steps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from time import monotonic, sleep


_CAMERA_LABELS = {"agentview": "overview", "robot0_eye_in_hand": "wrist"}


@dataclass
class MotionWatch:
    interruption: dict | None = None
    outcome: dict | None = None


def capture_rgb(backend) -> dict[str, Path]:
    """Capture both cameras on the simulator owner thread."""
    observation = backend.get_observation_data()
    if not observation.success or observation.value is None:
        return {}
    return {_CAMERA_LABELS[camera.camera_name]: Path(camera.rgb_path)
            for camera in observation.value.cameras if camera.camera_name in _CAMERA_LABELS}


@contextmanager
def watch_motion(backend, monitor, action_id: str, tool: str, arguments: dict, *,
                 allow_interrupt: bool = False, allow_held_interrupt: bool = False,
                 pace_hz: float | None = None):
    """Watch one motion. The model worker reads images and returns stop requests asynchronously.

    LIBERO calls its motion guard before each physics step. Returning False
    preserves the observation and step counter for the last completed step.
    The controller's existing rejection path unwinds the active operation
    after the arm target is reset to the measured pose with the grip preserved.
    """
    if pace_hz is not None and (not math.isfinite(pace_hz) or pace_hz <= 0):
        raise ValueError("pace_hz must be positive and finite")
    previous_guard = getattr(backend, "_motion_guard", None)
    watch = MotionWatch()
    first_step = backend.simulation_steps
    previous_tick = monotonic()
    carrying = tool == "test_lift" or (tool == "move_arm" and arguments.get("waypoint") in {"lift", "basket"})
    monitor.begin(action_id, tool, arguments, first_step)

    def checkpoint() -> bool:
        nonlocal previous_tick
        if previous_guard is not None and not previous_guard():
            return False
        if watch.interruption is not None:
            return False

        step = backend.simulation_steps
        if step > first_step and pace_hz is not None:
            # Pace a demonstration at the control clock. Never wait for inference.
            remaining = previous_tick + 1.0 / pace_hz - monotonic()
            if remaining > 0:
                sleep(remaining)

        state_result = backend.get_robot_state()
        state = state_result.value if state_result.success else None
        held_interrupt = allow_held_interrupt and carrying
        open_interrupt = allow_interrupt and tool != "set_gripper" and not carrying
        interruptible = bool(state is not None and (held_interrupt or (open_interrupt and state.gripper_open_fraction >= 0.75)))
        telemetry = {"interruptible": interruptible, "action_steps": step - first_step}
        if state is not None:
            telemetry.update(tcp_xyz_m=state.end_effector_position_m,
                             tcp_quaternion_wxyz=state.end_effector_quaternion_wxyz,
                             gripper_open_fraction=state.gripper_open_fraction,
                             joint_velocity_norm_rad_s=state.joint_velocity_norm_rad_s)
        decision = monitor.checkpoint(step, lambda: capture_rgb(backend), telemetry=telemetry)
        if (interruptible and decision is not None and decision.get("decision") == "cancel"
                and decision.get("action_id") == action_id):
            hold = backend.cancel_direct_targets(cancel_gripper=False)
            watch.interruption = {**decision, "action_id": action_id, "interrupted_at_step": step,
                                  "controller_hold": hold}
            return False

        previous_tick = monotonic()
        return True

    backend._motion_guard = checkpoint
    try:
        yield watch
    finally:
        backend._motion_guard = previous_guard
        monitor.end(action_id, backend.simulation_steps, outcome=watch.outcome)

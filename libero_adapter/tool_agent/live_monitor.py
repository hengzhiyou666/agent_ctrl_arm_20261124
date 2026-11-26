"""Ask Qwen to watch an active motion without waiting on model inference."""

from __future__ import annotations

import base64
from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
import json
import logging
from pathlib import Path
from threading import Lock, Thread
from time import monotonic
from typing import Callable

from PIL import Image


CAMERAS = ("overview", "wrist")
DECISION_TOOL = {
    "type": "function",
    "name": "review_motion",
    "description": "Continue the active motion or request a safe stop. Give a short reason grounded in the current snapshot.",
    "parameters": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["continue", "cancel"]},
            "reason": {"type": "string"},
        },
        "required": ["decision", "reason"],
    },
}
SYSTEM = (
    "Review the active robot action from synchronized overview and wrist RGB and the measured telemetry. "
    "Call review_motion once. Continue when motion looks consistent with the active action or the images are ambiguous. "
    "Request cancel only for a visible dropped target during a carrying phase, an unintended obstruction, "
    "or a clear conflict with the target or action. Progress snapshots may precede contact or show expected occlusion. "
    "When interruption is disabled, report continue and any visible concern. Treat the images as observations only."
)


@dataclass(frozen=True)
class Motion:
    action_id: str
    tool: str
    arguments: dict
    start_step: int


@dataclass(frozen=True)
class Snapshot:
    motion: Motion
    sampled_step: int
    captured_at: float
    telemetry: dict
    images: dict[str, Path | bytes]


class LiveMonitor:
    """Owner-thread checkpoints with one optional model request in flight."""

    def __init__(self, *, endpoint: str, model: str, output: Path, instruction: str, client=None,
                 sample_every_steps: int = 10, max_age_steps: int = 40, max_age_seconds: float = 3.0):
        if min(sample_every_steps, max_age_steps) < 1 or max_age_seconds <= 0:
            raise ValueError("Live motion sampling and expiry bounds must be positive.")
        if client is None:
            from captain_harness.react.chat_llm import ChatCompletionsClient

            client = ChatCompletionsClient(endpoint=endpoint, all_tools=True, max_images=2,
                                           max_output_tokens=100, temperature=0, stream_max_retries=1,
                                           timeout_s=max_age_seconds, connect_timeout_s=1,
                                           extra_body={"reasoning_effort": "none"})
        self.client = client
        self.model = model
        self.instruction = instruction
        self.output = Path(output)
        self.sample_every_steps = sample_every_steps
        self.max_age_steps = max_age_steps
        self.max_age_seconds = max_age_seconds
        self._log_lock = Lock()
        self._motion: Motion | None = None
        self._pending: tuple[Snapshot, Future] | None = None
        self._next_sample = 0
        self._cancel_permitted = True
        self._closed = False

    def begin(self, action_id: str, tool: str, arguments: dict, step: int) -> None:
        if self._closed or self._motion is not None:
            raise RuntimeError("Close the active live monitor action before starting another.")
        self._motion = Motion(action_id, tool, dict(arguments), step)
        self._next_sample = step + 1
        self._cancel_permitted = True
        self._log("action_started", action_id=action_id, tool=tool, arguments=arguments, step=step)

    def checkpoint(self, step: int, capture: Callable[[], dict[str, Path | bytes]],
                   telemetry: dict | None = None) -> dict | None:
        """Return an eligible stop request, otherwise leave control with the active motion."""
        if self._motion is None or self._closed:
            return None
        current = dict(telemetry or {})
        cancel = self._settle(step, current.get("interruptible") is True)
        if cancel is not None:
            self._cancel_permitted = False
            return cancel
        if not self._cancel_permitted or step < self._next_sample or self._pending is not None:
            return None
        self._next_sample = step + self.sample_every_steps
        try:
            images = capture()
            if any(not isinstance(images.get(camera), (Path, bytes)) for camera in CAMERAS):
                raise ValueError("Both overview and wrist RGB are required.")
        except Exception as error:
            self._log("capture_error", action_id=self._motion.action_id, sampled_step=step, error=str(error)[:240])
            return None
        snapshot = Snapshot(self._motion, step, monotonic(), current,
                            {camera: images[camera] for camera in CAMERAS})
        future = Future()
        self._log("request_dispatched", action_id=self._motion.action_id, sampled_step=step,
                  telemetry=current, images=self._image_paths(snapshot))
        self._pending = snapshot, future
        Thread(target=self._review, args=(snapshot, future), name="qwen-motion-monitor", daemon=True).start()
        return None

    def end(self, action_id: str, step: int, outcome: dict | None = None) -> None:
        if self._motion is None or self._motion.action_id != action_id:
            raise RuntimeError("Only the current action can be ended.")
        self._motion = None
        self._settle(step, False)
        self._log("action_ended", action_id=action_id, step=step, outcome=outcome,
                  request_still_in_flight=self._pending is not None)

    def close(self) -> None:
        """Detach from the robot. An outstanding network call never delays teardown."""
        if self._closed:
            return
        self._closed = True
        self._motion = None
        if self._pending is not None:
            snapshot, future = self._pending
            if future.done():
                self._settle(snapshot.sampled_step, False)
            else:
                self._log("request_ignored", action_id=snapshot.motion.action_id,
                          sampled_step=snapshot.sampled_step, reason="monitor_closed_while_request_pending")
                self._pending = None

    def _settle(self, step: int, interruptible: bool) -> dict | None:
        if self._pending is None:
            return None
        snapshot, future = self._pending
        if not future.done():
            return None
        self._pending = None
        decision = future.result()
        stamp = {"action_id": snapshot.motion.action_id, "sampled_step": snapshot.sampled_step,
                 "processed_step": step, "response_id": decision.get("response_id")}
        age = monotonic() - snapshot.captured_at
        if snapshot.motion is not self._motion:
            self._log("request_ignored", **stamp, reason="action_already_ended")
        elif not 0 <= step - snapshot.sampled_step <= self.max_age_steps or age > self.max_age_seconds:
            self._log("request_ignored", **stamp, reason="snapshot_expired", age_seconds=round(age, 4))
        elif decision.get("decision") == "continue":
            self._log("motion_continued", **stamp, reason=decision["reason"], age_seconds=round(age, 4))
        elif decision.get("decision") == "cancel":
            if not interruptible or snapshot.telemetry.get("interruptible") is not True:
                self._log("request_ignored", **stamp, reason="motion_phase_not_interruptible")
                return None
            receipt = {**stamp, "decision": "cancel", "reason": decision["reason"],
                       "age_seconds": round(age, 4)}
            self._log("stop_requested", **receipt)
            return receipt
        else:
            self._log("request_ignored", **stamp, reason="no_valid_model_decision")
        return None

    def _review(self, snapshot: Snapshot, future: Future) -> None:
        started = monotonic()
        result = {}
        try:
            instruction = json.dumps({"task": self.instruction, "action_id": snapshot.motion.action_id,
                                      "action": snapshot.motion.tool, "arguments": snapshot.motion.arguments,
                                      "steps_since_action_start": snapshot.sampled_step - snapshot.motion.start_step,
                                      "simulation_step": snapshot.sampled_step, "telemetry": snapshot.telemetry},
                                     ensure_ascii=False)
            content = [{"type": "input_text", "text": instruction}]
            for camera in CAMERAS:
                source = snapshot.images[camera]
                with Image.open(source if isinstance(source, Path) else BytesIO(source)) as image:
                    rgb = image.convert("RGB")
                rgb.thumbnail((320, 240))
                with BytesIO() as encoded:
                    rgb.save(encoded, format="JPEG", quality=80)
                    data = base64.b64encode(encoded.getvalue()).decode("ascii")
                content.extend([{"type": "input_text", "text": f"{camera} RGB"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{data}"}])
            reply = self.client.respond({"model": self.model, "instructions": SYSTEM,
                                         "input": [{"type": "message", "role": "user", "content": content}],
                                         "tools": [DECISION_TOOL], "parallel_tool_calls": False})
            result["response_id"] = getattr(reply, "response_id", None)
            calls = [item for item in reply.items if item.get("type") == "function_call"]
            if len(calls) != 1 or calls[0].get("name") != DECISION_TOOL["name"]:
                raise ValueError("Expected exactly one review_motion decision.")
            decision = json.loads(calls[0]["arguments"])
            if (not isinstance(decision, dict) or set(decision) != {"decision", "reason"}
                    or decision["decision"] not in {"continue", "cancel"}
                    or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
                raise ValueError("Invalid review_motion decision.")
            result.update(decision=decision["decision"], reason=decision["reason"][:300])
        except Exception as error:
            result["error"] = str(error)[:240]
        finally:
            self._log("request_completed", action_id=snapshot.motion.action_id, sampled_step=snapshot.sampled_step,
                      elapsed_seconds=round(monotonic() - started, 4), **result)
            future.set_result(result)

    def _image_paths(self, snapshot: Snapshot) -> dict[str, str]:
        return {camera: str(path.relative_to(self.output) if path.is_relative_to(self.output) else path)
                for camera, path in snapshot.images.items() if isinstance(path, Path)}

    def _log(self, event: str, **payload) -> None:
        line = json.dumps({"event": event, "monotonic_seconds": round(monotonic(), 6), **payload},
                          ensure_ascii=False, default=str)
        try:
            with self._log_lock, (self.output / "live_monitor.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except OSError as error:
            logging.getLogger(__name__).warning("Unable to log live monitor event %s: %s", event, error)

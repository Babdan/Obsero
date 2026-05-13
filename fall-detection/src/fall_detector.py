"""
Fall Detector — Main asynchronous detection orchestrator.

This is the primary module that ties the entire pipeline together:
    Camera → Pose → Normalize → Buffer → TCN Model → Rules → Confirm → Alarm

Runs as an async task, processing frames independently of the camera
capture loop. Supports both live camera feeds and video file input.
"""

import asyncio
import time
import cv2
import yaml
import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Union

from src.pose_extractor import PoseExtractor, PoseResult, NUM_KEYPOINTS
from src.skeleton_normalizer import SkeletonNormalizer
from src.keypoint_buffer import KeypointBuffer
from src.temporal_model import FallDetectionModel, load_model
from src.posture_rules import PostureRuleChecker, PostureCheckResult
from src.alarm_interface import AlarmInterface, FallAlarmEvent
from src.debug_logger import DebugLogger


class FallDetector:
    """
    Asynchronous fall detection module.

    Captures frames, extracts poses, normalizes skeletons, buffers
    keypoint sequences, runs temporal inference, checks posture rules,
    and emits alarms when falls are confirmed.

    This module is designed to run asynchronously alongside other
    system components without blocking.

    Args:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str):
        # Load configuration
        self._config_path = Path(config_path).expanduser().resolve()
        with open(self._config_path, "r") as f:
            self._config = yaml.safe_load(f)

        model_cfg = self._config["model"]
        self._allow_untrained = model_cfg.get("allow_untrained", False)
        self._weights_path = self._resolve_weights_path(model_cfg)

        # ── Initialize sub-modules ─────────────────────────────────────────
        self._pose_extractor = PoseExtractor(self._config["pose"])
        self._normalizer = SkeletonNormalizer(self._config["normalization"])

        buf_cfg = self._config["buffer"]
        self._buffer = KeypointBuffer(
            window_size=buf_cfg["window_size"],
            target_fps=buf_cfg["target_fps"],
            min_frames=buf_cfg["min_frames_for_prediction"],
            stride=buf_cfg.get("stride", 5),
        )

        self._alarm = AlarmInterface(self._config["alarm"])

        # ── Model ──────────────────────────────────────────────────────────
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[FallDetectionModel] = None

        if self._weights_path.exists():
            self._model = load_model(
                str(self._weights_path), model_cfg, self._device
            )
        elif self._allow_untrained:
            self._model = FallDetectionModel(model_cfg)
            self._model.to(self._device)
            self._model.eval()

        # ── Decision Logic ─────────────────────────────────────────────────
        dec_cfg = self._config["decision"]
        self._threshold = dec_cfg["model_threshold"]
        self._confirm_frames = dec_cfg["confirmation_frames"]
        self._confirm_window = dec_cfg["confirmation_window"]
        self._cooldown_sec = dec_cfg.get("cooldown_sec", 10.0)

        self._rule_checker = PostureRuleChecker(dec_cfg.get("rules", {}))

        # Confirmation tracking
        self._recent_probs: deque = deque(maxlen=self._confirm_window)
        self._last_alarm_time: float = 0.0

        # ── Camera ─────────────────────────────────────────────────────────
        cam_cfg = self._config["camera"]
        self._camera_source = cam_cfg["source"]
        self._capture_fps = cam_cfg.get("capture_fps", 30)
        self._resolution = cam_cfg.get("resolution", [640, 480])
        self._camera_id = "cam_0"

        # ── Debug ──────────────────────────────────────────────────────────
        self._logger = DebugLogger(
            self._config["debug"], camera_id=self._camera_id
        )

        # ── State ──────────────────────────────────────────────────────────
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None

        # Raw keypoints buffer for posture rules (un-normalized)
        self._raw_keypoints_history: deque = deque(
            maxlen=buf_cfg["window_size"]
        )
        self._raw_timestamps: deque = deque(maxlen=buf_cfg["window_size"])

    def _resolve_weights_path(self, model_cfg: dict) -> Path:
        """Resolve and validate the configured model checkpoint path."""
        weights_path = model_cfg.get("weights_path", "")
        if not weights_path:
            raise ValueError(
                "model.weights_path must point to a trained checkpoint. "
                "Set model.allow_untrained: true only for non-production tests."
            )

        weights_file = self._resolve_project_path(weights_path)
        if weights_file.exists() or self._allow_untrained:
            return weights_file

        raise FileNotFoundError(
            f"Trained model weights not found: {weights_file}. "
            "Train a model with `python -m training.train` or place the "
            "checkpoint at model.weights_path. Set model.allow_untrained: "
            "true only for non-production tests."
        )

    def _resolve_project_path(self, path_value: str) -> Path:
        """Resolve config-relative paths from the project root."""
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path

        config_dir = self._config_path.parent
        project_root = (
            config_dir.parent if config_dir.name == "config" else config_dir
        )
        return (project_root / path).resolve()

    async def start(
        self,
        camera_id: str = "cam_0",
        source: Optional[Union[int, str]] = None,
        show_preview: bool = False,
    ):
        """
        Start the fall detection loop.

        Args:
            camera_id: Identifier for this camera stream.
            source: Camera index or video file path. Uses config if None.
            show_preview: If True, show an OpenCV window with skeleton overlay.
        """
        self._camera_id = camera_id
        self._logger.set_camera_id(camera_id)
        self._running = True

        src = source if source is not None else self._camera_source
        self._capture = cv2.VideoCapture(src)

        if not self._capture.isOpened():
            self._logger.error(f"Cannot open camera/video: {src}")
            return

        # Set resolution
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

        self._logger.info(
            f"Fall detector started | camera={camera_id} | source={src} | "
            f"device={self._device} | model={self._config['model']['architecture']}"
        )

        # Compute frame interval
        frame_interval = 1.0 / self._capture_fps

        try:
            while self._running:
                loop_start = time.monotonic()

                ret, frame = self._capture.read()
                if not ret:
                    if isinstance(src, str):
                        # Video file ended
                        self._logger.info("Video file ended.")
                        break
                    else:
                        await asyncio.sleep(0.01)
                        continue

                timestamp_ms = int(time.time() * 1000)
                await self._process_frame(frame, timestamp_ms, show_preview)

                # Rate limiting
                elapsed = time.monotonic() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            self._logger.info("Interrupted by user.")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the detection loop and release resources."""
        self._running = False

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._pose_extractor.close()
        cv2.destroyAllWindows()

        self._logger.info(
            f"Fall detector stopped | {self._logger.get_stats()}"
        )

    async def _process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        show_preview: bool = False,
    ):
        """
        Process a single frame through the full detection pipeline.

        Steps:
            1. Pose extraction
            2. Skeleton normalization
            3. Buffer insertion
            4. Temporal model inference (if buffer ready)
            5. Posture rule checks
            6. Confirmation logic
            7. Alarm emission
        """
        # ── 1. Pose Extraction ─────────────────────────────────────────────
        pose_result = self._pose_extractor.extract(frame, timestamp_ms)

        if pose_result is None:
            self._buffer.push_empty(timestamp_ms)
            self._logger.log_frame(
                timestamp_ms, None, None, "no_person",
                buffer_length=self._buffer.length,
            )

            if show_preview:
                self._show_frame(frame, None, None, "No person")

            return

        raw_xy = pose_result.xy  # (17, 2) un-normalized

        # ── 2. Normalize ───────────────────────────────────────────────────
        normalized = self._normalizer.normalize(raw_xy.copy())

        # ── 3. Buffer ──────────────────────────────────────────────────────
        accepted = self._buffer.push(timestamp_ms, normalized)

        # Store raw keypoints for posture rules
        if accepted:
            self._raw_keypoints_history.append(raw_xy.copy())
            self._raw_timestamps.append(timestamp_ms)

        # ── 4 & 5. Inference + Rules ───────────────────────────────────────
        model_prob = None
        rules_result = None
        decision = "normal"

        if self._buffer.should_infer():
            model_prob = self._run_inference()
            rules_result = self._run_posture_checks()
            self._buffer.mark_inferred()

            # Track probability history
            self._recent_probs.append(model_prob)

            # Store latest for continuous display
            self._last_model_prob = model_prob
            self._last_rules_result = rules_result

            # ── Debug: Print every inference result to console ─────────
            above_ct = sum(1 for p in self._recent_probs if p >= self._threshold)
            rules_str = "N/A"
            if rules_result:
                r = rules_result
                rules_str = (
                    f"descent={r.rapid_descent} "
                    f"low={r.low_position} "
                    f"lying={r.lying_duration_met} "
                    f"aspect={r.aspect_ratio_horizontal} "
                    f"-> pass={r.overall_pass}"
                )
            print(
                f"  [INF] prob={model_prob:.3f} | "
                f"thresh={self._threshold} | "
                f"confirm={above_ct}/{self._confirm_frames} "
                f"(window={len(self._recent_probs)}/{self._confirm_window}) | "
                f"rules: {rules_str} | "
                f"buf={self._buffer.length}/{self._buffer.window_size}"
            )

            # ── 6. Confirmation ────────────────────────────────────────────
            decision = self._should_trigger_alarm(model_prob, rules_result)

            # ── 7. Alarm ──────────────────────────────────────────────────
            if decision in ("fall_confirmed", "fall_suspected"):
                confirmed = decision == "fall_confirmed"
                print(
                    f"  >>> ALARM: {decision.upper()} "
                    f"(prob={model_prob:.3f})"
                )

                event = self._alarm.create_event(
                    camera_id=self._camera_id,
                    confidence=model_prob,
                    confirmed=confirmed,
                    keypoints=raw_xy,
                    posture_checks=(
                        rules_result.to_dict() if rules_result else None
                    ),
                    frame=frame,
                    model_config={
                        "architecture": self._config["model"]["architecture"],
                        "window_size": self._buffer.window_size,
                        "threshold": self._threshold,
                    },
                )

                self._alarm.emit(event)
                self._logger.log_event(event.to_dict())

                # Save keypoint sequence for future training
                seq, ts = self._buffer.get_sequence()
                self._logger.save_sequence(
                    seq,
                    metadata={
                        "camera_id": self._camera_id,
                        "confidence": model_prob,
                        "timestamps": ts,
                        "posture": (
                            rules_result.to_dict() if rules_result else {}
                        ),
                    },
                    label="fall" if confirmed else "suspected",
                )

                self._last_alarm_time = time.time()

        # ── Logging ────────────────────────────────────────────────────────
        self._logger.log_frame(
            timestamp_ms,
            model_prob,
            rules_result.to_dict() if rules_result else None,
            decision,
            buffer_length=self._buffer.length,
        )

        if show_preview:
            self._show_frame(frame, pose_result, model_prob, decision, rules_result)

    def _run_inference(self) -> float:
        """Run the temporal model on the current buffer contents."""
        seq, timestamps = self._buffer.get_padded_sequence()[:2]

        with torch.no_grad():
            tensor = torch.tensor(
                seq, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            prob = self._model.predict_proba(tensor)

        return float(prob.item())

    def _run_posture_checks(self) -> PostureCheckResult:
        """Run posture rule checks on raw keypoint history."""
        if len(self._raw_keypoints_history) < 2:
            return PostureCheckResult()

        raw_seq = np.stack(list(self._raw_keypoints_history))
        timestamps = list(self._raw_timestamps)

        return self._rule_checker.check(raw_seq, timestamps)

    def _should_trigger_alarm(
        self,
        model_prob: float,
        rules_result: Optional[PostureCheckResult],
    ) -> str:
        """
        Determine if an alarm should be triggered.

        Confirmation logic:
            - Count how many of the recent N probabilities exceed threshold
            - If enough frames confirm AND rules pass -> "fall_confirmed"
            - If model says fall but rules fail -> "fall_suspected"
            - Otherwise -> "normal"

        Returns:
            "fall_confirmed", "fall_suspected", or "normal"
        """
        # Cooldown check
        if time.time() - self._last_alarm_time < self._cooldown_sec:
            return "normal"

        # Count frames above threshold in recent window
        above_threshold = sum(
            1 for p in self._recent_probs if p >= self._threshold
        )

        if above_threshold < self._confirm_frames:
            return "normal"

        # Model says fall -- check rules for confirmed vs suspected
        if rules_result is not None and rules_result.overall_pass:
            return "fall_confirmed"
        else:
            # Still trigger as suspected even without rules
            return "fall_suspected"

    def _show_frame(
        self,
        frame: np.ndarray,
        pose: Optional[PoseResult],
        model_prob: Optional[float],
        status: str,
        rules_result: Optional[PostureCheckResult] = None,
    ):
        """
        Show a preview frame with skeleton overlay and rich debug HUD.

        Displays continuously-updating model probability, posture rule
        states, confirmation counter, and buffer fill level.
        """
        display = frame.copy()
        h, w = display.shape[:2]

        if pose is not None:
            self._draw_skeleton(display, pose.xy)

        # Use last known values for continuous display
        prob = model_prob if model_prob is not None else getattr(self, '_last_model_prob', None)
        rules = rules_result if rules_result is not None else getattr(self, '_last_rules_result', None)

        # -- Status bar background --
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 175), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # -- Status color --
        if "confirmed" in status:
            status_color = (0, 0, 255)     # Red
            status_text = "!! FALL CONFIRMED !!"
        elif "suspected" in status:
            status_color = (0, 165, 255)   # Orange
            status_text = "! FALL SUSPECTED !"
        elif status == "no_person":
            status_color = (128, 128, 128) # Gray
            status_text = "No Person"
        else:
            status_color = (0, 255, 0)     # Green
            status_text = "Normal"

        # Line 1: Status
        cv2.putText(display, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        # Line 2: Model probability bar
        if prob is not None:
            bar_w = int(prob * 200)
            bar_color = (0, 255, 0) if prob < 0.5 else (
                (0, 165, 255) if prob < self._threshold else (0, 0, 255)
            )
            cv2.rectangle(display, (10, 40), (210, 58), (80, 80, 80), -1)
            cv2.rectangle(display, (10, 40), (10 + bar_w, 58), bar_color, -1)
            cv2.putText(display, f"Prob: {prob:.1%}", (220, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            # Threshold marker
            thresh_x = 10 + int(self._threshold * 200)
            cv2.line(display, (thresh_x, 38), (thresh_x, 60), (255, 255, 255), 2)

        # Line 3: Confirmation counter
        above_ct = sum(1 for p in self._recent_probs if p >= self._threshold)
        confirm_color = (0, 255, 0) if above_ct < self._confirm_frames else (0, 0, 255)
        cv2.putText(
            display,
            f"Confirm: {above_ct}/{self._confirm_frames} "
            f"(window: {len(self._recent_probs)}/{self._confirm_window})",
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confirm_color, 1,
        )

        # Line 4: Buffer fill
        buf_pct = self._buffer.length / max(self._buffer.window_size, 1)
        cv2.putText(
            display,
            f"Buffer: {self._buffer.length}/{self._buffer.window_size}",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # Line 5-6: Posture Rules
        if rules is not None:
            r = rules
            def rule_color(v): return (0, 255, 0) if v else (0, 0, 255)

            cv2.putText(display, "Rules:", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(display, f"Descent:{r.max_vertical_velocity:.3f}", (75, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, rule_color(r.rapid_descent), 1)
            cv2.putText(display, f"LowPos:{r.final_hip_height:.2f}", (240, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, rule_color(r.low_position), 1)
            cv2.putText(display, f"Lying:{r.time_below_threshold_sec:.1f}s", (75, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, rule_color(r.lying_duration_met), 1)
            cv2.putText(display, f"Aspect:{r.final_aspect_ratio:.2f}", (240, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, rule_color(r.aspect_ratio_horizontal), 1)

            pass_text = "PASS" if r.overall_pass else "FAIL"
            pass_color = (0, 255, 0) if r.overall_pass else (0, 0, 255)
            cv2.putText(display, f"Overall: {pass_text}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pass_color, 1)

        # Cooldown indicator
        cooldown_remaining = self._cooldown_sec - (time.time() - self._last_alarm_time)
        if cooldown_remaining > 0:
            cv2.putText(display, f"Cooldown: {cooldown_remaining:.1f}s",
                        (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Red border flash for 2 seconds after alarm
        alarm_elapsed = time.time() - self._last_alarm_time
        if self._last_alarm_time > 0 and alarm_elapsed < 2.0:
            border = 8
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
            cv2.rectangle(display, (border, border), (w - 1 - border, h - 1 - border), (0, 0, 255), border // 2)

        cv2.imshow("Fall Detection", display)
        cv2.waitKey(1)

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw skeleton connections on a frame."""
        h, w = frame.shape[:2]

        # COCO skeleton connections (pairs of keypoint indices)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),   # Head
            (5, 6),                              # Shoulders
            (5, 7), (7, 9),                      # Left arm
            (6, 8), (8, 10),                     # Right arm
            (5, 11), (6, 12),                    # Torso
            (11, 12),                            # Hips
            (11, 13), (13, 15),                  # Left leg
            (12, 14), (14, 16),                  # Right leg
        ]

        # Draw connections
        for i, j in connections:
            pt1 = keypoints[i]
            pt2 = keypoints[j]

            if pt1.sum() > 0 and pt2.sum() > 0:
                x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp.sum() > 0:
                x, y = int(kp[0] * w), int(kp[1] * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def get_status(self) -> dict:
        """Get current detector state for monitoring."""
        return {
            "running": self._running,
            "camera_id": self._camera_id,
            "device": str(self._device),
            "buffer_fill": self._buffer.length,
            "buffer_size": self._buffer.window_size,
            "recent_alarms": self._alarm.get_event_count(),
            "stats": self._logger.get_stats(),
        }

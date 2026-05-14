"""
Debug Logger — Comprehensive logging of keypoints, decisions, and events.

Saves keypoint sequences, model confidence, posture rule results,
and final decisions for debugging and future training data generation.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


class DebugLogger:
    """
    Logs detection pipeline data for debugging and future model training.

    Saves:
        - Keypoint sequences (.npy) on alarm events
        - Per-frame decisions to a JSON-lines log file
        - Summary statistics to console

    Args:
        config: Dictionary with debug logging parameters.
    """

    def __init__(self, config: dict, camera_id: str = "cam_0"):
        self._enabled = config.get("enabled", True)
        self._log_dir = Path(config.get("log_dir", "logs"))
        self._save_keypoints = config.get("save_keypoints", True)
        self._save_raw_frames = config.get("save_raw_frames", False)
        self._log_interval = config.get("log_interval_sec", 1.0)
        self._max_files = config.get("max_log_files", 1000)
        self._camera_id = camera_id

        # Create directories
        self._keypoint_dir = self._log_dir / "keypoints"
        self._event_dir = self._log_dir / "events"
        self._snapshot_dir = self._log_dir / "snapshots"

        if self._enabled:
            for d in [self._keypoint_dir, self._event_dir, self._snapshot_dir]:
                d.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self._logger = logging.getLogger("fall_detection")
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG if self._enabled else logging.WARNING)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

            # File handler
            if self._enabled:
                log_file = self._log_dir / "fall_detection.log"
                fh = logging.FileHandler(str(log_file), encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)-7s | %(message)s"
                    )
                )
                self._logger.addHandler(fh)

        # Stats tracking
        self._frame_count = 0
        self._detection_count = 0
        self._alarm_count = 0

    def set_camera_id(self, camera_id: str):
        """Update the camera id used in log messages and generated filenames."""
        self._camera_id = camera_id

    def log_frame(
        self,
        timestamp_ms: int,
        model_prob: Optional[float],
        rules_result: Optional[dict],
        decision: str,
        buffer_length: int = 0,
    ):
        """
        Log per-frame detection pipeline results.

        Args:
            timestamp_ms: Frame timestamp.
            model_prob: Model fall probability (None if not yet inferred).
            rules_result: Posture rule check results dict.
            decision: "fall_confirmed", "fall_suspected", "normal", or "no_person".
            buffer_length: Current keypoint buffer fill level.
        """
        self._frame_count += 1

        if decision in ("fall_confirmed", "fall_suspected"):
            self._detection_count += 1
            level = logging.WARNING if decision == "fall_suspected" else logging.ERROR
            self._logger.log(
                level,
                f"[{self._camera_id}] {decision.upper()} | "
                f"prob={model_prob:.3f} | "
                f"rules={rules_result} | "
                f"buffer={buffer_length}",
            )
        elif self._frame_count % max(1, int(30 * self._log_interval)) == 0:
            # Periodic status log
            prob_str = f"{model_prob:.3f}" if model_prob is not None else "N/A"
            self._logger.debug(
                f"[{self._camera_id}] frames={self._frame_count} | "
                f"prob={prob_str} | buffer={buffer_length}"
            )

    def log_event(self, event_dict: dict):
        """
        Log a complete alarm event to a JSON file.

        Args:
            event_dict: Serialized FallAlarmEvent dictionary.
        """
        if not self._enabled:
            return

        self._alarm_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"event_{self._camera_id}_{timestamp}.json"
        filepath = self._event_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(event_dict, f, indent=2, default=str)

        self._logger.info(f"Event logged: {filepath}")
        self._cleanup_old_files(self._event_dir, "*.json")

    def save_sequence(
        self,
        sequence: np.ndarray,
        metadata: dict,
        label: str,
    ) -> dict | None:
        """
        Save a keypoint sequence to disk for debugging or training.

        Args:
            sequence: (T, 34) keypoint array.
            metadata: Dictionary of additional context.
            label: "fall" or "normal" label for the sequence.
        """
        if not self._enabled or not self._save_keypoints:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"seq_{self._camera_id}_{label}_{timestamp}"
        sequence_path = self._keypoint_dir / f"{filename}.npy"
        metadata_path = self._keypoint_dir / f"{filename}_meta.json"

        # Save keypoints as numpy
        np.save(str(sequence_path), sequence)

        # Save metadata
        meta = {**metadata, "label": label, "shape": list(sequence.shape)}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        self._logger.debug(f"Sequence saved: {filename}")
        self._cleanup_old_files(self._keypoint_dir, "*.npy")
        return {
            "sequence_path": str(sequence_path),
            "metadata_path": str(metadata_path),
            "label": label,
            "shape": list(sequence.shape),
        }

    def save_snapshot(
        self, frame: np.ndarray, label: str
    ) -> Optional[str]:
        """
        Save a raw camera frame as a JPEG snapshot.

        Returns:
            Path to saved file, or None if saving is disabled.
        """
        if not self._enabled or not self._save_raw_frames:
            return None

        import cv2

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snap_{self._camera_id}_{label}_{timestamp}.jpg"
        filepath = self._snapshot_dir / filename

        cv2.imwrite(str(filepath), frame)
        self._cleanup_old_files(self._snapshot_dir, "*.jpg")
        return str(filepath)

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def get_stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "detections": self._detection_count,
            "alarms": self._alarm_count,
        }

    def _cleanup_old_files(self, directory: Path, pattern: str):
        """Remove oldest files if exceeding max_files limit."""
        files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
        while len(files) > self._max_files:
            files[0].unlink()
            files.pop(0)

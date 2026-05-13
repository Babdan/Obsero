"""
Posture Rules — Rule-based heuristic guards for fall confirmation.

These rules run alongside the temporal neural network and act as
guard conditions. Both the model AND the rules must agree to confirm
a fall, reducing false positives from the model alone.

Checks:
    1. Rapid vertical descent (hip drops fast)
    2. Low final body position (person is near ground)
    3. Lying duration (stays low for minimum time)
    4. Body aspect ratio (wider than tall → lying down)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.pose_extractor import (
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP,
    COCO_LEFT_SHOULDER,
    COCO_RIGHT_SHOULDER,
    COCO_LEFT_ANKLE,
    COCO_RIGHT_ANKLE,
    NUM_KEYPOINTS,
)


@dataclass
class PostureCheckResult:
    """Result of all posture rule checks for a keypoint sequence."""

    rapid_descent: bool = False
    low_position: bool = False
    lying_duration_met: bool = False
    aspect_ratio_horizontal: bool = False

    # Measurements for debugging
    max_vertical_velocity: float = 0.0
    final_hip_height: float = 0.0
    time_below_threshold_sec: float = 0.0
    final_aspect_ratio: float = 0.0

    @property
    def overall_pass(self) -> bool:
        """
        True if enough conditions are met to support a fall hypothesis.
        Requires at least 2 of the 4 checks to pass.
        """
        checks = [
            self.rapid_descent,
            self.low_position,
            self.lying_duration_met,
            self.aspect_ratio_horizontal,
        ]
        return sum(checks) >= 2

    def to_dict(self) -> dict:
        return {
            "rapid_descent": self.rapid_descent,
            "low_position": self.low_position,
            "lying_duration_met": self.lying_duration_met,
            "aspect_ratio_horizontal": self.aspect_ratio_horizontal,
            "overall_pass": self.overall_pass,
            "max_vertical_velocity": round(self.max_vertical_velocity, 4),
            "final_hip_height": round(self.final_hip_height, 4),
            "time_below_threshold_sec": round(self.time_below_threshold_sec, 3),
            "final_aspect_ratio": round(self.final_aspect_ratio, 4),
        }


class PostureRuleChecker:
    """
    Evaluates rule-based posture conditions on raw keypoint sequences.

    Uses un-normalized (image-space) keypoints so that absolute positions
    (e.g., "person is near the bottom of the frame") are meaningful.

    Args:
        config: Dictionary with rule thresholds from config.decision.rules
    """

    def __init__(self, config: dict):
        self._min_velocity = config.get("min_vertical_velocity", 0.15)
        self._max_hip_height = config.get("max_final_hip_height", 0.35)
        self._min_lying_sec = config.get("min_lying_duration_sec", 1.0)
        self._enabled = config.get("enable_rules", True)

        # Internal state for tracking lying duration
        self._low_since_ts: Optional[int] = None

    def check(
        self,
        raw_keypoints_seq: np.ndarray,
        timestamps_ms: list[int],
    ) -> PostureCheckResult:
        """
        Run all posture rule checks on a keypoint sequence.

        Args:
            raw_keypoints_seq: (T, 17, 2) raw image-normalized keypoints.
            timestamps_ms: List of T timestamps in milliseconds.

        Returns:
            PostureCheckResult with individual and overall results.
        """
        if not self._enabled:
            # If rules are disabled, return a permissive result
            return PostureCheckResult(
                rapid_descent=True,
                low_position=True,
                lying_duration_met=True,
                aspect_ratio_horizontal=True,
            )

        result = PostureCheckResult()
        T = len(timestamps_ms)

        if T < 2:
            return result

        # ── 1. Rapid vertical descent ──────────────────────────────────────
        hip_y_values = self._get_hip_heights(raw_keypoints_seq)
        velocities = np.diff(hip_y_values)  # Positive = downward movement

        if len(velocities) > 0:
            result.max_vertical_velocity = float(np.max(velocities))
            result.rapid_descent = result.max_vertical_velocity >= self._min_velocity

        # ── 2. Low final position ──────────────────────────────────────────
        # In image coordinates, y increases downward (0=top, 1=bottom).
        # A high y value means the person is low in the frame.
        # We check if hip is above max_hip_height (inverted: 1 - y < threshold)
        final_hip_y = hip_y_values[-1]
        result.final_hip_height = final_hip_y
        # Person is "low" if their hip Y is above (1 - threshold) in image space
        result.low_position = final_hip_y >= (1.0 - self._max_hip_height)

        # ── 3. Lying duration ──────────────────────────────────────────────
        # Check how long the person has been in the low position
        low_threshold = 1.0 - self._max_hip_height
        time_low_ms = 0

        # Scan backwards from the end to find how long hip has been low
        for i in range(T - 1, -1, -1):
            if hip_y_values[i] < low_threshold:
                # Found the most recent frame where person was NOT low
                time_low_ms = timestamps_ms[-1] - timestamps_ms[i]
                break
        else:
            # All frames are in the low position
            time_low_ms = timestamps_ms[-1] - timestamps_ms[0]

        result.time_below_threshold_sec = time_low_ms / 1000.0
        result.lying_duration_met = (
            result.time_below_threshold_sec >= self._min_lying_sec
        )

        # ── 4. Body aspect ratio (horizontal = lying) ─────────────────────
        final_frame = raw_keypoints_seq[-1]  # (17, 2)
        valid_kp = final_frame[final_frame.sum(axis=1) > 0]

        if len(valid_kp) >= 3:
            x_range = valid_kp[:, 0].max() - valid_kp[:, 0].min()
            y_range = valid_kp[:, 1].max() - valid_kp[:, 1].min()
            result.final_aspect_ratio = (
                x_range / max(y_range, 1e-6) if y_range > 0 else 0.0
            )
            # Width > height suggests lying down
            result.aspect_ratio_horizontal = result.final_aspect_ratio > 1.2
        else:
            result.final_aspect_ratio = 0.0

        return result

    def _get_hip_heights(self, keypoints_seq: np.ndarray) -> np.ndarray:
        """
        Extract mid-hip Y coordinate across all frames.

        Args:
            keypoints_seq: (T, 17, 2) array.

        Returns:
            (T,) array of mid-hip Y values.
        """
        left_hip_y = keypoints_seq[:, COCO_LEFT_HIP, 1]
        right_hip_y = keypoints_seq[:, COCO_RIGHT_HIP, 1]
        return (left_hip_y + right_hip_y) / 2.0

    def reset(self):
        """Reset internal tracking state."""
        self._low_since_ts = None

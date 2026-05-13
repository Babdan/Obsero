"""
Skeleton Normalizer — Coordinate normalization for camera/scale invariance.

Translates body keypoints so the mid-hip is at the origin, scales by
torso length, and optionally applies temporal smoothing to reduce
pose-estimation jitter.
"""

import numpy as np
from typing import Optional

from src.pose_extractor import (
    COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER,
    COCO_LEFT_HIP, COCO_RIGHT_HIP,
    NUM_KEYPOINTS,
)


class SkeletonNormalizer:
    """
    Normalizes raw COCO-17 keypoint coordinates for position, scale,
    and temporal smoothing invariance.

    Normalization steps:
        1. Translate so mid-hip = (0, 0)
        2. Scale by torso length (mid-hip to mid-shoulder distance)
        3. Apply exponential moving average smoothing (optional)

    Args:
        config: Dictionary with normalization parameters.
    """

    def __init__(self, config: dict):
        self._method = config.get("method", "hip_center_scale")
        self._smoothing = config.get("smoothing", "none")
        self._alpha = config.get("smoothing_alpha", 0.7)
        self._eps = config.get("epsilon", 1e-6)

        # Smoothing state
        self._prev_normalized: Optional[np.ndarray] = None
        self._prev_raw: Optional[np.ndarray] = None

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize a single frame of keypoint coordinates.

        Args:
            landmarks: (17, 2) array of (x, y) coordinates.

        Returns:
            Normalized (17, 2) array.
        """
        if landmarks.shape != (NUM_KEYPOINTS, 2):
            raise ValueError(
                f"Expected shape ({NUM_KEYPOINTS}, 2), got {landmarks.shape}"
            )

        normalized = self._apply_normalization(landmarks.copy())
        normalized = self._apply_smoothing(normalized)

        return normalized

    def _apply_normalization(self, kp: np.ndarray) -> np.ndarray:
        """Apply position and scale normalization."""
        if self._method == "hip_center_scale":
            return self._hip_center_scale(kp)
        elif self._method == "min_max":
            return self._min_max_normalize(kp)
        elif self._method == "z_score":
            return self._z_score_normalize(kp)
        else:
            return kp

    def _hip_center_scale(self, kp: np.ndarray) -> np.ndarray:
        """
        Hip-center normalization with torso-length scaling.

        1. mid_hip = (left_hip + right_hip) / 2
        2. mid_shoulder = (left_shoulder + right_shoulder) / 2
        3. torso_length = ||mid_shoulder - mid_hip||
        4. normalized = (kp - mid_hip) / torso_length
        """
        mid_hip = (kp[COCO_LEFT_HIP] + kp[COCO_RIGHT_HIP]) / 2.0
        mid_shoulder = (kp[COCO_LEFT_SHOULDER] + kp[COCO_RIGHT_SHOULDER]) / 2.0

        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        torso_length = max(torso_length, self._eps)

        # Translate to hip center
        kp = kp - mid_hip

        # Scale by torso length
        kp = kp / torso_length

        return kp

    def _min_max_normalize(self, kp: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range based on bounding box."""
        mins = kp.min(axis=0)
        maxs = kp.max(axis=0)
        ranges = maxs - mins
        ranges = np.where(ranges < self._eps, 1.0, ranges)
        return (kp - mins) / ranges

    def _z_score_normalize(self, kp: np.ndarray) -> np.ndarray:
        """Z-score normalization (zero mean, unit variance)."""
        mean = kp.mean(axis=0)
        std = kp.std(axis=0)
        std = np.where(std < self._eps, 1.0, std)
        return (kp - mean) / std

    def _apply_smoothing(self, normalized: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce jitter."""
        if self._smoothing == "none" or self._prev_normalized is None:
            self._prev_normalized = normalized.copy()
            return normalized

        if self._smoothing == "exponential":
            # EMA: smoothed = alpha * current + (1 - alpha) * previous
            smoothed = (
                self._alpha * normalized
                + (1 - self._alpha) * self._prev_normalized
            )
            self._prev_normalized = smoothed.copy()
            return smoothed

        elif self._smoothing == "moving_avg":
            # Simple average of current and previous
            smoothed = (normalized + self._prev_normalized) / 2.0
            self._prev_normalized = normalized.copy()
            return smoothed

        self._prev_normalized = normalized.copy()
        return normalized

    def reset(self):
        """Reset smoothing state (e.g., when tracking is lost)."""
        self._prev_normalized = None
        self._prev_raw = None

    def get_body_metrics(self, landmarks: np.ndarray) -> dict:
        """
        Compute useful body metrics from raw (un-normalized) keypoints.
        Useful for posture rule checks.

        Args:
            landmarks: (17, 2) raw (x, y) coordinates in image-normalized space.

        Returns:
            Dictionary with body measurement metrics.
        """
        mid_hip = (landmarks[COCO_LEFT_HIP] + landmarks[COCO_RIGHT_HIP]) / 2.0
        mid_shoulder = (
            landmarks[COCO_LEFT_SHOULDER] + landmarks[COCO_RIGHT_SHOULDER]
        ) / 2.0

        torso_length = float(np.linalg.norm(mid_shoulder - mid_hip))

        # Body bounding box from keypoints
        valid = landmarks[landmarks.sum(axis=1) > 0]
        if len(valid) > 0:
            bbox_min = valid.min(axis=0)
            bbox_max = valid.max(axis=0)
            bbox_width = float(bbox_max[0] - bbox_min[0])
            bbox_height = float(bbox_max[1] - bbox_min[1])
        else:
            bbox_width = 0.0
            bbox_height = 0.0

        return {
            "mid_hip_x": float(mid_hip[0]),
            "mid_hip_y": float(mid_hip[1]),
            "mid_shoulder_x": float(mid_shoulder[0]),
            "mid_shoulder_y": float(mid_shoulder[1]),
            "torso_length": torso_length,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "aspect_ratio": (
                bbox_width / max(bbox_height, self._eps)
                if bbox_height > 0
                else 0.0
            ),
        }

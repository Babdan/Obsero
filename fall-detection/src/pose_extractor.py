"""
Pose Extractor — MediaPipe Pose Landmarker wrapper for fall detection.

Uses the MediaPipe Tasks API (PoseLandmarker) to extract body keypoints
from video frames. Maps the 33 MediaPipe landmarks to the 17 COCO-convention
body keypoints used in fall detection.

Requires a .task model file (e.g., pose_landmarker_heavy.task).
"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ── MediaPipe landmark index → COCO 17 keypoint mapping ──────────────────────
# MediaPipe provides 33 landmarks; we select the 17 that match COCO convention.
MEDIAPIPE_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left_eye
    5: 2,    # right_eye
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Indices within COCO-17 for body structure references
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17

# Default model paths (tried in order)
_DEFAULT_MODEL_PATHS = [
    "models/pose_landmarker_heavy.task",
    "models/pose_landmarker_full.task",
    "models/pose_landmarker_lite.task",
]


@dataclass
class PoseResult:
    """Structured output from a single pose detection."""
    landmarks: np.ndarray           # Shape: (17, 4) — x, y, z, visibility
    timestamp_ms: int
    person_index: int = 0
    detection_confidence: float = 0.0
    raw_landmarks: list = field(default_factory=list)

    @property
    def xy(self) -> np.ndarray:
        """Return only (x, y) coordinates. Shape: (17, 2)."""
        return self.landmarks[:, :2]

    @property
    def visibility(self) -> np.ndarray:
        """Return visibility scores. Shape: (17,)."""
        return self.landmarks[:, 3]

    @property
    def mean_visibility(self) -> float:
        """Average visibility across all keypoints."""
        return float(np.mean(self.landmarks[:, 3]))


def _find_model_path(config: dict) -> str:
    """Find an available pose landmarker model file."""
    # Check config first
    model_path = config.get("model_path", "")
    if model_path and Path(model_path).exists():
        return model_path

    # Try default locations
    for path in _DEFAULT_MODEL_PATHS:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Pose landmarker model file not found. "
        "Download it with:\n"
        "  python -c \"import urllib.request; "
        "urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task', "
        "'models/pose_landmarker_heavy.task')\""
    )


class PoseExtractor:
    """
    Extracts body pose landmarks from video frames using MediaPipe
    PoseLandmarker (Tasks API).

    Args:
        config: Dictionary with pose configuration parameters.
                Expected keys: model_path, min_detection_confidence,
                min_tracking_confidence, static_image_mode
    """

    def __init__(self, config: dict):
        self._config = config

        model_path = _find_model_path(config)
        static_mode = config.get("static_image_mode", False)

        # Determine running mode
        if static_mode:
            running_mode = mp.tasks.vision.RunningMode.IMAGE
        else:
            running_mode = mp.tasks.vision.RunningMode.VIDEO

        # Build options
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=config.get(
                "min_detection_confidence", 0.5
            ),
            min_tracking_confidence=config.get(
                "min_tracking_confidence", 0.5
            ),
            min_pose_presence_confidence=config.get(
                "min_detection_confidence", 0.5
            ),
            num_poses=1,
        )

        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )
        self._running_mode = running_mode
        self._frame_count = 0

    def extract(
        self,
        frame: np.ndarray,
        timestamp_ms: int = 0,
    ) -> Optional[PoseResult]:
        """
        Extract pose landmarks from a single BGR frame.

        Args:
            frame: BGR image from OpenCV (H, W, 3).
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            PoseResult if a person is detected, None otherwise.
        """
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run detection
        if self._running_mode == mp.tasks.vision.RunningMode.IMAGE:
            results = self._landmarker.detect(mp_image)
        else:
            # VIDEO mode requires monotonically increasing timestamps
            self._frame_count += 1
            results = self._landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

        # Check if any poses detected
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Extract first person's landmarks
        raw_landmarks = results.pose_landmarks[0]
        landmarks = np.zeros((NUM_KEYPOINTS, 4), dtype=np.float32)

        for mp_idx, coco_idx in MEDIAPIPE_TO_COCO.items():
            lm = raw_landmarks[mp_idx]
            landmarks[coco_idx] = [lm.x, lm.y, lm.z, lm.visibility]

        return PoseResult(
            landmarks=landmarks,
            timestamp_ms=timestamp_ms,
            person_index=0,
            detection_confidence=float(np.mean(landmarks[:, 3])),
            raw_landmarks=list(raw_landmarks),
        )

    def extract_batch(
        self,
        frames: list[np.ndarray],
        timestamps_ms: list[int],
    ) -> list[Optional[PoseResult]]:
        """Extract poses from multiple frames (sequential, not batched)."""
        return [
            self.extract(frame, ts)
            for frame, ts in zip(frames, timestamps_ms)
        ]

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

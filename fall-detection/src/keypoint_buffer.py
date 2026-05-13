"""
Keypoint Buffer -- Thread-safe sliding window for temporal sequences.

Maintains a fixed-size ring buffer of normalized keypoint frames,
performing FPS subsampling and providing ready-to-infer numpy arrays
for the temporal model.

Stores raw position features (34) internally, but enriches output
with velocity and acceleration (102 total) on retrieval.
"""

import threading
import numpy as np
from collections import deque
from typing import Optional

from src.pose_extractor import NUM_KEYPOINTS
from src.motion_features import (
    compute_motion_features,
    compute_motion_features_incremental,
    TOTAL_FEATURES as ENRICHED_FEATURES_PER_FRAME,
)

# Raw features per frame: 17 keypoints x 2 (x, y) = 34
FEATURES_PER_FRAME = NUM_KEYPOINTS * 2


class KeypointBuffer:
    """
    Thread-safe sliding window buffer for keypoint sequences.

    Accepts normalized keypoint frames at arbitrary timestamps, subsamples
    to the target FPS, and provides the current window as a contiguous
    numpy array suitable for temporal model input.

    Args:
        window_size: Maximum number of frames in the buffer.
        target_fps: Target frames per second for subsampling.
        min_frames: Minimum frames needed before inference is allowed.
        stride: Run inference every N newly added frames.
    """

    def __init__(
        self,
        window_size: int = 30,
        target_fps: int = 10,
        min_frames: int = 15,
        stride: int = 5,
    ):
        self._window_size = window_size
        self._target_fps = target_fps
        self._min_frames = min_frames
        self._stride = stride

        self._min_interval_ms = 1000.0 / target_fps

        # Ring buffer: each entry is (timestamp_ms, keypoints_flat)
        self._buffer: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        self._last_accepted_ts: Optional[int] = None
        self._frames_since_inference = 0
        self._frames_without_person = 0

    def push(
        self,
        timestamp_ms: int,
        keypoints: np.ndarray,
    ) -> bool:
        """
        Add a normalized keypoint frame to the buffer.

        Applies FPS subsampling: only accepts the frame if enough time
        has elapsed since the last accepted frame.

        Args:
            timestamp_ms: Frame timestamp in milliseconds.
            keypoints: Normalized keypoints, shape (17, 2).

        Returns:
            True if the frame was accepted, False if skipped (too soon).
        """
        with self._lock:
            # FPS subsampling
            if self._last_accepted_ts is not None:
                elapsed = timestamp_ms - self._last_accepted_ts
                if elapsed < self._min_interval_ms:
                    return False

            # Flatten (17, 2) → (34,)
            flat = keypoints.flatten().astype(np.float32)
            if flat.shape[0] != FEATURES_PER_FRAME:
                raise ValueError(
                    f"Expected {FEATURES_PER_FRAME} features, "
                    f"got {flat.shape[0]}"
                )

            self._buffer.append((timestamp_ms, flat))
            self._last_accepted_ts = timestamp_ms
            self._frames_since_inference += 1
            self._frames_without_person = 0

            return True

    def push_empty(self, timestamp_ms: int):
        """Record that no person was detected in this frame."""
        with self._lock:
            self._frames_without_person += 1

    def get_sequence(
        self, enrich: bool = True,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Get the current buffer contents as a numpy array.

        Args:
            enrich: If True, append velocity + acceleration features.

        Returns:
            Tuple of:
                - keypoints: (T, F) float32 array, T <= window_size
                  F=102 if enrich=True, F=34 if enrich=False
                - timestamps: list of T timestamps in ms
        """
        with self._lock:
            if not self._buffer:
                feat_size = ENRICHED_FEATURES_PER_FRAME if enrich else FEATURES_PER_FRAME
                return np.zeros((0, feat_size), dtype=np.float32), []

            timestamps = [entry[0] for entry in self._buffer]
            keypoints = np.stack([entry[1] for entry in self._buffer])

            if enrich:
                keypoints = compute_motion_features(keypoints)

            return keypoints, timestamps

    def get_padded_sequence(
        self, enrich: bool = True,
    ) -> tuple[np.ndarray, list[int], int]:
        """
        Get buffer contents padded/truncated to exactly window_size frames.

        If fewer than window_size frames exist, zero-pad from the left.
        If enrich=True, appends velocity and acceleration features.

        Args:
            enrich: If True, append velocity + acceleration features.

        Returns:
            Tuple of:
                - keypoints: (window_size, F) float32 array
                  F=102 if enrich=True, F=34 if enrich=False
                - timestamps: list of actual timestamps
                - actual_length: number of real (non-padded) frames
        """
        with self._lock:
            feat_size = ENRICHED_FEATURES_PER_FRAME if enrich else FEATURES_PER_FRAME
            actual_len = len(self._buffer)

            if actual_len == 0:
                return (
                    np.zeros(
                        (self._window_size, feat_size),
                        dtype=np.float32,
                    ),
                    [],
                    0,
                )

            timestamps = [entry[0] for entry in self._buffer]
            keypoints = np.stack([entry[1] for entry in self._buffer])

            pad_left = 0
            if actual_len < self._window_size:
                # Left-pad raw positions with zeros
                pad_left = self._window_size - actual_len
                padding = np.zeros(
                    (pad_left, FEATURES_PER_FRAME), dtype=np.float32
                )
                keypoints = np.vstack([padding, keypoints])

            if enrich:
                # Compute motion features with padding awareness
                keypoints = compute_motion_features_incremental(
                    keypoints, pad_left=pad_left
                )

            return keypoints, timestamps, actual_len

    def is_ready(self) -> bool:
        """Check if enough frames are available for inference."""
        with self._lock:
            return len(self._buffer) >= self._min_frames

    def should_infer(self) -> bool:
        """Check if we should run inference (enough new frames since last)."""
        with self._lock:
            return (
                len(self._buffer) >= self._min_frames
                and self._frames_since_inference >= self._stride
            )

    def mark_inferred(self):
        """Reset the inference stride counter after inference runs."""
        with self._lock:
            self._frames_since_inference = 0

    def clear(self):
        """Clear the buffer and reset all state."""
        with self._lock:
            self._buffer.clear()
            self._last_accepted_ts = None
            self._frames_since_inference = 0
            self._frames_without_person = 0

    @property
    def length(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def frames_without_person(self) -> int:
        with self._lock:
            return self._frames_without_person

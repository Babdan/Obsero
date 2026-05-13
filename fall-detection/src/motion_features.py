"""
Motion Features -- Compute velocity and acceleration from keypoint sequences.

Transforms a raw position sequence (T, 34) into an enriched feature
sequence (T, 102) by appending first-order differences (velocity)
and second-order differences (acceleration).

Feature layout per frame:
    [0:34]   position    - normalized (x,y) of 17 joints
    [34:68]  velocity    - frame-to-frame position change (dx, dy)
    [68:102] acceleration - frame-to-frame velocity change (ddx, ddy)
"""

import numpy as np


# Feature dimensions
POSITION_FEATURES = 34    # 17 keypoints x 2
VELOCITY_FEATURES = 34    # same shape as position
ACCEL_FEATURES = 34       # same shape as velocity
TOTAL_FEATURES = POSITION_FEATURES + VELOCITY_FEATURES + ACCEL_FEATURES  # 102


def compute_motion_features(positions: np.ndarray) -> np.ndarray:
    """
    Compute position + velocity + acceleration features from raw positions.

    Args:
        positions: (T, 34) array of normalized keypoint coordinates.

    Returns:
        (T, 102) array with [position | velocity | acceleration] per frame.

    Notes:
        - Velocity at frame t = position[t] - position[t-1]
        - Acceleration at frame t = velocity[t] - velocity[t-1]
        - Frame 0 velocity = zeros (no previous frame)
        - Frames 0-1 acceleration = zeros (insufficient history)
    """
    T = positions.shape[0]
    if T == 0:
        return np.zeros((0, TOTAL_FEATURES), dtype=np.float32)

    # Velocity: first-order finite differences
    # velocity[0] = 0, velocity[t] = pos[t] - pos[t-1]
    velocity = np.zeros_like(positions)
    if T > 1:
        velocity[1:] = positions[1:] - positions[:-1]

    # Acceleration: second-order finite differences
    # accel[0] = accel[1] = 0, accel[t] = vel[t] - vel[t-1]
    acceleration = np.zeros_like(positions)
    if T > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]

    # Concatenate: [position | velocity | acceleration]
    features = np.concatenate(
        [positions, velocity, acceleration], axis=1
    ).astype(np.float32)

    return features


def compute_motion_features_incremental(
    positions: np.ndarray,
    pad_left: int = 0,
) -> np.ndarray:
    """
    Same as compute_motion_features, but handles left-padded sequences.

    For padded sequences (from the keypoint buffer), the padding region
    is all zeros. We compute velocity/acceleration only on the real
    frames, then re-pad.

    Args:
        positions: (T, 34) array, potentially with zero-padding on the left.
        pad_left: Number of leading frames that are padding (zeros).

    Returns:
        (T, 102) array with motion features, padding preserved as zeros.
    """
    T = positions.shape[0]
    if T == 0:
        return np.zeros((0, TOTAL_FEATURES), dtype=np.float32)

    if pad_left <= 0 or pad_left >= T:
        return compute_motion_features(positions)

    # Split into padding and real data
    real_positions = positions[pad_left:]
    real_features = compute_motion_features(real_positions)

    # Re-pad with zeros
    pad_features = np.zeros((pad_left, TOTAL_FEATURES), dtype=np.float32)
    return np.vstack([pad_features, real_features])

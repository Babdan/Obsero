"""
Run Detector — CLI entry point for the fall detection system.

Supports live camera feed and video file input with optional
skeleton visualization overlay.

Usage:
    # Live camera
    python run_detector.py --config config/fall_detection_config.yaml --camera 0

    # Video file
    python run_detector.py --config config/fall_detection_config.yaml --video path/to/test.mp4

    # With preview window
    python run_detector.py --config config/fall_detection_config.yaml --camera 0 --preview
"""

import sys
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fall_detector import FallDetector


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection System — Real-time pose-based fall detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/fall_detection_config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (0, 1, ...) for live feed",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file for offline testing",
    )
    parser.add_argument(
        "--rtsp",
        type=str,
        default=None,
        help="RTSP URL for IP camera stream",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show OpenCV preview window with skeleton overlay",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="cam_0",
        help="Identifier for this camera (for logs and alarms)",
    )
    args = parser.parse_args()

    # Determine video source
    source = None
    if args.video:
        source = args.video
        if not Path(source).exists():
            print(f"[ERROR] Video file not found: {source}")
            sys.exit(1)
    elif args.rtsp:
        source = args.rtsp
    elif args.camera is not None:
        source = args.camera
    # else: use config default

    # Validate config
    config_path = args.config
    if not Path(config_path).exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    print(f"\n[*] Fall Detection System")
    print(f"   Config:  {config_path}")
    print(f"   Source:  {source or 'config default'}")
    print(f"   Preview: {'yes' if args.preview else 'no'}")
    print(f"   Press Ctrl+C to stop\n")

    # Create and start detector
    detector = FallDetector(config_path)

    try:
        asyncio.run(
            detector.start(
                camera_id=args.camera_id,
                source=source,
                show_preview=args.preview,
            )
        )
    except KeyboardInterrupt:
        print("\n\n[STOP] Shutting down...")


if __name__ == "__main__":
    main()

"""Application entrypoint for minimal YOLOv8 person and luggage detection."""

from __future__ import annotations

import argparse

from app.adapters.detector import YoloV8Detector
from app.adapters.visualizer import DetectionVisualizer
from app.config.settings import build_settings
from app.pipeline.monitor import VideoMonitor

# Note: This file is intentionally kept minimal and focused on application bootstrapping.

def parse_args() -> argparse.Namespace:
    """Parse minimal runtime overrides."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 to detect people and luggage in a video source.",
    )
    parser.add_argument("--source", help="Video file path or camera index.")
    parser.add_argument("--model", help="Path to a YOLOv8 model file.")
    parser.add_argument("--conf", type=float, help="Confidence threshold.")
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save the annotated video to the configured output path.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the OpenCV preview window.",
    )
    return parser.parse_args()


def main() -> None:
    """Bootstrap and run the minimal application."""
    # Load the default runtime settings, then let CLI flags override them.
    args = parse_args()
    settings = build_settings()

    if args.source:
        settings.source = args.source
    if args.model:
        settings.model_path = args.model
    if args.conf is not None:
        settings.confidence_threshold = args.conf
    if args.save_output:
        settings.save_output = True
    if args.no_window:
        settings.show_window = False

    # Build the three core runtime components and hand control to the monitor loop.
    detector = YoloV8Detector(settings)
    visualizer = DetectionVisualizer()
    monitor = VideoMonitor(settings=settings, detector=detector, visualizer=visualizer)
    monitor.run()


if __name__ == "__main__":
    main()
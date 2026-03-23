"""Video monitoring loop for the minimal detection app."""

from __future__ import annotations

import cv2

from app.adapters.detector import YoloV8Detector
from app.adapters.visualizer import DetectionVisualizer
from app.config.settings import AppSettings



class VideoMonitor:
    """Run the end-to-end loop: read frames, detect, draw, and display."""

    def __init__(
        self,
        settings: AppSettings,
        detector: YoloV8Detector,
        visualizer: DetectionVisualizer,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._visualizer = visualizer

    def run(self) -> None:
        # Open the input source and prepare the optional output/display resources.
        capture = cv2.VideoCapture(self._parse_source(self._settings.source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {self._settings.source}")

        writer = self._create_writer(capture)
        if self._settings.show_window:
            cv2.namedWindow(self._settings.window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                # Process one frame at a time: read, detect, draw, then show/save it.
                success, frame = capture.read()
                if not success:
                    break

                detections = self._detector.detect(frame)
                annotated_frame = self._visualizer.draw(frame, detections)

                if self._settings.show_window:
                    cv2.imshow(
                        self._settings.window_name,
                        self._fit_frame_to_window(annotated_frame),
                    )

                if writer is not None:
                    writer.write(annotated_frame)

                if self._should_stop():
                    break
        finally:
            # Release native OpenCV resources even if the loop exits on an error.
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    def _create_writer(self, capture) -> cv2.VideoWriter | None:
        # Create the video writer only when annotated output is explicitly enabled.
        if not self._settings.save_output:
            return None

        self._settings.output_dir.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self._settings.output_path,
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {self._settings.output_path}")
        return writer

    def _parse_source(self, source: str) -> int | str:
        return int(source) if source.isdigit() else source

    def _should_stop(self) -> bool:
        # Keyboard exit is only relevant when a preview window is being shown.
        if not self._settings.show_window:
            return False
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def _fit_frame_to_window(self, frame):
        """Resize a frame to the current window while preserving aspect ratio."""
        _, _, window_width, window_height = cv2.getWindowImageRect(self._settings.window_name)
        frame_height, frame_width = frame.shape[:2]

        if window_width <= 0 or window_height <= 0:
            return frame

        scale = min(window_width / frame_width, window_height / frame_height)
        resized_width = max(1, int(frame_width * scale))
        resized_height = max(1, int(frame_height * scale))
        resized_frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        # Pad the resized frame back to the window size so the image is not distorted.
        pad_left = (window_width - resized_width) // 2
        pad_right = window_width - resized_width - pad_left
        pad_top = (window_height - resized_height) // 2
        pad_bottom = window_height - resized_height - pad_top

        return cv2.copyMakeBorder(
            resized_frame,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
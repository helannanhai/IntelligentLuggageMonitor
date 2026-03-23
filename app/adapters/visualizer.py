"""Rendering utilities for drawing detections on frames."""

from __future__ import annotations

import cv2

from app.config.settings import LUGGAGE_CLASS_IDS, PERSON_CLASS_ID
from app.core.types import Detection


PERSON_COLOR = (255, 0, 0)
LUGGAGE_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


class DetectionVisualizer:
    """Draw minimal overlays for person and luggage detections."""

    def draw(self, frame, detections: list[Detection]):
        # Render one visual block per detection: box first, then the text label.
        for detection in detections:
            color = self._get_color(detection.class_id)
            cv2.rectangle(
                frame,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                color,
                2,
            )
            cv2.putText(
                frame,
                detection.label,
                (detection.x1, max(20, detection.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
        return frame

    def _get_color(self, class_id: int) -> tuple[int, int, int]:
        # Keep class-to-color mapping in one place so all overlays stay consistent.
        if class_id == PERSON_CLASS_ID:
            return PERSON_COLOR
        if class_id in LUGGAGE_CLASS_IDS:
            return LUGGAGE_COLOR
        return (0, 255, 0)
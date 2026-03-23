"""YOLOv8 detector adapter."""

from __future__ import annotations

from typing import Iterable

from ultralytics import YOLO

from app.config.settings import LUGGAGE_CLASS_IDS, PERSON_CLASS_ID, AppSettings
from app.core.types import Detection


class YoloV8Detector:
    """Wrap the Ultralytics model behind a narrow detection interface."""

    def __init__(self, settings: AppSettings) -> None:
        # Load the model once and cache the class metadata needed for filtering.
        self._settings = settings
        self._model = YOLO(settings.model_path)
        self._class_names = self._model.names
        self._target_class_ids = sorted({PERSON_CLASS_ID, *LUGGAGE_CLASS_IDS})

    def detect(self, frame) -> list[Detection]:
        """Run inference and return filtered detections for people and luggage."""
        # Run a single-frame prediction and let YOLO keep only the target classes.
        result = self._model.predict(
            source=frame,
            conf=self._settings.confidence_threshold,
            imgsz=self._settings.image_size,
            classes=self._target_class_ids,
            verbose=False,
        )[0]

        # Convert the model output into the application's own detection objects.
        if result.boxes is None:
            return []

        return list(self._iter_detections(result.boxes))

    def _iter_detections(self, boxes) -> Iterable[Detection]:
        # Normalize Ultralytics box tensors into plain Python dataclasses.
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            class_name = str(self._class_names[class_id])
            yield Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
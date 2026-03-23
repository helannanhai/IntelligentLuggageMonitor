"""Shared typed models used by the application."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    """A single object detection emitted by the model."""

    # Plain detection data passed between the detector, visualizer, and pipeline.
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def label(self) -> str:
        """Return a compact user-facing label."""
        return f"{self.class_name} {self.confidence:.2f}"
"""Runtime settings for the minimal detector application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PERSON_CLASS_ID = 0
LUGGAGE_CLASS_IDS = frozenset({24, 26, 28})


@dataclass(slots=True)
class AppSettings:
    """Configuration needed to run the minimal detection pipeline."""

    model_path: str = "yolo26x.pt"
    # source: str = r"Dataset\JiMeng\1_4.mov"
    source: str = "0"
    # source: str = "1"
    confidence_threshold: float = 0.35
    image_size: int = 850
    show_window: bool = True
    window_name: str = "Luggage Monitor"
    save_output: bool = False
    output_path: str = r"outputs\minimal_detection.mp4"

    @property
    def output_dir(self) -> Path:
        """Return the output directory for optional video export."""
        return Path(self.output_path).parent


def build_settings() -> AppSettings:
    """Build the default application settings."""
    return AppSettings()

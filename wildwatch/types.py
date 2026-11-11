from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Categories used only for HUD color and grouping. They are assigned from the
# model's actual label, never invented as a species identity.
PREDATOR_LABELS = {
    "coyote", "wolf", "bear", "fox", "bobcat", "cougar", "mountain lion",
    "cat",
}
GAME_LABELS = {
    "deer", "whitetail", "white-tailed deer", "mule deer", "hog", "pig",
    "boar", "feral hog", "wild pig", "javelina", "cow", "horse", "sheep",
    "elk", "moose", "goat",
}
PERSON_LABELS = {"person", "people", "human"}
VEHICLE_LABELS = {
    "vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "boat",
    "airplane", "train",
}
ANIMAL_LABELS = {
    "animal", "animal (unid.)", "dog", "bird", "zebra", "giraffe", "elephant",
}


def category_for(label: str) -> str:
    key = label.strip().lower()
    # Prefer exact / contained wildlife names over the generic "animal" bucket.
    for name in PREDATOR_LABELS:
        if name in key:
            return "predator"
    for name in GAME_LABELS:
        if name in key:
            return "game"
    for name in PERSON_LABELS:
        if name in key:
            return "person"
    for name in VEHICLE_LABELS:
        if name in key:
            return "vehicle"
    for name in ANIMAL_LABELS:
        if name in key:
            return "animal"
    return "other"


@dataclass
class Detection:
    """One bounding box from a detector, optionally refined by a classifier."""

    xyxy: tuple[float, float, float, float]
    label: str
    confidence: float
    category: str = "other"
    raw_label: str = ""
    species_confidence: Optional[float] = None
    track_id: Optional[int] = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    velocity: Optional[tuple[float, float]] = None
    dwell_frames: int = 1
    threat: float = 0.0
    threat_level: str = ""
    zones: list[str] = field(default_factory=list)
    status: str = ""
    approaching: bool = False

    def __post_init__(self) -> None:
        if not self.raw_label:
            self.raw_label = self.label
        if self.category == "other":
            self.category = category_for(self.label)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Classification:
    label: str
    confidence: float
    extras: dict = field(default_factory=dict)


@dataclass
class Telemetry:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    timestamp: Optional[str] = None

    @property
    def has_fix(self) -> bool:
        return self.latitude is not None and self.longitude is not None

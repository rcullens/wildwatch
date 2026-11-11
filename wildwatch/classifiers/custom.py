from __future__ import annotations

from pathlib import Path

import numpy as np

from wildwatch.classifiers.base import Classifier
from wildwatch.types import Classification


class CustomImageClassifier(Classifier):
    """Ultralytics classification model (.pt) you trained yourself.

    Train an image classifier whose classes are the species you care about
    (for example coyote, deer, hog, other) and set:

        classifier: custom
        custom_classifier:
          weights: path/to/species_cls.pt

    See the README for a short training recipe.
    """

    name = "custom"

    def __init__(self, weights: str, labels: list[str] | None = None, device: str = "") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for classifier: custom") from exc
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(f"Custom classifier weights not found: {path}")
        print(f"[wildwatch] loading custom classifier {path}")
        self.model = YOLO(str(path))
        self.labels = list(labels or [])
        self.device = device or None
        self.enabled = True

    def classify(self, crop_bgr: np.ndarray) -> Classification | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        kwargs = dict(source=crop_bgr, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return None
        result = results[0]
        if getattr(result, "probs", None) is None:
            return None
        probs = result.probs
        top = int(probs.top1)
        conf = float(probs.top1conf)
        names = result.names or {}
        if self.labels:
            label = self.labels[top] if top < len(self.labels) else f"class_{top}"
        else:
            label = str(names.get(top, f"class_{top}"))
        return Classification(label=label, confidence=conf)

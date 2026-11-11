from __future__ import annotations

import numpy as np

from wildwatch.classifiers.base import Classifier
from wildwatch.types import Classification


class PassthroughClassifier(Classifier):
    """Default classifier. Does not invent a species.

    Generic MegaDetector "animal" boxes become "animal (unid.)".
    Hook a real model by setting classifier: speciesnet or classifier: custom.
    """

    name = "none"
    enabled = True

    def classify(self, crop_bgr: np.ndarray) -> Classification | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        return Classification(label="animal (unid.)", confidence=0.0)

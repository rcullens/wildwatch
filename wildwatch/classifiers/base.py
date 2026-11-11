from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wildwatch.types import Classification


class Classifier(ABC):
    """Species classifier applied to a cropped detection.

    Implementations must only return labels the underlying model can produce.
    """

    name: str = "classifier"
    enabled: bool = True

    @abstractmethod
    def classify(self, crop_bgr: np.ndarray) -> Classification | None:
        raise NotImplementedError

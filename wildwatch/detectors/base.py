from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wildwatch.types import Detection


class Detector(ABC):
    """Pluggable detection backend. Implementations must only emit labels
    the underlying model can actually produce."""

    name: str = "detector"

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None

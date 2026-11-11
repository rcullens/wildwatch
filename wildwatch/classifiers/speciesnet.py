from __future__ import annotations

import numpy as np

from wildwatch.classifiers.base import Classifier
from wildwatch.types import Classification


class SpeciesNetClassifier(Classifier):
    """Optional Google SpeciesNet wrapper.

    SpeciesNet is trained on camera-trap images of worldwide wildlife,
    including North American species such as coyote, white-tailed deer, and
    wild pig. It is a heavy dependency and is not installed by default.

        pip install speciesnet

    The public API has shifted across releases; this wrapper tries the
    documented predict-on-array paths and fails with a clear error if the
    installed version looks different. Aerial (nadir/oblique drone) photos
    are out of distribution relative to camera traps, so treat scores as a
    hint, not a guarantee.
    """

    name = "speciesnet"

    def __init__(self, device: str = "") -> None:
        try:
            import speciesnet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SpeciesNet is not installed. Install it with "
                "`pip install speciesnet` and see the README for how the "
                "classifier hooks in. Until then, use classifier: none."
            ) from exc
        self.model = self._load(device)
        self.enabled = True

    def _load(self, device: str):
        # Try a few known entry points so a minor package rename doesn't
        # silently break the hook.
        errors: list[str] = []
        try:
            from speciesnet import SpeciesNet

            kwargs = {}
            if device:
                kwargs["device"] = device
            return SpeciesNet(**kwargs) if kwargs else SpeciesNet()
        except Exception as exc:  # pragma: no cover - depends on the wheel
            errors.append(f"speciesnet.SpeciesNet: {exc}")
        try:
            from speciesnet.model import SpeciesNet

            return SpeciesNet()
        except Exception as exc:  # pragma: no cover
            errors.append(f"speciesnet.model.SpeciesNet: {exc}")
        raise RuntimeError(
            "Found the speciesnet package but could not construct a model. "
            "Update wildwatch/classifiers/speciesnet.py to match the API of "
            "the version you installed. Tried:\n  - " + "\n  - ".join(errors)
        )

    def classify(self, crop_bgr: np.ndarray) -> Classification | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        rgb = crop_bgr[:, :, ::-1]
        predict = (
            getattr(self.model, "predict", None)
            or getattr(self.model, "classify", None)
            or getattr(self.model, "single_image_classification", None)
        )
        if predict is None:
            raise RuntimeError(
                "SpeciesNet model has no predict/classify method. "
                "Adapt wildwatch/classifiers/speciesnet.py to your version."
            )
        result = predict(rgb)
        return _normalize_speciesnet_result(result)


def _normalize_speciesnet_result(result) -> Classification | None:
    if result is None:
        return None
    if isinstance(result, Classification):
        return result
    if isinstance(result, str):
        return Classification(label=result, confidence=0.0)
    if isinstance(result, dict):
        label = (
            result.get("prediction")
            or result.get("label")
            or result.get("class")
            or result.get("species")
        )
        conf = result.get("confidence", result.get("score", 0.0))
        if not label:
            # Some versions return {'predictions': [{'class': ..., 'score': ...}]}
            preds = result.get("predictions") or result.get("classifications") or []
            if preds:
                top = preds[0]
                if isinstance(top, dict):
                    label = top.get("class") or top.get("prediction") or top.get("label")
                    conf = top.get("score", top.get("confidence", 0.0))
        if not label:
            return None
        label = str(label)
        # SpeciesNet often uses "uuid;taxon;common_name" — keep the common name.
        if ";" in label:
            label = label.split(";")[-1]
        return Classification(label=label.replace("_", " "), confidence=float(conf or 0.0), extras=result)
    return None

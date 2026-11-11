from wildwatch.classifiers.base import Classifier
from wildwatch.classifiers.passthrough import PassthroughClassifier
from wildwatch.types import Detection

GENERIC_ANIMAL_LABELS = {"animal", "animal (unid.)"}


def build_classifier(cfg: dict) -> Classifier:
    name = str(cfg.get("classifier", "none")).strip().lower()
    if name in {"none", "passthrough", "off", ""}:
        return PassthroughClassifier()
    if name in {"speciesnet", "species"}:
        from wildwatch.classifiers.speciesnet import SpeciesNetClassifier

        return SpeciesNetClassifier(device=cfg.get("device") or "")
    if name in {"custom", "ultralytics"}:
        from wildwatch.classifiers.custom import CustomImageClassifier

        cc = cfg.get("custom_classifier") or {}
        weights = cc.get("weights") or ""
        if not weights:
            raise ValueError(
                "classifier=custom requires custom_classifier.weights in config.yaml"
            )
        return CustomImageClassifier(
            weights=weights,
            labels=list(cc.get("labels") or []),
            device=cfg.get("device") or "",
        )
    raise ValueError(
        f"Unknown classifier {name!r}. Use 'none', 'speciesnet', or 'custom'."
    )


def refine_detections(
    classifier: Classifier, frame_bgr, detections: list[Detection]
) -> list[Detection]:
    """Run the species classifier only on generic 'animal' boxes.

    Specific labels the detector already produced (person, vehicle, dog, bear,
    ...) are left untouched. We never overwrite a real class with a guess.
    """
    if not detections:
        return detections
    h, w = frame_bgr.shape[:2]
    refined: list[Detection] = []
    for det in detections:
        if det.raw_label.lower() not in GENERIC_ANIMAL_LABELS:
            refined.append(det)
            continue
        x1, y1, x2, y2 = det.xyxy
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(w, int(x2)), min(h, int(y2))
        if x2i <= x1i or y2i <= y1i or not classifier.enabled:
            det.label = "animal (unid.)"
            refined.append(det)
            continue
        crop = frame_bgr[y1i:y2i, x1i:x2i]
        result = classifier.classify(crop)
        if result is None or not result.label:
            det.label = "animal (unid.)"
            refined.append(det)
            continue
        det.label = result.label
        det.species_confidence = result.confidence
        det.category = "other"
        det.__post_init__()
        refined.append(det)
    return refined


__all__ = ["Classifier", "PassthroughClassifier", "build_classifier", "refine_detections"]

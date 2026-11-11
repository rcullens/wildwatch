from __future__ import annotations

import numpy as np

from wildwatch.detectors.base import Detector
from wildwatch.types import Detection

# MegaDetector class IDs are stable across V5/V6: 0=animal, 1=person, 2=vehicle.
CLASS_FALLBACK = {0: "animal", 1: "person", 2: "vehicle"}


class MegaDetector(Detector):
    """Microsoft MegaDetector v6 via PytorchWildlife.

    Emits only animal / person / vehicle. That is the right first stage for
    aerial or camera-trap wildlife: find the animal box, then classify the crop.
    It will not name coyote, deer, or hog on its own.
    """

    name = "megadetector"

    def __init__(
        self,
        version: str = "MDV6-yolov10-c",
        confidence: float = 0.35,
        device: str = "",
    ) -> None:
        try:
            import torch
            from PytorchWildlife.models import detection as pw_detection
        except ImportError as exc:
            raise ImportError(
                "PytorchWildlife is required for MegaDetector. "
                "Install the wildlife extras: pip install -r requirements-wildlife.txt"
            ) from exc
        if device:
            resolved = device
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.version = version
        self.confidence = confidence
        self.device = resolved
        print(
            f"[wildwatch] loading MegaDetector {version} on {resolved} "
            "(weights auto-download on first use)"
        )
        self.model = pw_detection.MegaDetectorV6(
            device=resolved, pretrained=True, version=version
        )
        self.class_names = getattr(self.model, "CLASS_NAMES", CLASS_FALLBACK)

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        # PytorchWildlife expects an RGB HWC array.
        rgb = frame_bgr[:, :, ::-1]
        # Newer PytorchWildlife uses det_conf_thres; older builds used conf_thres.
        try:
            result = self.model.single_image_detection(
                rgb, det_conf_thres=self.confidence
            )
        except TypeError:
            result = self.model.single_image_detection(
                rgb, conf_thres=self.confidence
            )
        detections: list[Detection] = []
        det = result.get("detections") if isinstance(result, dict) else None
        if det is None:
            return detections
        xyxy = getattr(det, "xyxy", None)
        confs = getattr(det, "confidence", None)
        class_ids = getattr(det, "class_id", None)
        if xyxy is None:
            return detections
        labels = result.get("labels") or []
        for i, box in enumerate(xyxy):
            conf = float(confs[i]) if confs is not None and i < len(confs) else 0.0
            cls_id = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else -1
            if i < len(labels) and labels[i]:
                # Labels sometimes come as "animal 0.87" — keep just the name.
                raw = str(labels[i]).split()[0]
            else:
                raw = str(self.class_names.get(cls_id, CLASS_FALLBACK.get(cls_id, "animal")))
            # Normalize MegaDetector's "animal" so the classifier can refine it.
            label = raw.lower()
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    label=label,
                    confidence=conf,
                    raw_label=label,
                )
            )
        return detections

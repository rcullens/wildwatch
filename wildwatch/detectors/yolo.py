from __future__ import annotations

import numpy as np

from wildwatch.detectors.base import Detector
from wildwatch.types import Detection


class YOLODetector(Detector):
    """Ultralytics YOLO. Default weights are a COCO nano model.

    COCO includes person, vehicles, and some animals (dog, horse, cow, sheep,
    bird, cat, bear, elephant, zebra, giraffe). It does not include coyote,
    whitetail deer, or feral hog. Those labels will never appear unless you
    point this detector at custom-trained weights that actually contain them.
    """

    name = "yolo"

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        track: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for the YOLO detector. "
                "Install with: pip install -r requirements.txt"
            ) from exc
        self.weights = weights
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device or None
        self.track_enabled = bool(track)
        self.tracker_name = tracker or "bytetrack.yaml"
        print(f"[wildwatch] loading YOLO weights {weights!r} (auto-download on first use)")
        self.model = YOLO(weights)
        # Trigger a tiny warmup later on first infer so startup stays honest.

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        kwargs = dict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        if self.track_enabled:
            kwargs["persist"] = True
            kwargs["tracker"] = self.tracker_name
            results = self.model.track(**kwargs)
        else:
            results = self.model.predict(**kwargs)
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        names = result.names or {}
        xyxy = result.boxes.xyxy
        confs = result.boxes.conf
        clss = result.boxes.cls
        ids = getattr(result.boxes, "id", None)
        if xyxy is None:
            return detections
        xyxy = xyxy.cpu().numpy()
        confs = confs.cpu().numpy() if confs is not None else np.zeros(len(xyxy))
        clss = clss.cpu().numpy() if clss is not None else np.zeros(len(xyxy))
        if ids is not None:
            ids = ids.cpu().numpy()
        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
            label = str(names.get(int(cls_id), f"class_{int(cls_id)}"))
            x1, y1, x2, y2 = (float(v) for v in box)
            track_id = None
            if ids is not None and i < len(ids) and ids[i] is not None:
                track_id = int(ids[i])
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    label=label,
                    confidence=float(conf),
                    raw_label=label,
                    track_id=track_id,
                )
            )
        return detections

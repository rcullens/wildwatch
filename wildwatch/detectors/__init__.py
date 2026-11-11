from wildwatch.detectors.base import Detector
from wildwatch.detectors.yolo import YOLODetector


def build_detector(cfg: dict) -> Detector:
    name = str(cfg.get("detector", "yolo")).strip().lower()
    device = cfg.get("device") or ""
    confidence = float(cfg.get("confidence", 0.35))
    imgsz = int(cfg.get("imgsz", 640))
    iou = float(cfg.get("iou", 0.45))
    tracking = cfg.get("tracking") or {}
    track_on = bool(tracking.get("enabled", True))
    backend = str(tracking.get("backend") or "auto").lower()
    if backend in {"iou", "off", "none"}:
        yolo_track, yolo_tracker = False, "bytetrack.yaml"
    elif backend in {"botsort", "bot-sort", "bot_sort"}:
        yolo_track, yolo_tracker = track_on, "botsort.yaml"
    else:
        yolo_track, yolo_tracker = track_on, "bytetrack.yaml"
    if name in {"yolo", "yolov8", "yolo11", "ultralytics"}:
        return YOLODetector(
            weights=cfg.get("model") or "yolo11n.pt",
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            device=device,
            track=yolo_track,
            tracker=yolo_tracker,
        )
    if name in {"megadetector", "md", "mdv6", "pytorchwildlife"}:
        from wildwatch.detectors.megadetector import MegaDetector

        version = (
            (cfg.get("megadetector") or {}).get("version") or "MDV6-yolov10-c"
        )
        return MegaDetector(
            version=version,
            confidence=confidence,
            device=device,
        )
    raise ValueError(
        f"Unknown detector {name!r}. Use 'yolo' or 'megadetector'."
    )


__all__ = ["Detector", "YOLODetector", "build_detector"]

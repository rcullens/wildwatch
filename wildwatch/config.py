from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "source": "samples/demo.mp4",
    "mode": "gui",
    "detector": "yolo",
    "model": "yolo11n.pt",
    "megadetector": {"version": "MDV6-yolov10-c"},
    "classifier": "none",
    "custom_classifier": {"weights": "", "labels": []},
    "confidence": 0.35,
    "iou": 0.45,
    "imgsz": 640,
    "device": "",
    "frame_skip": 0,
    "loop_file": False,
    "alert": {
        "species": [
            "coyote", "deer", "hog", "pig", "boar", "bear", "person", "animal",
        ],
        "min_confidence": 0.45,
        "beep": True,
        "snapshot": True,
        "cooldown_seconds": 3.0,
    },
    "paths": {
        "captures": "captures",
        "events": "captures/events.jsonl",
        "events_csv": "captures/events.csv",
        "zones": "captures/zones.yaml",
    },
    "server": {"host": "0.0.0.0", "port": 8080},
    "hud": {"compass": True, "reticle": True, "trails": True, "threat_list": True},
    "telemetry": {"srt_path": ""},
    "tracking": {
        "enabled": True,
        "backend": "auto",  # auto (ByteTrack for YOLO, IOU for MegaDetector) | bytetrack | botsort | iou
        "max_age": 30,
        "min_iou": 0.3,
        "trail_length": 30,
    },
    "threat": {
        "class_weights": {
            "person": 0.90,
            "vehicle": 0.85,
            "car": 0.85,
            "truck": 0.85,
            "bus": 0.85,
            "motorcycle": 0.80,
            "coyote": 0.90,
            "hog": 0.85,
            "pig": 0.85,
            "boar": 0.85,
            "bear": 0.90,
            "dog": 0.55,
            "deer": 0.40,
            "animal": 0.50,
            "animal (unid.)": 0.50,
        },
        "weights": {
            "class": 0.35,
            "confidence": 0.15,
            "proximity": 0.20,
            "dwell": 0.15,
            "approach": 0.15,
        },
        "proximity_area_ref": 0.12,
        "dwell_ref_frames": 45,
        "approach_min_px": 2.0,
        "approach_confirm": 5,
        "high": 0.70,
        "medium": 0.40,
        "approach_alerts": True,
        "zone_alerts": True,
    },
    "zones": {"items": []},
}


def _deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULTS)
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping, got {type(loaded)}")
    return _deep_update(cfg, loaded)


def apply_cli_overrides(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    """Overlay argparse namespace fields that the user actually set."""
    mapping = {
        "source": "source",
        "mode": "mode",
        "detector": "detector",
        "model": "model",
        "classifier": "classifier",
        "confidence": "confidence",
        "device": "device",
        "imgsz": "imgsz",
        "frame_skip": "frame_skip",
    }
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            cfg[key] = value
    if getattr(args, "host", None) is not None:
        cfg["server"]["host"] = args.host
    if getattr(args, "port", None) is not None:
        cfg["server"]["port"] = args.port
    if getattr(args, "no_beep", False):
        cfg["alert"]["beep"] = False
    if getattr(args, "loop", False):
        cfg["loop_file"] = True
    return cfg

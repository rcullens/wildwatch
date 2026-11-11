from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np

from wildwatch.alerts import AlertManager
from wildwatch.capture import StreamCapture
from wildwatch.classifiers import build_classifier, refine_detections
from wildwatch.detectors import build_detector
from wildwatch.events import EventLog
from wildwatch.hud import render
from wildwatch.telemetry import TelemetrySource
from wildwatch.threat import ThreatAssessor, apply_zones_and_threat
from wildwatch.tracker import IoUTracker
from wildwatch.types import Detection, Telemetry
from wildwatch.zones import ZoneStore


class Pipeline:
    """Capture → detect → classify → HUD → alerts. Shared by GUI and server."""

    def __init__(self, cfg: dict[str, Any], source_override: str | None = None) -> None:
        self.cfg = cfg
        self.source = source_override if source_override is not None else cfg["source"]
        self.detector = build_detector(cfg)
        self.classifier = build_classifier(cfg)
        self.capture = StreamCapture(
            self.source, loop_file=bool(cfg.get("loop_file")),
        )
        paths = cfg["paths"]
        self.event_log = EventLog(paths["events"], paths["events_csv"])
        alert = cfg["alert"]
        self.alerts = AlertManager(
            species=list(alert.get("species") or []),
            min_confidence=float(alert.get("min_confidence", 0.45)),
            beep=bool(alert.get("beep", True)),
            snapshot=bool(alert.get("snapshot", True)),
            cooldown_seconds=float(alert.get("cooldown_seconds", 3.0)),
            captures_dir=paths["captures"],
            event_log=self.event_log,
        )
        srt = (cfg.get("telemetry") or {}).get("srt_path") or ""
        self.telemetry = TelemetrySource(srt or None)
        self.frame_skip = max(0, int(cfg.get("frame_skip") or 0))
        self.hud_cfg = cfg.get("hud") or {}
        tracking = cfg.get("tracking") or {}
        self.tracker = IoUTracker(
            min_iou=float(tracking.get("min_iou", 0.3)),
            max_age=int(tracking.get("max_age", 30)),
            trail_length=int(tracking.get("trail_length", 30)),
        )
        self.zones = ZoneStore.load(cfg)
        self.assessor = ThreatAssessor(cfg.get("threat") or {})
        self.threat_cfg = cfg.get("threat") or {}
        self._last_detections: list[Detection] = []
        self._sa_banner: str | None = None
        self._started = time.monotonic()
        self._fps = 0.0
        self._fps_window: list[float] = []
        self.latest_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.stats: dict[str, Any] = {}
        self.lock = Lock()
        self.running = True
        self.device_label = self._device_label()

    def _device_label(self) -> str:
        configured = str(self.cfg.get("device") or "").strip()
        if configured:
            return configured
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def step(self) -> np.ndarray | None:
        frame = self.capture.read()
        if frame is None:
            return None
        t0 = time.monotonic()
        use_detector = (
            self.frame_skip == 0
            or (self.capture.frame_index - 1) % (self.frame_skip + 1) == 0
        )
        if use_detector:
            detections = self.detector.infer(frame)
            detections = refine_detections(self.classifier, frame, detections)
            if (self.cfg.get("tracking") or {}).get("enabled", True):
                detections = self.tracker.update(detections)
            h, w = frame.shape[:2]
            sa_events = apply_zones_and_threat(
                self.tracker.tracks,
                detections,
                self.zones,
                self.assessor,
                (w, h),
            )
            self._emit_sa_events(sa_events, detections, frame, tel_placeholder=None)
            self._last_detections = detections
        else:
            detections = self._last_detections
            sa_events = []
        elapsed = time.monotonic() - self._started
        tel: Telemetry = self.telemetry.at(elapsed)
        # Re-bind telemetry onto events that fired this frame if needed is handled below.
        annotated = render(
            frame,
            detections,
            fps=self._fps,
            frame_index=self.capture.frame_index,
            source=self.capture.name,
            detector_name=self.detector.name,
            device_label=self.device_label,
            telemetry=tel,
            banner=self.alerts.banner,
            draw_compass=bool(self.hud_cfg.get("compass", True)),
            draw_reticle=bool(self.hud_cfg.get("reticle", True)),
            draw_trails=bool(self.hud_cfg.get("trails", True)),
            zones=self.zones.zones,
            draft=self.zones.draft,
        )
        if use_detector:
            # Record SA events now that we have telemetry + the annotated frame.
            self._record_sa_events(sa_events, detections, annotated, tel)
        self.alerts.consider(
            detections,
            annotated,
            frame_index=self.capture.frame_index,
            source=self.capture.name,
            telemetry=tel,
        )
        dt = time.monotonic() - t0
        if dt > 0:
            self._fps_window.append(1.0 / dt)
            if len(self._fps_window) > 30:
                self._fps_window.pop(0)
            self._fps = sum(self._fps_window) / len(self._fps_window)
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        ranked = sorted(detections, key=lambda d: d.threat, reverse=True)
        with self.lock:
            self.latest_frame = annotated
            self.latest_jpeg = buf.tobytes() if ok else None
            self.stats = {
                "fps": round(self._fps, 2),
                "frame": self.capture.frame_index,
                "source": self.capture.name,
                "detector": self.detector.name,
                "classifier": self.classifier.name,
                "device": self.device_label,
                "counts": dict(Counter(d.label.lower() for d in detections)),
                "detections": [self._det_payload(d) for d in detections],
                "threats": [self._det_payload(d) for d in ranked],
                "zones": self.zones.as_list(),
                "draft": [list(p) for p in self.zones.draft],
                "banner": self.alerts.banner,
                "drops": self.capture.drops,
            }
        return annotated

    @staticmethod
    def _det_payload(d: Detection) -> dict[str, Any]:
        return {
            "label": d.label,
            "confidence": round(float(d.confidence), 3),
            "category": d.category,
            "xyxy": [round(float(v), 1) for v in d.xyxy],
            "track_id": d.track_id,
            "threat": d.threat,
            "threat_level": d.threat_level,
            "zones": list(d.zones),
            "status": d.status,
            "approaching": d.approaching,
            "trail": [[round(x, 1), round(y, 1)] for x, y in (d.trail or [])[-20:]],
        }

    def _emit_sa_events(self, *args, **kwargs) -> None:
        # Placeholder kept so the step body can stay linear; recording happens
        # after the annotated frame is rendered. See _record_sa_events.
        return None

    def _record_sa_events(self, sa_events, detections, frame_bgr, telemetry) -> None:
        if not sa_events:
            return
        threat_cfg = self.threat_cfg
        zone_on = bool(threat_cfg.get("zone_alerts", True))
        approach_on = bool(threat_cfg.get("approach_alerts", True))
        by_id = {d.track_id: d for d in detections if d.track_id is not None}
        for ev in sa_events:
            etype = ev["event_type"]
            if etype in {"zone_enter", "zone_exit"} and not zone_on:
                continue
            if etype == "approaching" and not approach_on:
                continue
            det = by_id.get(ev["track_id"])
            snapshot = ""
            if det is not None and self.alerts.snapshot_enabled:
                snapshot = str(self.alerts._save_snapshot(frame_bgr, det, self.capture.frame_index))
            self.event_log.record(
                det,
                frame_index=self.capture.frame_index,
                source=self.capture.name,
                snapshot=snapshot,
                telemetry=telemetry,
                event_type=etype,
                zone=ev.get("zone") or "",
                track_id=ev.get("track_id"),
                label=det.label if det is not None else "",
            )
            tid = ev.get("track_id")
            label = (det.label if det is not None else "track").upper()
            if etype == "zone_enter":
                self.alerts.announce(f"ZONE ENTER  {ev['zone'].upper()}  {label}  ID {tid}")
            elif etype == "zone_exit":
                self.alerts.announce(f"ZONE EXIT  {ev['zone'].upper()}  {label}  ID {tid}")
            elif etype == "approaching":
                target = ev.get("zone") or "center"
                self.alerts.announce(f"APPROACHING  {label}  ID {tid}  {target.upper()}")

    def run_loop(
        self,
        on_frame: Callable[[np.ndarray], bool] | None = None,
        max_frames: int = 0,
        save_dir: str | None = None,
        save_every: int = 1,
    ) -> dict[str, Any]:
        written = 0
        out_dir = Path(save_dir) if save_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.running:
                frame = self.step()
                if frame is None:
                    break
                if out_dir and (save_every <= 1 or self.capture.frame_index % save_every == 0):
                    path = out_dir / f"frame_{self.capture.frame_index:06d}.jpg"
                    cv2.imwrite(str(path), frame)
                    written += 1
                if on_frame is not None and not on_frame(frame):
                    break
                if max_frames and self.capture.frame_index >= max_frames:
                    break
        finally:
            self.close()
        return {
            "frames": self.capture.frame_index,
            "fps": self._fps,
            "saved": written,
            "events": len(self.event_log.recent),
        }

    def close(self) -> None:
        self.running = False
        self.capture.release()
        self.alerts.close()
        self.detector.close()

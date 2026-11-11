from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from wildwatch.events import EventLog
from wildwatch.types import Detection, Telemetry


def _label_matches(label: str, species: list[str]) -> bool:
    text = label.strip().lower()
    for item in species:
        key = item.strip().lower()
        if not key:
            continue
        if key in text or text in key:
            return True
    return False


def _write_beep_wav(path: Path, frequency: int = 880, duration: float = 0.18) -> None:
    rate = 22050
    n = int(rate * duration)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for i in range(n):
            # Short cosine envelope so it isn't a harsh click.
            env = 1.0
            if i < 200:
                env = i / 200
            elif i > n - 200:
                env = max(0.0, (n - i) / 200)
            sample = int(env * 20000 * np.sin(2 * np.pi * frequency * i / rate))
            wav.writeframes(struct.pack("<h", sample))


class AlertManager:
    def __init__(
        self,
        species: list[str],
        min_confidence: float,
        beep: bool,
        snapshot: bool,
        cooldown_seconds: float,
        captures_dir: str,
        event_log: EventLog,
    ) -> None:
        self.species = list(species)
        self.min_confidence = min_confidence
        self.beep_enabled = beep
        self.snapshot_enabled = snapshot
        self.cooldown = cooldown_seconds
        self.captures_dir = Path(captures_dir)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = event_log
        self._last_fired: dict[str, float] = {}
        self._banner: str | None = None
        self._banner_until = 0.0
        self._beep_wav: Path | None = None
        if self.beep_enabled:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            self._beep_wav = Path(handle.name)
            _write_beep_wav(self._beep_wav)

    @property
    def banner(self) -> str | None:
        if self._banner and time.time() < self._banner_until:
            return self._banner
        return None

    def consider(
        self,
        detections: list[Detection],
        frame_bgr: np.ndarray,
        frame_index: int,
        source: str,
        telemetry: Telemetry | None = None,
    ) -> list[dict]:
        events = []
        now = time.time()
        for det in detections:
            conf = det.species_confidence or det.confidence
            if conf < self.min_confidence:
                continue
            if not _label_matches(det.label, self.species):
                continue
            key = f"{det.track_id}:{det.label.lower()}"
            last = self._last_fired.get(key, 0.0)
            if now - last < self.cooldown:
                continue
            self._last_fired[key] = now
            snapshot_path = None
            if self.snapshot_enabled:
                snapshot_path = self._save_snapshot(frame_bgr, det, frame_index)
            event = self.event_log.record(
                det,
                frame_index=frame_index,
                source=source,
                snapshot=str(snapshot_path) if snapshot_path else "",
                telemetry=telemetry,
                event_type="alert",
            )
            events.append(event)
            tid = f" ID {det.track_id}" if det.track_id is not None else ""
            self.announce(f"{det.label.upper()}{tid}")
        return events

    def announce(self, text: str, hold: float = 2.5, beep: bool = True) -> None:
        self._banner = text
        self._banner_until = time.time() + hold
        if beep and self.beep_enabled:
            threading.Thread(target=self._beep, daemon=True).start()

    def _save_snapshot(self, frame_bgr: np.ndarray, det: Detection, frame_index: int) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in det.label.lower())
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.captures_dir / f"{stamp}_f{frame_index:06d}_{safe}.jpg"
        cv2.imwrite(str(path), frame_bgr)
        return path

    def _beep(self) -> None:
        if not self._beep_wav:
            return
        for binary in ("ffplay", "paplay", "aplay", "ffplay"):
            resolved = shutil.which(binary)
            if not resolved:
                continue
            cmd = [resolved]
            if binary == "ffplay":
                cmd += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
            cmd.append(str(self._beep_wav))
            try:
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
            except Exception:
                continue

    def close(self) -> None:
        if self._beep_wav and self._beep_wav.exists():
            try:
                self._beep_wav.unlink()
            except OSError:
                pass

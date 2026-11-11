from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from wildwatch.types import Detection, Telemetry


class EventLog:
    """Append-only JSONL + CSV event log, plus a recent-events ring buffer."""

    def __init__(self, jsonl_path: str, csv_path: str, maxlen: int = 400) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.recent: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._fields = [
            "timestamp", "event_type", "frame", "track_id", "label", "raw_label",
            "category", "confidence", "species_confidence", "threat", "threat_level",
            "zone", "status", "x1", "y1", "x2", "y2", "snapshot",
            "latitude", "longitude", "altitude_m", "source",
        ]
        self._csv_initialized = False
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            try:
                with self.csv_path.open() as fh:
                    header = fh.readline().strip().split(",")
                if header == self._fields:
                    self._csv_initialized = True
                else:
                    # Schema changed; start a sidecar rather than corrupt the old file.
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    self.csv_path = self.csv_path.with_name(self.csv_path.stem + f".{stamp}.csv")
            except OSError:
                pass

    def record(
        self,
        detection: Detection | None = None,
        frame_index: int = 0,
        source: str = "",
        snapshot: str | None = None,
        telemetry: Telemetry | None = None,
        event_type: str = "detection",
        zone: str = "",
        track_id: int | None = None,
        label: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        if detection is not None:
            x1, y1, x2, y2 = detection.xyxy
            label = label or detection.label
            track_id = detection.track_id if track_id is None else track_id
            raw_label = detection.raw_label
            category = detection.category
            confidence = round(float(detection.confidence), 4)
            species_conf = (
                None if detection.species_confidence is None
                else round(float(detection.species_confidence), 4)
            )
            threat = round(float(detection.threat), 3) if detection.threat else 0.0
            threat_level = detection.threat_level
            status = detection.status
            if not zone and detection.zones:
                zone = ",".join(detection.zones)
        else:
            x1 = y1 = x2 = y2 = ""
            raw_label = ""
            category = ""
            confidence = ""
            species_conf = None
            threat = ""
            threat_level = ""
            status = ""
        row: dict[str, Any] = {
            "timestamp": now,
            "event_type": event_type,
            "frame": frame_index,
            "track_id": "" if track_id is None else track_id,
            "label": label,
            "raw_label": raw_label,
            "category": category,
            "confidence": confidence,
            "species_confidence": species_conf,
            "threat": threat,
            "threat_level": threat_level,
            "zone": zone,
            "status": status,
            "x1": "" if x1 == "" else round(float(x1), 1),
            "y1": "" if y1 == "" else round(float(y1), 1),
            "x2": "" if x2 == "" else round(float(x2), 1),
            "y2": "" if y2 == "" else round(float(y2), 1),
            "snapshot": snapshot or "",
            "latitude": telemetry.latitude if telemetry else None,
            "longitude": telemetry.longitude if telemetry else None,
            "altitude_m": telemetry.altitude_m if telemetry else None,
            "source": source,
        }
        if extra:
            row.update(extra)
        with self.jsonl_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        write_header = not self._csv_initialized
        with self.csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow({k: row.get(k, "") for k in self._fields})
        self.recent.appendleft(row)
        return row

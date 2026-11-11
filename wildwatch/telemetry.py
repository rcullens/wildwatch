from __future__ import annotations

import re
from pathlib import Path

from wildwatch.types import Telemetry

# DJI Fly SRT blocks look like:
#   1
#   00:00:00,000 --> 00:00:01,000
#   <font>FrameCnt: 1, DiffTime: 33ms
#   2026-11-10 14:22:01.123
#   [iso: 100] [shutter: 1/200] ...
#   [latitude: 31.8123] [longitude: -97.2345] [rel_alt: 42.1 abs_alt: 210.0] ...
# Field names vary slightly across firmware / DJI Fly versions.

_LAT = re.compile(r"latitude\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I)
_LON = re.compile(r"longitude\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I)
_REL = re.compile(r"rel[_ ]?alt(?:itude)?\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I)
_ABS = re.compile(r"abs[_ ]?alt(?:itude)?\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I)
_HDR = re.compile(r"(?:heading|gb_yaw|yaw)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.I)
_TIME = re.compile(r"(\d{1,2}:\d{2}:\d{2}[.,]\d+)\s+-->\s+(\d{1,2}:\d{2}:\d{2}[.,]\d+)")


def _hms_to_seconds(text: str) -> float:
    text = text.replace(",", ".")
    parts = text.split(":")
    if len(parts) != 3:
        return 0.0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def parse_dji_srt(path: str | Path) -> list[tuple[float, Telemetry]]:
    """Parse a DJI SRT sidecar into (start_seconds, Telemetry) entries."""
    text = Path(path).read_text(errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    entries: list[tuple[float, Telemetry]] = []
    for block in blocks:
        time_match = _TIME.search(block)
        if not time_match:
            continue
        start = _hms_to_seconds(time_match.group(1))
        lat = _LAT.search(block)
        lon = _LON.search(block)
        rel = _REL.search(block)
        abs_alt = _ABS.search(block)
        heading = _HDR.search(block)
        alt = float(rel.group(1)) if rel else (float(abs_alt.group(1)) if abs_alt else None)
        entries.append(
            (
                start,
                Telemetry(
                    latitude=float(lat.group(1)) if lat else None,
                    longitude=float(lon.group(1)) if lon else None,
                    altitude_m=alt,
                    heading_deg=float(heading.group(1)) if heading else None,
                ),
            )
        )
    entries.sort(key=lambda item: item[0])
    return entries


class TelemetrySource:
    """Optional telemetry. Never invents a fix if none is available.

    Live Mini 3 RTMP does not carry SRT in-band. For recorded flights, pass
    the sidecar with --srt or telemetry.srt_path. To feed live telemetry
    later, subclass or replace this with a reader that implements at().
    """

    def __init__(self, srt_path: str | None = None) -> None:
        self.entries: list[tuple[float, Telemetry]] = []
        self.path = srt_path or ""
        if srt_path:
            self.entries = parse_dji_srt(srt_path)
            print(f"[wildwatch] loaded {len(self.entries)} SRT telemetry records from {srt_path}")

    def at(self, elapsed_seconds: float) -> Telemetry:
        if not self.entries:
            return Telemetry()
        # Last record whose start time is <= elapsed.
        chosen = self.entries[0][1]
        for start, tel in self.entries:
            if start <= elapsed_seconds:
                chosen = tel
            else:
                break
        return chosen

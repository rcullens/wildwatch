from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _norm_poly(points: list) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x, y = float(pt[0]), float(pt[1])
        # Accept either already-normalized 0-1 or accidental 0-100 percentages.
        out.append((x, y))
    return out


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting test. Coordinates must all be in the same space."""
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) == (y2 > y):
            continue
        denom = (y2 - y1)
        if denom == 0:
            continue
        xinters = (x2 - x1) * (y - y1) / denom + x1
        if x < xinters:
            inside = not inside
    return inside


def dist_to_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> float:
    """Euclidean distance in the same units as the polygon. 0 if inside."""
    if len(polygon) < 3:
        return float("inf")
    if point_in_polygon(x, y, polygon):
        return 0.0
    best = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            dist = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
            px, py = x1 + t * dx, y1 + t * dy
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if dist < best:
            best = dist
    return best


@dataclass
class Zone:
    name: str
    polygon: list[tuple[float, float]]  # normalized 0-1 in the video frame

    def contains_norm(self, nx: float, ny: float) -> bool:
        return point_in_polygon(nx, ny, self.polygon)

    def dist_norm(self, nx: float, ny: float) -> float:
        return dist_to_polygon(nx, ny, self.polygon)

    def pixel_polygon(self, width: int, height: int) -> list[tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in self.polygon]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "polygon": [list(p) for p in self.polygon]}


@dataclass
class ZoneStore:
    """Frame-space geofences (normalized polygons). Not GPS.

    Zones come from config.yaml and/or a sidecar YAML the GUI and dashboard
    write when you draw a polygon. No lat/lon is invented.
    """

    path: Path | None = None
    zones: list[Zone] = field(default_factory=list)
    draft: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def load(cls, cfg: dict[str, Any]) -> "ZoneStore":
        zone_cfg = cfg.get("zones") or {}
        items = zone_cfg.get("items") if isinstance(zone_cfg, dict) else zone_cfg
        if items is None:
            items = []
        path_str = (cfg.get("paths") or {}).get("zones") or "captures/zones.yaml"
        path = Path(path_str)
        store = cls(path=path)
        for item in items:
            store._add_raw(item, replace=False)
        if path.exists():
            loaded = yaml.safe_load(path.read_text()) or {}
            file_items = loaded.get("items", loaded if isinstance(loaded, list) else [])
            for item in file_items or []:
                store._add_raw(item, replace=True)
        return store

    def _add_raw(self, item: Any, replace: bool) -> None:
        if not isinstance(item, dict):
            return
        name = str(item.get("name") or "").strip()
        poly = _norm_poly(item.get("polygon") or [])
        if not name or len(poly) < 3:
            return
        # Clamp obviously-normalized values; leave 0-1 as 0-1.
        clamped = [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in poly]
        if replace:
            self.zones = [z for z in self.zones if z.name != name]
        elif any(z.name == name for z in self.zones):
            return
        self.zones.append(Zone(name=name, polygon=clamped))

    def add(self, name: str, polygon: list[tuple[float, float]], persist: bool = True) -> Zone:
        name = name.strip() or self._auto_name()
        poly = [(min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in polygon]
        if len(poly) < 3:
            raise ValueError("A zone needs at least 3 vertices")
        self.zones = [z for z in self.zones if z.name != name]
        zone = Zone(name=name, polygon=poly)
        self.zones.append(zone)
        self.draft = []
        if persist:
            self.save()
        return zone

    def remove(self, name: str, persist: bool = True) -> bool:
        before = len(self.zones)
        self.zones = [z for z in self.zones if z.name != name]
        if persist and len(self.zones) != before:
            self.save()
        return len(self.zones) != before

    def containing(self, nx: float, ny: float) -> list[Zone]:
        return [z for z in self.zones if z.contains_norm(nx, ny)]

    def add_draft(self, nx: float, ny: float) -> list[tuple[float, float]]:
        self.draft.append((min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))))
        return list(self.draft)

    def pop_draft(self) -> None:
        if self.draft:
            self.draft.pop()

    def cancel_draft(self) -> None:
        self.draft = []

    def close_draft(self, name: str = "", persist: bool = True) -> Zone:
        if len(self.draft) < 3:
            raise ValueError("Click at least 3 points before closing a zone")
        zone = self.add(name or self._auto_name(), list(self.draft), persist=persist)
        return zone

    def _auto_name(self) -> str:
        n = 1
        existing = {z.name for z in self.zones}
        while f"zone-{n}" in existing:
            n += 1
        return f"zone-{n}"

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [z.as_dict() for z in self.zones],
        }
        self.path.write_text(yaml.safe_dump(payload, sort_keys=False))

    def as_list(self) -> list[dict[str, Any]]:
        return [z.as_dict() for z in self.zones]

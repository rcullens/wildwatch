from __future__ import annotations

from typing import Any

from wildwatch.tracker import TrackState, centroid
from wildwatch.types import Detection
from wildwatch.zones import ZoneStore

DEFAULT_CLASS_WEIGHTS = {
    "person": 0.90,
    "vehicle": 0.85,
    "car": 0.85,
    "truck": 0.85,
    "bus": 0.85,
    "motorcycle": 0.80,
    "bicycle": 0.45,
    "coyote": 0.90,
    "hog": 0.85,
    "pig": 0.85,
    "boar": 0.85,
    "bear": 0.90,
    "dog": 0.55,
    "wolf": 0.90,
    "fox": 0.60,
    "deer": 0.40,
    "cow": 0.35,
    "horse": 0.35,
    "sheep": 0.30,
    "bird": 0.20,
    "animal": 0.50,
    "animal (unid.)": 0.50,
}

DEFAULT_WEIGHTS = {
    "class": 0.35,
    "confidence": 0.15,
    "proximity": 0.20,
    "dwell": 0.15,
    "approach": 0.15,
}


def _class_weight(label: str, table: dict[str, float]) -> float:
    key = (label or "").strip().lower()
    if key in table:
        return float(table[key])
    best = 0.35
    found = False
    for name, weight in table.items():
        if name and name in key:
            if not found or len(name) > 2:
                best = float(weight)
                found = True
    return best if found else 0.35


class ThreatAssessor:
    """Heuristic pixel-space threat score. Not a geospatial intercept.

    Score is a weighted sum of: class prior, detector confidence, box area
    (a proximity *proxy* — closer/larger in frame scores higher), dwell time,
    and whether the track is approaching the frame center or a drawn zone.
    There is no radio/GPS input here; we do not invent telemetry.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.class_weights = dict(DEFAULT_CLASS_WEIGHTS)
        self.class_weights.update({str(k).lower(): float(v) for k, v in (cfg.get("class_weights") or {}).items()})
        self.weights = dict(DEFAULT_WEIGHTS)
        self.weights.update({str(k): float(v) for k, v in (cfg.get("weights") or {}).items()})
        self.proximity_area_ref = float(cfg.get("proximity_area_ref", 0.12))
        self.dwell_ref_frames = max(1, int(cfg.get("dwell_ref_frames", 45)))
        self.approach_min_px = float(cfg.get("approach_min_px", 2.0))
        self.approach_confirm = max(1, int(cfg.get("approach_confirm", 5)))
        self.high = float(cfg.get("high", 0.70))
        self.medium = float(cfg.get("medium", 0.40))

    def update_approach(
        self,
        state: TrackState,
        frame_wh: tuple[int, int],
        zones: ZoneStore,
    ) -> None:
        w, h = frame_wh
        if w <= 0 or h <= 0:
            return
        cx, cy = state.center()
        mid = (w / 2.0, h / 2.0)
        dist_c = ((cx - mid[0]) ** 2 + (cy - mid[1]) ** 2) ** 0.5
        # Closing on the frame center: either distance shrank, or the velocity
        # has a component aimed at the center.
        speed = (state.vx ** 2 + state.vy ** 2) ** 0.5
        toward = 0.0
        if dist_c > 1e-3:
            toward = -((cx - mid[0]) * state.vx + (cy - mid[1]) * state.vy) / dist_c
        if toward >= self.approach_min_px or (
            len(state.trail) >= 2 and self._trail_closing(state.trail, mid, self.approach_min_px)
        ):
            state.closing_center += 1
        else:
            state.closing_center = 0

        nx, ny = cx / w, cy / h
        approaching_zone = ""
        for zone in zones.zones:
            dist = zone.dist_norm(nx, ny)
            # Convert normalized distance to a rough pixel scale so the same
            # threshold applies inside large frames.
            dist_px = dist * ((w + h) / 2.0)
            # Closing: compare to previous center if we have a trail point.
            prev = state.trail[-2] if len(state.trail) >= 2 else None
            closing = False
            if dist == 0:
                closing = False  # already inside; membership is a zone event
            elif prev is not None:
                prev_d = zone.dist_norm(prev[0] / w, prev[1] / h) * ((w + h) / 2.0)
                closing = (prev_d - dist_px) >= self.approach_min_px
            if closing:
                state.closing_zones[zone.name] = state.closing_zones.get(zone.name, 0) + 1
            else:
                state.closing_zones[zone.name] = 0
            if state.closing_zones.get(zone.name, 0) >= self.approach_confirm:
                approaching_zone = zone.name

        toward_center = state.closing_center >= self.approach_confirm and speed >= self.approach_min_px
        state.approaching = bool(toward_center or approaching_zone)
        if approaching_zone:
            state.approaching_what = f"zone:{approaching_zone}"
        elif toward_center:
            state.approaching_what = "center"
        else:
            state.approaching_what = ""

    def score(self, state: TrackState, frame_wh: tuple[int, int]) -> float:
        w, h = frame_wh
        area = max(0.0, (state.xyxy[2] - state.xyxy[0]) * (state.xyxy[3] - state.xyxy[1]))
        frame_area = max(1.0, float(w * h))
        proximity = min(1.0, area / (self.proximity_area_ref * frame_area))
        dwell = min(1.0, state.dwell_frames / float(self.dwell_ref_frames))
        conf = min(1.0, max(0.0, float(state.species_confidence or state.confidence)))
        class_w = _class_weight(state.label, self.class_weights)
        approach = 1.0 if state.approaching else 0.0
        parts = {
            "class": class_w,
            "confidence": conf,
            "proximity": proximity,
            "dwell": dwell,
            "approach": approach,
        }
        total_w = sum(self.weights.get(k, 0.0) for k in parts) or 1.0
        score = sum(self.weights.get(k, 0.0) * v for k, v in parts.items()) / total_w
        # A mild bump when already inside a drawn zone — still pixel-space only.
        if state.zones:
            score = min(1.0, score + 0.08)
        state.threat = round(float(score), 3)
        if state.threat >= self.high:
            state.threat_level = "high"
        elif state.threat >= self.medium:
            state.threat_level = "medium"
        else:
            state.threat_level = "low"
        bits = []
        if state.approaching:
            bits.append("APPROACHING")
        if state.zones:
            bits.append("IN " + ",".join(z.upper() for z in state.zones))
        state.status = "  ".join(bits)
        return state.threat

    @staticmethod
    def _trail_closing(trail: list[tuple[float, float]], target: tuple[float, float], min_px: float) -> bool:
        if len(trail) < 2:
            return False
        d0 = ((trail[-2][0] - target[0]) ** 2 + (trail[-2][1] - target[1]) ** 2) ** 0.5
        d1 = ((trail[-1][0] - target[0]) ** 2 + (trail[-1][1] - target[1]) ** 2) ** 0.5
        return (d0 - d1) >= min_px


def apply_zones_and_threat(
    tracks: dict[int, TrackState],
    detections: list[Detection],
    zones: ZoneStore,
    assessor: ThreatAssessor,
    frame_wh: tuple[int, int],
) -> list[dict[str, Any]]:
    """Update zone membership + scores. Returns zone enter/exit event payloads."""
    w, h = frame_wh
    events: list[dict[str, Any]] = []
    live = {d.track_id for d in detections if d.track_id is not None}
    for tid in live:
        state = tracks.get(tid)
        if state is None:
            continue
        cx, cy = state.center()
        nx, ny = (cx / w, cy / h) if w and h else (0.0, 0.0)
        current = [z.name for z in zones.containing(nx, ny)]
        prev = list(state.zones)
        state.prev_zones = prev
        state.zones = current
        entered = [n for n in current if n not in prev]
        exited = [n for n in prev if n not in current]
        assessor.update_approach(state, frame_wh, zones)
        assessor.score(state, frame_wh)
        for name in entered:
            events.append({"event_type": "zone_enter", "track_id": tid, "zone": name, "state": state})
        for name in exited:
            events.append({"event_type": "zone_exit", "track_id": tid, "zone": name, "state": state})
        # Fire at most once per target per approach_event_cooldown frames so box
        # jitter does not re-trigger APPROACHING every few frames.
        targets: list[str] = []
        if state.closing_center >= assessor.approach_confirm:
            targets.append("center")
        for zname, count in state.closing_zones.items():
            if count >= assessor.approach_confirm:
                targets.append(f"zone:{zname}")
        cooldown = int(getattr(assessor, "approach_event_cooldown", 45))
        for target in targets:
            last = state.approach_last.get(target, -10**9)
            if state.dwell_frames - last >= cooldown:
                events.append({"event_type": "approaching", "track_id": tid, "zone": target, "state": state})
                state.approach_last[target] = state.dwell_frames
    # Push scored fields back onto the current-frame detections.
    for det in detections:
        if det.track_id is None:
            continue
        state = tracks.get(det.track_id)
        if state is None:
            continue
        det.threat = state.threat
        det.threat_level = state.threat_level
        det.zones = list(state.zones)
        det.status = state.status
        det.approaching = state.approaching
        det.trail = list(state.trail)
        det.velocity = (state.vx, state.vy)
        det.dwell_frames = state.dwell_frames
    return events

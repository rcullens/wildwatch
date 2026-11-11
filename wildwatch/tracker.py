from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wildwatch.types import Detection


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def centroid(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class TrackState:
    track_id: int
    xyxy: tuple[float, float, float, float]
    label: str
    raw_label: str
    category: str
    confidence: float
    species_confidence: float | None
    hits: int = 1
    missed: int = 0
    dwell_frames: int = 1
    trail: list[tuple[float, float]] = field(default_factory=list)
    vx: float = 0.0
    vy: float = 0.0
    closing_center: int = 0
    closing_zones: dict[str, int] = field(default_factory=dict)
    zones: list[str] = field(default_factory=list)
    prev_zones: list[str] = field(default_factory=list)
    approaching: bool = False
    approaching_what: str = ""
    approach_last: dict[str, int] = field(default_factory=dict)
    threat: float = 0.0
    threat_level: str = "low"
    status: str = ""

    def center(self) -> tuple[float, float]:
        return centroid(self.xyxy)


class IoUTracker:
    """Lightweight IOU + centroid tracker for backends without native IDs.

    Used for MegaDetector and as a fallback when YOLO `model.track` does not
    assign an id on a given frame. This is not ByteTrack; it is a simple
    greedy IOU match so MegaDetector still gets persistent IDs.
    """

    def __init__(
        self,
        min_iou: float = 0.3,
        max_age: int = 30,
        trail_length: int = 30,
        next_id: int = 1,
    ) -> None:
        self.min_iou = min_iou
        self.max_age = max_age
        self.trail_length = trail_length
        self.next_id = next_id
        self.tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        # If the detector already assigned IDs (YOLO ByteTrack), sync state
        # to those IDs instead of inventing a second namespace.
        if any(d.track_id is not None for d in detections):
            return self._sync_native(detections)

        unmatched_tracks = set(self.tracks)
        unmatched_dets = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []
        for ti, track in self.tracks.items():
            for di in unmatched_dets:
                score = _iou(track.xyxy, detections[di].xyxy)
                if score >= self.min_iou:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        assigned: dict[int, int] = {}
        used_dets: set[int] = set()
        for score, ti, di in pairs:
            if ti not in unmatched_tracks or di in used_dets:
                continue
            assigned[ti] = di
            unmatched_tracks.discard(ti)
            used_dets.add(di)

        for ti, di in assigned.items():
            self._refresh(self.tracks[ti], detections[di])
            detections[di].track_id = ti
            self._copy_state(self.tracks[ti], detections[di])

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            state = self._new_state(tid, det)
            self.tracks[tid] = state
            det.track_id = tid
            self._copy_state(state, det)

        for ti in list(unmatched_tracks):
            self.tracks[ti].missed += 1
            if self.tracks[ti].missed > self.max_age:
                self.tracks.pop(ti, None)
        return detections

    def _sync_native(self, detections: list[Detection]) -> list[Detection]:
        seen: set[int] = set()
        for det in detections:
            if det.track_id is None:
                # Fill gaps with a fresh IOU match against existing tracks.
                best_id, best = None, self.min_iou
                for tid, track in self.tracks.items():
                    if tid in seen:
                        continue
                    score = _iou(track.xyxy, det.xyxy)
                    if score > best:
                        best, best_id = score, tid
                if best_id is None:
                    det.track_id = self.next_id
                    self.next_id += 1
                else:
                    det.track_id = best_id
            tid = int(det.track_id)
            seen.add(tid)
            if tid in self.tracks:
                self._refresh(self.tracks[tid], det)
            else:
                self.tracks[tid] = self._new_state(tid, det)
                if tid >= self.next_id:
                    self.next_id = tid + 1
            self._copy_state(self.tracks[tid], det)
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_age:
                    self.tracks.pop(tid, None)
        return detections

    def _new_state(self, tid: int, det: Detection) -> TrackState:
        cx, cy = centroid(det.xyxy)
        return TrackState(
            track_id=tid,
            xyxy=det.xyxy,
            label=det.label,
            raw_label=det.raw_label,
            category=det.category,
            confidence=det.confidence,
            species_confidence=det.species_confidence,
            trail=[(cx, cy)],
        )

    def _refresh(self, state: TrackState, det: Detection) -> None:
        px, py = state.center()
        state.xyxy = det.xyxy
        # Keep a stable display label unless the new one is more specific.
        if det.label and det.label.lower() not in {"animal (unid.)", "animal"}:
            state.label = det.label
        state.raw_label = det.raw_label or state.raw_label
        state.category = det.category or state.category
        state.confidence = det.confidence
        if det.species_confidence is not None:
            state.species_confidence = det.species_confidence
        cx, cy = centroid(det.xyxy)
        state.vx, state.vy = cx - px, cy - py
        state.trail.append((cx, cy))
        if len(state.trail) > self.trail_length:
            state.trail = state.trail[-self.trail_length:]
        state.hits += 1
        state.dwell_frames += 1
        state.missed = 0

    @staticmethod
    def _copy_state(state: TrackState, det: Detection) -> None:
        det.track_id = state.track_id
        det.trail = list(state.trail)
        det.velocity = (state.vx, state.vy)
        det.dwell_frames = state.dwell_frames
        det.threat = state.threat
        det.threat_level = state.threat_level
        det.zones = list(state.zones)
        det.status = state.status
        det.approaching = state.approaching

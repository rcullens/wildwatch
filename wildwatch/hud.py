from __future__ import annotations

from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from wildwatch.theme import (
    BGR_ACCENT,
    BGR_LINE,
    BGR_MUT,
    BGR_PANEL,
    BGR_PURPLE,
    BGR_SEND_ON,
    BGR_TXT,
    CATEGORY_COLOR,
)
from wildwatch.types import Detection, Telemetry

# Pocket LLM / Pixel Live Translate palette (BGR). See wildwatch/theme.py.
CYAN = BGR_ACCENT
CYAN_DIM = BGR_LINE
WHITE = BGR_TXT
AMBER = BGR_PURPLE
GRAY = BGR_MUT
BLACK = BGR_SEND_ON
PANEL = BGR_PANEL


def _put(img, text, org, color=WHITE, scale=0.5, thickness=1, align="left"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if align == "right":
        x = x - tw
    elif align == "center":
        x = x - tw // 2
    # Shadow for readability on bright ground.
    cv2.putText(img, text, (x + 1, y + 1), font, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _panel(img, x, y, w, h, alpha=0.55):
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    wash = np.zeros_like(roi)
    wash[:] = PANEL
    cv2.addWeighted(wash, alpha, roi, 1 - alpha, 0, roi)
    cv2.rectangle(img, (x, y), (x2 - 1, y2 - 1), CYAN_DIM, 1, cv2.LINE_AA)


def _draw_reticle(img) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    gap, arm, tick = 18, 42, 8
    color = CYAN
    # Corner brackets of a center reticle, leaving the middle open.
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x0, y0 = cx + sx * gap, cy + sy * gap
        cv2.line(img, (x0, y0), (x0 + sx * arm, y0), color, 1, cv2.LINE_AA)
        cv2.line(img, (x0, y0), (x0, y0 + sy * arm), color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, color, 1, cv2.LINE_AA)
    # Subtle mid-edge ticks.
    for x in (cx - 160, cx + 160):
        cv2.line(img, (x, cy - tick), (x, cy + tick), CYAN_DIM, 1, cv2.LINE_AA)
    for y in (cy - 90, cy + 90):
        cv2.line(img, (cx - tick, y), (cx + tick, y), CYAN_DIM, 1, cv2.LINE_AA)


def _draw_compass(img, heading_deg: float | None) -> None:
    h, w = img.shape[:2]
    band_h = 36
    _panel(img, 0, 0, w, band_h, alpha=0.62)
    cv2.line(img, (0, band_h), (w, band_h), CYAN_DIM, 1, cv2.LINE_AA)
    # A 120-degree window around the current heading, or a north-up strip
    # marked UNKNOWN when we have no telemetry.
    known = heading_deg is not None
    heading = float(heading_deg) % 360 if known else 0.0
    window = 90.0  # degrees visible
    cx = w // 2
    px_per_deg = (w * 0.72) / window

    def wrap(deg: float) -> float:
        return deg % 360

    # Ticks every 5°, labels every 15°.
    start = heading - window / 2
    for offset in range(int(-window / 2), int(window / 2) + 1):
        deg = wrap(heading + offset)
        x = int(cx + offset * px_per_deg)
        if x < 40 or x > w - 40:
            continue
        major = abs(deg % 15) < 0.51 or abs(deg % 15 - 15) < 0.51
        tick_h = 10 if major else 5
        color = CYAN if major else CYAN_DIM
        cv2.line(img, (x, band_h - 2), (x, band_h - 2 - tick_h), color, 1, cv2.LINE_AA)
        if abs(deg % 30) < 0.51 or abs(deg % 30 - 30) < 0.51:
            label = {0: "N", 90: "E", 180: "S", 270: "W"}.get(int(round(deg)) % 360, f"{int(round(deg))%360:03d}")
            _put(img, label, (x, 16), CYAN if label in "NESW" else WHITE, 0.4, 1, align="center")

    # Center caret.
    pts = np.array([[cx, band_h + 6], [cx - 6, band_h - 2], [cx + 6, band_h - 2]], np.int32)
    cv2.fillConvexPoly(img, pts, CYAN)
    hdg_text = f"HDG {heading:05.1f}°" if known else "HDG ---"
    _put(img, hdg_text, (12, 22), CYAN, 0.5, 1)
    _put(img, "WILDWATCH", (w - 12, 22), CYAN, 0.5, 1, align="right")


def _threat_color(level: str, fallback):
    from wildwatch.theme import BGR_DANGER, BGR_PURPLE, BGR_STATUS_OK
    return {
        "high": BGR_DANGER,
        "medium": BGR_PURPLE,
        "low": BGR_STATUS_OK,
    }.get((level or "").lower(), fallback)


def _draw_zones(img, zones, draft) -> None:
    from wildwatch.theme import BGR_PURPLE, BGR_ACCENT
    h, w = img.shape[:2]
    for zone in zones or []:
        pts = zone.pixel_polygon(w, h)
        if len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [arr], BGR_PURPLE)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [arr], True, BGR_PURPLE, 2, cv2.LINE_AA)
        _put(img, zone.name.upper(), (pts[0][0] + 6, pts[0][1] + 18), BGR_PURPLE, 0.5, 1)
    if draft and len(draft) >= 1:
        pts = [(int(x * w), int(y * h)) for x, y in draft]
        arr = np.array(pts, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [arr], False, BGR_ACCENT, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, pt, 4, BGR_ACCENT, -1, cv2.LINE_AA)


def _draw_trails(img, detections: list[Detection]) -> None:
    for det in detections:
        if not det.trail or len(det.trail) < 2:
            continue
        color = _threat_color(det.threat_level, CATEGORY_COLOR.get(det.category, GRAY))
        pts = np.array(det.trail, dtype=np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        if det.velocity and det.trail:
            cx, cy = det.trail[-1]
            vx, vy = det.velocity
            scale = 4.0
            end = (int(cx + vx * scale), int(cy + vy * scale))
            cv2.arrowedLine(img, (int(cx), int(cy)), end, color, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_boxes(img, detections: list[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = CATEGORY_COLOR.get(det.category, GRAY)
        if det.approaching or det.threat_level == "high":
            color = _threat_color(det.threat_level or "high", color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        tick = max(6, int(0.08 * min(x2 - x1, y2 - y1)))
        for (ax, ay, dx, dy) in (
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ):
            cv2.line(img, (ax, ay), (ax + dx * tick, ay), color, 2, cv2.LINE_AA)
            cv2.line(img, (ax, ay), (ax, ay + dy * tick), color, 2, cv2.LINE_AA)
        conf = det.species_confidence if det.species_confidence else det.confidence
        tid = f"#{det.track_id} " if det.track_id is not None else ""
        threat = f"  T{det.threat:.2f}" if det.threat else ""
        label = f"{tid}{det.label.upper()}  {conf:.2f}{threat}"
        if det.approaching:
            label += "  APPR"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(0, y1 - th - 8)
        cv2.rectangle(img, (x1, ty), (x1 + tw + 8, ty + th + 8), color, -1)
        cv2.putText(img, label, (x1 + 4, ty + th + 3), font, 0.5, BLACK, 1, cv2.LINE_AA)


def _fmt_coord(value: float | None, kind: str) -> str:
    if value is None:
        return "LAT ---" if kind == "lat" else "LON ---"
    hemi = ""
    if kind == "lat":
        hemi = "N" if value >= 0 else "S"
    else:
        hemi = "E" if value >= 0 else "W"
    return f"{kind.upper()} {abs(value):.5f}{hemi}"


def _draw_telemetry_panel(
    img,
    fps: float,
    frame_index: int,
    source: str,
    counts: Counter,
    detections: list[Detection],
    telemetry: Telemetry | None,
    detector_name: str,
    device_label: str,
) -> None:
    h, w = img.shape[:2]
    panel_w, panel_h = 280, 168
    x, y = 10, h - panel_h - 10
    _panel(img, x, y, panel_w, panel_h, alpha=0.66)
    yy = y + 22
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _put(img, stamp, (x + 10, yy), CYAN, 0.45, 1)
    yy += 18
    _put(img, f"FPS {fps:5.1f}   FRAME {frame_index}", (x + 10, yy), WHITE, 0.45, 1)
    yy += 18
    src = source if len(source) < 36 else "…" + source[-34:]
    _put(img, f"SRC {src}", (x + 10, yy), WHITE, 0.4, 1)
    yy += 18
    _put(img, f"DET {detector_name}  {device_label}", (x + 10, yy), WHITE, 0.4, 1)
    yy += 20
    tel = telemetry or Telemetry()
    alt = f"ALT {tel.altitude_m:.0f}m" if tel.altitude_m is not None else "ALT ---"
    _put(img, _fmt_coord(tel.latitude, "lat"), (x + 10, yy), WHITE, 0.4, 1)
    yy += 16
    _put(img, f"{_fmt_coord(tel.longitude, 'lon')}   {alt}", (x + 10, yy), WHITE, 0.4, 1)
    yy += 20
    if counts:
        parts = [f"{name} {n}" for name, n in counts.most_common(5)]
        _put(img, "  ".join(parts).upper(), (x + 10, yy), AMBER, 0.4, 1)
    else:
        _put(img, "NO CONTACTS", (x + 10, yy), CYAN_DIM, 0.4, 1)

    # Right-side ranked threat list (track ID, class, level, zone, status).
    ranked = sorted(detections, key=lambda d: d.threat, reverse=True)
    list_w = 300
    rows = max(1, min(10, len(ranked)))
    list_h = min(h - 60, 36 + 20 * rows)
    lx, ly = w - list_w - 10, 48
    _panel(img, lx, ly, list_w, list_h, alpha=0.66)
    _put(img, "THREATS", (lx + 10, ly + 18), CYAN, 0.45, 1)
    if not ranked:
        _put(img, "NO CONTACTS", (lx + 10, ly + 40), CYAN_DIM, 0.4, 1)
    for i, det in enumerate(ranked[:10]):
        tid = f"#{det.track_id}" if det.track_id is not None else "#-"
        level = (det.threat_level or "low")[0].upper()
        zone = det.zones[0] if det.zones else "-"
        status = "APPR" if det.approaching else ("IN" if det.zones else "")
        line = f"{tid:<4} {det.label.upper()[:10]:<10} {level} {det.threat:0.2f}  {zone[:8]:<8} {status}"
        color = _threat_color(det.threat_level, CATEGORY_COLOR.get(det.category, GRAY))
        _put(img, line, (lx + 10, ly + 40 + i * 20), color, 0.38, 1)


def _draw_banner(img, text: str) -> None:
    from wildwatch.theme import BGR_DANGER
    h, w = img.shape[:2]
    banner_h = 54
    y = 44
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y), (w, y + banner_h), BGR_DANGER, -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.line(img, (0, y), (w, y), BGR_DANGER, 2)
    cv2.line(img, (0, y + banner_h), (w, y + banner_h), BGR_DANGER, 2)
    _put(img, f"ALERT  {text}", (w // 2, y + 36), WHITE, 1.1, 2, align="center")


def render(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    fps: float,
    frame_index: int,
    source: str,
    detector_name: str,
    device_label: str = "",
    telemetry: Telemetry | None = None,
    banner: str | None = None,
    draw_compass: bool = True,
    draw_reticle: bool = True,
    draw_trails: bool = True,
    zones=None,
    draft=None,
) -> np.ndarray:
    img = frame_bgr.copy()
    if draw_compass:
        heading = telemetry.heading_deg if telemetry else None
        _draw_compass(img, heading)
    if draw_reticle:
        _draw_reticle(img)
    _draw_zones(img, zones, draft)
    if draw_trails:
        _draw_trails(img, detections)
    _draw_boxes(img, detections)
    counts: Counter = Counter(det.label.lower() for det in detections)
    _draw_telemetry_panel(
        img,
        fps=fps,
        frame_index=frame_index,
        source=source,
        counts=counts,
        detections=detections,
        telemetry=telemetry,
        detector_name=detector_name,
        device_label=device_label,
    )
    if banner:
        _draw_banner(img, banner)
    return img

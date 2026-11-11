#!/usr/bin/env python3
"""Build a short sample clip so Wildwatch can be tested without a drone.

Tries, in order:
  1. A small public sample video if a URL is reachable.
  2. Public stills (Ultralytics demo images, Wikimedia wildlife photos)
     composed into a short pan-and-scan MP4 that YOLO can actually detect.
  3. A fully synthetic clip (moving shapes on a field). This still proves
     the pipeline runs end-to-end, but YOLO will usually report no contacts.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "samples" / "demo.mp4"

# Small, publicly fetchable stills. Used only to produce a local test clip.
STILLS = [
    # Person + vehicle — reliable COCO detections for the smoke test.
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
    # White-tailed deer, public domain (US FWS / Wikimedia).
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/White-tailed_deer.jpg/1280px-White-tailed_deer.jpg",
]

VIDEO_CANDIDATES = [
    # Optional: a short official sample if the network allows it.
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
]


def _fetch(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wildwatch-sample/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        print(f"  skip {url} ({exc})")
        return None


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _letterbox(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - tw) // 2)
    y0 = max(0, (nh - th) // 2)
    crop = resized[y0:y0 + th, x0:x0 + tw]
    if crop.shape[0] != th or crop.shape[1] != tw:
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        return canvas
    return crop


def _write_from_stills(images: list[np.ndarray], out: Path, size=(1280, 720), fps=24, seconds_each=3.0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out}")
    frames_each = int(fps * seconds_each)
    for img in images:
        base = _letterbox(img, size)
        h, w = base.shape[:2]
        for i in range(frames_each):
            # Slow pan so the detector sees motion, not a single frozen still.
            shift = int(18 * np.sin(2 * np.pi * i / frames_each))
            m = np.float32([[1, 0, shift], [0, 1, shift // 3]])
            frame = cv2.warpAffine(base, m, (w, h), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    writer.release()


def _synthetic(out: Path, size=(1280, 720), fps=24, seconds=5.0) -> None:
    """Fallback clip: a field with moving blobs. Pipeline-complete, few detections."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, size)
    w, h = size
    n = int(fps * seconds)
    rng = np.random.default_rng(0)
    # A grassy gradient background.
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :, 1] = np.clip(70 + yy * 0.08, 0, 255).astype(np.uint8)
    bg[:, :, 0] = np.clip(40 + xx * 0.02, 0, 255).astype(np.uint8)
    bg[:, :, 2] = np.clip(40 + (h - yy) * 0.03, 0, 255).astype(np.uint8)
    blobs = [
        {"xy": np.array([200.0, 400.0]), "v": np.array([3.0, -0.4]), "r": 28, "c": (40, 80, 40)},
        {"xy": np.array([900.0, 500.0]), "v": np.array([-2.2, 0.6]), "r": 36, "c": (30, 60, 90)},
        {"xy": np.array([600.0, 300.0]), "v": np.array([1.4, 1.1]), "r": 22, "c": (20, 20, 20)},
    ]
    for i in range(n):
        frame = bg.copy()
        # Simple horizon / tree line.
        cv2.rectangle(frame, (0, 0), (w, 180), (90, 90, 70), -1)
        for b in blobs:
            b["xy"] = b["xy"] + b["v"]
            x, y = b["xy"]
            if x < 40 or x > w - 40:
                b["v"][0] *= -1
            if y < 200 or y > h - 40:
                b["v"][1] *= -1
            cv2.ellipse(frame, (int(x), int(y)), (b["r"] * 2, b["r"]), 0, 0, 360, b["c"], -1)
        writer.write(frame)
    writer.release()
    _ = rng  # reserved if we later jitter the blobs



def _crop_person(img: np.ndarray) -> np.ndarray | None:
    """Best-effort person-sized crop from a demo still (center-weighted)."""
    h, w = img.shape[:2]
    # zidane.jpg / bus.jpg: take a tall mid slice that usually contains a person.
    x0, x1 = int(w * 0.28), int(w * 0.72)
    y0, y1 = int(h * 0.05), int(h * 0.98)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img
    return crop


def write_threat_clip(images: list[np.ndarray], out: Path, size=(1280, 720), fps=24, seconds=6.0) -> None:
    """Move a person-like crop from the left toward the center and into a right-hand zone.

    The companion zones file (samples/threat_test_zones.yaml) covers x=0.45-0.95,
    y=0.25-0.85. The crop starts left of that box and walks into it, also closing
    on the frame center, so tracking, zone-enter, and APPROACHING can fire.
    """
    if not images:
        raise ValueError("need at least one still to build the threat test clip")
    # Prefer a standing-person still (zidane is index 1 when all three downloaded).
    src = images[1] if len(images) > 1 else images[0]
    crop = _crop_person(src)
    ch, cw = crop.shape[:2]
    target_h = int(size[1] * 0.55)
    scale = target_h / max(1, ch)
    crop = cv2.resize(crop, (max(8, int(cw * scale)), target_h), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = int(fps * seconds)
    w, h = size
    # Field + sky background.
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :, 1] = np.clip(70 + yy * 0.08, 0, 255).astype(np.uint8)
    bg[:, :, 0] = np.clip(40 + xx * 0.02, 0, 255).astype(np.uint8)
    bg[:, :, 2] = np.clip(50 + (h - yy) * 0.03, 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (0, 0), (w, 200), (90, 90, 70), -1)
    # Start left, end inside the zone and closer to center.
    x_start, x_end = 40, int(w * 0.62) - cw // 2
    y = int(h * 0.38)
    for i in range(n):
        t = i / max(1, n - 1)
        x = int(x_start + (x_end - x_start) * t)
        frame = bg.copy()
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + cw), min(h, y + ch)
        roi = frame[y1:y2, x1:x2]
        piece = crop[: roi.shape[0], : roi.shape[1]]
        frame[y1:y2, x1:x2] = piece
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create samples/demo.mp4 for Wildwatch tests.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)

    if not args.synthetic_only:
        for url in VIDEO_CANDIDATES:
            print(f"trying video {url}")
            data = _fetch(url)
            if not data or len(data) < 10_000:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            # Confirm OpenCV can actually read it.
            cap = cv2.VideoCapture(str(out))
            ok, _ = cap.read()
            cap.release()
            if ok:
                print(f"wrote {out} from remote sample video ({len(data)} bytes)")
                # Still try to build the moving-person threat clip from stills.
                stills = []
                for still_url in STILLS:
                    raw = _fetch(still_url)
                    if not raw:
                        continue
                    img = _decode_image(raw)
                    if img is not None:
                        stills.append(img)
                if stills:
                    threat_out = out.parent / "threat_test.mp4"
                    try:
                        write_threat_clip(stills, threat_out)
                        print(f"wrote {threat_out} (moving-person threat test)")
                    except Exception as exc:
                        print(f"could not write threat test clip: {exc}")
                return 0
            out.unlink(missing_ok=True)

        images: list[np.ndarray] = []
        for url in STILLS:
            print(f"trying still {url}")
            data = _fetch(url)
            if not data:
                continue
            img = _decode_image(data)
            if img is not None:
                images.append(img)
                print(f"  got {img.shape[1]}x{img.shape[0]}")
        if images:
            _write_from_stills(images, out)
            print(f"wrote {out} from {len(images)} stills")
            threat_out = out.parent / "threat_test.mp4"
            try:
                write_threat_clip(images, threat_out)
                print(f"wrote {threat_out} (moving-person threat test)")
            except Exception as exc:
                print(f"could not write threat test clip: {exc}")
            return 0
        print("could not download stills or video; falling back to a synthetic clip")

    _synthetic(out)
    print(f"wrote synthetic {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

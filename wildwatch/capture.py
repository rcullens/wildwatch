from __future__ import annotations

import time
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import cv2
import numpy as np

Source = Union[int, str]


def parse_source(value: str | int) -> Source:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def source_kind(source: Source) -> str:
    if isinstance(source, int):
        return "device"
    text = str(source)
    if text.startswith("/dev/video"):
        return "device"
    parsed = urlparse(text)
    if parsed.scheme in {"rtmp", "rtsp", "rtsps", "udp", "http", "https", "tcp"}:
        return "url"
    return "file"


def is_live_source(source: Source) -> bool:
    return source_kind(source) in {"device", "url"}


def _open(source: Source) -> cv2.VideoCapture:
    kind = source_kind(source)
    if kind == "url":
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
    elif kind == "device":
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(str(source), cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(str(source))
    else:
        cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(source))
    if cap.isOpened():
        # Keep the buffer shallow so a live stream does not lag behind reality.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class StreamCapture:
    """OpenCV capture with reconnect for live sources and optional file looping."""

    def __init__(
        self,
        source: str | int,
        reconnect_delay: float = 2.0,
        loop_file: bool = False,
        open_timeout: float = 20.0,
    ) -> None:
        self.source = parse_source(source)
        self.reconnect_delay = reconnect_delay
        self.loop_file = loop_file
        self.open_timeout = open_timeout
        self.live = is_live_source(self.source)
        self.cap: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.drops = 0
        self._ended = False
        self._open_or_raise(first=True)

    @property
    def name(self) -> str:
        return str(self.source)

    @property
    def ended(self) -> bool:
        return self._ended

    def _open_or_raise(self, first: bool = False) -> None:
        if not self.live and isinstance(self.source, str):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(
                    f"Video file not found: {path}. "
                    "Pass --source PATH, or generate a demo clip with "
                    "python scripts/make_sample_video.py"
                )
        deadline = time.time() + (self.open_timeout if first else max(self.open_timeout, 5.0))
        last_error = "unknown"
        while True:
            cap = _open(self.source)
            if cap.isOpened():
                self.cap = cap
                return
            cap.release()
            last_error = f"could not open {self.source!r}"
            if not self.live:
                raise RuntimeError(
                    f"Failed to open video source {self.source!r}. "
                    "Check the path and that ffmpeg is installed."
                )
            if first and time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out opening live source {self.source!r}. "
                    "If this is the Mini 3 RTMP feed, start scripts/start-rtmp.sh "
                    "and begin the DJI Fly livestream first. "
                    f"Last error: {last_error}"
                )
            print(f"[wildwatch] source unavailable ({last_error}); retrying in {self.reconnect_delay:.1f}s")
            time.sleep(self.reconnect_delay)

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if the file has ended."""
        if self._ended:
            return None
        assert self.cap is not None
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.frame_index += 1
            return frame
        if not self.live:
            if self.loop_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.frame_index += 1
                    return frame
            self._ended = True
            return None
        self.drops += 1
        print(f"[wildwatch] stream dropped (count={self.drops}); reconnecting")
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(self.reconnect_delay)
        self._open_or_raise(first=False)
        return self.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

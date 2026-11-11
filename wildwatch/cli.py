from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wildwatch import __version__
from wildwatch.config import apply_cli_overrides, load_config


def _default_config_path() -> Path | None:
    here = Path(__file__).resolve().parent.parent / "config.yaml"
    if here.exists():
        return here
    cwd = Path.cwd() / "config.yaml"
    return cwd if cwd.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wildwatch",
        description=(
            "Wildwatch — real-time aerial awareness and wildlife identification. "
            "Ingest a DJI Mini 3 stream (RTMP/RTSP), a capture device, or a file, "
            "and overlay Atlas-style detections."
        ),
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--source",
        "-s",
        default=None,
        help="Video source: file path, /dev/videoN, device index, or URL "
        "(rtmp://, rtsp://, udp://, http://)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["gui", "browser", "headless"],
        default=None,
        help="gui = OpenCV window, browser = Chrome dashboard, headless = no display",
    )
    parser.add_argument(
        "--detector",
        "-d",
        choices=["yolo", "megadetector"],
        default=None,
        help="yolo (default, COCO) or megadetector (recommended wildlife mode)",
    )
    parser.add_argument("--model", default=None, help="YOLO weights name or path")
    parser.add_argument(
        "--classifier",
        choices=["none", "speciesnet", "custom"],
        default=None,
        help="Species classifier. Default none labels animals as 'animal (unid.)'",
    )
    parser.add_argument("--confidence", type=float, default=None, help="Detection threshold")
    parser.add_argument("--device", default=None, help="cpu, cuda, or a device index")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        dest="frame_skip",
        help="Run detection every N+1 frames (helps on CPU)",
    )
    parser.add_argument("--host", default=None, help="Browser-mode bind address")
    parser.add_argument("--port", type=int, default=None, help="Browser-mode port")
    parser.add_argument("--no-beep", action="store_true", help="Disable audible alerts")
    parser.add_argument("--loop", action="store_true", help="Loop file sources")
    parser.add_argument("--srt", default=None, help="Optional DJI SRT sidecar for recorded video")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (headless tests). 0 = run until the source ends.",
    )
    parser.add_argument(
        "--save-annotated",
        default=None,
        help="Directory to write annotated JPEG frames (headless / tests)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Write every Nth annotated frame when --save-annotated is set",
    )
    parser.add_argument(
        "--zones-file",
        default=None,
        help="YAML file of frame-space geofences (items: [{name, polygon: [[x,y],...]}]), normalized 0-1",
    )
    parser.add_argument("--version", action="version", version=f"wildwatch {__version__}")
    return parser


def _run_gui(pipeline) -> dict:
    import cv2

    window = "WILDWATCH"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(
        "[wildwatch] GUI keys: Q/Esc quit  S snapshot  "
        "click=add geofence vertex  Z close zone  U undo  C cancel  D delete last zone"
    )

    def _norm_click(x, y, frame) -> tuple[float, float]:
        rw = rh = 0
        try:
            _rx, _ry, rw, rh = cv2.getWindowImageRect(window)
        except Exception:
            pass
        if rw > 1 and rh > 1:
            nx, ny = x / rw, y / rh
        else:
            h, w = frame.shape[:2]
            nx, ny = x / max(1, w), y / max(1, h)
        return (min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny)))

    def on_mouse(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame = pipeline.latest_frame
        if frame is None:
            return
        nx, ny = _norm_click(x, y, frame)
        draft = pipeline.zones.add_draft(nx, ny)
        print(f"[wildwatch] draft vertex ({nx:.3f}, {ny:.3f})  n={len(draft)}  (Z close, U undo, C cancel)")

    cv2.setMouseCallback(window, on_mouse)

    def on_frame(frame) -> bool:
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            return False
        if key == ord("s"):
            from datetime import datetime

            dest = Path(pipeline.cfg["paths"]["captures"])
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / f"manual_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[wildwatch] saved {path}")
        elif key in {ord("z"), 13}:  # Z or Enter closes the draft zone
            try:
                zone = pipeline.zones.close_draft()
                print(f"[wildwatch] closed geofence {zone.name!r} with {len(zone.polygon)} vertices")
            except ValueError as exc:
                print(f"[wildwatch] {exc}")
        elif key == ord("u"):
            pipeline.zones.pop_draft()
        elif key == ord("c"):
            pipeline.zones.cancel_draft()
            print("[wildwatch] cancelled draft zone")
        elif key == ord("d"):
            if pipeline.zones.zones:
                name = pipeline.zones.zones[-1].name
                pipeline.zones.remove(name)
                print(f"[wildwatch] removed zone {name!r}")
        return True

    try:
        return pipeline.run_loop(on_frame=on_frame)
    finally:
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config) if args.config else _default_config_path()
    try:
        cfg = load_config(config_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cfg = apply_cli_overrides(cfg, args)
    if args.srt:
        cfg.setdefault("telemetry", {})["srt_path"] = args.srt
    if args.zones_file:
        import yaml

        zpath = Path(args.zones_file)
        if not zpath.exists():
            print(f"error: zones file not found: {zpath}", file=sys.stderr)
            return 2
        loaded = yaml.safe_load(zpath.read_text()) or {}
        items = loaded.get("items", loaded if isinstance(loaded, list) else [])
        cfg.setdefault("zones", {}).setdefault("items", []).extend(items or [])

    from wildwatch.pipeline import Pipeline

    try:
        pipeline = Pipeline(cfg)
    except Exception as exc:
        print(f"error: failed to start pipeline: {exc}", file=sys.stderr)
        return 1

    mode = str(cfg.get("mode") or "gui").lower()
    print(
        f"[wildwatch] source={pipeline.capture.name}  detector={pipeline.detector.name}  "
        f"classifier={pipeline.classifier.name}  mode={mode}  device={pipeline.device_label}"
    )
    if mode == "browser":
        from wildwatch.server import run_browser

        host = cfg["server"]["host"]
        port = int(cfg["server"]["port"])
        run_browser(pipeline, host, port)
        return 0
    if mode == "headless":
        result = pipeline.run_loop(
            max_frames=args.max_frames,
            save_dir=args.save_annotated,
            save_every=max(1, args.save_every),
        )
        print(
            f"[wildwatch] done  frames={result['frames']}  "
            f"fps={result['fps']:.2f}  saved={result['saved']}  "
            f"recent_events={result['events']}"
        )
        return 0
    if mode == "gui":
        result = _run_gui(pipeline)
        print(
            f"[wildwatch] done  frames={result['frames']}  fps={result['fps']:.2f}"
        )
        return 0
    print(f"error: unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

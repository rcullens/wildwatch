# Wildwatch

Real-time aerial awareness for a live drone feed, with a path to wildlife species identification.

Wildwatch is a local Linux desktop app in the spirit of an Anduril Atlas-style situational-awareness overlay: bounding boxes, a compass strip, a center reticle, a telemetry panel, and alert banners. It is built to take a **DJI Mini 3** live stream on an Ubuntu laptop, run object detection on every frame, and (when you add a species model) name North American wildlife such as coyotes, whitetail deer, and feral hogs.

It runs entirely on the laptop. No cloud inference, no account, no telemetry sent off-machine.

## What works today — and what does not

Be precise about labels. The app will **never invent a species name** a model cannot actually produce.

| Mode | What it detects | Coyote / deer / hog? |
|---|---|---|
| **YOLO** (`yolo11n` / `yolov8n`, default) | COCO classes: person, vehicles, dog, cat, horse, sheep, cow, bird, bear, elephant, zebra, giraffe, … | **No.** A coyote may be boxed as `dog`. A deer or hog may be missed or mislabeled as a COCO animal. Use this to get running immediately. |
| **MegaDetector v6** (`--detector megadetector`) | `animal`, `person`, `vehicle` only. Excellent at finding an animal at a distance (it is a camera-trap detector; aerial views are related but not identical). | **Not by name.** Generic animal boxes are labeled `animal (unid.)` until you attach a classifier. This is the recommended wildlife *finder*. |
| **Classifier: none** (default) | Does not change detector labels except to rewrite MegaDetector's `animal` as `animal (unid.)`. | No species names. |
| **Classifier: SpeciesNet** (optional) | Google SpeciesNet, trained on camera-trap images including North American wildlife. | Possible, after you install it. Camera-trap photos are not the same as Mini 3 oblique/nadir footage, so treat scores as a hint. |
| **Classifier: custom** | An Ultralytics image classifier *you* trained on coyote / deer / hog / other crops. | Yes, for whatever classes are in *your* weights. This is the honest path to reliable species ID from a drone. |

Alert banners, beeps, snapshots, and the event log fire only on labels the active model really emitted, matched against the list in `config.yaml`.

## Input modes

A single `--source` argument accepts all of these:

- **File path** — offline testing, e.g. `samples/demo.mp4`
- **Device index or path** — HDMI capture dongle, e.g. `0` or `/dev/video0`
- **URL** — `rtmp://`, `rtsp://`, `udp://`, `http://`, `https://`

OpenCV reads every source with its FFmpeg backend (V4L2 for local devices). Live sources reconnect automatically if the stream drops.

## DJI Mini 3: getting the live feed onto the laptop

The Mini 3's **OcuSync radio link** (aircraft ↔ remote controller) is encrypted. Wildwatch does not decrypt, sniff, or pull that radio link. The supported path is DJI's own **Live Streaming** feature in the DJI Fly app, which **pushes** a standard, **unencrypted RTMP** stream to a URL you supply, for example `rtmp://<laptop-lan-ip>:1935/live/stream`. That is a push from the phone, not a pull from the drone.

The phone (or RC that is running DJI Fly) and the laptop must be on the **same Wi-Fi network or phone hotspot**. Open the laptop firewall for the ingest port: `sudo ufw allow 1935/tcp`. If you point DJI Fly at `127.0.0.1`, the phone will try to stream to itself — use the laptop's LAN IP.

The Mini 3 does not show up as a Linux webcam by itself. There are two practical paths.

### Path A — RTMP from DJI Fly (recommended)

Official livestreaming on the Mini 3 uses the **RC-N1** (the stick controller) plus **DJI Fly on a phone** connected to the controller over USB. The phone and the laptop must be on the same Wi-Fi network or hotspot.

The screen-equipped **DJI RC** does **not** expose DJI Fly's custom RTMP livestream path. If that is the controller you have, use Path B (HDMI capture, if you have an HDMI source) or a phone + RC-N1.

On the laptop:

```bash
cd ~/wildwatch          # or wherever you cloned this repo
./scripts/start-rtmp.sh
```

The script downloads [MediaMTX](https://github.com/bluenviron/mediamtx) (a single binary, once) and prints two URLs. Leave that terminal running.

In DJI Fly, open the camera view → **⋯** → **Transmission** → **Live Streaming Platforms** → **RTMP**. Paste the LAN URL the script printed, for example:

```
rtmp://192.168.1.42:1935/live/stream
```

Use the laptop's LAN address, not `127.0.0.1` — the phone cannot reach the laptop's localhost. If the laptop's firewall is on:

```bash
sudo ufw allow 1935/tcp
```

Start the livestream in DJI Fly, then on the laptop:

```bash
source .venv/bin/activate
python -m wildwatch --source rtmp://127.0.0.1:1935/live/stream --mode gui
# or the auto-converted RTSP twin:
python -m wildwatch --source rtsp://127.0.0.1:8554/live/stream --mode gui
```

### Path B — HDMI capture dongle (fallback)

The Mini 3 / RC-N1 has no HDMI output of its own. This path is only useful if some *other* device in the chain already exposes HDMI (a monitor loop, a capture-friendly controller, etc.). Plug that dongle into the laptop and:

```bash
# list devices
v4l2-ctl --list-devices
python -m wildwatch --source /dev/video0 --mode gui
# or
python -m wildwatch --source 0 --mode gui
```

## Install (Ubuntu 24.04)

```bash
git clone https://github.com/rcullens/wildwatch.git
cd wildwatch
./setup.sh
source .venv/bin/activate
```

`setup.sh` installs ffmpeg and a few system libraries, creates `.venv`, and `pip install`s the core requirements (Ultralytics YOLO, OpenCV, FastAPI). YOLO weights are **not** in the repo; Ultralytics downloads `yolo11n.pt` on first run.

Optional MegaDetector (recommended wildlife finder):

```bash
source .venv/bin/activate
pip install -r requirements-wildlife.txt
```

## Run

Generate a demo clip and smoke-test without a drone:

```bash
python scripts/make_sample_video.py
python -m wildwatch --source samples/demo.mp4 --mode headless --max-frames 90 --save-annotated captures/smoke
```

**GUI** (OpenCV window on the laptop). Press `q` or `Esc` to quit, `s` to snapshot:

```bash
python -m wildwatch --source samples/demo.mp4 --mode gui --loop
python -m wildwatch --source rtmp://127.0.0.1:1935/live/stream --mode gui
```

**Browser** (headless capture + a dark dashboard in Chrome at http://127.0.0.1:8080/):

```bash
python -m wildwatch --source rtmp://127.0.0.1:1935/live/stream --mode browser
# or, to test with a file:
python -m wildwatch --source samples/demo.mp4 --mode browser --loop
```

**MegaDetector** instead of stock YOLO:

```bash
python -m wildwatch --source rtmp://127.0.0.1:1935/live/stream --detector megadetector --mode gui
```

Useful flags: `--device cpu` (force CPU), `--frame-skip 2` (detect every 3rd frame if the HUD stutters on CPU), `--model yolo11s.pt` (larger YOLO, better with a GPU), `--no-beep`, `--config path/to/config.yaml`.

## Configuration

`config.yaml` is the default config; CLI flags override it. The interesting knobs:

- `detector`: `yolo` or `megadetector`
- `model`: YOLO weights (`yolo11n.pt` default; point this at your own trained detector `.pt` if you train coyote/deer/hog boxes directly)
- `megadetector.version`: `MDV6-yolov10-c` is the compact CPU variant; `MDV6-yolov10-e` is more accurate on a GPU
- `classifier`: `none` (default), `speciesnet`, or `custom`
- `alert.species`: substring match against the **actual** label. With stock YOLO, `person` and `bear` can fire; `coyote` will not until a model emits that word. With MegaDetector and no classifier, keep `animal` in the list to snapshot every animal box.
- `telemetry.srt_path`: optional DJI `.SRT` sidecar for *recorded* flights (lat/lon/alt/heading). Live Mini 3 RTMP does not include telemetry in-band; the HUD shows `---` rather than inventing a fix.
- `tracking`: ByteTrack (YOLO `model.track`) or a simple IOU tracker (MegaDetector). Persistent track IDs, trails, and motion vectors.
- `threat`: heuristic **pixel-space** score from class weight, confidence, box area (a closeness *proxy*), dwell time, and approach. This is not a geospatial intercept and does not use GPS.
- `zones.items`: frame-space geofence polygons, coordinates normalized 0–1. Draw more in the GUI (click vertices, `Z` to close) or on the browser dashboard. They are not lat/lon fences.

Snapshots of alert contacts go to `captures/` with timestamped names. A running log (`captures/events.jsonl`, `captures/events.csv`) records `alert`, `zone_enter`, `zone_exit`, and `approaching` events, including track ID and threat score. Drawn zones persist to `captures/zones.yaml`.

## Tracking, threat scoring, and geofences

- **Tracking.** YOLO uses Ultralytics built-in ByteTrack (or BoT-SORT via `tracking.backend: botsort`) through `model.track`, so IDs persist across frames. MegaDetector has no native tracker; Wildwatch runs a lightweight IOU/centroid tracker so that path still gets stable IDs, trails, and dwell time.
- **Threat score.** A weighted sum (see `threat.weights` in `config.yaml`). Class priors default to person / vehicle / hog / coyote / bear high, deer lower — but a class weight only applies when the *active model* actually emits that label. Stock YOLO will never score "coyote". Larger boxes score as "closer" only as an image-size proxy. No radio range, no GPS.
- **Approach.** A track is flagged `APPROACHING` when its motion vector consistently closes on the **frame center** or on a **drawn zone** for `approach_confirm` frames. That is image-space motion, not "heading toward the operator" in the world.
- **Geofences.** Polygons in the video frame. Zone-entry and zone-exit write event-log rows and raise a banner. Draw them live: GUI left-click vertices then `Z`; dashboard click the video then "Close zone".

GUI keys: `Q`/`Esc` quit, `S` snapshot, click to add a vertex, `Z` close zone, `U` undo vertex, `C` cancel draft, `D` delete last zone.

## Adding real species identification

Three options, in increasing order of work and of fit to *drone* footage.

### 1. Drop in SpeciesNet (camera-trap model, already trained)

```bash
pip install speciesnet
# config.yaml
classifier: speciesnet
```

or `--classifier speciesnet`. The wrapper lives in `wildwatch/classifiers/speciesnet.py`. SpeciesNet's public API has moved across releases; if a newer wheel looks different, adapt that one file. This is the fastest way to get names like "coyote" on a *camera-trap-like* crop. Overhead Mini 3 frames will be out of distribution.

### 2. Train a small crop classifier (recommended for this drone)

1. Fly and record. Run Wildwatch with `--detector megadetector` and `animal` in the alert list so every animal box is snapshotted under `captures/`.
2. Sort a few hundred crops into folders: `data/cls/coyote/`, `data/cls/deer/`, `data/cls/hog/`, `data/cls/other/`.
3. Train an Ultralytics classifier:

```bash
yolo classify train data=data/cls model=yolo11n-cls.pt epochs=50 imgsz=224
```

4. Point Wildwatch at the resulting weights:

```yaml
detector: megadetector
classifier: custom
custom_classifier:
  weights: runs/classify/train/weights/best.pt
```

Alerts for `coyote`, `deer`, and `hog` will then start to fire, because those words will be real model outputs.

### 3. Train a custom detector

If you would rather have one model that both boxes and names the animal, train YOLO on labeled Mini 3 frames (`coyote`, `deer`, `hog`, `person`, `vehicle`) and set:

```yaml
detector: yolo
model: path/to/best.pt
```

The HUD, alerts, snapshots, and event log work unchanged — they key off the label string.

To add a completely different classifier backend, implement `Classifier` in `wildwatch/classifiers/` (see `base.py`) and register it in `build_classifier`.

## Architecture

```
DJI Mini 3 ──(DJI Fly RTMP)──► MediaMTX ──► Wildwatch
HDMI dongle ── /dev/video0 ─────────────►    │
video file  ─────────────────────────────►    │
                                              ▼
                                    StreamCapture (reconnect)
                                              ▼
                          Detector (YOLO or MegaDetector)
                                              ▼
                      Classifier (none / SpeciesNet / custom)
                                              ▼
                     Tracker (ByteTrack or IOU) + trails + IDs
                                              ▼
                  Geofences + approach + heuristic threat score
                                              ▼
                         HUD + alerts + event log + snapshots
                                              ▼
                         OpenCV window    or    FastAPI / Chrome
```

The detection backends share a `Detector` interface (`wildwatch/detectors/`). The species step shares a `Classifier` interface (`wildwatch/classifiers/`). The pipeline (`wildwatch/pipeline.py`) is used by both the GUI and the browser server.

## Visual theme

The browser dashboard and the on-video HUD use the same dark cyan + purple palette as Pocket LLM and Pixel Live Translate. Tokens were copied, not approximated, from:

- `rcullens/pocket-llm-ui` `styles.css` `:root` (the source of truth; also noted there as ported from ProxyScrapeStudio)
- `rcullens/pixel-live-translate` `app/src/main/res/values/colors.xml` and `Theme.kt` (comment: "Tokens copied from pocket-llm-ui styles.css")

| Token | Hex |
|---|---|
| `--bg` | `#14122B` |
| `--panel` | `#1C1A3A` |
| `--panel2` | `#221F47` |
| `--line` | `#332E66` |
| `--txt` | `#E8E6F5` |
| `--mut` | `#9C97C9` |
| `--accent` | `#22D3EE` |
| `--purple` | `#7C4DFF` |
| `--danger` | `#F472B6` |
| `--user` | `#2A2656` |
| `--bot` | `#1C1A3A` |
| send-on | `#0B0A1A` |
| status ok | `#34D399` |
| gradient | `#22D3EE` → `#7C4DFF` |
| gradient hover | `#4FE0F3` → `#9470FF` |

Detection-category colors on the HUD are mapped only onto those existing tokens (person = accent, animal = status ok, predator = danger, game = purple, vehicle = hover cyan, other = muted). The shared definitions live in `wildwatch/theme.py`.

## Hardware notes

Wildwatch is intended to run on a CPU or iGPU. The default `yolo11n` (and MegaDetector `MDV6-yolov10-c`) are the small weights. On a CPU-only laptop, expect on the order of a few to ~15 FPS at 640 px, depending on the machine; use `--frame-skip` if the overlay stutters. A discrete NVIDIA GPU is used automatically when PyTorch can see CUDA.

## License

MIT. Third-party weights have their own licenses: Ultralytics YOLO (AGPL for the default builds), MegaDetector v6 compact YOLOv10 (AGPL), with MIT/Apache MegaDetector variants available if you need them — see the [MegaDetector model zoo](https://microsoft.github.io/MegaDetector/model_zoo/).

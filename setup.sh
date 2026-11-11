#!/usr/bin/env bash
# Install system packages and a Python venv for Wildwatch on Ubuntu 24.04.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "[wildwatch] installing system packages (sudo required for apt)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    ffmpeg v4l-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl ca-certificates
else
  echo "[wildwatch] apt-get not found; install python3, ffmpeg, and v4l-utils yourself."
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install -e .

echo
echo "[wildwatch] core install is ready."
echo "  source $ROOT/.venv/bin/activate"
echo "  python -m wildwatch --help"
echo
echo "Optional wildlife extras (MegaDetector):"
echo "  pip install -r requirements-wildlife.txt"
echo
echo "Generate a demo clip and do a headless smoke test:"
echo "  python scripts/make_sample_video.py"
echo "  python -m wildwatch --source samples/demo.mp4 --mode headless --max-frames 60 --save-annotated captures/smoke"
echo
echo "For the Mini 3 RTMP ingest:"
echo "  ./scripts/start-rtmp.sh"

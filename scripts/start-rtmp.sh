#!/usr/bin/env bash
# Download (once) and run MediaMTX so the DJI Mini 3 / DJI Fly app can
# publish a custom RTMP stream to this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/mediamtx"
CONF="$ROOT/mediamtx.yml"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) MTX_ARCH="linux_amd64" ;;
  aarch64|arm64) MTX_ARCH="linux_arm64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "[wildwatch] MediaMTX binary not found; downloading the latest release…"
  if ! command -v curl >/dev/null || ! command -v python3 >/dev/null; then
    echo "Need curl and python3 to download MediaMTX." >&2
    exit 1
  fi
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  API="https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
  TAG="$(curl -fsSL "$API" | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'])")"
  URL="https://github.com/bluenviron/mediamtx/releases/download/${TAG}/mediamtx_${TAG}_${MTX_ARCH}.tar.gz"
  echo "[wildwatch] fetching $URL"
  curl -fsSL "$URL" -o "$TMP/mediamtx.tgz"
  tar -xzf "$TMP/mediamtx.tgz" -C "$TMP"
  install -m 0755 "$TMP/mediamtx" "$BIN"
  echo "[wildwatch] installed $BIN ($TAG)"
fi

# Best-effort LAN address so the phone (DJI Fly) can reach this laptop.
LAN_IP="$(python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
except Exception:
    print("YOUR_LAN_IP")
finally:
    s.close()
PY
)"

echo
echo "=============================================================="
echo " MediaMTX is starting."
echo
echo " In DJI Fly on the phone (same Wi-Fi as this laptop):"
echo "   Camera view → ⋯ → Transmission → Live Streaming Platforms"
echo "   → RTMP, then enter:"
echo
echo "     rtmp://${LAN_IP}:1935/live/stream"
echo
echo " On this laptop, run Wildwatch against:"
echo
echo "     rtmp://127.0.0.1:1935/live/stream"
echo "     # or  rtsp://127.0.0.1:8554/live/stream"
echo
echo " Mini 3 note: official livestreaming uses the RC-N1 (or similar)"
echo " plus DJI Fly on a phone. The screen-only DJI RC does not expose"
echo " this custom RTMP path. If 1935 is firewalled:"
echo "     sudo ufw allow 1935/tcp"
echo "=============================================================="
echo

cd "$ROOT"
exec "$BIN" "$CONF"

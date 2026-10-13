#!/usr/bin/env bash
# Install the SynthFace 3D pipeline toolchain on a fresh Linux machine (Google Colab).
#
#   bash colab/setup_colab.sh [INSTALL_DIR]        # default INSTALL_DIR=/content/synthface_tools
#
# Installs exactly the versions used on the dev machine, every download checked by sha256/md5:
#   Blender 5.2.2 LTS (official Linux build)          -> INSTALL_DIR/blender-5.2.2-linux-x64/blender
#   MPFB 2.0.17 (extensions.blender.org, GPL tool)    -> ~/.config/blender/5.2/extensions/blender_org/mpfb
#   MakeHuman asset packs (CC0): system assets, skins01, skins02
#                                                     -> ~/.config/blender/5.2/extensions/.user/blender_org/mpfb/data
#   Poly Haven HDRIs (CC0, assets/hdri/hdris.json)    -> assets/hdri/
# Every asset here already has its row in assets/LICENSES.csv. Nothing else is downloaded.
# Set SKIP_APT=1 to skip the apt-get step (not root / libraries already present).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${1:-/content/synthface_tools}"
DL="$INSTALL_DIR/downloads"
BLENDER_SERIES="5.2"
BLENDER_VERSION="5.2.2"
BLENDER_DIR="$INSTALL_DIR/blender-$BLENDER_VERSION-linux-x64"
BLENDER_CONFIG="$HOME/.config/blender/$BLENDER_SERIES"
MPFB_DIR="$BLENDER_CONFIG/extensions/blender_org/mpfb"
MPFB_DATA="$BLENDER_CONFIG/extensions/.user/blender_org/mpfb/data"

# name|url|sha256
DOWNLOADS=(
  "blender-$BLENDER_VERSION-linux-x64.tar.xz|https://download.blender.org/release/Blender$BLENDER_SERIES/blender-$BLENDER_VERSION-linux-x64.tar.xz|84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"
  "add-on-mpfb-v2.0.17.zip|https://extensions.blender.org/download/sha256:4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87/add-on-mpfb-v2.0.17.zip|4f0a879d64a39bf646fbf5f53601ac678855da329d650617dca5737548239a87"
  "makehuman_system_assets_cc0.zip|https://files.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip|b542127a8e25547c7c29c19f2d1d2adb9a664c80396ecd694095dbc8028a0107"
  "skins01_cc0.zip|https://files.makehumancommunity.org/asset_packs/skins01/skins01_cc0.zip|7495ab99287053bd19ff1636114e64b608994d9f7437fea6cc75ea387f96dba9"
  "skins02_cc0.zip|https://files.makehumancommunity.org/asset_packs/skins02/skins02_cc0.zip|1613f1ef3afca53094511d26620ed7cf1d2dedc29ed3d384d60bdebe250698ae"
)

log() { printf '\n== %s\n' "$*"; }

fetch() {  # fetch NAME URL SHA256  (skips the download if a verified copy exists)
  local name="$1" url="$2" sha="$3" file="$DL/$1"
  if [[ -f "$file" ]] && echo "$sha  $file" | sha256sum -c --quiet - 2>/dev/null; then
    echo "ok (cached) $name"; return
  fi
  curl -fL --retry 3 --progress-bar -o "$file" "$url"
  echo "$sha  $file" | sha256sum -c --quiet - || { echo "SHA256 MISMATCH: $name" >&2; rm -f "$file"; exit 1; }
  echo "ok $name"
}

mkdir -p "$DL"

if [[ "${SKIP_APT:-0}" != "1" ]] && command -v apt-get >/dev/null && [[ $(id -u) -eq 0 ]]; then
  log "system libraries for headless Blender"
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    libxrender1 libxi6 libxkbcommon0 libxkbcommon-x11-0 libsm6 libice6 libgl1 libegl1 \
    libxxf86vm1 libxfixes3 libxext6 libx11-6 libxrandr2 libxinerama1 libxcursor1 libwayland-client0 >/dev/null
fi

log "downloads (sha256 checked)"
for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name url sha <<<"$entry"
  fetch "$name" "$url" "$sha"
done

log "Blender $BLENDER_VERSION"
if [[ ! -x "$BLENDER_DIR/blender" ]]; then
  tar -xJf "$DL/blender-$BLENDER_VERSION-linux-x64.tar.xz" -C "$INSTALL_DIR"
fi
"$BLENDER_DIR/blender" --version | head -1

log "MPFB 2.0.17 -> $MPFB_DIR"
rm -rf "$MPFB_DIR"
mkdir -p "$MPFB_DIR"
unzip -q -o "$DL/add-on-mpfb-v2.0.17.zip" -d "$MPFB_DIR"

log "MakeHuman asset packs -> $MPFB_DATA"
mkdir -p "$MPFB_DATA"
for pack in makehuman_system_assets_cc0.zip skins01_cc0.zip skins02_cc0.zip; do
  unzip -q -o "$DL/$pack" -d "$MPFB_DATA"
done

log "HDRIs -> assets/hdri"
python3 - "$ROOT" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
hdri_dir = Path(sys.argv[1]) / "assets" / "hdri"
for h in json.loads((hdri_dir / "hdris.json").read_text()):
    path = hdri_dir / h["file"]
    if not (path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == h["md5"]):
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(path), h["url"]], check=True)
        if hashlib.md5(path.read_bytes()).hexdigest() != h["md5"]:
            path.unlink()
            sys.exit(f"MD5 MISMATCH: {h['file']}")
    print("ok", h["file"])
EOF

log "check: MPFB loads headless and finds the assets"
cd "$ROOT"
"$BLENDER_DIR/blender" --background --python-expr "
import sys; sys.path[:0] = ['blender', '.']
import human
human.load_mpfb()
AssetService = human.service('assetservice', 'AssetService')
for kind in ('skins', 'eyes', 'eyebrows', 'eyelashes', 'hair'):
    print('ASSETS', kind, len(AssetService.list_mhclo_assets(kind) if kind != 'skins' else AssetService.list_mhmat_assets(kind)))
" 2>&1 | grep -E "^ASSETS|Error|Traceback" || true

echo
echo "BLENDER_PATH=$BLENDER_DIR/blender"

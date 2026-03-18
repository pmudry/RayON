#!/usr/bin/env bash
# Download a set of free CC0 HDRI environment maps from Poly Haven (polyhaven.com)
# All images are CC0 — no attribution required.
#
# Usage:
#   cd resources/hdri && bash download_hdri.sh [RESOLUTION]
#
#   RESOLUTION: 1k | 2k | 4k | 8k  (default: 4k)
#   Note: 8k files are ~40–100 MB each; 4k is ~10–25 MB each.
#
# Requires: curl

set -euo pipefail

RES="${1:-8k}"
BASE="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/${RES}"

STEMS=(
    "venice_sunset"
    "kloppenheim_06"
    "autumn_crossing"
    "studio_small_03"
    "sunflowers_puresky"
    "rosendal_plains_2"
)

echo "Downloading ${#STEMS[@]} HDRIs at ${RES} resolution to $(pwd)..."

for stem in "${STEMS[@]}"; do
    file="${stem}_${RES}.hdr"
    if [[ -f "$file" ]]; then
        echo "  [skip]   $file (already exists)"
    else
        echo "  [fetch]  $file"
        curl -fL --progress-bar -o "$file" "$BASE/$file"
    fi
done

echo "Done. Launch RayON with -m 3 and use Numpad +/- to cycle skies."

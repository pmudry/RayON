#!/usr/bin/env bash
set -euo pipefail

# RayON Raytracer helper script — made for this project
# Compare the two most recent image files using ImageMagick RMSE and generate a visual diff.
# Usage: compare_last_two_images.sh [--open|--no-open] [directory]
# Examples:
#   scripts/compare_last_two_images.sh
#   scripts/compare_last_two_images.sh --no-open build/res

usage() {
  cat <<EOF
RayON util -- Compare last two rendered images
Usage: $(basename "$0") [--open|--no-open] [directory]
  --open      Open newest image after comparison (default)
  --no-open   Do not open image
  directory   Directory to search (default: .)
Compares two most recent image files by mtime using ImageMagick compare -metric RMSE and writes a highlighted diff PNG.
EOF
}

# Compare the two most recent image files in a directory using ImageMagick `convert` with RMSE metric.
# Usage: compare_last_two_images.sh [directory]
# Default directory is current working directory.

OPEN=1
DIR="."
# Parse args: [--no-open|--open] [directory]
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --no-open) OPEN=0; shift ;;
    --open) OPEN=1; shift ;;
    *) DIR="$1"; shift ;;
  esac
done

if [[ ! -d "$DIR" ]]; then
  echo "Directory not found: $DIR" >&2
  exit 2
fi

# Find two newest images (non-recursive), handling spaces via line-based parsing
mapfile -t latest < <(
  find "$DIR" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \
       -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' -o -iname '*.gif' \
       -o -iname '*.ppm' -o -iname '*.pgm' -o -iname '*.pnm' -o -iname '*.exr' \) \
    -printf '%T@|%p\n' |
  sort -t'|' -k1,1nr |
  head -n 2 |
  cut -d'|' -f2-
)

if (( ${#latest[@]} < 2 )); then
  echo "Not enough image files in $DIR" >&2
  exit 3
fi

# Newest first; compare newest (0) vs second newest (1)
img_newest="${latest[0]}"
img_second="${latest[1]}"

# Ensure ImageMagick compare exists
if ! command -v compare >/dev/null 2>&1; then
  echo "ImageMagick 'compare' not found in PATH" >&2
  exit 4
fi

# Perform RMSE comparison using convert; print normalized RMSE via %[distortion]
# Tolerate non-zero exit statuses from ImageMagick by appending || true
result=$(compare "$img_second" "$img_newest" -metric RMSE -format '%[distortion]' info: 2>&1 || true)

DIR_ABS=$(cd "$DIR" && pwd)
timestamp=$(date +%Y%m%d_%H%M%S)
diff_dir="$DIR_ABS/.comparison_diffs"
mkdir -p "$diff_dir"
diff_image="$diff_dir/comparison_diff_$timestamp.png"

# Generate highlighted diff image; tolerate non-zero exit when images differ
compare "$img_second" "$img_newest" -compose src -highlight-color red -lowlight-color black "$diff_image" 2>/dev/null || true

printf 'Second newest: %s\nNewest:        %s\nRMSE:          %s\nDiff image:    %s\n' "$img_second" "$img_newest" "$result" "$diff_image"

# Optionally open newest image using xdg-open (Linux) or open (macOS)
if (( OPEN )); then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$img_newest" >/dev/null 2>&1 &
    xdg-open "$diff_image" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then
    open "$img_newest" >/dev/null 2>&1 &
    open "$diff_image" >/dev/null 2>&1 &
  fi
fi

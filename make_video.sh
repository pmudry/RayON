#!/bin/bash
# Create a high-quality H.265 video from PNG files in the res directory using ffmpeg

set -e

INPUT_DIR="$(dirname "$0")/res"
OUTPUT="output.mp4"
FRAMERATE=60

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    exit 1
fi

# Create video from PNG sequence
ffmpeg -y -framerate "$FRAMERATE" -start_number 0 -i "$INPUT_DIR/output%d.png" \
    -c:v libx265 -preset medium -crf 23 -pix_fmt yuv420p "$OUTPUT"

echo "Video created: $OUTPUT"

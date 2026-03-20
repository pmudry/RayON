#!/usr/bin/env bash
# render_comparisons.sh — Render 4 scenes × 4 sampler/MIS configurations
# Produces 16 images in images/comparisons/ for the visual-comparisons doc page.
#
# Usage:  cd <project-root> && bash scripts/render_comparisons.sh
# Requires: build/rayon already compiled

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/rayon"
OUT="$ROOT/images/comparisons"
RENDERED="$ROOT/build/rendered_images"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: $BIN not found or not executable. Build first." >&2
    exit 1
fi

mkdir -p "$OUT"

# ── Scene definitions ──────────────────────────────────────────────────────────
#   name        yaml_file                                           spp
SCENES=(
    "04_statue   resources/scenes/04_obj_statue.yaml                 64"
    "06_caustics  resources/scenes/06_caustics_chapel.yaml            512"
    "09_colorbleed resources/scenes/09_color_bleed_box.yaml           64"
    "default      resources/scenes/default_scene_no_ambient.yaml      64"
)

# ── Sampler / MIS configurations ───────────────────────────────────────────────
#   label         extra CLI flags
CONFIGS=(
    "pcg_nomis    --sampler pcg --no-mis"
    "sobol_nomis  --sampler sobol --no-mis"
    "pcg_mis      --sampler pcg"
    "sobol_mis    --sampler sobol"
)

RESOLUTION="1280x720"
total=${#SCENES[@]}
total_configs=${#CONFIGS[@]}
render_num=0
total_renders=$((total * total_configs))

for scene_line in "${SCENES[@]}"; do
    read -r scene_name scene_yaml scene_spp <<< "$scene_line"

    for config_line in "${CONFIGS[@]}"; do
        # Split: first token is label, rest are flags
        config_label="${config_line%% *}"
        config_flags="${config_line#* }"

        render_num=$((render_num + 1))
        dest="$OUT/${scene_name}_${config_label}.png"

        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "  [$render_num/$total_renders]  $scene_name  ×  $config_label"
        echo "  scene : $scene_yaml"
        echo "  spp   : $scene_spp"
        echo "  flags : $config_flags"
        echo "  dest  : $dest"
        echo "═══════════════════════════════════════════════════════════════"

        # Run the renderer
        (cd "$ROOT/build" && ./rayon -m 2 \
            -s "$scene_spp" \
            -r "$RESOLUTION" \
            --scene "../$scene_yaml" \
            $config_flags)

        # Find the most recent rendered image (newest by modification time)
        latest=$(ls -t "$RENDERED"/*.png 2>/dev/null | head -1)

        if [[ -z "$latest" ]]; then
            echo "ERROR: No rendered image found after render." >&2
            exit 1
        fi

        mv "$latest" "$dest"
        echo "  ✓ Saved $dest"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  All $total_renders renders complete.  Images in: $OUT"
echo "════════════════════════════════════════════════════════════════════"
ls -lh "$OUT"

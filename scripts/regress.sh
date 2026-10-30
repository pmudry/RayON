#!/bin/bash
# Image regression test for RayON: renders a fixed set of scenes in the offline modes and compares
# each image with its reference in tests/regression/m<mode>/<scene>.png (RMSE, ImageMagick).
# Renders are deterministic (same binary, same GPU: RMSE 0), so any difference is a real change.
#
# Usage: scripts/regress.sh [--update] [--modes "2 4"] [--tolerance 0.001] [scene ...]
#   --update      Re-render the references instead of checking against them
#   --modes       Offline modes to test (default: "2 4", or "2" when built without OptiX)
#   --tolerance   Largest normalised RMSE accepted (default: 0.001; other GPUs and drivers round
#                 slightly differently)
#   scene ...     Scene names of resources/scenes/ (default: the list below)
# Exit code: 0 when every image matches, 1 otherwise. Diff images of failures land in the
# output directory printed at the end.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
REFS="$ROOT/tests/regression"
RES="320x180"
SPP=64

SCENES=(default_scene 02_displacement_garden 04_obj_dragon 05_material_laboratory 11_soap_bubbles
        12_clearcoat_pokemonball 14_texture_test2 16_nee_small_light 17_cloud_bunny 18_sdf_shapes
        20_emissive_mesh 21_caustics)
UPDATE=0
MODES=""
TOL=0.001
ARGS_SCENES=()

while [ $# -gt 0 ]; do
   case "$1" in
   --update) UPDATE=1 ;;
   --modes) MODES="$2"; shift ;;
   --tolerance) TOL="$2"; shift ;;
   -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
   *) ARGS_SCENES+=("$1") ;;
   esac
   shift
done
[ ${#ARGS_SCENES[@]} -gt 0 ] && SCENES=("${ARGS_SCENES[@]}")

if [ ! -x "$BUILD/rayon" ]; then
   echo "No $BUILD/rayon: build first (make -C build -j)" >&2
   exit 1
fi
if ! command -v compare >/dev/null; then
   echo "ImageMagick 'compare' is required" >&2
   exit 1
fi
if [ -z "$MODES" ]; then
   MODES="2"
   [ -f "$BUILD/optix_programs.ptx" ] && MODES="2 4"
fi

OUT="$(mktemp -d "${TMPDIR:-/tmp}/rayon-regress.XXXXXX")"
cd "$BUILD" || exit 1

# The first OptiX run after a PTX change compiles the module: do it outside the timings
case " $MODES " in *" 4 "*) ./rayon -m 4 -s 1 -r 64x36 -o "$OUT/warmup.png" >/dev/null 2>&1 ;; esac

failures=0
printf "%-28s %-5s %-10s %s\n" "scene" "mode" "time" "result"
for mode in $MODES; do
   mkdir -p "$REFS/m$mode"
   for scene in "${SCENES[@]}"; do
      yaml="$ROOT/resources/scenes/$scene.yaml"
      if [ ! -f "$yaml" ]; then
         printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "-" "MISSING SCENE"
         failures=$((failures + 1))
         continue
      fi
      img="$OUT/m${mode}_$scene.png"
      start=$(date +%s.%N)
      ./rayon -m "$mode" -s "$SPP" -r "$RES" --scene "../resources/scenes/$scene.yaml" -o "$img" \
         >"$OUT/m${mode}_$scene.log" 2>&1
      secs=$(echo "$(date +%s.%N) - $start" | bc)
      t=$(printf "%.2fs" "$secs")
      ref="$REFS/m$mode/$scene.png"

      if [ ! -f "$img" ]; then
         printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "RENDER FAILED (log: $OUT/m${mode}_$scene.log)"
         failures=$((failures + 1))
      elif [ "$UPDATE" -eq 1 ]; then
         cp "$img" "$ref"
         printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "reference updated"
      elif [ ! -f "$ref" ]; then
         printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "NO REFERENCE (run with --update)"
         failures=$((failures + 1))
      else
         # compare prints "absolute (normalised)" on stderr and exits 1 when the images differ
         rmse=$(compare -metric RMSE "$ref" "$img" "$OUT/diff_m${mode}_$scene.png" 2>&1 | sed -n 's/.*(\(.*\)).*/\1/p')
         if [ -z "$rmse" ]; then
            printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "COMPARE FAILED (size changed?)"
            failures=$((failures + 1))
         elif awk -v r="$rmse" -v t="$TOL" 'BEGIN { exit !(r <= t) }'; then
            printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "ok (rmse $rmse)"
            rm -f "$OUT/diff_m${mode}_$scene.png"
         else
            printf "%-28s %-5s %-10s %s\n" "$scene" "m$mode" "$t" "DIFFERS (rmse $rmse > $TOL)"
            failures=$((failures + 1))
         fi
      fi
   done
done

echo
if [ "$failures" -eq 0 ]; then
   echo "All images match."
   rm -rf "$OUT"
   exit 0
fi
echo "$failures failure(s). Renders, logs and diff images in $OUT"
exit 1

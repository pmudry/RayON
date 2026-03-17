#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Milestone Explorer — goto_milestone.sh
#
# Checks out a historical milestone of RayON, builds the renderer, and runs it.
# Restores the original branch and stash automatically on exit.
#
# Usage:
#   ./scripts/milestones/goto_milestone.sh <N> [options]
#   ./scripts/milestones/goto_milestone.sh --list
#
# Options:
#   --offline      Force an offline render (produces a PNG) even for milestones
#                  whose default demo mode is the interactive SDL window.
#   --no-restore   Leave the repo at the milestone commit after the run.
#                  Useful for poking around. Restore manually with:
#                    git checkout <original-branch> && git stash pop
#   --help         Show this help message.
#   --list         Print all milestones and exit.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build_milestone"

# ── Milestone table ────────────────────────────────────────────────────────────
# Each milestone is described by seven parallel arrays (1-indexed; [0] unused).
#
# COMMIT        — git hash to check out
# EXECUTABLE    — name of the built binary
# MAKE_TARGET   — CMake/make target to build
# DEMO_MODE     — "offline" or "interactive"
# STDIN_CHOICE  — digit to echo to stdin for method selection (empty = no stdin)
#                 Only used in the pre-flag eras (milestones 1–7).
# RUN_ARGS      — arguments passed to the executable (for the default demo mode)
# OFFLINE_ARGS  — arguments to use when --offline is forced on an interactive ms
# DESCRIPTION   — human-readable label

declare -a    COMMIT=( ""
    "b6af112"   # 1
    "4740f42"   # 2
    "671f69d"   # 3
    "2fa8d0f"   # 4
    "fb88041"   # 5
    "c9bf459"   # 6
    "2dfd645"   # 7
    "a52ee87"   # 8
    "ada3e77"   # 9
    "264e525"   # 10
    "8623b79"   # 11
    "e383e83"   # 12
    "1f7a83d"   # 13
    "8ec565e"   # 14
    "d831935"   # 15
)

declare -a EXECUTABLE=( ""
    "v0_single_threaded"  # 1
    "v0_single_threaded"  # 2
    "v0_single_threaded"  # 3
    "v0_single_threaded"  # 4
    "302_raytracer"       # 5
    "302_raytracer"       # 6
    "302_raytracer"       # 7
    "302_raytracer"       # 8
    "302_raytracer"       # 9
    "rayon"               # 10
    "rayon"               # 11
    "rayon"               # 12
    "rayon"               # 13
    "rayon"               # 14
    "rayon"               # 15
)

declare -a MAKE_TARGET=( ""
    "v0_single_threaded"  # 1
    "v0_single_threaded"  # 2
    "v0_single_threaded"  # 3
    "v0_single_threaded"  # 4
    "302_raytracer"       # 5
    "302_raytracer"       # 6
    "302_raytracer"       # 7
    "302_raytracer"       # 8
    "302_raytracer"       # 9
    "rayon"               # 10
    "rayon"               # 11
    "rayon"               # 12
    "rayon"               # 13
    "rayon"               # 14
    "rayon"               # 15
)

# "offline" = batch render to PNG; "interactive" = SDL window
declare -a DEMO_MODE=( ""
    "offline"      # 1
    "offline"      # 2
    "offline"      # 3
    "offline"      # 4
    "offline"      # 5
    "interactive"  # 6
    "interactive"  # 7
    "interactive"  # 8
    "interactive"  # 9
    "interactive"  # 10
    "interactive"  # 11
    "interactive"  # 12
    "interactive"  # 13
    "offline"      # 14
    "offline"      # 15
)

# Digit piped to stdin for the pre-flag era interactive method menus.
# Empty ("") = binary takes no stdin input.
# "2" = select CUDA offline render; "3" = select CUDA + SDL interactive.
declare -a STDIN_CHOICE=( ""
    ""   # 1 — main() with no args, no menu
    ""   # 2 — main() with no args, no menu
    "2"  # 3 — has method menu, default CUDA
    "2"  # 4 — has method menu, default CUDA
    "2"  # 5 — has method menu (0/1/2), CUDA = 2
    "3"  # 6 — has method menu (0/1/2/3), SDL = 3
    "3"  # 7 — has method menu (0/1/2/3), SDL = 3
    ""   # 8 — has -m flag (no stdin menu)
    ""   # 9
    ""   # 10
    ""   # 11
    ""   # 12
    ""   # 13
    ""   # 14
    ""   # 15
)

# Run arguments for the default demo mode.
# __SCENE__ is a placeholder for --scene <file> (expanded below).
declare -a RUN_ARGS=( ""
    ""                                                                     # 1
    ""                                                                     # 2
    ""                                                                     # 3
    ""                                                                     # 4
    "-r 360 -s 32"                                                         # 5
    "-r 720"                                                               # 6
    "-r 720 --scene resources/bvh_test_scene.yaml"                        # 7
    "-m 3 -r 720"                                                          # 8
    "-m 3 -r 720"                                                          # 9
    "-m 3 -r 720"                                                          # 10
    "-m 3 -r 720 --scene resources/scenes/01_anisotropic_metals_test.yaml"  # 11
    "-m 3 -r 720 --scene resources/scenes/03_platonic_solids.yaml"        # 12
    "-m 3 -r 720 --scene resources/scenes/12_clearcoat_pokemonball.yaml"  # 13
    "-m 4 -r 720 -s 64"                                                    # 14
    "-m 4 -r 720 -s 64"                                                    # 15
)

# Run arguments when --offline is forced on an interactive milestone.
declare -a OFFLINE_ARGS=( ""
    ""                    # 1 — already offline
    ""                    # 2 — already offline
    ""                    # 3 — already offline
    ""                    # 4 — already offline
    ""                    # 5 — already offline
    "-r 720 -s 128"       # 6 — CUDA batch render (stdin "2" replaces "3")
    "-r 720 -s 128 --scene resources/bvh_test_scene.yaml"  # 7
    "-m 2 -r 720 -s 128"  # 8
    "-m 2 -r 720 -s 128"  # 9
    "-m 2 -r 720 -s 128"  # 10
    "-m 2 -r 720 -s 128 --scene resources/scenes/01_anisotropic_metals_test.yaml"  # 11
    "-m 2 -r 720 -s 128 --scene resources/scenes/03_platonic_solids.yaml"          # 12
    "-m 2 -r 720 -s 128 --scene resources/scenes/12_clearcoat_pokemonball.yaml"    # 13
    ""                    # 14 — already offline OptiX
    ""                    # 15 — already offline OptiX
)

# Stdin choice override for --offline mode on interactive pre-flag milestones.
declare -a OFFLINE_STDIN=( ""
    "" "" "" ""  # 1-4
    ""           # 5
    "2"          # 6 — use CUDA offline instead of SDL
    "2"          # 7
    "" "" "" "" "" "" "" ""  # 8-15
)

declare -a DESCRIPTION=( ""
    "First sphere — depth pseudo-shading (Sep 8, 2025)"           # 1
    "Normals, perspective, mirror reflections (Sep 11, 2025)"     # 2
    "First CUDA kernel — 38k rays/min (Sep 16, 2025)"             # 3
    "Golf ball procedural displacement (Sep 19, 2025)"            # 4
    "Material system — Lambertian + Metal (Sep 26, 2025)"         # 5
    "Interactive SDL2 window — progressive accumulation (Nov 5)"  # 6
    "BVH + Cornell box + SDF shapes + DOF (Nov 7, 2025)"          # 7
    "RayON 1.0 — named project, GPL licence (Nov 17, 2025)"       # 8
    "Multi-platform stabilisation — AMD64 + ARM64 (Nov 27, 2025)" # 9
    "Dear ImGui GUI, v1.5.0 — CUDA 2.24× speedup (Mar 10, 2026)"  # 10
    "Anisotropic GGX metals (Mar 13, 2026)"                        # 11
    "Triangle/OBJ pipeline — Platonic solids (Mar 13, 2026)"      # 12
    "Thin-film interference + clear-coat (Mar 13, 2026)"          # 13
    "First NVIDIA OptiX render (Mar 15, 2026)"                     # 14
    "OptiX 4× speedup on dragon mesh (Mar 15, 2026)"              # 15
)

NUM_MILESTONES=15

# ── Helper functions ────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") <N> [--offline] [--no-restore] [--help]
       $(basename "$0") --list

Check out milestone N, build, and run the RayON renderer at that point in history.

Options:
  --offline      Produce a PNG render instead of opening the SDL window.
  --no-restore   Do not restore the original branch after running.
  --list         Print all 15 milestones and exit.
  --help         Show this help.

Examples:
  $(basename "$0") 7              # Cornell box + BVH, interactive SDL
  $(basename "$0") 3 --offline    # First CUDA render, writes a PNG
  $(basename "$0") 15             # OptiX dragon mesh render
EOF
}

list_milestones() {
    printf "\n  %-4s  %-10s  %-12s  %s\n" "No." "Commit" "Mode" "Description"
    printf "  %-4s  %-10s  %-12s  %s\n" "----" "----------" "------------" "-------------------------------------------"
    for i in $(seq 1 $NUM_MILESTONES); do
        printf "  %-4s  %-10s  %-12s  %s\n" \
            "$i" "${COMMIT[$i]}" "${DEMO_MODE[$i]}" "${DESCRIPTION[$i]}"
    done
    echo ""
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ── Parse arguments ──────────────────────────────────────────────────────────

MILESTONE_N=""
FORCE_OFFLINE=false
NO_RESTORE=false

for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --list)    list_milestones; exit 0 ;;
        --offline) FORCE_OFFLINE=true ;;
        --no-restore) NO_RESTORE=true ;;
        [0-9]|[0-9][0-9]) MILESTONE_N="$arg" ;;
        *) die "Unknown argument: '$arg'. Run with --help for usage." ;;
    esac
done

[[ -z "$MILESTONE_N" ]] && { usage; exit 1; }
[[ "$MILESTONE_N" -lt 1 || "$MILESTONE_N" -gt $NUM_MILESTONES ]] && \
    die "Milestone $MILESTONE_N out of range. Valid: 1–$NUM_MILESTONES."

# ── Resolve config for the requested milestone ────────────────────────────────

MS_COMMIT="${COMMIT[$MILESTONE_N]}"
MS_EXE="${EXECUTABLE[$MILESTONE_N]}"
MS_TARGET="${MAKE_TARGET[$MILESTONE_N]}"
MS_MODE="${DEMO_MODE[$MILESTONE_N]}"
MS_STDIN="${STDIN_CHOICE[$MILESTONE_N]}"
MS_ARGS="${RUN_ARGS[$MILESTONE_N]}"
MS_DESC="${DESCRIPTION[$MILESTONE_N]}"

# Apply --offline override for interactive milestones
if $FORCE_OFFLINE && [[ "$MS_MODE" == "interactive" ]]; then
    MS_MODE="offline"
    MS_ARGS="${OFFLINE_ARGS[$MILESTONE_N]}"
    # Override stdin for pre-flag milestones 6-7
    if [[ -n "${OFFLINE_STDIN[$MILESTONE_N]}" ]]; then
        MS_STDIN="${OFFLINE_STDIN[$MILESTONE_N]}"
    fi
fi

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  RayON Milestone Explorer"
echo "  Milestone $MILESTONE_N of $NUM_MILESTONES — $MS_DESC"
echo "  Commit : $MS_COMMIT"
echo "  Mode   : $MS_MODE"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── Save git state ────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

ORIGINAL_BRANCH="$(git branch --show-current 2>/dev/null || echo "")"
ORIGINAL_HEAD="$(git rev-parse HEAD)"
STASHED=false

if ! git diff --quiet HEAD 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    echo "→ Stashing uncommitted changes..."
    git stash push --include-untracked --quiet \
        --message "goto_milestone: auto-stash before milestone $MILESTONE_N"
    STASHED=true
fi

# ── Cleanup trap ──────────────────────────────────────────────────────────────

cleanup() {
    local EXIT_CODE=$?
    if $NO_RESTORE; then
        echo ""
        echo "──────────────────────────────────────────────────────────────────"
        echo "  --no-restore set. Repository left at milestone $MILESTONE_N."
        echo "  To return to your branch, run:"
        if [[ -n "$ORIGINAL_BRANCH" ]]; then
            echo "    git checkout $ORIGINAL_BRANCH"
        else
            echo "    git checkout $ORIGINAL_HEAD"
        fi
        $STASHED && echo "    git stash pop"
        echo "──────────────────────────────────────────────────────────────────"
    else
        echo ""
        echo "→ Restoring repository state..."
        if [[ -n "$ORIGINAL_BRANCH" ]]; then
            git checkout "$ORIGINAL_BRANCH" --quiet 2>/dev/null || \
            git checkout "$ORIGINAL_HEAD" --quiet 2>/dev/null || true
        else
            git checkout "$ORIGINAL_HEAD" --quiet 2>/dev/null || true
        fi
        $STASHED && git stash pop --quiet 2>/dev/null || true
        echo "  Repository back on: ${ORIGINAL_BRANCH:-$ORIGINAL_HEAD}"
    fi
    echo ""
}

trap cleanup EXIT

# ── Checkout milestone commit ─────────────────────────────────────────────────

echo "→ Checking out $MS_COMMIT …"
git checkout "$MS_COMMIT" --quiet
echo "  Done."
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────

# Ensure output dirs used by old-era code exist
mkdir -p "$REPO_ROOT/res"

# Use a dedicated build dir so the main build is never touched
echo "→ Configuring build in build_milestone/ …"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$REPO_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=0 \
    -DENABLE_CLANG_TIDY=OFF \
    --no-warn-unused-cli \
    -Wno-dev \
    -DCMAKE_RULE_MESSAGES=OFF \
    2>&1 | grep -v "^--" | grep -Ev "^[[:space:]]*$" || true

echo ""
echo "→ Building target '$MS_TARGET' (this may take a minute) …"
make "$MS_TARGET" -j"$(nproc)" 2>&1 | tail -5
echo "  Build succeeded."
echo ""

EXE_PATH="$BUILD_DIR/$MS_EXE"
[[ -x "$EXE_PATH" ]] || die "Built binary not found at $EXE_PATH"

# ── Run ───────────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

echo "══════════════════════════════════════════════════════════════════"
if [[ "$MS_MODE" == "interactive" ]]; then
    echo "  Launching interactive SDL window."
    echo "  Controls: left-click = orbit · right-click = pan · scroll = zoom"
    echo "  Press Q or close the window to exit."
else
    echo "  Running offline render. Output will be written to the build dir"
    echo "  or to res/ (early milestones)."
fi
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Construct the command. For milestones without a -m flag, we pipe a single
# digit to stdin to select the rendering method from the interactive menu.
if [[ -n "$MS_STDIN" ]]; then
    # Pre-flag era: pipe method selection via stdin, then let SDL/render run.
    # Process substitution keeps stdin open long enough for SDL to start.
    echo "  > echo \"$MS_STDIN\" | $MS_EXE $MS_ARGS"
    echo ""
    # shellcheck disable=SC2086
    echo "$MS_STDIN" | "$EXE_PATH" $MS_ARGS
else
    echo "  > $MS_EXE $MS_ARGS"
    echo ""
    # shellcheck disable=SC2086
    "$EXE_PATH" $MS_ARGS
fi

echo ""
echo "  Renderer exited."

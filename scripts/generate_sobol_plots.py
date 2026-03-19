#!/usr/bin/env python3
"""
Generate Sobol' sampler documentation plots for website/docs/how-it-works/sobol-sampling.md.

Produces four figures:
  1. sobol_vs_pcg_2d.png     — 2D scatter: PCG vs Sobol first 256 samples
  2. sobol_convergence.png   — Integration error vs sample count (log-log)
  3. sobol_stratification.png — 1D projection per dimension (pixel AA, DOF, bounce)
  4. sobol_scramble.png      — 4 pixel hashes side-by-side showing scramble diversity

Outputs to: website/docs/assets/images/sampling/
"""

import math
import random
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
except ImportError:
    print("matplotlib / numpy required.  Run: uv pip install matplotlib numpy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Minimal Sobol' implementation matching sobol_sampler.cuh
# (CPU reference — used only for plot generation)
# ---------------------------------------------------------------------------

NBITS = 32

# Joe-Kuo 2010 table (dimensions 1..14 — enough for the plots)
_JK_DATA = [
    (1, 0,  [1]),
    (2, 1,  [1, 1]),
    (3, 1,  [1, 1, 1]),
    (3, 2,  [1, 1, 3]),
    (4, 1,  [1, 1, 3, 3]),
    (4, 4,  [1, 1, 3, 5]),
    (5, 2,  [1, 1, 1, 15, 17]),
    (5, 4,  [1, 1, 1, 3, 1]),
    (5, 7,  [1, 1, 3, 3, 9]),
    (6, 11, [1, 1, 3, 5, 27, 13]),
    (6, 13, [1, 1, 1, 5, 7, 59]),
    (6, 14, [1, 1, 3, 3, 9, 8]),
    (7, 14, [1, 1, 1, 3, 11, 53, 83]),
    (7, 16, [1, 1, 3, 1, 19, 25, 119]),
]


def _build_directions(s, a, m):
    v = [0] * (NBITS + 1)
    for i in range(1, s + 1):
        v[i] = m[i - 1] << (NBITS - i)
    for i in range(s + 1, NBITS + 1):
        v[i] = v[i - s] ^ (v[i - s] >> s)
        for k in range(1, s):
            if (a >> (s - 1 - k)) & 1:
                v[i] ^= v[i - k]
    return [v[i] & 0xFFFFFFFF for i in range(1, NBITS + 1)]


# Dimension 0: van der Corput
_vdc = [(1 << (NBITS - i)) & 0xFFFFFFFF for i in range(1, NBITS + 1)]
_DIRECTIONS = [_vdc] + [_build_directions(s, a, m) for s, a, m in _JK_DATA]


def _gray(n):
    return n ^ (n >> 1)


def sobol_uint(n, dim):
    g = _gray(n)
    result = 0
    for bit in range(NBITS):
        if (g >> bit) & 1:
            result ^= _DIRECTIONS[dim][bit]
    return result


def _lk_hash(x):
    x = (x ^ (x >> 17)) * 0xBF324C81
    x = (x ^ (x >> 11)) * 0x68BC6E26
    x = x ^ (x >> 16)
    return x & 0xFFFFFFFF


def _rev32(x):
    result = 0
    for _ in range(32):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def owen_scramble(x, seed):
    x = _rev32(x)
    x ^= x * 0x3D20ADEA
    x += seed
    x *= (seed >> 16) | 1
    x ^= x * 0x05526C56
    x ^= x * 0x53A22864
    return _rev32(x)


def sobol_float(sample_idx, dim, pixel_hash):
    raw = sobol_uint(sample_idx, dim)
    dim_seed = pixel_hash ^ (dim * 0x9E3779B9 & 0xFFFFFFFF)
    scrambled = owen_scramble(raw, dim_seed)
    return (scrambled >> 8) * (1.0 / (1 << 24))


def pcg_float(seed_ref):
    # Simple LCG as stand-in for PCG
    seed_ref[0] = (seed_ref[0] * 1664525 + 1013904223) & 0xFFFFFFFF
    return seed_ref[0] / 2**32


# ---------------------------------------------------------------------------
# Colour palette (dark theme compatible)
# ---------------------------------------------------------------------------

BLUE   = "#4C9BE8"
ORANGE = "#F08030"
GREEN  = "#58B04A"
PURPLE = "#9B59B6"
GREY   = "#888888"

plt.rcParams.update({
    "figure.facecolor": "#1E1E2E",
    "axes.facecolor":   "#181825",
    "axes.edgecolor":   "#45475A",
    "axes.labelcolor":  "#CDD6F4",
    "xtick.color":      "#CDD6F4",
    "ytick.color":      "#CDD6F4",
    "text.color":       "#CDD6F4",
    "grid.color":       "#313244",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#181825",
    "legend.edgecolor": "#45475A",
    "font.size":        11,
})

OUT = "website/docs/assets/images/sampling"
N_SAMPLES = 256


# ---------------------------------------------------------------------------
# Figure 1: 2D scatter comparison PCG vs Sobol  (dims 0 & 1)
# ---------------------------------------------------------------------------

def fig_sobol_vs_pcg():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Sample distribution — PCG vs Sobol'", fontsize=14)

    pixel_hash = 0xDEADBEEF

    xs_pcg, ys_pcg = [], []
    seed = [42]
    for _ in range(N_SAMPLES):
        xs_pcg.append(pcg_float(seed))
        ys_pcg.append(pcg_float(seed))

    xs_sob = [sobol_float(i, 0, pixel_hash) for i in range(N_SAMPLES)]
    ys_sob = [sobol_float(i, 1, pixel_hash) for i in range(N_SAMPLES)]

    for ax, xs, ys, color, title in [
        (axes[0], xs_pcg, ys_pcg, BLUE,   "PCG (pseudo-random)"),
        (axes[1], xs_sob, ys_sob, ORANGE, "Sobol' (quasi-random)"),
    ]:
        ax.scatter(xs, ys, s=6, c=color, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("dim 0  (pixel AA  x)")
        ax.set_ylabel("dim 1  (pixel AA  y)")
        ax.grid(True)

        # Draw 4×4 strata grid faintly
        for k in range(1, 4):
            ax.axhline(k / 4, color=GREY, lw=0.5, alpha=0.4)
            ax.axvline(k / 4, color=GREY, lw=0.5, alpha=0.4)

    fig.tight_layout()
    path = f"{OUT}/sobol_vs_pcg_2d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Convergence — integrating f(x,y) = sin(πx)·sin(πy) over [0,1]²
# ---------------------------------------------------------------------------

def fig_convergence():
    truth = (2 / math.pi) ** 2  # exact integral

    ns = [2**k for k in range(2, 13)]

    def estimate_pcg(n):
        seed = [7777]
        total = 0.0
        for _ in range(n):
            x = pcg_float(seed)
            y = pcg_float(seed)
            total += math.sin(math.pi * x) * math.sin(math.pi * y)
        return abs(total / n - truth)

    def estimate_sobol(n, pixel_hash=0xCAFEBABE):
        total = 0.0
        for i in range(n):
            x = sobol_float(i, 0, pixel_hash)
            y = sobol_float(i, 1, pixel_hash)
            total += math.sin(math.pi * x) * math.sin(math.pi * y)
        return abs(total / n - truth)

    errs_pcg  = [estimate_pcg(n)   for n in ns]
    errs_sob  = [estimate_sobol(n) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, errs_pcg,  "o-", color=BLUE,   label="PCG  O(N⁻¹/²)")
    ax.loglog(ns, errs_sob,  "s-", color=ORANGE, label="Sobol  O((log N)²/N)")

    # Reference slopes
    n_ref = np.array(ns, dtype=float)
    ax.loglog(n_ref, 0.5 * n_ref**(-0.5), "--", color=BLUE,   alpha=0.4, lw=1, label="N⁻¹/² slope")
    ax.loglog(n_ref, 2.0 * (np.log2(n_ref)**2) / n_ref, "--", color=ORANGE, alpha=0.4, lw=1,
              label="(log N)²/N slope")

    ax.set_xlabel("Sample count  N")
    ax.set_ylabel("Absolute integration error")
    ax.set_title("Convergence: integrating sin(πx)·sin(πy) over [0,1]²", fontsize=12)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    path = f"{OUT}/sobol_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: 1D projection per dimension — shows stratification
# ---------------------------------------------------------------------------

def fig_stratification():
    pixel_hash = 0x1234ABCD
    dims = {
        "dim 0 — pixel AA x": 0,
        "dim 1 — pixel AA y": 1,
        "dim 2 — lens u": 2,
        "dim 3 — lens v": 3,
        "dim 4 — bounce": 4,
    }

    n = 64
    fig, axes = plt.subplots(len(dims), 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"1D projections — first {n} Sobol' samples per dimension", fontsize=13)

    for ax, (label, d) in zip(axes, dims.items()):
        vals = [sobol_float(i, d, pixel_hash) for i in range(n)]
        ax.scatter(range(n), vals, s=10, c=ORANGE, alpha=0.9)
        ax.set_ylabel(label, fontsize=9, labelpad=4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.axhline(0.5, color=GREY, lw=0.5, alpha=0.3)
        ax.grid(True, axis="y")

    axes[-1].set_xlabel("Sample index  i")
    fig.tight_layout()
    path = f"{OUT}/sobol_stratification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Owen scrambling diversity — four pixel hashes side-by-side
# ---------------------------------------------------------------------------

def fig_scramble():
    hashes = [
        (0x00000000, "pixel (0,0)"),
        (0xDEADBEEF, "pixel (100,50)"),
        (0xCAFEBABE, "pixel (200,150)"),
        (0x1337C0DE, "pixel (300,200)"),
    ]

    n = 128
    fig, axes = plt.subplots(1, len(hashes), figsize=(12, 4), sharey=True)
    fig.suptitle("Owen scrambling — same Sobol' base per pixel, independent sequences", fontsize=12)

    colors = [BLUE, ORANGE, GREEN, PURPLE]

    for ax, (ph, label), color in zip(axes, hashes, colors):
        xs = [sobol_float(i, 0, ph) for i in range(n)]
        ys = [sobol_float(i, 1, ph) for i in range(n)]
        ax.scatter(xs, ys, s=5, c=color, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("dim 0")
        ax.grid(True)
        for k in range(1, 4):
            ax.axhline(k / 4, color=GREY, lw=0.4, alpha=0.3)
            ax.axvline(k / 4, color=GREY, lw=0.4, alpha=0.3)

    axes[0].set_ylabel("dim 1")
    fig.tight_layout()
    path = f"{OUT}/sobol_scramble.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)

    print("Generating Sobol' documentation plots …")
    fig_sobol_vs_pcg()
    fig_convergence()
    fig_stratification()
    fig_scramble()
    print("Done.")

#!/usr/bin/env python3

# # Example usage:
# # Full run (CUDA, ~1-2sec/thumbnail)
# python3 scripts/generate_material_gallery.py --method 2 --samples 96 --resolution 180

# # Use sphere instead of PokemonBall
# python3 scripts/generate_material_gallery.py --use-sphere --method 2

# # Different OBJ model
# python3 scripts/generate_material_gallery.py --obj-file resources/models/suzanne.obj \
#   --obj-position 0 1.0 0 --obj-scale 0.8

# # Quick test of 5 materials
# python3 scripts/generate_material_gallery.py --limit 5 --method 1 --samples 32

# # Just regenerate HTML (no re-render)
# python3 scripts/generate_material_gallery.py --skip-render
# #
"""
RayON Material Gallery Generator
==================================
Scans all YAML scene files, extracts every unique material (deduplicates by
type + visual parameters), renders a small thumbnail for each one using the
rayon binary, and produces a self-contained HTML gallery page.

Usage (run from repo root):
    python3 scripts/generate_material_gallery.py [OPTIONS]

Options:
    --scenes-dir DIR        YAML scenes directory   (default: resources/scenes)
    --rayon-bin  PATH       rayon executable         (default: build/rayon)
    --output-dir DIR        gallery output directory (default: material_gallery)
    --obj-file   PATH       OBJ model to use        (default: resources/models/PokemonBall.obj)
    --obj-position X Y Z    World position applied to model (default: 0.08 0.95 0.33)
    --obj-scale  S          Uniform scale for the model    (default: 0.35)
    --use-sphere            Use a sphere instead of an OBJ model
    --samples    N          Samples-per-pixel for thumbnails (default: 96)
    --resolution H          Vertical resolution: 180/360/720 (default: 180)
    --method     M          Rendering method 0=CPU 1=CPUpar 2=CUDA (default: 2)
    --skip-render           Skip rendering, regenerate HTML from existing thumbnails
    --title      TEXT       Gallery page title (default: "RayON Material Gallery")
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install it with:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Material fingerprinting / deduplication
# ---------------------------------------------------------------------------

# Properties that define a material's visual identity (order matters for display)
_VISUAL_KEYS = [
    "type", "albedo", "roughness", "metallic", "refractive_index",
    "transmission", "anisotropy", "preset",
    "emission", "color", "emission_intensity",
    "film_thickness", "film_ior",
    "coat_roughness",
]

# Fields to exclude from the dedup key (meta/name only)
_EXCLUDE_FROM_KEY = {"name"}


def _normalise_value(v):
    """Recursively round floats for stable hashing."""
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [_normalise_value(i) for i in v]
    return v


def material_fingerprint(mat: dict) -> str:
    """Return a stable hex digest that identifies a material's visual appearance."""
    data = {k: _normalise_value(v) for k, v in sorted(mat.items())
            if k not in _EXCLUDE_FROM_KEY}
    serialised = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialised.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# YAML scene scanning
# ---------------------------------------------------------------------------

def load_scene_materials(scene_path: Path) -> list[dict]:
    """Return the list of material dicts from a YAML scene file, or [] on error."""
    try:
        with open(scene_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            return []
        raw_mats = doc.get("materials", []) or []
        result = []
        for m in raw_mats:
            if isinstance(m, dict) and "type" in m and "name" in m:
                result.append(m)
        return result
    except Exception as e:
        print(f"  [warn] Could not parse {scene_path.name}: {e}", file=sys.stderr)
        return []


def collect_unique_materials(scenes_dir: Path) -> list[dict]:
    """
    Scan all *.yaml files in scenes_dir, collect every material definition,
    and return a deduplicated list (first occurrence wins).

    Each entry is augmented with:
        _fp       – fingerprint hex string
        _source   – source YAML filename (stem)
    """
    seen_fps: dict[str, dict] = {}  # fp -> first-seen material

    yaml_files = sorted(scenes_dir.glob("*.yaml")) + sorted(scenes_dir.glob("*.yml"))
    if not yaml_files:
        print(f"[warn] No YAML files found in {scenes_dir}", file=sys.stderr)

    for scene_path in yaml_files:
        mats = load_scene_materials(scene_path)
        for m in mats:
            fp = material_fingerprint(m)
            if fp not in seen_fps:
                entry = dict(m)
                entry["_fp"] = fp
                entry["_source"] = scene_path.stem
                seen_fps[fp] = entry
                print(f"  + {entry['type']:20s}  {entry['name']:30s}  [{scene_path.stem}]")

    return list(seen_fps.values())


# ---------------------------------------------------------------------------
# Thumbnail scene YAML generation
# ---------------------------------------------------------------------------

# Camera and lighting setup designed for a ~2-unit-tall object near the origin
_THUMBNAIL_CAMERA = {
    "position": [0.0, 2.0, 5.0],
    "look_at":  [0.0, 0.9, 0.0],
    "fov":      38,
}

_THUMBNAIL_SETTINGS = {
    "background_color":     [0.06, 0.06, 0.08],
    "background_intensity": 0.0,
    "ambient_light":        0.03,
    "use_bvh":              False,
    "adaptive_sampling":    False,
}


def _vec3_str(v) -> str:
    if isinstance(v, (list, tuple)):
        return f"[{v[0]}, {v[1]}, {v[2]}]"
    return str(v)


def _mat_to_yaml_block(mat: dict, name: str = "showcase") -> str:
    """Serialise a material dict back to YAML lines, renaming it to `name`."""
    lines = [f'  - name: "{name}"']
    for k, v in mat.items():
        if k.startswith("_") or k == "name":
            continue
        if isinstance(v, list):
            lines.append(f"    {k}: {_vec3_str(v)}")
        elif isinstance(v, str):
            lines.append(f'    {k}: "{v}"')
        elif isinstance(v, bool):
            lines.append(f"    {k}: {'true' if v else 'false'}")
        else:
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def build_thumbnail_scene(
    mat: dict,
    obj_path: str,          # absolute path or "" for sphere
    obj_position: list,     # [x, y, z]
    obj_scale: float,
    use_sphere: bool,
) -> str:
    """Return a complete YAML scene string for rendering a thumbnail of `mat`."""

    cam = _THUMBNAIL_CAMERA
    st  = _THUMBNAIL_SETTINGS

    mat_block = _mat_to_yaml_block(mat, name="showcase")

    if use_sphere or not obj_path:
        subject_geom = (
            "  - type: sphere\n"
            "    material: showcase\n"
            "    center: [0.0, 0.9, 0.0]\n"
            "    radius: 0.85\n"
        )
    else:
        sx = obj_scale
        px, py, pz = obj_position
        subject_geom = (
            f'  - type: "obj"\n'
            f'    material: "showcase"\n'
            f'    file: "{obj_path}"\n'
            f"    position: [{px}, {py}, {pz}]\n"
            f"    scale: [{sx}, {sx}, {sx}]\n"
        )

    return f"""\
# Auto-generated thumbnail scene — do not edit
camera:
  position: {_vec3_str(cam["position"])}
  look_at: {_vec3_str(cam["look_at"])}
  fov: {cam["fov"]}

settings:
  background_color: {_vec3_str(st["background_color"])}
  background_intensity: {st["background_intensity"]}
  ambient_light: {st["ambient_light"]}
  use_bvh: {"true" if st["use_bvh"] else "false"}
  adaptive_sampling: {"true" if st["adaptive_sampling"] else "false"}

materials:
{_mat_to_yaml_block({"type": "lambertian", "albedo": [0.28, 0.28, 0.30]}, "ground")}

  - name: "key_light"
    type: "light"
    emission: [22.0, 19.0, 14.0]

  - name: "fill_light"
    type: "light"
    emission: [7.0, 7.0, 9.0]

{mat_block}

geometry:
  # Ground plane
  - type: rectangle
    material: ground
    corner: [-500, -1.0, -500]
    u: [1000, 0, 0]
    v: [0, 0, 1000]

  # Key light — overhead, warm, slightly in front
  - type: rectangle
    material: key_light
    corner: [-2.0, 5.0, -1.5]
    u: [4.0, 0, 0]
    v: [0, 0, 2.5]
    visible: false

  # Fill light — behind camera, cool
  - type: rectangle
    material: fill_light
    corner: [-1.5, 2.0, 4.5]
    u: [3.0, 0, 0]
    v: [0, 0, 1.0]
    visible: false

  # Subject object
{subject_geom}
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_thumbnail(
    scene_yaml: str,
    rayon_bin: Path,
    build_dir: Path,
    samples: int,
    resolution: int,
    method: int,
) -> Path | None:
    """
    Write scene_yaml to a temp file, invoke rayon, and return the path to
    the rendered PNG (build_dir/rendered_images/latest.png).
    Returns None on failure.
    """
    # Write temp scene file in the build directory so relative paths inside
    # the scene (if any) resolve from there.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=build_dir, delete=False, encoding="utf-8"
    ) as tf:
        tf.write(scene_yaml)
        tmp_scene = tf.name

    try:
        cmd = [
            str(rayon_bin),
            "--scene", tmp_scene,
            "-m", str(method),
            "-s", str(samples),
            "-r", str(resolution),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(build_dir),
        )
        if result.returncode != 0:
            print(f"\n  [error] rayon exited {result.returncode}", file=sys.stderr)
            print(result.stderr[-800:], file=sys.stderr)
            return None

        latest = build_dir / "rendered_images" / "latest.png"
        if not latest.exists():
            print("  [error] rayon succeeded but latest.png not found", file=sys.stderr)
            return None
        return latest

    finally:
        os.unlink(tmp_scene)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg:      #0d0d0f;
    --surface: #17171b;
    --border:  #2a2a32;
    --text:    #d8d8e0;
    --sub:     #888896;
    --accent:  #5b8af5;
    --tag-bg:  #1e2235;
    --tag-txt: #7aa4f0;
    --radius:  10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    padding: 2rem;
  }}

  header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}
  header h1 {{
    font-size: 1.9rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
  }}
  header p {{ color: var(--sub); margin-top: 0.35rem; }}

  /* Filter bar */
  .filters {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.8rem;
    align-items: center;
  }}
  .filters span {{ color: var(--sub); margin-right: 0.4rem; font-size: 0.85rem; }}
  .filter-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--sub);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    cursor: pointer;
    font-size: 0.82rem;
    transition: all 0.15s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    background: var(--tag-bg);
    color: var(--tag-txt);
    border-color: var(--accent);
  }}

  /* Search */
  .search-wrap {{
    flex: 1;
    max-width: 300px;
    margin-left: auto;
  }}
  .search-wrap input {{
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 20px;
    padding: 0.35rem 1rem;
    font-size: 0.85rem;
    outline: none;
  }}
  .search-wrap input::placeholder {{ color: var(--sub); }}

  /* Grid */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1.2rem;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform 0.15s, border-color 0.15s;
    cursor: pointer;
  }}
  .card:hover {{
    transform: translateY(-3px);
    border-color: var(--accent);
  }}

  .card-thumb {{
    width: 100%;
    aspect-ratio: 16/9;
    object-fit: cover;
    display: block;
    background: #111;
  }}
  .card-thumb.missing {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: #333;
  }}

  .card-body {{
    padding: 0.75rem;
  }}
  .card-name {{
    font-weight: 600;
    font-size: 0.88rem;
    color: #fff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.3rem;
  }}
  .card-type {{
    display: inline-block;
    background: var(--tag-bg);
    color: var(--tag-txt);
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 4px;
    padding: 1px 7px;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .card-params {{
    font-size: 0.76rem;
    color: var(--sub);
  }}
  .card-params span {{ display: block; }}
  .card-source {{
    font-size: 0.7rem;
    color: #444;
    margin-top: 0.4rem;
  }}

  /* Modal */
  .modal-overlay {{
    display: none;
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.82);
    z-index: 100;
    align-items: center;
    justify-content: center;
  }}
  .modal-overlay.open {{ display: flex; }}
  .modal {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    max-width: 720px;
    width: 90vw;
    overflow: hidden;
  }}
  .modal img {{
    width: 100%;
    display: block;
  }}
  .modal-info {{
    padding: 1.2rem 1.5rem;
  }}
  .modal-info h2 {{ font-size: 1.2rem; color: #fff; }}
  .modal-type {{
    display: inline-block;
    background: var(--tag-bg);
    color: var(--tag-txt);
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 4px;
    padding: 2px 10px;
    margin: 0.5rem 0;
    text-transform: uppercase;
  }}
  .modal-details {{
    font-size: 0.82rem;
    color: var(--sub);
    margin-top: 0.6rem;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.2rem 1rem;
  }}
  .modal-details .key {{ color: var(--text); font-weight: 500; }}
  .modal-close {{
    float: right;
    background: none;
    border: none;
    color: var(--sub);
    font-size: 1.4rem;
    cursor: pointer;
    margin-top: -0.2rem;
  }}
  .modal-close:hover {{ color: #fff; }}

  footer {{
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    color: var(--sub);
    font-size: 0.78rem;
    text-align: center;
  }}
</style>
</head>
<body>

<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>

<div class="filters">
  <span>Filter by type:</span>
  <button class="filter-btn active" data-type="all" onclick="filterType(this,'all')">All ({total})</button>
{type_buttons}
  <div class="search-wrap">
    <input type="text" id="search" placeholder="Search name…" oninput="filterSearch(this.value)" />
  </div>
</div>

<div class="grid" id="mat-grid">
{cards}
</div>

<!-- Modal -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal" id="modal-inner">
    <img id="modal-img" src="" alt="">
    <div class="modal-info">
      <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">✕</button>
      <h2 id="modal-name"></h2>
      <span class="modal-type" id="modal-type"></span>
      <div class="modal-details" id="modal-details"></div>
    </div>
  </div>
</div>

<footer>
  Generated by RayON generate_material_gallery.py &mdash; {timestamp}
  &mdash; {total} unique materials from {num_scenes} scene files
</footer>

<script>
const allCards = Array.from(document.querySelectorAll('.card'));
let currentType = 'all';
let currentSearch = '';

function filterType(btn, type) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentType = type;
  applyFilter();
}}

function filterSearch(val) {{
  currentSearch = val.toLowerCase();
  applyFilter();
}}

function applyFilter() {{
  allCards.forEach(c => {{
    const matchType   = currentType === 'all' || c.dataset.type === currentType;
    const matchSearch = !currentSearch || c.dataset.name.includes(currentSearch);
    c.style.display = (matchType && matchSearch) ? '' : 'none';
  }});
}}

function openModal(fp) {{
  const data = window._matData[fp];
  if (!data) return;
  document.getElementById('modal-img').src  = data.thumb;
  document.getElementById('modal-name').textContent = data.name;
  document.getElementById('modal-type').textContent = data.type;
  const det = document.getElementById('modal-details');
  det.innerHTML = '';
  for (const [k, v] of Object.entries(data.params)) {{
    det.innerHTML += `<span class="key">${{k}}</span><span>${{JSON.stringify(v)}}</span>`;
  }}
  det.innerHTML += `<span class="key">source</span><span>${{data.source}}</span>`;
  document.getElementById('modal').classList.add('open');
}}

function closeModal(e) {{
  if (e.target.id === 'modal') document.getElementById('modal').classList.remove('open');
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('modal').classList.remove('open');
}});

window._matData = {mat_data_json};
</script>
</body>
</html>
"""


def _colour_swatch(albedo) -> str:
    """Return a CSS hex colour from an albedo list, or empty string."""
    if isinstance(albedo, list) and len(albedo) == 3:
        r, g, b = [min(255, int(round(c * 255))) for c in albedo]
        return f"#{r:02x}{g:02x}{b:02x}"
    return ""


def _param_summary(mat: dict) -> list[tuple[str, str]]:
    """Return a short list of (key, display_value) for the card."""
    skip = {"name", "type", "_fp", "_source"}
    items = []
    for k in _VISUAL_KEYS:
        if k in mat and k not in skip:
            v = mat[k]
            if isinstance(v, list):
                items.append((k, f"[{', '.join(str(round(x,3)) if isinstance(x,float) else str(x) for x in v)}]"))
            elif isinstance(v, float):
                items.append((k, str(round(v, 4))))
            else:
                items.append((k, str(v)))
    return items[:5]  # cap at 5 lines per card


def build_html(
    materials: list[dict],
    thumbnails: dict[str, str],   # fp -> relative path from HTML file
    title: str,
    num_scenes: int,
) -> str:
    from datetime import datetime

    all_types = sorted({m.get("type", "unknown") for m in materials})
    total = len(materials)

    # Type filter buttons
    type_btns = []
    for t in all_types:
        count = sum(1 for m in materials if m.get("type") == t)
        type_btns.append(
            f'  <button class="filter-btn" data-type="{t}" onclick="filterType(this,\'{t}\')">'
            f"{t} ({count})</button>"
        )

    # Cards + JS data
    mat_data = {}
    card_html = []

    for mat in materials:
        fp    = mat["_fp"]
        name  = mat.get("name", fp)
        mtype = mat.get("type", "unknown")
        thumb = thumbnails.get(fp, "")
        params = {k: mat[k] for k in _VISUAL_KEYS if k in mat and k not in {"name", "type"}}

        mat_data[fp] = {
            "name":   name,
            "type":   mtype,
            "thumb":  thumb,
            "source": mat.get("_source", ""),
            "params": params,
        }

        # Thumb HTML
        if thumb:
            thumb_html = f'<img class="card-thumb" src="{thumb}" alt="{name}" loading="lazy">'
        else:
            thumb_html = '<div class="card-thumb missing">🎨</div>'

        # Param lines
        param_lines = "".join(
            f"<span>{k}: <b>{v}</b></span>"
            for k, v in _param_summary(mat)
        )

        # Colour dot for albedo
        swatch = _colour_swatch(mat.get("albedo"))
        dot = (f'<span style="display:inline-block;width:10px;height:10px;'
               f'border-radius:50%;background:{swatch};'
               f'vertical-align:middle;margin-right:4px"></span>') if swatch else ""

        card_html.append(
            f'  <div class="card" data-type="{mtype}" data-name="{name.lower()}" '
            f'onclick="openModal(\'{fp}\')">\n'
            f"    {thumb_html}\n"
            f'    <div class="card-body">\n'
            f'      <div class="card-name">{dot}{name}</div>\n'
            f'      <span class="card-type">{mtype}</span>\n'
            f'      <div class="card-params">{param_lines}</div>\n'
            f'      <div class="card-source">{mat.get("_source","")}.yaml</div>\n'
            f"    </div>\n"
            f"  </div>"
        )

    unique_scenes = num_scenes
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subtitle = (
        f"{total} unique materials &mdash; types: "
        + ", ".join(f"<b>{t}</b>" for t in all_types)
    )

    return _HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        total=total,
        num_scenes=unique_scenes,
        type_buttons="\n".join(type_btns),
        cards="\n".join(card_html),
        mat_data_json=json.dumps(mat_data, indent=2),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a material gallery from RayON YAML scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--scenes-dir",    default="resources/scenes",
                   help="Directory containing .yaml scene files (default: resources/scenes)")
    p.add_argument("--rayon-bin",     default="build/rayon",
                   help="Path to rayon binary (default: build/rayon)")
    p.add_argument("--output-dir",    default="material_gallery",
                   help="Output directory for the gallery (default: material_gallery)")
    p.add_argument("--obj-file",      default="resources/models/PokemonBall.obj",
                   help="OBJ model for thumbnails (default: resources/models/PokemonBall.obj)")
    p.add_argument("--obj-position",  nargs=3, type=float, default=[0.08, 0.95, 0.33],
                   metavar=("X", "Y", "Z"),
                   help="World position applied to the OBJ model (default: 0.08 0.95 0.33)")
    p.add_argument("--obj-scale",     type=float, default=0.35,
                   help="Uniform scale for the OBJ model (default: 0.35)")
    p.add_argument("--use-sphere",    action="store_true",
                   help="Use a unit sphere instead of an OBJ model")
    p.add_argument("--samples",       type=int, default=96,
                   help="Samples per pixel for thumbnails (default: 96)")
    p.add_argument("--resolution",    type=int, default=180, choices=[180, 360, 720],
                   help="Vertical resolution for thumbnails (default: 180)")
    p.add_argument("--method",        type=int, default=2, choices=[0, 1, 2],
                   help="Rendering method: 0=CPU, 1=CPUpar, 2=CUDA (default: 2)")
    p.add_argument("--skip-render",   action="store_true",
                   help="Skip rendering, regenerate HTML from existing thumbnails")
    p.add_argument("--limit",         type=int, default=0,
                   help="Limit rendering to first N materials (0 = all, useful for quick tests)")
    p.add_argument("--title",         default="RayON Material Gallery",
                   help='Gallery page title (default: "RayON Material Gallery")')
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scenes_dir = (repo_root / args.scenes_dir).resolve()
    rayon_bin  = (repo_root / args.rayon_bin).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    build_dir  = rayon_bin.parent.resolve()

    obj_file_rel = args.obj_file
    if not args.use_sphere:
        obj_path = (repo_root / obj_file_rel).resolve()
        if not obj_path.exists():
            print(f"[warn] OBJ file not found: {obj_path}", file=sys.stderr)
            print("       Falling back to sphere.", file=sys.stderr)
            args.use_sphere = True
            obj_abs = ""
        else:
            obj_abs = str(obj_path)
    else:
        obj_abs = ""

    if not args.skip_render and not rayon_bin.exists():
        print(f"[error] rayon binary not found at: {rayon_bin}", file=sys.stderr)
        print("        Build it first with:  cd build && make -j8", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = output_dir / "thumbnails"
    thumbs_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    print("\n=== Scanning YAML scenes ===")
    materials = collect_unique_materials(scenes_dir)
    print(f"\nFound {len(materials)} unique materials across all scenes.\n")

    if not materials:
        print("[error] No materials found. Check --scenes-dir path.", file=sys.stderr)
        sys.exit(1)

    # Count source scenes
    source_scenes = {m["_source"] for m in materials}

    # ------------------------------------------------------------------
    thumbnails: dict[str, str] = {}   # fp -> relative path from HTML

    if not args.skip_render:
        render_list = materials[:args.limit] if args.limit > 0 else materials
        # Filter out already-cached thumbnails to estimate remaining time
        pending = [m for m in render_list if not (thumbs_dir / f"{m['_fp']}.png").exists()]
        print("=== Rendering thumbnails ===")
        if pending:
            avg_secs = max(5, args.samples // 8)  # rough estimate
            est_min  = len(pending) * avg_secs // 60
            print(f"  {len(pending)} thumbnails to render"
                  f" (~{avg_secs}s each with -m{args.method}, est. {est_min}+ min)")
            print("  Tip: use --method 2 (CUDA) for faster renders,"
                  " or --limit N to test a subset.\n")
        total = len(render_list)
        for i, mat in enumerate(render_list, 1):
            fp   = mat["_fp"]
            name = mat.get("name", fp)
            mtype = mat.get("type", "unknown")

            out_png = thumbs_dir / f"{fp}.png"

            # If thumb already exists, skip re-render
            if out_png.exists():
                print(f"  [{i:3d}/{total}]  (cached)  {mtype:20s}  {name}")
                thumbnails[fp] = f"thumbnails/{fp}.png"
                continue

            print(f"  [{i:3d}/{total}]  rendering  {mtype:20s}  {name} … ", end="", flush=True)
            t0 = time.time()

            scene_yaml = build_thumbnail_scene(
                mat,
                obj_path=obj_abs,
                obj_position=args.obj_position,
                obj_scale=args.obj_scale,
                use_sphere=args.use_sphere,
            )

            latest = render_thumbnail(
                scene_yaml=scene_yaml,
                rayon_bin=rayon_bin,
                build_dir=build_dir,
                samples=args.samples,
                resolution=args.resolution,
                method=args.method,
            )

            if latest:
                shutil.copy2(latest, out_png)
                elapsed = time.time() - t0
                print(f"done ({elapsed:.1f}s)")
                thumbnails[fp] = f"thumbnails/{fp}.png"
            else:
                print("FAILED")

    else:
        print("=== --skip-render set, scanning existing thumbnails ===")
        for mat in materials:
            fp = mat["_fp"]
            out_png = thumbs_dir / f"{fp}.png"
            if out_png.exists():
                thumbnails[fp] = f"thumbnails/{fp}.png"

    # ------------------------------------------------------------------
    print("\n=== Generating HTML gallery ===")
    html = build_html(
        materials=materials,
        thumbnails=thumbnails,
        title=args.title,
        num_scenes=len(source_scenes),
    )

    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    rendered_count = len(thumbnails)
    failed_count   = len(materials) - rendered_count
    print(f"  Written:  {html_path}")
    print(f"  Thumbnails: {rendered_count} rendered, {failed_count} missing")
    print(f"\nOpen in browser:  {html_path}\n")


if __name__ == "__main__":
    main()

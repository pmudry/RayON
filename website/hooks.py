"""
MkDocs hook — copy image assets from the repo root into docs/assets/images/
before every build (local `mkdocs serve/build` and CI alike).
"""
import os
import shutil


def on_pre_build(config, **kwargs):
    docs_dir = config["docs_dir"]
    # docs_dir is <repo>/website/docs — go up two levels to reach repo root
    repo_root = os.path.dirname(os.path.dirname(docs_dir))

    dest_base = os.path.join(docs_dir, "assets", "images")

    copies = [
        # (source path relative to repo root, destination path relative to docs/assets/images/)
        ("images/samples",                        "samples"),
        ("images/for_project",                    "for_project"),
        ("images/dev",                            "dev"),
        ("material_gallery/thumbnails",           "thumbnails"),
        ("explanations/lambert sampling",         "sampling"),
    ]

    single_files = [
        ("images/real_time_raytrace.png", "real_time_raytrace.png"),
    ]

    for src_rel, dst_rel in copies:
        src = os.path.join(repo_root, src_rel)
        dst = os.path.join(dest_base, dst_rel)
        if not os.path.isdir(src):
            continue
        for dirpath, _dirnames, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            dest_dir = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(dest_dir, exist_ok=True)
            for fname in filenames:
                if fname.lower().endswith(".png"):
                    shutil.copy2(os.path.join(dirpath, fname), os.path.join(dest_dir, fname))

    for src_rel, dst_rel in single_files:
        src = os.path.join(repo_root, src_rel)
        dst = os.path.join(dest_base, dst_rel)
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

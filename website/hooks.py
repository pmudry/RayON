"""
MkDocs hooks:
  on_pre_build  — copy image assets from repo root into docs/assets/images/
  on_page_content — unwrap glightbox <a> wrappers inside img-comparison-slider
                    elements so the web component slots work correctly.
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
        ("images/comparisons",                    "comparisons"),
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


def on_post_page(output, page, config, **kwargs):
    """
    The glightbox plugin wraps every <img> in an <a class="glightbox"> at
    build time (in its own on_post_page handler).  Inside <img-comparison-slider>
    this breaks the web component: it expects <img slot="first"> and
    <img slot="second"> as direct children, but finds <a> wrappers instead.

    This hook also runs in on_post_page but is loaded after all plugins (hooks
    are last in the MkDocs event chain), so it sees the glightbox-modified HTML
    and can unwrap those <a> tags inside slider elements.

    Uses selectolax which is already a transitive dependency of mkdocs-glightbox.
    """
    if "img-comparison-slider" not in output:
        return output  # Fast path — skip pages that have no sliders

    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(output)
    changed = False

    for slider in tree.css("img-comparison-slider"):
        for a_tag in slider.css("a.glightbox"):
            img = a_tag.css_first("img")
            if img:
                a_tag.replace_with(img)
                changed = True

    return tree.html if changed else output

#!/usr/bin/env python3
"""Generate a simple grid texture PNG without external dependencies.

Writes a 512x512 RGBA PNG with a 16x16 grid pattern.
Usage: python3 scripts/generate_grid_texture.py
Output: resources/textures/grid_512.png
"""

import struct
import zlib
import os

def write_png(filename, width, height, pixels):
    """Write RGBA pixels (flat list of RGBA tuples) as a PNG file."""
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(c[4:]) & 0xFFFFFFFF)

    # Build raw image data: one filter byte (0=none) per row
    raw_rows = []
    for y in range(height):
        row = b'\x00'  # filter type None
        for x in range(width):
            r, g, b, a = pixels[y * width + x]
            row += bytes([r, g, b, a])
        raw_rows.append(row)

    compressed = zlib.compress(b''.join(raw_rows), 9)

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    png += chunk(b'IDAT', compressed)
    png += chunk(b'IEND', b'')

    with open(filename, 'wb') as f:
        f.write(png)
    print(f"Written: {filename}  ({width}x{height})")


def make_grid_texture(width=512, height=512, cells=16, line_frac=0.08):
    """Generate a grid texture.

    Colors:
    - Cell interiors: alternating light-grey (#e0e0e0) and white (#f8f8f8)
    - Grid lines: dark-grey (#333333)
    """
    pixels = []
    cell_w = width / cells
    cell_h = height / cells
    line_px_x = max(1, int(cell_w * line_frac))
    line_px_y = max(1, int(cell_h * line_frac))

    DARK   = (51,  51,  51,  255)
    LIGHT  = (224, 224, 224, 255)
    LIGHT2 = (248, 248, 248, 255)

    for y in range(height):
        for x in range(width):
            # Position within cell
            cx = int(x / cell_w)
            cy = int(y / cell_h)
            lx = x - int(cx * cell_w)
            ly = y - int(cy * cell_h)
            cw = int((cx + 1) * cell_w) - int(cx * cell_w)
            ch = int((cy + 1) * cell_h) - int(cy * cell_h)

            on_line_x = lx < line_px_x or lx >= cw - line_px_x
            on_line_y = ly < line_px_y or ly >= ch - line_px_y

            if on_line_x or on_line_y:
                pixels.append(DARK)
            elif (cx + cy) % 2 == 0:
                pixels.append(LIGHT)
            else:
                pixels.append(LIGHT2)

    return pixels


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'textures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'grid_512.png')

    pixels = make_grid_texture(512, 512, cells=16, line_frac=0.06)
    write_png(out_path, 512, 512, pixels)

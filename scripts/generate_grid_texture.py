#!/usr/bin/env python3
"""Generate grid texture PNGs without external dependencies.

Outputs:
  resources/textures/grid_512.png   (512x512,   fast reference)
  resources/textures/grid_4k.png    (4096x4096, high-res default)

Usage: python3 scripts/generate_grid_texture.py
"""

import struct
import zlib
import os


def write_png(filename, width, height, pixels_rgba):
    """Write a flat RGBA bytearray as a PNG file."""
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(c[4:]) & 0xFFFFFFFF)

    # Build raw scanlines: each row prefixed with filter byte 0 (None)
    row_size = width * 4
    raw = bytearray(height * (1 + row_size))
    for y in range(height):
        base = y * (1 + row_size)
        raw[base] = 0  # filter = None
        raw[base + 1 : base + 1 + row_size] = pixels_rgba[y * row_size : (y + 1) * row_size]

    compressed = zlib.compress(bytes(raw), 6)  # level 6: good ratio, not too slow

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
    png += chunk(b'IDAT', compressed)
    png += chunk(b'IEND', b'')

    with open(filename, 'wb') as f:
        f.write(png)
    print(f'Written: {filename}  ({width}x{height})')


def make_grid_texture(width, height, cells=16, line_frac=0.06):
    """Return a flat RGBA bytearray for a grid texture.

    Colors:
      Cell interiors: alternating light-grey (#e0e0e0) and white (#f8f8f8)
      Grid lines:     dark-grey (#333333)
    """
    cell_w = width / cells
    cell_h = height / cells
    line_px_x = max(1, int(cell_w * line_frac))
    line_px_y = max(1, int(cell_h * line_frac))

    DARK   = bytes([51,  51,  51,  255])
    LIGHT  = bytes([224, 224, 224, 255])
    LIGHT2 = bytes([248, 248, 248, 255])

    buf = bytearray(width * height * 4)

    for y in range(height):
        cy = int(y / cell_h)
        ly = y - int(cy * cell_h)
        ch = int((cy + 1) * cell_h) - int(cy * cell_h)
        on_line_y = ly < line_px_y or ly >= ch - line_px_y

        row_base = y * width * 4
        for x in range(width):
            cx = int(x / cell_w)
            lx = x - int(cx * cell_w)
            cw_px = int((cx + 1) * cell_w) - int(cx * cell_w)
            on_line_x = lx < line_px_x or lx >= cw_px - line_px_x

            if on_line_x or on_line_y:
                color = DARK
            elif (cx + cy) % 2 == 0:
                color = LIGHT
            else:
                color = LIGHT2

            buf[row_base + x * 4 : row_base + x * 4 + 4] = color

    return buf


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'textures')
    os.makedirs(out_dir, exist_ok=True)

    # 512x512 reference
    buf = make_grid_texture(512, 512, cells=16, line_frac=0.06)
    write_png(os.path.join(out_dir, 'grid_512.png'), 512, 512, buf)

    # 4K high-res
    import time
    t0 = time.time()
    buf4k = make_grid_texture(4096, 4096, cells=16, line_frac=0.04)
    write_png(os.path.join(out_dir, 'grid_4k.png'), 4096, 4096, buf4k)
    print(f'4K generation took {time.time() - t0:.1f}s')

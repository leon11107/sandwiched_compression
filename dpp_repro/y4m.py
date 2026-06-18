"""Minimal correct y4m reader (yuvio.get_reader treats .y4m as HEADERLESS raw
yuv420p — the 60-byte header + 6-byte per-frame FRAME markers land in the
pixel stream and shift every frame; found 2026-06-11 via byte-level diff)."""
from __future__ import annotations
import numpy as np


def read_y4m(path):
    """Yields (y, u, v) uint8 planes; asserts 4:2:0 8-bit."""
    with open(path, "rb") as f:
        header = f.readline().decode()
        fields = dict((t[0], t[1:]) for t in header.split()[1:] if t)
        w, h = int(fields["W"]), int(fields["H"])
        assert fields.get("C", "420jpeg").startswith("420"), header
        ysz, csz = w * h, (w // 2) * (h // 2)
        while True:
            marker = f.readline()
            if not marker:
                return
            assert marker.startswith(b"FRAME"), marker[:20]
            buf = f.read(ysz + 2 * csz)
            if len(buf) < ysz + 2 * csz:
                return
            a = np.frombuffer(buf, np.uint8)
            yield (a[:ysz].reshape(h, w),
                   a[ysz:ysz + csz].reshape(h // 2, w // 2),
                   a[ysz + csz:].reshape(h // 2, w // 2))


def y4m_header(path):
    with open(path, "rb") as f:
        return f.readline()

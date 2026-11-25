# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from echo_control import art


def _parse_png_ihdr(data: bytes) -> tuple[int, int, int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    length, chunk_type = struct.unpack(">I4s", data[8:16])
    assert chunk_type == b"IHDR"
    assert length == 13
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return width, height, bit_depth, color_type


def test_project_trajectory_bounds_and_shape() -> None:
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(16, 64)).astype(np.float32)
    pts = art.project_trajectory(vecs)
    assert pts.shape == (16, 2)
    assert float(pts.min()) >= 0.0
    assert float(pts.max()) <= 1.0


def test_project_trajectory_handles_single_and_empty() -> None:
    single = art.project_trajectory(np.ones((1, 16), dtype=np.float32))
    assert single.shape == (1, 2)
    assert np.allclose(single, [[0.5, 0.5]])

    empty = art.project_trajectory(np.zeros((0, 16), dtype=np.float32))
    assert empty.shape == (1, 2)
    assert np.allclose(empty, [[0.5, 0.5]])


def test_latent_palette_is_deterministic_and_varies_with_vectors() -> None:
    rng = np.random.default_rng(7)
    m1 = rng.normal(size=32).astype(np.float32)
    m2 = rng.normal(size=32).astype(np.float32)
    p1a = art.latent_palette(m1, m1, m1)
    p1b = art.latent_palette(m1, m1, m1)
    p2 = art.latent_palette(m2, m2, m2)
    assert p1a == p1b
    assert p1a != p2
    for rgb in p1a:
        assert len(rgb) == 3
        assert all(0 <= c <= 255 for c in rgb)


def test_render_latent_art_produces_valid_png() -> None:
    rng = np.random.default_rng(99)
    vecs = np.cumsum(rng.normal(scale=0.3, size=(12, 32)), axis=0).astype(np.float32)
    png_bytes = art.render_latent_art(vecs, mean=vecs.mean(axis=0), size=128)
    w, h, bit_depth, color_type = _parse_png_ihdr(png_bytes)
    assert (w, h, bit_depth, color_type) == (128, 128, 8, 2)

    # Verify IDAT decompresses to expected scanline byte count: h * (1 + w * 3)
    pos = 8
    idat = bytearray()
    while pos < len(png_bytes):
        length, ctype = struct.unpack(">I4s", png_bytes[pos : pos + 8])
        payload = png_bytes[pos + 8 : pos + 8 + length]
        if ctype == b"IDAT":
            idat.extend(payload)
        pos += 12 + length
    raw = zlib.decompress(bytes(idat))
    assert len(raw) == 128 * (1 + 128 * 3)


def test_ensure_latent_art_caches_file(tmp_path: Path) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: int = 0

        def chunk_vectors(self, path: str) -> np.ndarray:
            _ = path
            self.calls += 1
            return np.linspace(-1.0, 1.0, 20 * 16, dtype=np.float32).reshape(20, 16)

        def track(self, path: str) -> tuple[dict[str, object], dict[str, np.ndarray]] | None:
            v = np.ones(16, dtype=np.float32)
            return {"path": path}, {"mean": v, "intro": v, "outro": -v}

    store = FakeStore()
    out1 = art.ensure_latent_art("/music/song.flac", store, cache_dir=tmp_path, size=64)
    assert out1 is not None
    assert out1.exists()
    assert out1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert store.calls == 1

    # Second call hits cache without re-querying store
    out2 = art.ensure_latent_art("/music/song.flac", store, cache_dir=tmp_path, size=64)
    assert out2 == out1
    assert store.calls == 1

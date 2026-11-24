# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate latent-seeded placeholder album art from a track's CLAP trajectory."""

from __future__ import annotations

import colorsys
import hashlib
import logging
import struct
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DATA_DIR

_log = logging.getLogger(__name__)
ART_CACHE_DIR = DATA_DIR / "art"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


class ArtStore(Protocol):
    def chunk_vectors(self, path: str) -> np.ndarray: ...
    def track(self, path: str) -> tuple[dict[str, object], dict[str, np.ndarray]] | None: ...


def project_trajectory(vectors: np.ndarray, *, padding: float = 0.14) -> np.ndarray:
    """Project an (N, D) latent trajectory to 2D points normalized within [0, 1]."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)

    centered = arr - arr.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2] if vt.shape[0] >= 2 else np.vstack([vt[0], np.zeros_like(vt[0])])
    coords = centered @ basis.T

    # Normalize each axis independently if it has spread, preserving aspect when small.
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    if span < 1e-6:
        return np.full((arr.shape[0], 2), 0.5, dtype=np.float32)

    center = (mins + maxs) * 0.5
    scale = (1.0 - 2.0 * padding) / span
    norm = 0.5 + (coords - center) * scale
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return (round(r * 255), round(g * 255), round(b * 255))


def _vector_phase(vec: np.ndarray | None, salt: int) -> float:
    if vec is None or vec.size == 0:
        return (salt * 0.37) % 1.0
    v = np.asarray(vec, dtype=np.float32).ravel()
    weights = np.cos(np.arange(v.size, dtype=np.float32) * (1.7 + salt * 0.61) + salt)
    val = float(np.dot(v, weights)) / max(float(np.linalg.norm(v)) * 2.0, 1e-6)
    return (val + 0.5) % 1.0


def latent_palette(
    mean: np.ndarray | None,
    intro: np.ndarray | None = None,
    outro: np.ndarray | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Derive a deterministic (background, intro_color, outro_color) RGB palette."""
    base_hue = _vector_phase(mean, 1)
    intro_shift = 0.12 + 0.18 * _vector_phase(intro if intro is not None else mean, 2)
    outro_shift = 0.25 + 0.25 * _vector_phase(outro if outro is not None else mean, 3)
    bg = _hsl_to_rgb(base_hue, 0.32, 0.08)
    start = _hsl_to_rgb(base_hue + intro_shift, 0.85, 0.62)
    end = _hsl_to_rgb(base_hue + intro_shift + outro_shift, 0.85, 0.60)
    return bg, start, end


def _encode_png(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array as an 8-bit PNG."""
    height, width, _ = rgb.shape
    raw_rows = np.hstack([np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)])
    compressed = zlib.compress(raw_rows.tobytes(), level=6)

    def chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def _interpolate_path(pts: np.ndarray, samples_per_seg: int = 16) -> np.ndarray:
    if pts.shape[0] <= 1:
        return pts
    out: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False, dtype=np.float32)[:, None]
    for i in range(pts.shape[0] - 1):
        out.append(pts[i] * (1.0 - t) + pts[i + 1] * t)
    out.append(pts[-1:])
    return np.vstack(out)


def render_latent_art(
    vectors: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    intro: np.ndarray | None = None,
    outro: np.ndarray | None = None,
    size: int = 512,
) -> bytes:
    """Render a square PNG visualizing the track's latent trajectory."""
    bg, start_rgb, end_rgb = latent_palette(mean, intro, outro)
    pts = project_trajectory(vectors) * (size - 1)
    dense = _interpolate_path(pts, samples_per_seg=max(8, size // 32))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = (size - 1) * 0.5
    dist_center = np.sqrt((xx - cx) ** 2 + (yy - cx) ** 2) / max(size * 0.5, 1.0)

    # Subtle radial background gradient + faint orbital guide rings
    canvas = np.empty((size, size, 3), dtype=np.float32)
    bg_arr = np.array(bg, dtype=np.float32)
    glow_bg = np.clip(bg_arr * 1.9 + 14.0, 0.0, 255.0)
    radial = np.exp(-dist_center * dist_center * 2.2)[..., None]
    canvas[:] = bg_arr * (1.0 - radial) + glow_bg * radial

    for ring_r in (0.35, 0.65):
        ring_alpha = np.exp(-((dist_center - ring_r) ** 2) / (0.004 ** 2)) * 0.14
        canvas += ring_alpha[..., None] * 140.0

    # Draw trajectory segments with soft glow + crisp core
    start_arr = np.array(start_rgb, dtype=np.float32)
    end_arr = np.array(end_rgb, dtype=np.float32)
    n_pts = dense.shape[0]
    sigma_glow = max(size * 0.022, 3.0)
    sigma_core = max(size * 0.0055, 1.2)

    if n_pts == 1:
        d2 = (xx - dense[0, 0]) ** 2 + (yy - dense[0, 1]) ** 2
        glow = np.exp(-d2 / (2.0 * (sigma_glow * 1.5) ** 2))[..., None]
        core = np.exp(-d2 / (2.0 * (sigma_core * 1.8) ** 2))[..., None]
        canvas = canvas * (1.0 - glow * 0.45) + start_arr * (glow * 0.45)
        canvas = canvas * (1.0 - core) + start_arr * core
    else:
        # Compute min squared distance to sampled points in chunks to keep memory small.
        min_d2 = np.full((size, size), np.inf, dtype=np.float32)
        nearest_t = np.zeros((size, size), dtype=np.float32)
        t_vals = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
        for idx in range(n_pts):
            px, py = float(dense[idx, 0]), float(dense[idx, 1])
            d2 = (xx - px) ** 2 + (yy - py) ** 2
            mask = d2 < min_d2
            min_d2[mask] = d2[mask]
            nearest_t[mask] = t_vals[idx]

        color_field = start_arr * (1.0 - nearest_t[..., None]) + end_arr * nearest_t[..., None]
        glow = np.exp(-min_d2 / (2.0 * sigma_glow * sigma_glow))[..., None] * 0.45
        core = np.exp(-min_d2 / (2.0 * sigma_core * sigma_core))[..., None] * 0.95
        canvas = canvas * (1.0 - glow) + color_field * glow
        canvas = canvas * (1.0 - core) + color_field * core

        # Intro (hollow ring) and outro (bright dot) markers
        ix, iy = float(pts[0, 0]), float(pts[0, 1])
        ox, oy = float(pts[-1, 0]), float(pts[-1, 1])
        r_mark = max(size * 0.014, 2.5)
        intro_d = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
        intro_ring = np.exp(-((intro_d - r_mark) ** 2) / (max(r_mark * 0.35, 1.0) ** 2))[..., None]
        canvas = canvas * (1.0 - intro_ring) + np.array([245.0, 245.0, 250.0]) * intro_ring

        outro_d2 = (xx - ox) ** 2 + (yy - oy) ** 2
        outro_dot = np.exp(-outro_d2 / (2.0 * (r_mark * 0.7) ** 2))[..., None]
        canvas = canvas * (1.0 - outro_dot) + end_arr * outro_dot

    rgb = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    return _encode_png(rgb)


def ensure_latent_art(
    path: str,
    store: ArtStore | None,
    *,
    cache_dir: Path = ART_CACHE_DIR,
    size: int = 512,
) -> Path | None:
    """Return a cached PNG path of the track's latent trajectory, generating it if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{path}:{size}".encode()).hexdigest()[:24]
    out_path = cache_dir / f"{digest}.png"
    if out_path.exists():
        return out_path

    vectors = np.zeros((0, 1), dtype=np.float32)
    mean = intro = outro = None
    if store is not None:
        try:
            vectors = store.chunk_vectors(path)
        except (KeyError, ValueError, RuntimeError, OSError) as err:
            _log.debug("chunk_vectors failed for %s: %s", path, err)
            vectors = np.zeros((0, 1), dtype=np.float32)
        try:
            found = store.track(path)
            if found is not None:
                _, vecs = found
                mean = vecs.get("mean")
                intro = vecs.get("intro")
                outro = vecs.get("outro")
        except (KeyError, ValueError, RuntimeError, OSError) as err:
            _log.debug("track lookup failed for %s: %s", path, err)

    png_bytes = render_latent_art(vectors, mean=mean, intro=intro, outro=outro, size=size)
    out_path.write_bytes(png_bytes)
    return out_path

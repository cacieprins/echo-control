# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Topographic heat maps and full-library 2D atlas visualization (SVG / PNG)."""

from __future__ import annotations

import hashlib
import html
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import CACHE_DIR, STATE_DIR, Config, WanderConfig
from .cooldown import HeatField, TrailLog
from .errors import EchoControlError
from .manifold import CorpusManifold
from .store import ChunkIndex


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).astype(np.float32) if norm > 1e-8 else arr

ATLAS_WIDTH = 1520
ATLAS_HEIGHT = 920
TOPO_WIDTH = 1280
TOPO_HEIGHT = 680

_PALETTE = [
    "#f43f5e",
    "#38bdf8",
    "#2dd4bf",
    "#a855f7",
    "#22c55e",
    "#fb923c",
    "#ec4899",
    "#818cf8",
    "#06b6d4",
    "#facc15",
    "#f59e0b",
    "#ef4444",
    "#84cc16",
    "#34d399",
    "#c084fc",
    "#60a5fa",
    "#fb7185",
    "#a3e635",
    "#e879f9",
    "#22d3ee",
]

_DOPPLER_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (11, 16, 30)),     # clear sky / background
    (0.08, (29, 78, 216)),    # deep radar blue
    (0.20, (14, 165, 233)),   # cyan-blue
    (0.34, (34, 197, 94)),    # radar green
    (0.50, (250, 204, 21)),   # vivid yellow
    (0.64, (249, 115, 22)),   # storm orange
    (0.78, (239, 68, 68)),    # severe red
    (0.90, (236, 72, 153)),   # extreme pink / magenta
    (1.00, (255, 255, 255)),  # white-hot core
]


def _doppler_color(t: float) -> str:
    """Map normalized intensity t in [0, 1] to a NEXRAD-style Doppler radar hex color."""
    clamped = min(1.0, max(0.0, t))
    for i in range(len(_DOPPLER_STOPS) - 1):
        t0, c0 = _DOPPLER_STOPS[i]
        t1, c1 = _DOPPLER_STOPS[i + 1]
        if clamped <= t1:
            f = (clamped - t0) / max(1e-9, t1 - t0)
            r = round(c0[0] + f * (c1[0] - c0[0]))
            g = round(c0[1] + f * (c1[1] - c0[1]))
            b = round(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#ffffff"


def _artist_of(path: str, library_roots: list[Path] | None = None) -> str:
    rel = path
    if library_roots:
        for root in library_roots:
            prefix = str(root).rstrip("/") + "/"
            if rel.startswith(prefix):
                rel = rel[len(prefix) :]
                break
    else:
        parts = Path(path).parts
        if len(parts) >= 3:
            rel = "/".join(parts[-3:])
    segs = [s for s in rel.split("/") if s]
    if not segs:
        return Path(path).stem
    top = segs[0]
    if " - " in top and len(segs) == 2:
        top = top.split(" - ", 1)[0]
    return top


def _layout_fingerprint(paths: list[str]) -> str:
    h = hashlib.sha256()
    h.update(f"tsne-v1:{len(paths)}\n".encode())
    for p in paths:
        h.update(p.encode("utf-8", errors="surrogateescape"))
        h.update(b"\n")
    return h.hexdigest()


def _parse_hhmmss(token: str) -> int | None:
    parts = token.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


def load_session_runs(
    log_dir: Path, date_str: str | None = None, *, gap_seconds: int = 1800
) -> list[list[str]]:
    """Read committed track paths from `<log_dir>/YYYY-MM-DD.log`, split into separate sessions.

    A new session starts whenever a line is logged as a seed/hard-cut
    (`[score=0.00 intro=0.00`) or when more than `gap_seconds` elapses between commits.
    """
    day = date_str or datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
    log_path = log_dir / f"{day}.log"
    if not log_path.exists():
        return []
    runs: list[list[str]] = []
    prev_sec: int | None = None
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        idx = line.rfind("]  ")
        if idx == -1:
            continue
        track_path = line[idx + 3 :].strip()
        if not track_path:
            continue
        ts_sec = _parse_hhmmss(line.split(None, 1)[0])
        is_seed = "[score=0.00 intro=0.00" in line
        big_gap = (
            prev_sec is not None
            and ts_sec is not None
            and ts_sec >= prev_sec
            and (ts_sec - prev_sec) > gap_seconds
        )
        if not runs or is_seed or big_gap:
            runs.append([track_path])
        else:
            runs[-1].append(track_path)
        prev_sec = ts_sec
    return runs


def load_session_paths(log_dir: Path, date_str: str | None = None) -> list[str]:
    """Read committed track paths from `<log_dir>/YYYY-MM-DD.log` as a flat list."""
    return [p for run in load_session_runs(log_dir, date_str) for p in run]


def compute_tsne_layout(
    manifold: CorpusManifold,
    *,
    cache_path: Path | None = None,
    perplexity: float = 35.0,
) -> np.ndarray:
    """Compute (or load cached) 2D cosine t-SNE coordinates for `manifold.centered`."""
    n = len(manifold.paths)
    if n == 0:
        return np.empty((0, 2), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32)

    fp = _layout_fingerprint(manifold.paths)
    if cache_path is not None and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if str(cached["fingerprint"].item()) == fp:
                    coords = np.asarray(cached["coords"], dtype=np.float32)
                    if coords.shape == (n, 2):
                        return coords
        except (OSError, ValueError, KeyError):
            pass

    from sklearn.manifold import TSNE  # pyright: ignore[reportMissingTypeStubs]

    eff_perp = min(perplexity, max(2.0, (n - 1) / 3.0))
    init_mode = "pca" if n >= 4 and manifold.centered.shape[1] >= 2 else "random"
    tsne: Any = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=eff_perp,
        init=init_mode,
        learning_rate="auto",
        random_state=42,
        max_iter=750,
    )
    coords: np.ndarray = np.asarray(tsne.fit_transform(manifold.centered), dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, fingerprint=np.array(fp), coords=coords)
            os.replace(tmp, cache_path)
        except OSError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return coords


def _build_heat_field(
    manifold: CorpusManifold, session_paths: list[str], wcfg: WanderConfig
) -> HeatField:
    hf = HeatField.from_config(wcfg)
    for p in session_paths:
        v = manifold.mean_for(p)
        if v is not None:
            hf.record(v)
    return hf


def _eval_field_weighted(
    hf: HeatField, wcfg: WanderConfig, vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = vecs.shape[0]
    if n == 0 or not hf._history:  # pyright: ignore[reportPrivateUsage]
        zeros = np.zeros(n, dtype=np.float32)
        return zeros, zeros, zeros
    history = hf._history  # pyright: ignore[reportPrivateUsage]
    step = hf._step  # pyright: ignore[reportPrivateUsage]
    h_vecs = np.stack([h[0] for h in history], axis=0)
    h_weights = np.array(
        [h[2] * math.exp(-(step - h[1]) / hf.micro_tau) for h in history],
        dtype=np.float32,
    )
    dists = np.maximum(0.0, 1.0 - vecs @ h_vecs.T)
    micro = np.exp(-dists / (hf.micro_sigma**2)) @ h_weights
    macro = np.zeros(n, dtype=np.float32)
    macro_c = hf._macro_centroid  # pyright: ignore[reportPrivateUsage]
    if macro_c is not None and hf.macro_beta > 0.0:
        macro_d = np.maximum(0.0, 1.0 - vecs @ macro_c)
        macro = hf.macro_beta * np.exp(-macro_d / (hf.macro_sigma**2))
    w = wcfg.heat_weight
    return (w * (micro + macro)).astype(np.float32), (w * micro).astype(np.float32), (w * macro).astype(np.float32)


def _catmull_rom_bezier(
    pts: list[tuple[float, float]], *, tension: float = 0.35
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """Convert a polyline into cubic Bezier segments `(p0, cp1, cp2, p1)` via Catmull-Rom."""
    m = len(pts)
    if m < 2:
        return []
    out: list[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = []
    scale = tension / 3.0
    for i in range(m - 1):
        p_prev = pts[i - 1] if i > 0 else pts[i]
        p0 = pts[i]
        p1 = pts[i + 1]
        p_next = pts[i + 2] if i + 2 < m else pts[i + 1]
        cp1 = (p0[0] + (p1[0] - p_prev[0]) * scale, p0[1] + (p1[1] - p_prev[1]) * scale)
        cp2 = (p1[0] - (p_next[0] - p0[0]) * scale, p1[1] - (p_next[1] - p0[1]) * scale)
        out.append((p0, cp1, cp2, p1))
    return out


def _heat_color(val: float) -> str:
    t = min(1.0, max(0.0, (val / 0.38) ** 0.52))
    return _doppler_color(t)


def _marching_squares(
    z_grid: np.ndarray, gx: np.ndarray, gy: np.ndarray, level: float
) -> list[tuple[float, float, float, float]]:
    segs: list[tuple[float, float, float, float]] = []
    ny, nx = z_grid.shape
    for j in range(ny - 1):
        y0, y1 = float(gy[j]), float(gy[j + 1])
        for i in range(nx - 1):
            v00 = float(z_grid[j, i])
            v10 = float(z_grid[j, i + 1])
            v11 = float(z_grid[j + 1, i + 1])
            v01 = float(z_grid[j + 1, i])
            idx = (
                (1 if v00 >= level else 0)
                | (2 if v10 >= level else 0)
                | (4 if v11 >= level else 0)
                | (8 if v01 >= level else 0)
            )
            if idx == 0 or idx == 15:
                continue

            def _interp(a: float, b: float, va: float, vb: float) -> float:
                if abs(vb - va) < 1e-12:
                    return 0.5 * (a + b)
                return a + (level - va) / (vb - va) * (b - a)

            x0, x1 = float(gx[i]), float(gx[i + 1])
            pts: dict[int, tuple[float, float]] = {}
            if (v00 >= level) != (v10 >= level):
                pts[0] = (_interp(x0, x1, v00, v10), y0)
            if (v10 >= level) != (v11 >= level):
                pts[1] = (x1, _interp(y0, y1, v10, v11))
            if (v01 >= level) != (v11 >= level):
                pts[2] = (_interp(x0, x1, v01, v11), y1)
            if (v00 >= level) != (v01 >= level):
                pts[3] = (x0, _interp(y0, y1, v00, v01))
            keys = list(pts.keys())
            if len(keys) == 2:
                pa, pb = pts[keys[0]], pts[keys[1]]
                segs.append((pa[0], pa[1], pb[0], pb[1]))
            elif len(keys) == 4:
                p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
                segs.append((p0[0], p0[1], p1[0], p1[1]))
                segs.append((p2[0], p2[1], p3[0], p3[1]))
    return segs


def _cluster_cache_path(cache_path: Path, k: int) -> Path:
    return cache_path.with_name(f"{cache_path.stem}_k{k}.npz")


def _compute_clusters(
    manifold: CorpusManifold, k_eff: int, *, cache_path: Path | None = None
) -> np.ndarray:
    """Compute (or load cached) 512D spherical k-means cluster labels."""
    n = len(manifold.paths)
    fp = _layout_fingerprint(manifold.paths)
    c_path = _cluster_cache_path(cache_path, k_eff) if cache_path is not None else None
    if c_path is not None and c_path.exists():
        try:
            with np.load(c_path, allow_pickle=False) as cached:
                if str(cached["fingerprint"].item()) == fp:
                    lbl = np.asarray(cached["labels"], dtype=np.int32)
                    if lbl.shape == (n,):
                        return lbl
        except (OSError, ValueError, KeyError):
            pass

    from sklearn.cluster import KMeans  # pyright: ignore[reportMissingTypeStubs]

    km: Any = cast("Any", KMeans)(n_clusters=k_eff, random_state=42, n_init=10)
    labels: np.ndarray = np.asarray(km.fit_predict(manifold.centered), dtype=np.int32)

    if c_path is not None:
        c_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = c_path.with_name(f"{c_path.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, fingerprint=np.array(fp), labels=labels)
            os.replace(tmp, c_path)
        except OSError:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return labels


def _compute_coastlines_svg(
    pts_2d: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    mx0: int,
    my0: int,
    mw: int,
    mh: int,
) -> list[str]:
    """Compute library density coastline SVG <path> elements."""
    n_tracks = pts_2d.shape[0]
    nx_g, ny_g = 90, 72
    gx = np.linspace(xmin, xmax, nx_g)
    gy = np.linspace(ymin, ymax, ny_g)
    gxx, gyy = np.meshgrid(gx, gy)
    dx_s = (xmax - xmin) * 0.030
    dy_s = (ymax - ymin) * 0.030
    dens = np.zeros((ny_g, nx_g), dtype=np.float32)
    for start in range(0, n_tracks, 512):
        end = min(start + 512, n_tracks)
        px = pts_2d[start:end, 0][:, None, None]
        py = pts_2d[start:end, 1][:, None, None]
        d2 = ((gxx[None, :, :] - px) / dx_s) ** 2 + ((gyy[None, :, :] - py) / dy_s) ** 2
        dens += np.exp(-0.5 * d2).sum(axis=0)
    dens_norm = dens / max(1e-6, float(np.percentile(dens, 98)))

    def mx(x: float) -> float:
        return mx0 + (x - xmin) / (xmax - xmin) * mw

    def my(y: float) -> float:
        return my0 + (ymax - y) / (ymax - ymin) * mh

    out: list[str] = []
    for lvl, op in [(0.13, 0.24), (0.34, 0.16), (0.64, 0.12)]:
        segs = _marching_squares(dens_norm, gx, gy, lvl)
        if segs:
            d_str = "".join(
                f"M{mx(x0):.1f},{my(y0):.1f}L{mx(x1):.1f},{my(y1):.1f}" for x0, y0, x1, y1 in segs
            )
            out.append(f'<path d="{d_str}" fill="none" stroke="#64748b" stroke-opacity="{op}" stroke-width="0.9"/>')
    return out


def render_library_atlas_svg(
    manifold: CorpusManifold,
    trails: TrailLog,
    session_paths: list[str],
    wcfg: WanderConfig,
    *,
    clusters: int = 15,
    highlight: str | None = None,
    library_roots: list[Path] | None = None,
    coords: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    coastlines_svg: list[str] | None = None,
    cache_path: Path | None = None,
    now: float | None = None,
    label_mode: str = "clusters",
    session_runs: list[list[str]] | None = None,
    next_path: str | None = None,
) -> str:
    """Render the full-library 2D spherical-cluster atlas + active heat overlay as SVG."""
    n_tracks = len(manifold.paths)
    if n_tracks < 3:
        raise EchoControlError("need at least 3 indexed tracks to render a library atlas")

    t_now = time.time() if now is None else now
    hf = _build_heat_field(manifold, session_paths, wcfg)
    whf_all, _, _ = _eval_field_weighted(hf, wcfg, manifold.means)
    nh_all = np.array([trails.node_heat(p, now=t_now) for p in manifold.paths], dtype=np.float32)
    tot_all = whf_all + nh_all

    artists = [_artist_of(p, library_roots) for p in manifold.paths]
    global_counts = Counter(artists)
    k_eff = max(1, min(clusters, n_tracks))

    if labels is None:
        labels = _compute_clusters(manifold, k_eff, cache_path=cache_path)
    pts_2d = (
        coords
        if coords is not None
        else compute_tsne_layout(manifold, cache_path=cache_path)
    )

    cluster_list: list[dict[str, Any]] = []
    for k in range(k_eff):
        idxs = np.where(labels == k)[0]
        cnt = Counter(artists[i] for i in idxs if artists[i] != "Various Artists")
        scored: list[tuple[float, str]] = []
        min_c = 3 if len(idxs) >= 15 else 1
        for a, c_in in cnt.items():
            if c_in < min_c:
                continue
            purity = c_in / max(1, global_counts[a])
            scored.append((c_in * (purity**0.7), a))
        if not scored:
            scored = [(float(c_in), a) for a, c_in in cnt.items()]
        scored.sort(reverse=True)
        top_a = [a for _, a in scored[:3]] or ["Mixed"]

        c_vec = manifold.centered[idxs].mean(axis=0)
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm > 1e-8:
            c_vec = c_vec / c_norm
        coh = float((manifold.centered[idxs] @ c_vec).mean())

        sub2d = pts_2d[idxs]
        d2 = np.sum((sub2d[:, None, :] - sub2d[None, :, :]) ** 2, axis=2)
        bw2 = float(np.percentile(d2, 25)) + 1e-4
        core_i = int(np.argmax(np.exp(-0.5 * d2 / bw2).sum(axis=1)))
        cx, cy = float(sub2d[core_i, 0]), float(sub2d[core_i, 1])

        cluster_list.append({
            "k": k,
            "count": len(idxs),
            "coh": coh,
            "top_a": top_a,
            "cx": cx,
            "cy": cy,
            "color": _PALETTE[k % len(_PALETTE)],
            "n_hot": int(np.sum(tot_all[idxs] > 0.05)),
        })

    cluster_list.sort(key=lambda d: -d["count"])
    for rank, cl in enumerate(cluster_list, 1):
        cl["rank"] = rank
    mean_coh = float(np.mean([cl["coh"] for cl in cluster_list]))

    w_svg, h_svg = ATLAS_WIDTH, ATLAS_HEIGHT
    mx0, my0, mw, mh = 36, 72, 1010, 812
    sx0, sy0, sw, sh = 1066, 72, 418, 812

    xmin, xmax = float(pts_2d[:, 0].min()), float(pts_2d[:, 0].max())
    ymin, ymax = float(pts_2d[:, 1].min()), float(pts_2d[:, 1].max())
    span_x = max(1e-3, xmax - xmin)
    span_y = max(1e-3, ymax - ymin)
    xmin -= span_x * 0.07
    xmax += span_x * 0.07
    ymin -= span_y * 0.07
    ymax += span_y * 0.07

    def mx(x: float) -> float:
        return mx0 + (x - xmin) / (xmax - xmin) * mw

    def my(y: float) -> float:
        return my0 + (ymax - y) / (ymax - ymin) * mh

    # Smooth Doppler heat field on 2D grid (evaluated only over active warm/hot tracks)
    nx_g, ny_g = 96, 76
    gx = np.linspace(xmin, xmax, nx_g)
    gy = np.linspace(ymin, ymax, ny_g)
    gxx, gyy = np.meshgrid(gx, gy)
    dx_h = (xmax - xmin) * 0.036
    dy_h = (ymax - ymin) * 0.036
    heat_grid = np.zeros((ny_g, nx_g), dtype=np.float32)
    hot_idx = np.where(tot_all > 0.02)[0]
    if len(hot_idx) > 0:
        px_h = pts_2d[hot_idx, 0][:, None, None]
        py_h = pts_2d[hot_idx, 1][:, None, None]
        hw = tot_all[hot_idx][:, None, None]
        d2_h = ((gxx[None, :, :] - px_h) / dx_h) ** 2 + ((gyy[None, :, :] - py_h) / dy_h) ** 2
        heat_grid = (np.exp(-0.5 * d2_h) * hw).sum(axis=0).astype(np.float32)
    h_max = float(heat_grid.max())
    heat_norm = (heat_grid / h_max) if h_max > 1e-6 else heat_grid
    coast_paths = (
        coastlines_svg
        if coastlines_svg is not None
        else _compute_coastlines_svg(pts_2d, xmin, xmax, ymin, ymax, mx0, my0, mw, mh)
    )

    # Repulsion for non-overlapping cluster labels
    lbl_pos = [[mx(cl["cx"]), my(cl["cy"])] for cl in cluster_list]
    for _ in range(40):
        for i in range(len(lbl_pos)):
            for j in range(i + 1, len(lbl_pos)):
                dx = lbl_pos[i][0] - lbl_pos[j][0]
                dy = lbl_pos[i][1] - lbl_pos[j][1]
                dist = math.hypot(dx / 135.0, dy / 32.0)
                if 1e-6 < dist < 1.0:
                    push = (1.0 - dist) * 0.45
                    lbl_pos[i][0] += (dx / dist) * push * 18.0
                    lbl_pos[i][1] += (dy / dist) * push * 12.0
                    lbl_pos[j][0] -= (dx / dist) * push * 18.0
                    lbl_pos[j][1] -= (dy / dist) * push * 12.0
        for p in lbl_pos:
            p[0] = max(mx0 + 85.0, min(mx0 + mw - 85.0, p[0]))
            p[1] = max(my0 + 24.0, min(my0 + mh - 20.0, p[1]))

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_svg}" height="{h_svg}" viewBox="0 0 {w_svg} {h_svg}">',
        """<defs>
  <style>
    .bg { fill: #090c14; }
    .panel { fill: #0d121f; stroke: #222c42; stroke-width: 1.2; }
    .title { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 20px; font-weight: 700; fill: #f1f5f9; }
    .subtitle { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 12px; fill: #94a3b8; }
    .sec { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 13.5px; font-weight: 700; fill: #e2e8f0; }
    .reg { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 10.5px; font-weight: 700; }
    .badge { font-family: "DejaVu Sans", sans-serif; font-size: 9.5px; font-weight: 700; fill: #090c14; }
    .lbl { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 10.5px; fill: #e2e8f0; }
    .lbl-dim { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 9.6px; fill: #94a3b8; }
    .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 9.5px; fill: #94a3b8; }
  </style>
  <clipPath id="mapClip"><rect x="36" y="72" width="1010" height="812" rx="6"/></clipPath>
  <filter id="radarBlur" x="-5%" y="-5%" width="110%" height="110%">
    <feGaussianBlur stdDeviation="4.5"/>
  </filter>
  <marker id="nextArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M1,1.5 L9,5 L1,8.5 L3,5 Z" fill="#fffbeb" fill-opacity="0.9"/>
  </marker>
</defs>""",
        f'<rect class="bg" x="0" y="0" width="{w_svg}" height="{h_svg}"/>',
        f'<text class="title" x="36" y="34">echo-control Library Atlas — {k_eff} Spherical 512D Clusters ({n_tracks:,} Tracks)</text>',
        f'<text class="subtitle" x="36" y="54">Clustered in 512D centered CLAP space (mean cohesion {mean_coh:.2f}) • Distinctive artists • Session walk: {len(session_paths)} tracks</text>',
        f'<rect class="panel" x="{mx0}" y="{my0}" width="{mw}" height="{mh}" rx="6"/>',
        '<g clip-path="url(#mapClip)">',
    ]

    # Smooth Doppler weather-radar heat overlay (blue -> green -> yellow -> orange -> red -> pink -> white)
    cw_h = mw / (nx_g - 1) + 0.8
    ch_h = mh / (ny_g - 1) + 0.8
    svg.append('<g filter="url(#radarBlur)">')
    for j in range(ny_g - 1):
        y_px = my(float(gy[j + 1]))
        for i in range(nx_g - 1):
            hv = 0.25 * (
                float(heat_norm[j, i])
                + float(heat_norm[j, i + 1])
                + float(heat_norm[j + 1, i])
                + float(heat_norm[j + 1, i + 1])
            )
            if hv > 0.035:
                t_dop = 0.08 + 0.92 * (hv**0.68)
                col = _doppler_color(t_dop)
                op = min(0.72, 0.14 + 0.58 * (hv**0.55))
                svg.append(
                    f'<rect x="{mx(float(gx[i])):.1f}" y="{y_px:.1f}" '
                    + f'width="{cw_h:.1f}" height="{ch_h:.1f}" fill="{col}" fill-opacity="{op:.3f}"/>'
                )
    svg.append("</g>")

    svg.extend(coast_paths)

    sess_set = set(session_paths)
    hl_lower = highlight.lower() if highlight else None
    hl_set: set[int] = (
        {i for i, p in enumerate(manifold.paths) if hl_lower in p.lower()}
        if hl_lower
        else set()
    )
    color_by_k = {cl["k"]: cl["color"] for cl in cluster_list}

    for i in range(n_tracks):
        p = manifold.paths[i]
        if p in sess_set or i in hl_set:
            continue
        px, py = mx(float(pts_2d[i, 0])), my(float(pts_2d[i, 1]))
        col = color_by_k[int(labels[i])]
        if nh_all[i] > 0.01:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.9" fill="#f59e0b" fill-opacity="0.88" stroke="#090c14" stroke-width="0.6"/>')
        else:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.9" fill="{col}" fill-opacity="0.62"/>')

    for i in hl_set:
        if manifold.paths[i] in sess_set:
            continue
        px, py = mx(float(pts_2d[i, 0])), my(float(pts_2d[i, 1]))
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="#ffffff" stroke="#ec4899" stroke-width="1.3"/>')

    # Build per-session index lists so we never draw lines across separate sessions
    raw_runs = session_runs if session_runs is not None else ([session_paths] if session_paths else [])
    indexed_runs: list[list[int]] = [
        [manifold.index[p] for p in run if p in manifold.index] for run in raw_runs
    ]
    indexed_runs = [r for r in indexed_runs if r]
    prev_runs = indexed_runs[:-1]
    curr_run = indexed_runs[-1] if indexed_runs else []
    sess_indices = [idx for run in indexed_runs for idx in run]

    # 1. Previous sessions: subtle cool-slate backdrop splines & muted dots
    n_prev = len(prev_runs)
    for run_i, run in enumerate(prev_runs):
        run_rec = (run_i + 1) / max(1, n_prev + 1)
        if len(run) >= 2:
            run_pts = [(mx(float(pts_2d[idx, 0])), my(float(pts_2d[idx, 1]))) for idx in run]
            beziers = _catmull_rom_bezier(run_pts, tension=0.35)
            n_seg = len(beziers)
            for seg_i, (p0, cp1, cp2, p1) in enumerate(beziers):
                seg_rec = (seg_i + 1) / max(1, n_seg)
                op = (0.04 + 0.11 * seg_rec) * (0.55 + 0.45 * run_rec)
                sw = 0.7 + 0.35 * seg_rec
                svg.append(
                    f'<path d="M{p0[0]:.1f},{p0[1]:.1f} '
                    + f'C{cp1[0]:.1f},{cp1[1]:.1f} {cp2[0]:.1f},{cp2[1]:.1f} {p1[0]:.1f},{p1[1]:.1f}" '
                    + f'fill="none" stroke="#94a3b8" stroke-opacity="{op:.3f}" '
                    + f'stroke-width="{sw:.2f}" stroke-linecap="round"/>'
                )
        for idx in run:
            px, py = mx(float(pts_2d[idx, 0])), my(float(pts_2d[idx, 1]))
            if idx in hl_set:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.0" fill="none" stroke="#ec4899" stroke-opacity="0.65" stroke-width="1.3"/>')
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.0" fill="#cbd5e1" fill-opacity="{0.22 + 0.16 * run_rec:.2f}"/>')

    # 2. Current session: bright foreground fading spline & vivid gold/cream dots
    if len(curr_run) >= 2:
        curr_pts = [(mx(float(pts_2d[idx, 0])), my(float(pts_2d[idx, 1]))) for idx in curr_run]
        beziers = _catmull_rom_bezier(curr_pts, tension=0.35)
        n_seg = len(beziers)
        for seg_i, (p0, cp1, cp2, p1) in enumerate(beziers):
            rec = (seg_i + 1) / max(1, n_seg)
            op = 0.42 + 0.54 * (rec**0.65)
            sw = 1.4 + 1.6 * (rec**0.85)
            col = "#fffbeb" if rec > 0.75 else "#fef08a"
            svg.append(
                f'<path d="M{p0[0]:.1f},{p0[1]:.1f} '
                + f'C{cp1[0]:.1f},{cp1[1]:.1f} {cp2[0]:.1f},{cp2[1]:.1f} {p1[0]:.1f},{p1[1]:.1f}" '
                + f'fill="none" stroke="{col}" stroke-opacity="{op:.3f}" '
                + f'stroke-width="{sw:.2f}" stroke-linecap="round"/>'
            )

    n_c = len(curr_run)
    for step_i, idx in enumerate(curr_run):
        px, py = mx(float(pts_2d[idx, 0])), my(float(pts_2d[idx, 1]))
        rec = (step_i + 1) / max(1, n_c)
        r = 3.0 + 2.2 * (rec**0.9)
        dot_op = 0.55 + 0.45 * (rec**0.7)
        if idx in hl_set:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r + 2.2:.1f}" fill="none" stroke="#ec4899" stroke-width="1.7"/>')
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="#fff8be" fill-opacity="{dot_op:.2f}" stroke="#090c14" stroke-opacity="{dot_op:.2f}" stroke-width="0.9"/>')

    now_label_svg = ""
    if curr_run:
        s0, s_last = curr_run[0], curr_run[-1]
        svg.append(f'<circle cx="{mx(float(pts_2d[s0, 0])):.1f}" cy="{my(float(pts_2d[s0, 1])):.1f}" r="7" fill="none" stroke="#38bdf8" stroke-width="1.8"/>')
        lx, ly = mx(float(pts_2d[s_last, 0])), my(float(pts_2d[s_last, 1]))
        # Leading arrow from NOW -> Up Next (no ring/circle around Up Next)
        if next_path and next_path in manifold.index:
            n_idx = manifold.index[next_path]
            nx_u, ny_u = mx(float(pts_2d[n_idx, 0])), my(float(pts_2d[n_idx, 1]))
            dx_u, dy_u = nx_u - lx, ny_u - ly
            dist_u = math.hypot(dx_u, dy_u)
            if dist_u > 12.0:
                start_trim = 8.5
                end_trim = 5.0
                x_s = lx + (dx_u / dist_u) * start_trim
                y_s = ly + (dy_u / dist_u) * start_trim
                x_e = nx_u - (dx_u / dist_u) * end_trim
                y_e = ny_u - (dy_u / dist_u) * end_trim
                svg.append(
                    f'<line x1="{x_s:.1f}" y1="{y_s:.1f}" x2="{x_e:.1f}" y2="{y_e:.1f}" '
                    + 'stroke="#fffbeb" stroke-opacity="0.82" stroke-width="1.8" '
                    + 'stroke-dasharray="5,4" stroke-linecap="round" marker-end="url(#nextArrow)"/>'
                )
        svg.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="8" fill="none" stroke="#ffffff" stroke-width="2.0"/>')
        if label_mode != "recent":
            last_name = Path(manifold.paths[s_last]).stem
            dx_l = -155.0 if lx > mx0 + mw - 180 else 14.0
            dy_l = 22.0 if ly > my0 + 60 else -14.0
            now_label_svg = (
                f'<text class="reg" x="{lx + dx_l:.1f}" y="{ly + dy_l:.1f}" fill="#fff8be" '
                + f'stroke="#090c14" stroke-width="4.0" paint-order="stroke">NOW: {html.escape(last_name[:28])}</text>'
            )

    recent_anchors: list[tuple[int, float, float, str, str, float, int]] = []
    if label_mode == "recent":
        # Highlight the last 3 played tracks of the current session; their cards & pointer lines go in the right sidebar
        recent_top = curr_run[-3:]
        n_rec = len(recent_top)
        for r_i, idx in enumerate(recent_top):
            age = n_rec - 1 - r_i  # 0 = NOW, 1 = -1, 2 = -2
            nx_p = mx(float(pts_2d[idx, 0]))
            ny_p = my(float(pts_2d[idx, 1]))
            art = _artist_of(manifold.paths[idx], library_roots)
            stem = Path(manifold.paths[idx]).stem
            col = "#fffbeb" if age == 0 else ("#fef08a" if age == 1 else "#cbd5e1")
            op = 0.98 if age == 0 else (0.86 if age == 1 else 0.74)
            svg.append(
                f'<circle cx="{nx_p:.1f}" cy="{ny_p:.1f}" r="{6.5 if age == 0 else 5.0:.1f}" '
                + f'fill="none" stroke="{col}" stroke-opacity="{op:.2f}" stroke-width="{2.0 if age == 0 else 1.5:.1f}"/>'
            )
            recent_anchors.append((age, nx_p, ny_p, art, stem, op, idx))
        # Order newest-first (age 0, 1, 2) to match sidebar order
        recent_anchors.sort(key=lambda t: t[0])
    else:
        for cl, (lx, ly) in zip(cluster_list, lbl_pos, strict=True):
            cx_p, cy_p = mx(cl["cx"]), my(cl["cy"])
            if math.hypot(lx - cx_p, ly - cy_p) > 22.0:
                svg.append(f'<line x1="{cx_p:.1f}" y1="{cy_p:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="{cl["color"]}" stroke-opacity="0.45" stroke-width="1"/>')
            svg.append(f'<circle cx="{cx_p:.1f}" cy="{cy_p:.1f}" r="8.5" fill="{cl["color"]}" stroke="#090c14" stroke-width="1.6"/>')
            svg.append(f'<text class="badge" x="{cx_p:.1f}" y="{cy_p + 3.2:.1f}" text-anchor="middle">{cl["rank"]}</text>')
            short_lbl = f"{cl['rank']}. " + " · ".join(cl["top_a"][:2])
            svg.append(
                f'<text class="reg" x="{lx:.1f}" y="{ly - 12:.1f}" text-anchor="middle" '
                + f'fill="{cl["color"]}" stroke="#090c14" stroke-width="4.0" paint-order="stroke">{html.escape(short_lbl)}</text>'
            )

    if now_label_svg:
        svg.append(now_label_svg)
    svg.append("</g>")

    # Right Sidebar
    svg.append(f'<rect class="panel" x="{sx0}" y="{sy0}" width="{sw}" height="{sh}" rx="6"/>')
    if label_mode == "recent":
        svg.append(f'<text class="sec" x="{sx0 + 16}" y="{sy0 + 28}">Recent Trail ({len(sess_indices)} played)</text>')
        svg.append(f'<text class="lbl-dim" x="{sx0 + 16}" y="{sy0 + 45}">Newest first • current session</text>')

        # Top 3 highlighted cards in sidebar
        card_y = sy0 + 62
        card_h = 62
        card_gap = 12
        for age, _nx_p, _ny_p, art, stem, _op, idx in recent_anchors:
            col = "#fffbeb" if age == 0 else ("#fef08a" if age == 1 else "#cbd5e1")
            badge_txt = "NOW" if age == 0 else f"-{age}"
            cx_card = sx0 + 14
            cw_card = sw - 28
            svg.append(
                f'<rect x="{cx_card}" y="{card_y}" width="{cw_card}" height="{card_h}" rx="5" '
                + f'fill="#111827" stroke="{col}" stroke-opacity="{0.85 if age == 0 else 0.45:.2f}" '
                + f'stroke-width="{1.5 if age == 0 else 1.0:.1f}"/>'
            )
            svg.append(
                f'<rect x="{cx_card + 10}" y="{card_y + 10}" width="38" height="18" rx="3" '
                + f'fill="{col}" fill-opacity="{0.95 if age == 0 else 0.80:.2f}"/>'
            )
            svg.append(
                f'<text class="badge" x="{cx_card + 29}" y="{card_y + 22.5}" text-anchor="middle">{badge_txt}</text>'
            )
            title_txt = stem if len(stem) <= 34 else stem[:33] + "…"
            art_txt = art if len(art) <= 38 else art[:37] + "…"
            svg.append(
                f'<text class="lbl" x="{cx_card + 56}" y="{card_y + 23}" font-weight="700" fill="{col}">{html.escape(title_txt)}</text>'
            )
            svg.append(
                f'<text class="lbl-dim" x="{cx_card + 56}" y="{card_y + 39}">{html.escape(art_txt)}</text>'
            )
            svg.append(
                f'<text class="mono" x="{cx_card + 56}" y="{card_y + 53}">'
                + f'heat {tot_all[idx]:.2f} (field {whf_all[idx]:.2f} + node {nh_all[idx]:.2f})</text>'
            )
            card_y += card_h + card_gap

        # Earlier tracks in the session below the 3 callout cards
        earlier = list(reversed(sess_indices[:-3]))[:10]
        if earlier:
            ey = card_y + 12
            svg.append(f'<line x1="{sx0 + 16}" y1="{ey - 10}" x2="{sx0 + sw - 16}" y2="{ey - 10}" stroke="#1e293b" stroke-width="1.0"/>')
            svg.append(f'<text class="lbl-dim" x="{sx0 + 16}" y="{ey + 6}" font-weight="700">Earlier in session</text>')
            ey += 26
            for offset_i, idx in enumerate(earlier, 3):
                if ey > sy0 + sh - 135:
                    break
                art = _artist_of(manifold.paths[idx], library_roots)
                stem = Path(manifold.paths[idx]).stem
                line_txt = f"{art} — {stem}" if art.lower() not in stem.lower() else stem
                if len(line_txt) > 44:
                    line_txt = line_txt[:43] + "…"
                svg.append(f'<text class="mono" x="{sx0 + 18}" y="{ey}">-{offset_i:<2d}</text>')
                svg.append(f'<text class="lbl-dim" x="{sx0 + 44}" y="{ey}">{html.escape(line_txt)}</text>')
                ey += 22
    else:
        svg.append(f'<text class="sec" x="{sx0 + 16}" y="{sy0 + 26}">{k_eff} Spherical 512D Clusters</text>')
        svg.append(f'<text class="lbl-dim" x="{sx0 + 16}" y="{sy0 + 42}">Distinctive artists (purity-weighted) • cohesion</text>')

        max_rows = min(len(cluster_list), 16)
        row_step = min(41, int((sh - 185) / max(1, max_rows)))
        ry = sy0 + 64
        for cl in cluster_list[:max_rows]:
            svg.append(f'<circle cx="{sx0 + 24}" cy="{ry - 3}" r="8.5" fill="{cl["color"]}"/>')
            svg.append(f'<text class="badge" x="{sx0 + 24}" y="{ry + 0.3}" text-anchor="middle">{cl["rank"]}</text>')
            top_str = ", ".join(cl["top_a"])
            if len(top_str) > 42:
                top_str = top_str[:41] + "…"
            svg.append(f'<text class="lbl" x="{sx0 + 40}" y="{ry}" font-weight="700">{html.escape(top_str)}</text>')
            svg.append(f'<text class="mono" x="{sx0 + 40}" y="{ry + 14}">{cl["count"]:3d} tracks • coh {cl["coh"]:.2f} • {cl["n_hot"]:2d} warm/hot</text>')
            ry += row_step

    ly_leg = sy0 + sh - 122
    svg.append(f'<line x1="{sx0 + 16}" y1="{ly_leg - 10}" x2="{sx0 + sw - 16}" y2="{ly_leg - 10}" stroke="#1e293b" stroke-width="1.2"/>')
    # Doppler radar colorbar in legend
    cb_x, cb_y, cb_w, cb_h = sx0 + 16, ly_leg + 2, 148, 10
    for step_c in range(40):
        t_c = 0.08 + 0.92 * (step_c / 39.0)
        svg.append(
            f'<rect x="{cb_x + step_c * (cb_w / 40):.1f}" y="{cb_y}" '
            + f'width="{cb_w / 40 + 0.6:.1f}" height="{cb_h}" fill="{_doppler_color(t_c)}"/>'
        )
    svg.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#334155" stroke-width="0.9"/>')
    svg.append(f'<text class="lbl-dim" x="{cb_x + cb_w + 10}" y="{cb_y + 9}">Doppler heat (cool → hot)</text>')
    svg.append(f'<circle cx="{sx0 + 24}" cy="{ly_leg + 29}" r="3.8" fill="#fff8be"/><text class="lbl-dim" x="{sx0 + 38}" y="{ly_leg + 32}">Fading spline: session walk (bright = recent, white ring = NOW)</text>')
    svg.append(f'<circle cx="{sx0 + 24}" cy="{ly_leg + 50}" r="3.2" fill="#f59e0b"/><text class="lbl-dim" x="{sx0 + 38}" y="{ly_leg + 53}">Amber dots: recent trails (trails.json)</text>')
    if highlight:
        svg.append(f'<circle cx="{sx0 + 24}" cy="{ly_leg + 71}" r="3.8" fill="#ffffff" stroke="#ec4899" stroke-width="1.3"/><text class="lbl-dim" x="{sx0 + 38}" y="{ly_leg + 74}">Pink ring: "{html.escape(highlight)}" ({len(hl_set)} tracks)</text>')
    svg.append(f'<circle cx="{sx0 + 24}" cy="{ly_leg + 92}" r="5.5" fill="none" stroke="#38bdf8" stroke-width="1.6"/><text class="lbl-dim" x="{sx0 + 38}" y="{ly_leg + 95}">Blue ring: start of session walk</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def render_focus_topo_svg(
    manifold: CorpusManifold,
    trails: TrailLog,
    session_paths: list[str],
    wcfg: WanderConfig,
    focus: str,
    *,
    now: float | None = None,
) -> str:
    """Render a local 2D spherical topographic heat map + radial profile around tracks matching `focus`."""
    t_now = time.time() if now is None else now
    needle = focus.lower()
    focus_indices = [i for i, p in enumerate(manifold.paths) if needle in p.lower()]
    if not focus_indices:
        raise EchoControlError(f"no indexed tracks matched focus {focus!r}")

    hf = _build_heat_field(manifold, session_paths, wcfg)
    whf_all, _, _ = _eval_field_weighted(hf, wcfg, manifold.means)
    nh_all = np.array([trails.node_heat(p, now=t_now) for p in manifold.paths], dtype=np.float32)
    sess_set = set(session_paths)

    # Pick 3 anchor vectors to define the 2D spherical slice (preferring hottest/played focus tracks)
    ranked_focus = sorted(focus_indices, key=lambda i: -(float(whf_all[i]) + float(nh_all[i])))
    if len(ranked_focus) >= 3:
        v1, v2, v3 = (manifold.means[ranked_focus[j]] for j in range(3))
        c = _unit(v1 + v2 + v3)
        t1 = v1 - float(v1 @ c) * c
        e1 = _unit(t1)
        t2 = v2 - float(v2 @ c) * c
        t2 = t2 - float(t2 @ e1) * e1
        e2 = _unit(t2)
    else:
        c = _unit(manifold.means[ranked_focus].mean(axis=0))
        all_cos = manifold.means @ c
        nbrs = np.argsort(-all_cos)[: min(30, len(manifold.paths))]
        tangents = manifold.means[nbrs] - (manifold.means[nbrs] @ c)[:, None] * c[None, :]
        _, _, vt = np.linalg.svd(tangents, full_matrices=False)
        e1 = _unit(vt[0])
        e2 = _unit(vt[1] if vt.shape[0] > 1 else np.roll(e1, 1))

    def proj_2d(v: np.ndarray) -> tuple[float, float]:
        vc = max(1e-4, float(v @ c))
        return float(v @ e1) / vc, float(v @ e2) / vc

    w_svg, h_svg = TOPO_WIDTH, TOPO_HEIGHT
    mx0, my0, mw, mh = 60, 75, 620, 540
    rx0, ry0, rw, rh = 755, 75, 480, 540
    xmin, xmax = -0.95, 0.95
    ymin, ymax = -0.82, 0.82

    nx_g, ny_g = 120, 104
    xs = np.linspace(xmin, xmax, nx_g)
    ys = np.linspace(ymin, ymax, ny_g)
    xx, yy = np.meshgrid(xs, ys)
    grid_vecs = c[None, None, :] + xx[:, :, None] * e1[None, None, :] + yy[:, :, None] * e2[None, None, :]
    norms = np.linalg.norm(grid_vecs, axis=2, keepdims=True)
    grid_unit = (grid_vecs / np.maximum(1e-8, norms)).reshape(-1, c.shape[0]).astype(np.float32)
    z_tot, _, _ = _eval_field_weighted(hf, wcfg, grid_unit)
    z_grid = z_tot.reshape(ny_g, nx_g)

    def map_x(x: float) -> float:
        return mx0 + (x - xmin) / (xmax - xmin) * mw

    def map_y(y: float) -> float:
        return my0 + (ymax - y) / (ymax - ymin) * mh

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_svg}" height="{h_svg}" viewBox="0 0 {w_svg} {h_svg}">',
        """<defs>
  <style>
    .bg { fill: #0b0e17; }
    .panel { fill: #101522; stroke: #252e45; stroke-width: 1.2; }
    .title { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 18px; font-weight: 700; fill: #eef2ff; }
    .subtitle { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 12px; fill: #94a3b8; }
    .sec { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 13.5px; font-weight: 700; fill: #e2e8f0; }
    .lbl { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 10.5px; fill: #e2e8f0; }
    .lbl-dim { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 9.5px; fill: #94a3b8; }
    .lbl-hot { font-family: "DejaVu Sans", "Noto Sans CJK JP", sans-serif; font-size: 10.5px; font-weight: 700; fill: #fff8be; }
    .axis { font-family: "DejaVu Sans Mono", monospace; font-size: 10px; fill: #94a3b8; }
  </style>
  <clipPath id="mapClip"><rect x="60" y="75" width="620" height="540" rx="4"/></clipPath>
</defs>""",
        f'<rect class="bg" x="0" y="0" width="{w_svg}" height="{h_svg}"/>',
        f'<text class="title" x="60" y="34">Topographic Heat Map Around "{html.escape(focus)}" ({len(focus_indices)} Tracks)</text>',
        f'<text class="subtitle" x="60" y="54">2D spherical slice (512D CLAP mean space) • Session: {len(session_paths)} tracks • heat_weight={wcfg.heat_weight}, micro_σ={wcfg.micro_sigma}, macro_σ={wcfg.macro_sigma}</text>',
        f'<rect class="panel" x="{mx0}" y="{my0}" width="{mw}" height="{mh}" rx="4"/>',
        '<g clip-path="url(#mapClip)">',
    ]

    cell_w = mw / (nx_g - 1) + 0.6
    cell_h = mh / (ny_g - 1) + 0.6
    for j in range(ny_g - 1):
        y_px = map_y(float(ys[j + 1]))
        for i in range(nx_g - 1):
            val = 0.25 * (z_grid[j, i] + z_grid[j, i + 1] + z_grid[j + 1, i] + z_grid[j + 1, i + 1])
            svg.append(f'<rect x="{map_x(float(xs[i])):.1f}" y="{y_px:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{_heat_color(float(val))}"/>')

    cx_px, cy_px = map_x(0.0), map_y(0.0)
    for cos_ring in [0.95, 0.90, 0.85, 0.80]:
        r = math.sqrt(1.0 / (cos_ring**2) - 1.0)
        rx_r = r / (xmax - xmin) * mw
        ry_r = r / (ymax - ymin) * mh
        svg.append(f'<ellipse cx="{cx_px:.1f}" cy="{cy_px:.1f}" rx="{rx_r:.1f}" ry="{ry_r:.1f}" fill="none" stroke="#ffffff" stroke-opacity="0.10" stroke-dasharray="3,4" stroke-width="1"/>')
        svg.append(f'<text class="axis" x="{cx_px + rx_r * 0.72:.1f}" y="{cy_px - ry_r * 0.72:.1f}" fill-opacity="0.55">cos {cos_ring:.2f}</text>')

    for lvl, col, op, sw in [
        (0.015, "#cbd5e1", 0.35, 0.8),
        (0.025, "#e2e8f0", 0.45, 0.9),
        (0.050, "#fbbf24", 0.60, 1.1),
        (0.100, "#fde047", 0.75, 1.2),
        (0.200, "#fef08a", 0.85, 1.3),
        (0.300, "#ffffff", 0.92, 1.4),
    ]:
        segs = _marching_squares(z_grid, xs, ys, lvl)
        if segs:
            d_str = "".join(f"M{map_x(x0):.1f},{map_y(y0):.1f}L{map_x(x1):.1f},{map_y(y1):.1f}" for x0, y0, x1, y1 in segs)
            svg.append(f'<path d="{d_str}" fill="none" stroke="{col}" stroke-opacity="{op}" stroke-width="{sw}"/>')

    # Nearby non-focus neighbors
    focus_set = set(focus_indices)
    all_cos_c = manifold.means @ c
    nbr_indices = [int(i) for i in np.argsort(-all_cos_c) if int(i) not in focus_set][:80]
    for i in nbr_indices:
        x_p, y_p = proj_2d(manifold.means[i])
        if xmin <= x_p <= xmax and ymin <= y_p <= ymax:
            px, py = map_x(x_p), map_y(y_p)
            if manifold.paths[i] in sess_set:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#38bdf8" stroke="#0b0e17" stroke-width="1.2"/>')
            elif nh_all[i] > 0.01:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#a78bfa" fill-opacity="0.85" stroke="#0b0e17" stroke-width="0.9"/>')
            else:
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="#94a3b8" fill-opacity="0.45"/>')

    # Focus tracks
    for i in focus_indices:
        x_p, y_p = proj_2d(manifold.means[i])
        if not (xmin <= x_p <= xmax and ymin <= y_p <= ymax):
            continue
        px, py = map_x(x_p), map_y(y_p)
        if manifold.paths[i] in sess_set:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="none" stroke="#fff8be" stroke-width="1.6" stroke-opacity="0.85"/>')
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#ffffff" stroke="#0b0e17" stroke-width="1.2"/>')
        elif nh_all[i] > 0.01:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="#f59e0b" stroke="#0b0e17" stroke-width="1.2"/>')
        else:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="#22d3ee" fill-opacity="0.85" stroke="#0b0e17" stroke-width="1.1"/>')

    # Label top 5 hottest focus tracks
    for i in ranked_focus[:5]:
        x_p, y_p = proj_2d(manifold.means[i])
        if not (xmin <= x_p <= xmax and ymin <= y_p <= ymax):
            continue
        px, py = map_x(x_p), map_y(y_p)
        stem = Path(manifold.paths[i]).stem[:24]
        txt = f"{stem} ({whf_all[i]:.3f}" + (f" + {nh_all[i]:.2f}n)" if nh_all[i] > 0.001 else ")")
        cls = "lbl-hot" if manifold.paths[i] in sess_set else "lbl"
        dx_l = -140.0 if px > mx0 + mw - 170 else 10.0
        svg.append(f'<text class="{cls}" x="{px + dx_l:.1f}" y="{py - 9:.1f}" stroke="#0b0e17" stroke-width="3.2" paint-order="stroke">{html.escape(txt)}</text>')

    svg.append("</g>")

    # Colorbar
    cbx, cby, cbw, cbh = mx0, my0 + mh + 18, 280, 12
    for k in range(100):
        svg.append(f'<rect x="{cbx + k * (cbw / 100):.1f}" y="{cby}" width="{cbw / 100 + 0.8:.1f}" height="{cbh}" fill="{_heat_color((k / 99.0) * 0.38)}"/>')
    svg.append(f'<rect x="{cbx}" y="{cby}" width="{cbw}" height="{cbh}" fill="none" stroke="#334155" stroke-width="1"/>')
    for v_tick in [0.00, 0.05, 0.10, 0.20, 0.38]:
        tx = cbx + (v_tick / 0.38) * cbw
        svg.append(f'<text class="axis" x="{tx:.1f}" y="{cby + 24}" text-anchor="middle">{v_tick:.2f}</text>')
    svg.append(f'<text class="lbl-dim" x="{cbx + cbw + 14}" y="{cby + 10}">Session Field Heat (field + n = persistent node heat)</text>')

    # Right panel: Radial Cross-Section
    svg.append(f'<rect class="panel" x="{rx0}" y="{ry0}" width="{rw}" height="{rh}" rx="4"/>')
    svg.append(f'<text class="sec" x="{rx0 + 20}" y="{ry0 + 28}">Radial Kernel Profile (heat_weight × T)</text>')
    svg.append(f'<text class="lbl-dim" x="{rx0 + 20}" y="{ry0 + 45}">Micro (σ={wcfg.micro_sigma}) vs. Macro EMA (σ={wcfg.macro_sigma}, β={wcfg.macro_beta})</text>')

    px0, py0, pw, ph = rx0 + 56, ry0 + 72, rw - 82, rh - 155
    dmax, ymax_p = 0.30, 0.42

    def px_x(d: float) -> float:
        return px0 + (d / dmax) * pw

    def px_y(val: float) -> float:
        return py0 + (1.0 - min(ymax_p, max(0.0, val)) / ymax_p) * ph

    for yv in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        yy_p = px_y(yv)
        svg.append(f'<line x1="{px0}" y1="{yy_p:.1f}" x2="{px0 + pw}" y2="{yy_p:.1f}" stroke="#1e293b" stroke-width="1"/>')
        svg.append(f'<text class="axis" x="{px0 - 8}" y="{yy_p + 3:.1f}" text-anchor="end">{yv:.2f}</text>')
    for dv in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        xx_p = px_x(dv)
        svg.append(f'<line x1="{xx_p:.1f}" y1="{py0}" x2="{xx_p:.1f}" y2="{py0 + ph}" stroke="#1e293b" stroke-width="1"/>')
        svg.append(f'<text class="axis" x="{xx_p:.1f}" y="{py0 + ph + 16}" text-anchor="middle">{1.0 - dv:.2f}</text>')
    svg.append(f'<text class="lbl-dim" x="{px0 + pw / 2:.1f}" y="{py0 + ph + 34}" text-anchor="middle">Cosine Similarity to Hottest Focus Track</text>')

    ds = np.linspace(0.0, dmax, 200)
    mic_c = wcfg.heat_weight * np.exp(-ds / (wcfg.micro_sigma**2))
    mac_c = wcfg.heat_weight * wcfg.macro_beta * np.exp(-ds / (wcfg.macro_sigma**2))
    tot_c = mic_c + mac_c

    def _poly(xs_a: np.ndarray, ys_a: np.ndarray) -> str:
        return " ".join(f"{px_x(float(x)):.1f},{px_y(float(y)):.1f}" for x, y in zip(xs_a, ys_a, strict=True))

    svg.append(f'<polygon points="{px_x(0):.1f},{px_y(0):.1f} {_poly(ds, tot_c)} {px_x(dmax):.1f},{px_y(0):.1f}" fill="#f97316" fill-opacity="0.16"/>')
    svg.append(f'<polyline points="{_poly(ds, mac_c)}" fill="none" stroke="#a855f7" stroke-width="2" stroke-dasharray="5,4"/>')
    svg.append(f'<polyline points="{_poly(ds, mic_c)}" fill="none" stroke="#f97316" stroke-width="2"/>')
    svg.append(f'<polyline points="{_poly(ds, tot_c)}" fill="none" stroke="#fff8be" stroke-width="2.2"/>')

    # Plot actual focus tracks relative to hottest focus track
    hot_v = manifold.means[ranked_focus[0]]
    for i in focus_indices:
        d_act = max(0.0, 1.0 - float(manifold.means[i] @ hot_v))
        if d_act <= dmax:
            col = "#ffffff" if manifold.paths[i] in sess_set else ("#f59e0b" if nh_all[i] > 0.01 else "#22d3ee")
            r = 4.5 if manifold.paths[i] in sess_set else 3.5
            svg.append(f'<circle cx="{px_x(d_act):.1f}" cy="{px_y(float(whf_all[i])):.1f}" r="{r}" fill="{col}" stroke="#0b0e17" stroke-width="1"/>')

    svg.append("</svg>")
    return "\n".join(svg)


def write_atlas(svg_text: str, output: Path, *, width: int, height: int) -> Path:
    """Write SVG directly, or rasterize to PNG via `rsvg-convert` if `output` ends in `.png`."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".png":
        output.write_text(svg_text, encoding="utf-8")
        return output

    rsvg = shutil.which("rsvg-convert")
    if rsvg is None:
        raise EchoControlError(
            "rsvg-convert is not installed (install librsvg, or pass --output *.svg)"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".svg", encoding="utf-8", delete=False) as tf:
        tf.write(svg_text)
        tmp_svg = Path(tf.name)
    try:
        r = subprocess.run(
            [rsvg, "-w", str(width), "-h", str(height), str(tmp_svg), "-o", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            raise EchoControlError(f"rsvg-convert failed: {r.stderr.strip()}")
    finally:
        tmp_svg.unlink(missing_ok=True)
    return output


def generate_map(
    cfg: Config,
    *,
    output: Path | None = None,
    clusters: int = 15,
    highlight: str | None = None,
    focus: str | None = None,
    date: str | None = None,
    open_after: bool = False,
    label_mode: str = "clusters",
) -> Path:
    """Connect to the local chunk index, render the requested map, and save it."""
    from .strategies import open_store

    store = open_store(cfg)
    if not isinstance(store, ChunkIndex):
        raise EchoControlError("`echo-control map` requires [model] strategy = 'chunks'")
    if not store.exists():
        raise EchoControlError("the index is empty; start the daemon or run `echo-control scan`")

    manifold = CorpusManifold.from_store(
        store,
        neighbors=cfg.wander.csls_neighbors,
        cache_path=CACHE_DIR / f"manifold_{cfg.qdrant.collection}.npz",
    )
    trails = TrailLog.load(STATE_DIR / "trails.json", cfg.wander)
    session_runs = load_session_runs(STATE_DIR / "logs", date)
    session_paths = [p for run in session_runs for p in run]

    default_ext = ".png" if shutil.which("rsvg-convert") else ".svg"
    if output is None:
        stem = f"topo_{focus.replace('/', '_')}" if focus else "library_atlas"
        output = Path.cwd() / f"{stem}{default_ext}"

    if focus:
        svg_text = render_focus_topo_svg(manifold, trails, session_paths, cfg.wander, focus)
        written = write_atlas(svg_text, output, width=TOPO_WIDTH, height=TOPO_HEIGHT)
    else:
        tsne_cache = CACHE_DIR / f"atlas_tsne_{cfg.qdrant.collection}.npz"
        svg_text = render_library_atlas_svg(
            manifold,
            trails,
            session_paths,
            cfg.wander,
            clusters=clusters,
            highlight=highlight,
            library_roots=cfg.library.paths,
            cache_path=tsne_cache,
            label_mode=label_mode,
            session_runs=session_runs,
        )
        written = write_atlas(svg_text, output, width=ATLAS_WIDTH, height=ATLAS_HEIGHT)

    if open_after:
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen(
                [opener, str(written)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    return written


class LiveAtlasRenderer:
    """Fast in-memory Doppler atlas renderer for live mpv overlay toggling."""

    def __init__(
        self,
        *,
        manifold: CorpusManifold,
        trails: TrailLog,
        cfg: Config,
        log_dir: Path,
        cache_dir: Path = CACHE_DIR,
        clusters: int = 15,
    ) -> None:
        self.manifold: CorpusManifold = manifold
        self.trails: TrailLog = trails
        self.cfg: Config = cfg
        self.log_dir: Path = log_dir
        self.cache_dir: Path = cache_dir
        self.clusters: int = clusters
        self._coords: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._coastlines: list[str] | None = None
        self._slot: int = 0

    def render(
        self, current_path: str | None = None, next_path: str | None = None
    ) -> Path | None:
        if len(self.manifold.paths) < 3 or shutil.which("rsvg-convert") is None:
            return None
        tsne_cache = self.cache_dir / f"atlas_tsne_{self.cfg.qdrant.collection}.npz"
        if self._coords is None:
            self._coords = compute_tsne_layout(self.manifold, cache_path=tsne_cache)
        if self._labels is None:
            k_eff = max(1, min(self.clusters, len(self.manifold.paths)))
            self._labels = _compute_clusters(self.manifold, k_eff, cache_path=tsne_cache)
        if self._coastlines is None:
            xmin, xmax = float(self._coords[:, 0].min()), float(self._coords[:, 0].max())
            ymin, ymax = float(self._coords[:, 1].min()), float(self._coords[:, 1].max())
            span_x = max(1e-3, xmax - xmin)
            span_y = max(1e-3, ymax - ymin)
            self._coastlines = _compute_coastlines_svg(
                self._coords,
                xmin - span_x * 0.07,
                xmax + span_x * 0.07,
                ymin - span_y * 0.07,
                ymax + span_y * 0.07,
                36,
                72,
                1010,
                812,
            )
        session_runs = load_session_runs(self.log_dir)
        if current_path:
            if not session_runs:
                session_runs = [[current_path]]
            elif session_runs[-1][-1] != current_path:
                session_runs[-1].append(current_path)
        session_paths = [p for run in session_runs for p in run]
        svg = render_library_atlas_svg(
            self.manifold,
            self.trails,
            session_paths,
            self.cfg.wander,
            clusters=self.clusters,
            library_roots=self.cfg.library.paths,
            coords=self._coords,
            labels=self._labels,
            coastlines_svg=self._coastlines,
            label_mode="recent",
            session_runs=session_runs,
            next_path=next_path,
        )
        self._slot = 1 - self._slot
        out = self.cache_dir / f"live_atlas_{self._slot}.png"
        return write_atlas(svg, out, width=ATLAS_WIDTH, height=ATLAS_HEIGHT)

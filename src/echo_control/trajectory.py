# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How a track moves through embedding space, from its ordered unit-norm chunk vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .segments import normalize

MAD_SCALE = 1.4826
# Windows this far below the track's median loudness at either end count as fade.
FADE_DB = 12.0


def angles(vecs: np.ndarray) -> np.ndarray:
    """Angle in radians between each consecutive pair of unit vectors."""
    if len(vecs) < 2:
        return np.zeros(0)
    dots = np.einsum("ij,ij->i", vecs[:-1], vecs[1:])
    return np.arccos(np.clip(dots, -1.0, 1.0))


def speeds(vecs: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Angular speed (rad/s) arriving at each chunk; the first chunk has none (nan)."""
    out = np.full(len(vecs), np.nan)
    if len(vecs) > 1:
        dt = np.maximum(np.diff(starts), 1e-6)
        out[1:] = angles(vecs) / dt
    return out


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) < 2:
        return x.copy()
    half = width // 2
    padded = np.pad(x, half, mode="edge")
    return np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


def find_spikes(x: np.ndarray, k: float, min_gap: int) -> list[int]:
    """Indices of local peaks more than k robust deviations above the median, keeping the
    tallest when peaks fall within min_gap samples of each other."""
    if len(x) < 3:
        return []
    med = float(np.median(x))
    mad = MAD_SCALE * float(np.median(np.abs(x - med)))
    threshold = med + k * max(mad, 1e-9)
    peaks = [
        i
        for i in range(len(x))
        if x[i] > threshold
        and (i == 0 or x[i] > x[i - 1])
        and (i == len(x) - 1 or x[i] >= x[i + 1])
    ]
    kept: list[int] = []
    for i in sorted(peaks, key=lambda i: -x[i]):
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(i)
    return sorted(kept)


def summarize(
    vecs: np.ndarray,
    starts: np.ndarray,
    window_seconds: float,
    smooth: int = 5,
    k: float = 4.0,
) -> dict[str, Any]:
    """Per-track trajectory features. Speeds are smoothed before looking for spikes (section
    changes) so CLAP's window-to-window jitter doesn't register as structure; a median
    filter would erase hard cuts, which move the embedding for only a sample or two."""
    starts = np.asarray(starts, dtype=np.float64)
    speed = speeds(vecs, starts)[1:]
    if not len(speed):
        return {
            "speed_mean": 0.0, "speed_std": 0.0, "spikes": 0, "spike_times": [],
            "tortuosity": None, "intro_len": None,
        }
    smoothed = moving_average(speed, smooth)
    hop = float(np.median(np.diff(starts)))
    spikes = find_spikes(smoothed, k, max(1, round(window_seconds / hop)))
    # A boundary moves the embedding fastest while it crosses the middle of the windows.
    mid = (starts[:-1] + starts[1:]) / 2 + window_seconds / 2
    times = [round(float(mid[i]), 2) for i in spikes]
    displacement = float(np.arccos(np.clip(vecs[0] @ vecs[-1], -1.0, 1.0)))
    path = float(angles(vecs).sum())
    return {
        "speed_mean": round(float(speed.mean()), 5),
        "speed_std": round(float(speed.std()), 5),
        "spikes": len(spikes),
        "spike_times": times,
        "tortuosity": round(path / displacement, 3) if displacement > 1e-3 else None,
        "intro_len": times[0] if times else None,
    }


def aligned_similarity(a: np.ndarray, b: np.ndarray, max_shift: int = 8) -> float:
    """Mean window-by-window similarity of two tracks at their best time offset (up to
    max_shift windows either way, for different lead-ins). Two files of one recording line
    up almost exactly; tracks that merely sound alike on average don't."""
    best = -1.0
    for k in range(-max_shift, max_shift + 1):
        x, y = (a[k:], b) if k >= 0 else (a, b[-k:])
        n = min(len(x), len(y))
        if n >= min(4, len(a), len(b)) and n > 0:
            best = max(best, float(np.einsum("ij,ij->i", x[:n], y[:n]).mean()))
    return best


@dataclass
class Edge:
    """Where a track sits and where it's heading over its first or last stretch."""

    position: np.ndarray
    direction: np.ndarray | None
    drift: float


def _edge(vecs: np.ndarray, t: np.ndarray) -> Edge:
    position = normalize(vecs.mean(axis=0))
    if len(vecs) < 3:
        return Edge(position, None, 0.0)
    # Least-squares slope of the vectors over time, projected onto the sphere's tangent
    # plane at the position: the net drift, which window-to-window jitter mostly cancels out of.
    tc = t - t.mean()
    slope = (tc[:, None] * vecs).sum(axis=0) / float(tc @ tc)
    tangent = slope - (slope @ position) * position
    drift = float(np.linalg.norm(tangent))
    return Edge(position, tangent / drift if drift > 1e-9 else None, drift)


def edges(
    vecs: np.ndarray,
    starts: np.ndarray,
    silent: list[bool],
    db: list[float],
    edge_seconds: float,
    fade_db: float = FADE_DB,
) -> tuple[Edge, Edge]:
    """Intro and outro: the first and last edge_seconds of the track at full level, skipping
    leading/trailing silence and fades so every outro doesn't look like the same fade-out."""
    starts = np.asarray(starts, dtype=np.float64)
    db_arr = np.asarray(db, dtype=np.float64)
    loud = ~np.asarray(silent, dtype=bool)
    ref = float(np.median(db_arr[loud])) if loud.any() else float(np.median(db_arr))
    full = np.flatnonzero((db_arr >= ref - fade_db) & loud)
    if not len(full):
        full = np.arange(len(vecs))
    first, last = int(full[0]), int(full[-1])
    intro = np.flatnonzero((starts >= starts[first]) & (starts < starts[first] + edge_seconds))
    outro = np.flatnonzero((starts <= starts[last]) & (starts > starts[last] - edge_seconds))
    return _edge(vecs[intro], starts[intro]), _edge(vecs[outro], starts[outro])

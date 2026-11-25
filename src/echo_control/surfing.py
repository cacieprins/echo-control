# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Interactive spherical bisection channel-surfing on the centered corpus manifold."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .cooldown import HeatField, TrailLog
from .manifold import CorpusManifold

_EPS = 1e-8


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).astype(np.float32) if norm > _EPS else arr


def _principal_axis(mat: np.ndarray, iters: int = 3) -> np.ndarray:
    """Approximate top principal direction of rows in `mat` via fast power iteration."""
    if mat.shape[0] <= 1:
        return _unit(mat[0]) if mat.shape[0] == 1 else np.zeros(mat.shape[1], dtype=np.float32)
    centered = mat - np.mean(mat, axis=0, keepdims=True)
    # Deterministic seed from the row with largest norm
    norms = np.linalg.norm(centered, axis=1)
    best_row = int(np.argmax(norms))
    if float(norms[best_row]) <= _EPS:
        return _unit(mat[0])
    v = _unit(centered[best_row])
    for _ in range(iters):
        proj = centered @ v
        nxt = centered.T @ proj
        norm = float(np.linalg.norm(nxt))
        if norm <= _EPS:
            break
        v = (nxt / norm).astype(np.float32)
    return v


@dataclass(frozen=True)
class SurfResult:
    path: str
    mode: str
    viable_count: int
    total_count: int
    wrapped: bool = False


@dataclass
class ChannelSurfer:
    """Spherical version-space binary search across centered track embeddings.

    Supports both 1-key dwell-time `[b]` (cold < 3s median halfspace cut,
    warm 3s-15s near-half + principal-axis refinement, settle >= 15s reset)
    and explicit 2-key `[v]` geodesic midpoint (`SLERP(0.5)`) bracketing.
    """

    manifold: CorpusManifold
    trails: TrailLog | None = None
    heat_field: HeatField | None = None
    cold_seconds: float = 3.0
    settle_seconds: float = 15.0
    opening_cut: float = 0.15
    min_viable: int = 8
    active: bool = False
    viable: np.ndarray = field(init=False)
    auditioned: set[int] = field(default_factory=set)
    prev_idx: int | None = None
    curr_idx: int | None = None
    last_tap_ts: float | None = None
    last_mode: str = ""

    def __post_init__(self) -> None:
        self.viable = np.ones(len(self.manifold.paths), dtype=bool)

    def reset(self) -> None:
        self.active = False
        self.viable = np.ones(len(self.manifold.paths), dtype=bool)
        self.auditioned.clear()
        self.prev_idx = None
        self.curr_idx = None
        self.last_tap_ts = None
        self.last_mode = ""

    def check_settled(self, now: float | None = None) -> bool:
        if not self.active or self.last_tap_ts is None:
            return False
        t = time.monotonic() if now is None else now
        if (t - self.last_tap_ts) >= self.settle_seconds:
            self.reset()
            return True
        return False

    def status_badge(self, now: float | None = None) -> str:
        if self.check_settled(now) or not self.active:
            return ""
        return f"surf: {int(np.sum(self.viable))}/{len(self.manifold.paths)} ({self.last_mode})"

    def _pick_best(self, indices: np.ndarray, target_dir: np.ndarray) -> int:
        sims = self.manifold.centered[indices] @ target_dir
        if self.trails is not None:
            penalties = np.array(
                [self.trails.node_heat(self.manifold.paths[int(i)]) for i in indices],
                dtype=np.float32,
            )
            scores = sims - penalties
        else:
            scores = sims
        return int(indices[int(np.argmax(scores))])

    def _inject_rejection_spike(self, track_path: str) -> None:
        if self.heat_field is None:
            return
        mean_vec = self.manifold.mean_for(track_path)
        if mean_vec is not None:
            self.heat_field.inject_spike(mean_vec)

    def _reset_viable_minus_auditioned(self, exclude_idx: int | None) -> np.ndarray:
        n = len(self.manifold.paths)
        mask = np.ones(n, dtype=bool)
        for idx in self.auditioned:
            if 0 <= idx < n:
                mask[idx] = False
        if exclude_idx is not None and 0 <= exclude_idx < n:
            mask[exclude_idx] = False
        if not np.any(mask):
            mask = np.ones(n, dtype=bool)
            if exclude_idx is not None and 0 <= exclude_idx < n:
                mask[exclude_idx] = False
        return mask

    def bail(self, current_path: str, *, now: float | None = None) -> SurfResult | None:
        n = len(self.manifold.paths)
        if n < 2:
            return None
        t = time.monotonic() if now is None else now
        self._inject_rejection_spike(current_path)

        cur_idx = self.manifold.index.get(current_path)
        if cur_idx is None:
            cur_idx = self.curr_idx

        is_fresh = (
            not self.active
            or self.last_tap_ts is None
            or (t - self.last_tap_ts) >= self.settle_seconds
        )

        if is_fresh:
            self.reset()
            if cur_idx is not None:
                self.auditioned.add(cur_idx)
                self.viable[cur_idx] = False
                p0 = self.manifold.centered[cur_idx]
                far_mask = self.viable & ((self.manifold.centered @ p0) <= self.opening_cut)
                if np.any(far_mask):
                    self.viable = far_mask
                target = -p0
            else:
                target = _unit(-self.manifold.mu)
            surv = np.flatnonzero(self.viable)
            if len(surv) == 0:
                self.viable = self._reset_viable_minus_auditioned(cur_idx)
                surv = np.flatnonzero(self.viable)
            chosen = self._pick_best(surv, target)
            self.prev_idx = cur_idx
            self.curr_idx = chosen
            self.auditioned.add(chosen)
            self.viable[chosen] = False
            self.active = True
            self.last_tap_ts = t
            self.last_mode = "antipode"
            return SurfResult(
                path=self.manifold.paths[chosen],
                mode="antipode",
                viable_count=int(np.sum(self.viable)),
                total_count=n,
                wrapped=False,
            )

        assert self.last_tap_ts is not None
        dt = t - self.last_tap_ts
        if cur_idx is not None:
            self.auditioned.add(cur_idx)
            self.viable[cur_idx] = False

        wrapped = False
        if int(np.sum(self.viable)) < self.min_viable:
            self.viable = self._reset_viable_minus_auditioned(cur_idx)
            wrapped = True

        surv = np.flatnonzero(self.viable)
        if len(surv) == 0:
            return None

        p_k = (
            self.manifold.centered[cur_idx]
            if cur_idx is not None
            else _unit(np.mean(self.manifold.centered[surv], axis=0))
        )
        m_k = _unit(np.sum(self.manifold.centered[surv], axis=0))
        normal_raw = p_k - m_k
        n_k = _unit(normal_raw) if float(np.linalg.norm(normal_raw)) > _EPS else p_k

        proj = self.manifold.centered[surv] @ n_k
        theta_med = float(np.median(proj))

        if dt < self.cold_seconds:
            # Cold reject: discard the half closer to p_k
            keep_mask = proj < theta_med
            if not np.any(keep_mask):
                keep_mask = proj <= theta_med
            kept = surv[keep_mask]
            new_viable = np.zeros(n, dtype=bool)
            new_viable[kept] = True
            self.viable = new_viable
            chosen = self._pick_best(kept, -n_k)
            mode = "wrap" if wrapped else "cold"
        else:
            # Warm linger: keep the near half and step along its top principal variance axis
            keep_mask = proj >= theta_med
            kept = surv[keep_mask] if np.any(keep_mask) else surv
            new_viable = np.zeros(n, dtype=bool)
            new_viable[kept] = True
            self.viable = new_viable
            sub = self.manifold.centered[kept]
            v1 = _principal_axis(sub, iters=3)
            if float(p_k @ v1) > 0.0:
                v1 = -v1
            mean_sub = _unit(np.mean(sub, axis=0))
            target = _unit(mean_sub + 0.5 * v1)
            chosen = self._pick_best(kept, target)
            mode = "wrap" if wrapped else "warm"

        self.prev_idx = cur_idx
        self.curr_idx = chosen
        self.auditioned.add(chosen)
        self.viable[chosen] = False
        self.last_tap_ts = t
        self.last_mode = mode
        return SurfResult(
            path=self.manifold.paths[chosen],
            mode=mode,
            viable_count=int(np.sum(self.viable)),
            total_count=n,
            wrapped=wrapped,
        )

    def warmer(self, current_path: str, *, now: float | None = None) -> SurfResult | None:
        """Explicit `[v]` geodesic midpoint (SLERP 0.5) between previous and current probes."""
        n = len(self.manifold.paths)
        if n < 2:
            return None
        t = time.monotonic() if now is None else now
        if self.check_settled(t) or not self.active or self.prev_idx is None:
            return self.bail(current_path, now=t)

        cur_idx = self.manifold.index.get(current_path)
        if cur_idx is None:
            cur_idx = self.curr_idx
        if cur_idx is None:
            return self.bail(current_path, now=t)

        self.auditioned.add(cur_idx)
        self.viable[cur_idx] = False

        surv = np.flatnonzero(self.viable)
        wrapped = False
        if len(surv) == 0:
            self.viable = self._reset_viable_minus_auditioned(cur_idx)
            surv = np.flatnonzero(self.viable)
            wrapped = True
        if len(surv) == 0:
            return None

        p_prev = self.manifold.centered[self.prev_idx]
        p_curr = self.manifold.centered[cur_idx]
        mid_raw = np.asarray(p_prev + p_curr, dtype=np.float32)
        if float(np.linalg.norm(mid_raw)) > 1e-6:
            q_mid = _unit(mid_raw)
        else:
            q_mid = _principal_axis(self.manifold.centered[surv], iters=3)

        chosen = self._pick_best(surv, q_mid)
        self.prev_idx = cur_idx
        self.curr_idx = chosen
        self.auditioned.add(chosen)
        self.viable[chosen] = False
        self.last_tap_ts = t
        self.last_mode = "midpoint"
        return SurfResult(
            path=self.manifold.paths[chosen],
            mode="midpoint",
            viable_count=int(np.sum(self.viable)),
            total_count=n,
            wrapped=wrapped,
        )

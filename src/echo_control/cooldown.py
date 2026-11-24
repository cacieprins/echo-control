# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dynamic dual-scale von Mises-Fisher heat field, persistent anti-Hebbian trail log,
cooldown scoring, and close-call Boltzmann sampling."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .config import WanderConfig
from .manifold import CorpusManifold

_EPS = 1e-8
_PRUNE_THRESHOLD = 1e-3
_MICRO_PRUNE = 1e-4


class _Scored(Protocol):
    score: float


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).astype(np.float32) if norm > _EPS else arr


@dataclass
class HeatField:
    """In-session repulsive scalar potential T(x) on the unit sphere."""

    micro_sigma: float = 0.10
    micro_tau: float = 15.0
    macro_sigma: float = 0.30
    macro_beta: float = 0.20
    macro_alpha: float = 0.80
    _step: int = 0
    _history: list[tuple[np.ndarray, int, float]] = field(default_factory=list)
    _macro_centroid: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: WanderConfig) -> HeatField:
        return cls(
            micro_sigma=cfg.micro_sigma,
            micro_tau=cfg.micro_tau,
            macro_sigma=cfg.macro_sigma,
            macro_beta=cfg.macro_beta,
            macro_alpha=cfg.macro_alpha,
        )

    def record(self, vec: np.ndarray, *, weight: float = 1.0, update_macro: bool = True) -> None:
        u = _unit(vec)
        self._step += 1
        self._history.append((u, self._step, float(weight)))
        self._prune_history()
        if update_macro:
            if self._macro_centroid is None:
                self._macro_centroid = u
            else:
                blended = self.macro_alpha * self._macro_centroid + (1.0 - self.macro_alpha) * u
                self._macro_centroid = _unit(blended)

    def inject_spike(self, vec: np.ndarray, weight: float = 1.5) -> None:
        """Inject immediate micro-kernel repulsion (e.g. on channel-flip bail) without moving macro EMA."""
        self.record(vec, weight=weight, update_macro=False)

    def _prune_history(self) -> None:
        kept: list[tuple[np.ndarray, int, float]] = []
        for u, step, base_w in self._history:
            w = base_w * math.exp(-(self._step - step) / self.micro_tau)
            if w >= _MICRO_PRUNE:
                kept.append((u, step, base_w))
        self._history = kept

    def heat(self, vec: np.ndarray) -> float:
        u = _unit(vec)
        total = 0.0
        micro_denom = self.micro_sigma * self.micro_sigma
        for h_vec, step, base_w in self._history:
            w = base_w * math.exp(-(self._step - step) / self.micro_tau)
            dist = max(0.0, 1.0 - float(u @ h_vec))
            total += w * math.exp(-dist / micro_denom)
        if self._macro_centroid is not None and self.macro_beta > 0.0:
            macro_denom = self.macro_sigma * self.macro_sigma
            dist = max(0.0, 1.0 - float(u @ self._macro_centroid))
            total += self.macro_beta * math.exp(-dist / macro_denom)
        return total


def _decay(weight: float, elapsed_seconds: float, halflife_seconds: float) -> float:
    if weight <= 0.0:
        return 0.0
    if elapsed_seconds <= 0.0:
        return weight
    return weight * (0.5 ** (elapsed_seconds / halflife_seconds))


@dataclass
class TrailLog:
    """Persistent anti-Hebbian node heat H_j and directed edge wear W_{i->j}."""

    path: Path | None = None
    node_heat_step: float = 0.15
    node_halflife_hours: float = 60.0
    edge_wear_step: float = 0.25
    edge_halflife_days: float = 14.0
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)

    @property
    def _node_halflife_s(self) -> float:
        return self.node_halflife_hours * 3600.0

    @property
    def _edge_halflife_s(self) -> float:
        return self.edge_halflife_days * 86400.0

    @classmethod
    def load(cls, path: Path, cfg: WanderConfig | None = None) -> TrailLog:
        c = cfg or WanderConfig()
        log = cls(
            path=path,
            node_heat_step=c.node_heat_step,
            node_halflife_hours=c.node_halflife_hours,
            edge_wear_step=c.edge_wear_step,
            edge_halflife_days=c.edge_halflife_days,
        )
        if not path.exists():
            return log
        try:
            raw = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return log
        raw_nodes = raw.get("nodes")
        if isinstance(raw_nodes, dict):
            for k, entry in cast("dict[str, Any]", raw_nodes).items():
                if isinstance(entry, dict):
                    ed = cast("dict[str, Any]", entry)
                    w, ts = ed.get("w"), ed.get("ts")
                    if isinstance(w, (int, float)) and isinstance(ts, (int, float)):
                        log.nodes[str(k)] = (float(w), float(ts))
        raw_edges = raw.get("edges")
        if isinstance(raw_edges, dict):
            for src, targets in cast("dict[str, Any]", raw_edges).items():
                if not isinstance(targets, dict):
                    continue
                inner: dict[str, tuple[float, float]] = {}
                for dst, entry in cast("dict[str, Any]", targets).items():
                    if isinstance(entry, dict):
                        ed = cast("dict[str, Any]", entry)
                        w, ts = ed.get("w"), ed.get("ts")
                        if isinstance(w, (int, float)) and isinstance(ts, (int, float)):
                            inner[str(dst)] = (float(w), float(ts))
                if inner:
                    log.edges[str(src)] = inner
        return log

    def node_heat(self, track_path: str, now: float | None = None) -> float:
        entry = self.nodes.get(track_path)
        if entry is None:
            return 0.0
        t = time.time() if now is None else now
        return _decay(entry[0], t - entry[1], self._node_halflife_s)

    def edge_wear(self, prev_path: str, next_path: str, now: float | None = None) -> float:
        targets = self.edges.get(prev_path)
        if not targets:
            return 0.0
        entry = targets.get(next_path)
        if entry is None:
            return 0.0
        t = time.time() if now is None else now
        return _decay(entry[0], t - entry[1], self._edge_halflife_s)

    def commit(
        self,
        track_path: str,
        prev_path: str | None = None,
        *,
        now: float | None = None,
        save: bool = True,
    ) -> None:
        t = time.time() if now is None else now
        cur_node = self.node_heat(track_path, now=t)
        self.nodes[track_path] = (cur_node + self.node_heat_step, t)
        if prev_path is not None and prev_path != track_path:
            cur_edge = self.edge_wear(prev_path, track_path, now=t)
            self.edges.setdefault(prev_path, {})[track_path] = (cur_edge + self.edge_wear_step, t)
        self.prune(now=t)
        if save and self.path is not None:
            self.save()

    def prune(self, now: float | None = None) -> None:
        t = time.time() if now is None else now
        self.nodes = {
            k: (w, ts)
            for k, (w, ts) in self.nodes.items()
            if _decay(w, t - ts, self._node_halflife_s) >= _PRUNE_THRESHOLD
        }
        pruned_edges: dict[str, dict[str, tuple[float, float]]] = {}
        for src, targets in self.edges.items():
            kept = {
                dst: (w, ts)
                for dst, (w, ts) in targets.items()
                if _decay(w, t - ts, self._edge_halflife_s) >= _PRUNE_THRESHOLD
            }
            if kept:
                pruned_edges[src] = kept
        self.edges = pruned_edges

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "nodes": {k: {"w": round(w, 6), "ts": ts} for k, (w, ts) in self.nodes.items()},
            "edges": {
                src: {dst: {"w": round(w, 6), "ts": ts} for dst, (w, ts) in targets.items()}
                for src, targets in self.edges.items()
            },
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


@dataclass
class CooldownScorer:
    """Combines static CSLS hubness, dynamic dual-scale heat, and persistent trail wear."""

    manifold: CorpusManifold
    heat_field: HeatField
    trails: TrailLog
    cfg: WanderConfig = field(default_factory=WanderConfig)

    def penalty(
        self,
        prev_path: str,
        cand_path: str,
        cand_mean: np.ndarray,
        *,
        now: float | None = None,
    ) -> tuple[float, float, float]:
        """Return (total_penalty, csls_term, heat_term)."""
        csls_val = self.manifold.csls_for(cand_path)
        csls_term = self.cfg.csls_gamma * csls_val
        heat_val = self.heat_field.heat(cand_mean)
        heat_term = self.cfg.heat_weight * heat_val
        node_h = self.trails.node_heat(cand_path, now=now)
        edge_w = self.trails.edge_wear(prev_path, cand_path, now=now)
        total = csls_term + heat_term + node_h + edge_w
        return total, csls_term, heat_term + node_h + edge_w

    def commit_play(
        self,
        path: str,
        prev_path: str | None = None,
        vec: np.ndarray | None = None,
        *,
        persist: bool = True,
        now: float | None = None,
    ) -> None:
        v = vec if vec is not None else self.manifold.mean_for(path)
        if v is not None:
            self.heat_field.record(v)
        if persist:
            self.trails.commit(path, prev_path, now=now, save=True)


def sample_next[T: _Scored](
    ranked: list[T],
    cfg: WanderConfig,
    rng: np.random.Generator | None = None,
) -> T:
    """Masked top-k Boltzmann sampling within softmax_margin of the top score."""
    if len(ranked) == 1 or cfg.softmax_temp <= 0.0:
        return ranked[0]
    best = ranked[0].score
    pool = [t for t in ranked[: cfg.softmax_k] if t.score >= best - cfg.softmax_margin]
    if len(pool) <= 1:
        return ranked[0]
    scores = np.array([t.score for t in pool], dtype=np.float64)
    logits = (scores - best) / cfg.softmax_temp
    weights = np.exp(logits)
    probs = weights / np.sum(weights)
    gen = rng if rng is not None else np.random.default_rng()
    idx = int(gen.choice(len(pool), p=probs))
    return pool[idx]

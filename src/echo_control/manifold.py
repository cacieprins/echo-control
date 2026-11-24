# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Static corpus manifold geometry: centering around the global CLAP cone mean
and Cross-Domain Similarity Local Scaling (CSLS) hubness penalties."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

_EPS = 1e-8
_CSLS_BATCH = 512


class ManifoldStore(Protocol):
    def all_track_vectors(self) -> tuple[list[str], np.ndarray, np.ndarray]: ...


def _normalize_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms > _EPS, norms, 1.0)
    return (mat / safe).astype(np.float32), norms.reshape(-1).astype(np.float32)


def _compute_csls(intros: np.ndarray, neighbors: int) -> np.ndarray:
    """Mean cosine similarity of each intro vector to its m nearest neighbors (excluding self)."""
    n = intros.shape[0]
    if n <= 1 or neighbors <= 0:
        return np.zeros(n, dtype=np.float32)
    m = min(neighbors, n - 1)
    unit_intros, _ = _normalize_rows(intros)
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CSLS_BATCH):
        end = min(start + _CSLS_BATCH, n)
        sims = unit_intros[start:end] @ unit_intros.T
        rows = np.arange(end - start)
        sims[rows, start + rows] = -np.inf
        if m < n - 1:
            top = np.partition(sims, -m, axis=1)[:, -m:]
        else:
            # All other tracks (exclude -inf diagonal)
            mask = np.ones_like(sims, dtype=bool)
            mask[rows, start + rows] = False
            top = sims[mask].reshape(end - start, n - 1)
        out[start:end] = np.mean(top, axis=1, dtype=np.float32)
    return out


def _fingerprint(paths: list[str], neighbors: int) -> str:
    h = hashlib.sha256()
    h.update(f"v1:{len(paths)}:{neighbors}\n".encode())
    for p in paths:
        h.update(p.encode("utf-8", errors="surrogateescape"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class CorpusManifold:
    """Centered unit track embeddings and static CSLS hubness scores for the corpus."""

    paths: list[str]
    means: np.ndarray
    mu: np.ndarray
    centered: np.ndarray
    median_radius: float
    csls: np.ndarray
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {p: i for i, p in enumerate(self.paths)}

    @classmethod
    def from_arrays(
        cls,
        paths: list[str],
        means: np.ndarray,
        intros: np.ndarray,
        *,
        neighbors: int = 20,
    ) -> CorpusManifold:
        n = len(paths)
        if n == 0:
            empty_2d = np.empty((0, 0), dtype=np.float32)
            empty_1d = np.empty(0, dtype=np.float32)
            return cls([], empty_2d, empty_1d, empty_2d, 1.0, empty_1d)
        unit_means, _ = _normalize_rows(np.asarray(means, dtype=np.float32))
        mu = np.mean(unit_means, axis=0, dtype=np.float32)
        shifted = unit_means - mu[None, :]
        centered, norms = _normalize_rows(shifted)
        median_radius = float(np.median(norms))
        csls = _compute_csls(np.asarray(intros, dtype=np.float32), neighbors)
        return cls(
            paths=list(paths),
            means=unit_means,
            mu=mu,
            centered=centered,
            median_radius=median_radius,
            csls=csls,
        )

    @classmethod
    def from_store(
        cls,
        store: ManifoldStore,
        *,
        neighbors: int = 20,
        cache_path: Path | None = None,
    ) -> CorpusManifold:
        paths, means, intros = store.all_track_vectors()
        if not paths:
            return cls.from_arrays(paths, means, intros, neighbors=neighbors)

        fp = _fingerprint(paths, neighbors)
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cached_fp = str(data["fingerprint"].item())
                    if cached_fp == fp:
                        return cls(
                            paths=paths,
                            means=np.asarray(data["means"], dtype=np.float32),
                            mu=np.asarray(data["mu"], dtype=np.float32),
                            centered=np.asarray(data["centered"], dtype=np.float32),
                            median_radius=float(data["median_radius"].item()),
                            csls=np.asarray(data["csls"], dtype=np.float32),
                        )
            except (OSError, KeyError, ValueError):
                pass

        manifold = cls.from_arrays(paths, means, intros, neighbors=neighbors)
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
                np.savez(
                    tmp,
                    fingerprint=np.array(fp),
                    means=manifold.means,
                    mu=manifold.mu,
                    centered=manifold.centered,
                    median_radius=np.array(manifold.median_radius, dtype=np.float32),
                    csls=manifold.csls,
                )
                os.replace(tmp, cache_path)
            except OSError:
                pass
        return manifold

    def csls_for(self, path: str) -> float:
        idx = self.index.get(path)
        return float(self.csls[idx]) if idx is not None else 0.0

    def mean_for(self, path: str) -> np.ndarray | None:
        idx = self.index.get(path)
        return self.means[idx] if idx is not None else None

    def centered_for(self, path: str) -> np.ndarray | None:
        idx = self.index.get(path)
        return self.centered[idx] if idx is not None else None

    def raw_probe(self, centered_dir: np.ndarray) -> np.ndarray:
        """Map a centered unit direction back into raw Qdrant query coordinates."""
        q = self.mu + self.median_radius * np.asarray(centered_dir, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        return (q / norm).astype(np.float32) if norm > _EPS else q.astype(np.float32)

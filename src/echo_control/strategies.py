# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Indexing and search strategies (mean, clusters, chunks)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import trajectory
from .config import Config, ModelConfig, StrategyName
from .records import chunk_payload, chunk_track_payload, multivector_payload
from .segments import (
    Windows,
    dense_chunks,
    load_clips,
    normalize,
    sliding_chunks,
    summarize,
)
from .store import (
    ChunkIndex,
    ChunkStore,
    MultivectorIndex,
    MultivectorStore,
    VectorIndex,
)

Decoded = tuple[Any, float]


@runtime_checkable
class AudioEncoder(Protocol):
    dim: int
    device: str

    def embed_clips(self, clips: list[np.ndarray]) -> np.ndarray: ...
    def embed_windows(self, windows: Windows) -> np.ndarray: ...


@runtime_checkable
class Strategy(Protocol):
    name: StrategyName
    supports_wander: bool

    def signature(self, cfg: ModelConfig) -> str: ...
    def decode(self, path: Path, cfg: ModelConfig) -> Decoded: ...
    def embed_and_upsert(
        self,
        path: str,
        st: os.stat_result,
        decoded: Decoded,
        cfg: ModelConfig,
        encoder: AudioEncoder,
        store: VectorIndex,
    ) -> None: ...
    def open_store(self, cfg: Config) -> MultivectorStore | ChunkStore: ...


class MeanStrategy:
    name: StrategyName = StrategyName.MEAN
    supports_wander: bool = False

    def signature(self, cfg: ModelConfig) -> str:
        return f"{cfg.name}|mean|{cfg.windows}x{cfg.window_seconds:g}"

    def decode(self, path: Path, cfg: ModelConfig) -> tuple[list[np.ndarray], float]:
        return load_clips(path, cfg.windows, cfg.window_seconds)

    def embed_and_upsert(
        self,
        path: str,
        st: os.stat_result,
        decoded: Decoded,
        cfg: ModelConfig,
        encoder: AudioEncoder,
        store: VectorIndex,
    ) -> None:
        clips, duration = decoded
        if not clips:
            raise ValueError("no decodable audio")
        embs = encoder.embed_clips(clips)
        vectors = [normalize(embs.mean(axis=0)).tolist()]
        assert isinstance(store, MultivectorIndex)
        store.upsert(
            path,
            vectors,
            multivector_payload(path, st, duration, [1.0], self.signature(cfg)),
        )

    def open_store(self, cfg: Config) -> MultivectorStore:
        return MultivectorStore(cfg.qdrant, share_weight=cfg.search.share_weight)


class ClustersStrategy:
    name: StrategyName = StrategyName.CLUSTERS
    supports_wander: bool = False

    def signature(self, cfg: ModelConfig) -> str:
        return (
            f"{cfg.name}|clusters|{cfg.window_seconds:g}|{cfg.max_chunks}|{cfg.silence_db:g}"
            f"|{cfg.max_clusters}|{cfg.min_cluster_share:g}|{cfg.merge_threshold:g}"
        )

    def decode(self, path: Path, cfg: ModelConfig) -> tuple[list[np.ndarray], float]:
        return dense_chunks(path, cfg.window_seconds, cfg.max_chunks, cfg.silence_db)

    def embed_and_upsert(
        self,
        path: str,
        st: os.stat_result,
        decoded: Decoded,
        cfg: ModelConfig,
        encoder: AudioEncoder,
        store: VectorIndex,
    ) -> None:
        clips, duration = decoded
        if not clips:
            raise ValueError("no decodable audio")
        embs = encoder.embed_clips(clips)
        centroids, shares = summarize(
            embs, cfg.max_clusters, cfg.min_cluster_share, cfg.merge_threshold
        )
        rounded_shares = [round(float(s), 4) for s in shares]
        assert isinstance(store, MultivectorIndex)
        store.upsert(
            path,
            centroids.tolist(),
            multivector_payload(path, st, duration, rounded_shares, self.signature(cfg)),
        )

    def open_store(self, cfg: Config) -> MultivectorStore:
        return MultivectorStore(cfg.qdrant, share_weight=cfg.search.share_weight)


class ChunksStrategy:
    name: StrategyName = StrategyName.CHUNKS
    supports_wander: bool = True

    def signature(self, cfg: ModelConfig) -> str:
        return (
            f"{cfg.name}|chunks|{cfg.window_seconds:g}|{cfg.hop_seconds:g}|{cfg.silence_db:g}"
            f"|{cfg.edge_seconds:g}"
        )

    def decode(self, path: Path, cfg: ModelConfig) -> tuple[Windows, float]:
        return sliding_chunks(path, cfg.window_seconds, cfg.hop_seconds, cfg.silence_db)

    def embed_and_upsert(
        self,
        path: str,
        st: os.stat_result,
        decoded: Decoded,
        cfg: ModelConfig,
        encoder: AudioEncoder,
        store: VectorIndex,
    ) -> None:
        windows, duration = decoded
        vectors = encoder.embed_windows(windows)
        spans = [windows.span(i) for i in range(len(windows))]
        starts = np.array([s for s, _ in spans])
        speed = trajectory.speeds(vectors, starts)
        chunk_payloads = [
            chunk_payload(
                path=path,
                idx=i,
                total=len(windows),
                start=start,
                end=end,
                db=windows.db[i],
                silent=windows.silent[i],
                speed=None if np.isnan(speed[i]) else float(speed[i]),
            )
            for i, (start, end) in enumerate(spans)
        ]
        loud = ~np.array(windows.silent)
        mean = normalize(vectors[loud].mean(axis=0) if loud.any() else vectors.mean(axis=0))
        intro, outro = trajectory.edges(
            vectors, starts, windows.silent, windows.db, cfg.edge_seconds
        )
        track_vectors: dict[str, Any] = {"mean": mean.tolist()}
        for name, edge in (("intro", intro), ("outro", outro)):
            track_vectors[name] = edge.position.tolist()
            if edge.direction is not None:
                track_vectors[f"{name}_dir"] = edge.direction.tolist()
        assert isinstance(store, ChunkIndex)
        store.replace_track(
            path,
            vectors,
            chunk_payloads,
            track_vectors,
            chunk_track_payload(
                path=path,
                st=st,
                duration=duration,
                signature=self.signature(cfg),
                n_chunks=len(windows),
                silent=not loud.any(),
                intro_drift=intro.drift,
                outro_drift=outro.drift,
                trajectory_summary=trajectory.summarize(vectors, starts, cfg.window_seconds),
            ),
        )

    def open_store(self, cfg: Config) -> ChunkStore:
        sustain = max(1, round(cfg.search.sustain_seconds / cfg.model.hop_seconds))
        return ChunkStore(cfg.qdrant, sustain)


_STRATEGIES: dict[str, Strategy] = {
    StrategyName.MEAN: MeanStrategy(),
    StrategyName.CLUSTERS: ClustersStrategy(),
    StrategyName.CHUNKS: ChunksStrategy(),
}


def get_strategy(cfg_or_name: ModelConfig | str) -> Strategy:
    name = cfg_or_name.strategy if isinstance(cfg_or_name, ModelConfig) else cfg_or_name
    try:
        return _STRATEGIES[name]
    except KeyError as e:
        raise ValueError(
            f"[model] strategy must be 'mean', 'clusters' or 'chunks', not {name!r}"
        ) from e


def open_store(cfg: Config) -> MultivectorStore | ChunkStore:
    return get_strategy(cfg.model).open_store(cfg)

"""In-memory NumPy implementations of VectorIndex and ChunkIndex for fast unit testing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from echo_control.store import SEARCH_OVERFETCH, Candidate, Hit


@dataclass
class _MultiPoint:
    vectors: np.ndarray
    payload: dict[str, Any]


@dataclass
class _ChunkRecord:
    vector: np.ndarray
    payload: dict[str, Any]


@dataclass
class _TrackRecord:
    vectors: dict[str, np.ndarray]
    payload: dict[str, Any]


class InMemoryVectorIndex:
    """In-memory implementation of VectorIndex matching Store (multivector MaxSim)."""

    def __init__(self, share_weight: float = 0.0):
        self.share_weight: float = share_weight
        self.dim: int | None = None
        self._points: dict[str, _MultiPoint] = {}

    def ensure(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            self._points.clear()
        self.dim = dim

    def drop(self) -> None:
        self._points.clear()
        self.dim = None

    def exists(self) -> bool:
        return self.dim is not None

    def count(self) -> int:
        return len(self._points) if self.exists() else 0

    def upsert(self, path: str, vectors: list[list[float]], payload: dict[str, Any]) -> None:
        self._points[path] = _MultiPoint(
            vectors=np.asarray(vectors, dtype=np.float32),
            payload=dict(payload),
        )

    def delete_paths(self, paths: list[str]) -> None:
        for p in paths:
            self._points.pop(p, None)

    def iter_payloads(self) -> Iterator[dict[str, Any]]:
        for pt in list(self._points.values()):
            yield dict(pt.payload)

    def payloads(self, paths: list[str]) -> list[dict[str, Any]]:
        return [dict(self._points[p].payload) for p in paths if p in self._points]

    def search(self, vector: list[float], limit: int, share_weight: float | None = None) -> list[Hit]:
        weight = self.share_weight if share_weight is None else share_weight
        rerank = weight > 0
        q = np.asarray(vector, dtype=np.float32)
        scored: list[tuple[float, str, _MultiPoint]] = []
        for path, pt in self._points.items():
            sims = pt.vectors @ q
            max_sim = float(np.max(sims)) if len(sims) else 0.0
            scored.append((max_sim, path, pt))
        scored.sort(key=lambda item: -item[0])
        candidates = scored[: limit * SEARCH_OVERFETCH if rerank else limit]

        hits: list[Hit] = []
        for max_sim, _path, pt in candidates:
            score = max_sim
            pl = pt.payload
            if rerank:
                sims = pt.vectors @ q
                shares = np.asarray(pl.get("shares") or [1.0] * len(sims))
                if len(shares) == len(sims):
                    score = float(np.max(sims * shares**weight))
            hits.append(Hit(path=str(pl.get("path", "")), score=score, duration=pl.get("duration")))
        if rerank:
            hits.sort(key=lambda h: -h.score)
        return hits[:limit]


class InMemoryChunkIndex:
    """In-memory implementation of ChunkIndex matching ChunkStore."""

    def __init__(self, sustain_windows: int = 1):
        self.sustain_windows: int = sustain_windows
        self.dim: int | None = None
        self._tracks: dict[str, _TrackRecord] = {}
        self._chunks: dict[str, list[_ChunkRecord]] = {}

    def ensure(self, dim: int) -> None:
        if self.dim is not None and self.dim != dim:
            self._tracks.clear()
            self._chunks.clear()
        self.dim = dim

    def drop(self) -> None:
        self._tracks.clear()
        self._chunks.clear()
        self.dim = None

    def exists(self) -> bool:
        return self.dim is not None

    def count(self) -> int:
        return len(self._tracks) if self.exists() else 0

    def chunk_count(self) -> int:
        return sum(len(chunks) for chunks in self._chunks.values())

    def replace_track(
        self,
        path: str,
        vectors: np.ndarray,
        chunk_payloads: list[dict[str, Any]],
        track_vectors: dict[str, Any],
        track_payload: dict[str, Any],
    ) -> None:
        self._chunks[path] = [
            _ChunkRecord(
                vector=np.asarray(v, dtype=np.float32),
                payload=dict(pl),
            )
            for v, pl in zip(vectors, chunk_payloads, strict=True)
        ]
        self._tracks[path] = _TrackRecord(
            vectors={
                k: np.asarray(v, dtype=np.float32)
                for k, v in track_vectors.items()
                if v is not None
            },
            payload=dict(track_payload),
        )

    def delete_paths(self, paths: list[str]) -> None:
        for p in paths:
            self._tracks.pop(p, None)
            self._chunks.pop(p, None)

    def iter_payloads(self) -> Iterator[dict[str, Any]]:
        for tr in list(self._tracks.values()):
            yield dict(tr.payload)

    def payloads(self, paths: list[str]) -> list[dict[str, Any]]:
        return [dict(self._tracks[p].payload) for p in paths if p in self._tracks]

    def track(self, path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        rec = self._tracks.get(path)
        if rec is None:
            return None
        return dict(rec.payload), {k: v.copy() for k, v in rec.vectors.items()}

    def chunk_vectors(self, path: str) -> np.ndarray:
        chunks = list(self._chunks.get(path, []))
        chunks.sort(key=lambda c: c.payload.get("idx", 0))
        return np.array([c.vector for c in chunks], dtype=np.float32)

    def intros_near(self, vector: np.ndarray, limit: int, exclude: list[str]) -> list[Candidate]:
        excluded = set(exclude)
        q = np.asarray(vector, dtype=np.float32)
        cands: list[Candidate] = []
        for path, rec in self._tracks.items():
            if path in excluded or rec.payload.get("silent") is True:
                continue
            intro = rec.vectors["intro"]
            sim = float(intro @ q)
            cands.append(Candidate.of(dict(rec.payload), sim, rec.vectors))
        cands.sort(key=lambda c: -c.intro_sim)
        return cands[:limit]

    def intros_of(self, vector: np.ndarray, paths: list[str]) -> list[Candidate]:
        q = np.asarray(vector, dtype=np.float32)
        cands: list[Candidate] = []
        for path in paths:
            rec = self._tracks.get(path)
            if rec is None:
                continue
            sim = float(rec.vectors["intro"] @ q)
            cands.append(Candidate.of(dict(rec.payload), sim, rec.vectors))
        return cands

    def _groups(
        self, vector: list[float], limit: int, only_paths: set[str] | None = None
    ) -> list[Hit]:
        q = np.asarray(vector, dtype=np.float32)
        groups: list[tuple[float, Hit]] = []
        for path, chunks in self._chunks.items():
            if only_paths is not None and path not in only_paths:
                continue
            loud = [c for c in chunks if not c.payload.get("silent")]
            if not loud:
                continue
            scored = [(float(c.vector @ q), c) for c in loud]
            scored.sort(key=lambda item: -item[0])
            top = scored[: self.sustain_windows]
            best_sim, best_chunk = top[0]
            mean_sim = float(np.mean([s for s, _ in top]))
            groups.append((
                best_sim,
                Hit(
                    path=str(best_chunk.payload.get("path", path)),
                    score=mean_sim,
                    duration=None,
                    start=best_chunk.payload.get("start"),
                ),
            ))
        groups.sort(key=lambda item: -item[0])
        return [hit for _, hit in groups[:limit]]

    def sustained(self, vector: list[float], paths: list[str]) -> dict[str, float]:
        if not paths:
            return {}
        return {h.path: h.score for h in self._groups(vector, len(paths), set(paths))}

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        hits = self._groups(
            vector, limit * SEARCH_OVERFETCH if self.sustain_windows > 1 else limit
        )
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]


class FakeAudioEncoder:
    """Deterministic in-memory audio encoder for unit testing Indexer and Strategies."""

    def __init__(self, dim: int = 8, device: str = "cpu"):
        self.dim: int = dim
        self.device: str = device

    def embed_clips(self, clips: list[np.ndarray]) -> np.ndarray:
        if not clips:
            raise ValueError("no decodable audio")
        out = np.zeros((len(clips), self.dim), dtype=np.float32)
        for i, c in enumerate(clips):
            out[i, i % self.dim] = 1.0
            if len(c):
                out[i, 0] += float(np.mean(c)) * 0.01
        norms = np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)
        return (out / norms).astype(np.float32)

    def embed_windows(self, windows: Any) -> np.ndarray:
        if not len(windows):
            raise ValueError("no decodable audio")
        return self.embed_clips([windows.clip(i) for i in range(len(windows))])

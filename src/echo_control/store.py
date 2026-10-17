# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from qdrant_client import QdrantClient, models

from .config import QdrantConfig
from .errors import EchoControlError

log = logging.getLogger(__name__)

_NS = uuid.UUID("5f0c7c1e-6d7a-4f55-9d0e-6563686f6374")


class QdrantUnavailable(EchoControlError):
    pass


UPSERT_BATCH = 256
SCROLL_BATCH = 1000
SEARCH_OVERFETCH = 3
TRACK_VECTORS = ("mean", "intro", "outro", "intro_dir", "outro_dir")


def point_id(path: str) -> str:
    return str(uuid.uuid5(_NS, path))


def chunk_id(path: str, idx: int) -> str:
    return str(uuid.uuid5(_NS, f"{path}#{idx}"))


@dataclass
class Hit:
    path: str
    score: float
    duration: float | None
    start: float | None = None


@dataclass
class Candidate:
    """A possible next track: its payload, how close its intro is, and its vectors."""

    payload: dict[str, Any]
    intro_sim: float
    intro_dir: np.ndarray | None
    mean: np.ndarray
    intro: np.ndarray

    @property
    def position(self) -> float:
        return self.intro_sim

    @classmethod
    def of(cls, payload: dict[str, Any], intro_sim: float, vectors: dict[str, Any]) -> Candidate:
        d = vectors.get("intro_dir")
        return cls(
            payload,
            intro_sim,
            None if d is None else np.asarray(d, np.float32),
            np.asarray(vectors["mean"], np.float32),
            np.asarray(vectors["intro"], np.float32),
        )


@runtime_checkable
class VectorIndex(Protocol):
    def ensure(self, dim: int) -> None: ...
    def drop(self) -> None: ...
    def exists(self) -> bool: ...
    def count(self) -> int: ...
    def delete_paths(self, paths: list[str]) -> None: ...
    def iter_payloads(self) -> Iterator[dict[str, Any]]: ...
    def payloads(self, paths: list[str]) -> list[dict[str, Any]]: ...
    def search(self, vector: list[float], limit: int) -> list[Hit]: ...


@runtime_checkable
class MultivectorIndex(VectorIndex, Protocol):
    def upsert(self, path: str, vectors: list[list[float]], payload: dict[str, Any]) -> None: ...


@runtime_checkable
class ChunkIndex(VectorIndex, Protocol):
    def replace_track(
        self,
        path: str,
        vectors: np.ndarray,
        chunk_payloads: list[dict[str, Any]],
        track_vectors: dict[str, Any],
        track_payload: dict[str, Any],
    ) -> None: ...
    def track(self, path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None: ...
    def chunk_vectors(self, path: str) -> np.ndarray: ...
    def intros_near(self, vector: np.ndarray, limit: int, exclude: list[str]) -> list[Candidate]: ...
    def intros_of(self, vector: np.ndarray, paths: list[str]) -> list[Candidate]: ...
    def sustained(self, vector: list[float], paths: list[str]) -> dict[str, float]: ...


def is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/readyz", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_ready(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    delay = 0.5
    while not is_ready(url):
        if time.monotonic() > deadline:
            raise QdrantUnavailable(f"Qdrant not reachable at {url} after {timeout:.0f}s")
        time.sleep(delay)
        delay = min(delay * 2, 5)


def _iter_payloads(client: QdrantClient, collection: str) -> Iterator[dict[str, Any]]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection, limit=SCROLL_BATCH, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            yield p.payload or {}
        if offset is None:
            break


def _payloads(client: QdrantClient, collection: str, paths: list[str]) -> list[dict[str, Any]]:
    points = client.retrieve(collection, ids=[point_id(p) for p in paths], with_payload=True)
    return [pt.payload for pt in points if pt.payload]


class MultivectorStore:
    def __init__(self, cfg: QdrantConfig, share_weight: float = 0.0):
        self.cfg: QdrantConfig = cfg
        self.share_weight: float = share_weight
        self.collection: str = cfg.collection
        self.client: QdrantClient = QdrantClient(url=cfg.url, timeout=30)

    def ensure(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors
            if (
                isinstance(vectors, models.VectorParams)
                and vectors.multivector_config is not None
                and vectors.size == dim
            ):
                return
            log.warning(
                "collection %r has an outdated layout or dimension; recreating it",
                self.collection,
            )
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        self.client.create_payload_index(self.collection, "path", models.PayloadSchemaType.KEYWORD)

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def upsert(self, path: str, vectors: list[list[float]], payload: dict[str, Any]) -> None:
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id(path), vector=vectors, payload=payload)],
        )

    def delete_paths(self, paths: list[str]) -> None:
        if paths:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[point_id(p) for p in paths]),
            )

    def iter_payloads(self) -> Iterator[dict[str, Any]]:
        return _iter_payloads(self.client, self.collection)

    def payloads(self, paths: list[str]) -> list[dict[str, Any]]:
        return _payloads(self.client, self.collection, paths)

    def search(self, vector: list[float], limit: int, share_weight: float | None = None) -> list[Hit]:
        """MaxSim over each track's vectors. With share_weight > 0, candidates are re-ranked
        by max(similarity * share ** share_weight), so a vector covering a sliver of a track
        (a bridge, one section of a DJ mix) counts for less than one covering most of it."""
        weight = self.share_weight if share_weight is None else share_weight
        rerank = weight > 0
        res = self.client.query_points(
            self.collection,
            query=[vector],
            limit=limit * SEARCH_OVERFETCH if rerank else limit,
            with_payload=True,
            with_vectors=rerank,
        )
        hits: list[Hit] = []
        q = np.asarray(vector, dtype=np.float32)
        for p in res.points:
            score = p.score
            pl = p.payload or {}
            if rerank:
                sims = np.asarray(p.vector, dtype=np.float32) @ q
                shares = np.asarray(pl.get("shares") or [1.0] * len(sims))
                if len(shares) == len(sims):
                    score = float(np.max(sims * shares**weight))
            hits.append(Hit(path=str(pl.get("path", "")), score=score, duration=pl.get("duration")))
        if rerank:
            hits.sort(key=lambda h: -h.score)
        return hits[:limit]


Store = MultivectorStore


class ChunkStore:
    """One point per window in `<collection>_chunks`, plus one per track in
    `<collection>_tracks` holding the scan state and the track's mean vector. The track point
    is written after its chunks, so a track whose chunks were only partly written has no
    current track record and is redone on the next scan."""

    def __init__(self, cfg: QdrantConfig, sustain_windows: int = 1):
        self.cfg: QdrantConfig = cfg
        self.sustain_windows: int = sustain_windows
        self.collection: str = cfg.collection
        self.tracks: str = f"{cfg.collection}_tracks"
        self.chunks: str = f"{cfg.collection}_chunks"
        self.client: QdrantClient = QdrantClient(url=cfg.url, timeout=30)

    def ensure(self, dim: int) -> None:
        tracks_exist = self.client.collection_exists(self.tracks)
        chunks_exist = self.client.collection_exists(self.chunks)
        if tracks_exist or chunks_exist:
            tracks_vecs = (
                self.client.get_collection(self.tracks).config.params.vectors
                if tracks_exist
                else None
            )
            chunks_vecs = (
                self.client.get_collection(self.chunks).config.params.vectors
                if chunks_exist
                else None
            )
            tracks_ok = (
                isinstance(tracks_vecs, dict)
                and set(tracks_vecs) == set(TRACK_VECTORS)
                and all(v.size == dim for v in tracks_vecs.values())
            )
            chunks_ok = isinstance(chunks_vecs, models.VectorParams) and chunks_vecs.size == dim
            if not (tracks_ok and chunks_ok):
                log.warning(
                    "%r has an outdated layout or dimension; recreating the chunk index",
                    self.tracks,
                )
                self.drop()
                tracks_exist = chunks_exist = False
        if not tracks_exist:
            self.client.create_collection(
                self.tracks,
                vectors_config={
                    name: models.VectorParams(size=dim, distance=models.Distance.COSINE)
                    for name in TRACK_VECTORS
                },
            )
            self.client.create_payload_index(self.tracks, "path", models.PayloadSchemaType.KEYWORD)
        if not chunks_exist:
            self.client.create_collection(
                self.chunks,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE, on_disk=True
                ),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, always_ram=True
                    )
                ),
            )
            self.client.create_payload_index(self.chunks, "path", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.chunks, "silent", models.PayloadSchemaType.BOOL)

    def drop(self) -> None:
        for name in (self.chunks, self.tracks):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    def exists(self) -> bool:
        return self.client.collection_exists(self.tracks)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.tracks, exact=True).count

    def replace_track(
        self,
        path: str,
        vectors: np.ndarray,
        chunk_payloads: list[dict[str, Any]],
        track_vectors: dict[str, Any],
        track_payload: dict[str, Any],
    ) -> None:
        """track_vectors holds TRACK_VECTORS entries; a missing direction is left out."""
        self._delete_chunks([path])
        points = [
            models.PointStruct(id=chunk_id(path, i), vector=v.tolist(), payload=pl)
            for i, (v, pl) in enumerate(zip(vectors, chunk_payloads, strict=True))
        ]
        for i in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(self.chunks, points=points[i : i + UPSERT_BATCH])
        self.client.upsert(
            self.tracks,
            points=[
                models.PointStruct(id=point_id(path), vector=track_vectors, payload=track_payload)
            ],
        )

    def _delete_chunks(self, paths: list[str]) -> None:
        self.client.delete(
            self.chunks,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="path", match=models.MatchAny(any=paths))]
                )
            ),
        )

    def delete_paths(self, paths: list[str]) -> None:
        if paths:
            self.client.delete(
                self.tracks,
                points_selector=models.PointIdsList(points=[point_id(p) for p in paths]),
            )
            self._delete_chunks(paths)

    def iter_payloads(self) -> Iterator[dict[str, Any]]:
        return _iter_payloads(self.client, self.tracks)

    def payloads(self, paths: list[str]) -> list[dict[str, Any]]:
        return _payloads(self.client, self.tracks, paths)

    def track(self, path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        points = self.client.retrieve(
            self.tracks, ids=[point_id(path)], with_payload=True, with_vectors=True
        )
        if not points:
            return None
        pv = points[0].vector
        vecs = {k: np.asarray(v, dtype=np.float32) for k, v in pv.items()} if isinstance(pv, dict) else {}
        return points[0].payload or {}, vecs

    def chunk_vectors(self, path: str) -> np.ndarray:
        """The track's window vectors in time order."""
        all_points: list[models.Record] = []
        offset = None
        filt = models.Filter(
            must=[models.FieldCondition(key="path", match=models.MatchValue(value=path))]
        )
        while True:
            points, offset = self.client.scroll(
                self.chunks,
                scroll_filter=filt,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=["idx"],
                with_vectors=True,
            )
            all_points.extend(points)
            if offset is None:
                break
        all_points.sort(key=lambda p: (p.payload or {}).get("idx", 0))
        return np.array([p.vector for p in all_points], dtype=np.float32)

    @staticmethod
    def _to_candidates(
        points: list[models.ScoredPoint] | list[models.Record],
        ref_vector: np.ndarray,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for p in points:
            if isinstance(p.vector, dict):
                sim = (
                    p.score
                    if isinstance(p, models.ScoredPoint)
                    else float(np.asarray(p.vector["intro"], np.float32) @ ref_vector)
                )
                out.append(Candidate.of(p.payload or {}, sim, p.vector))
        return out

    def intros_near(self, vector: np.ndarray, limit: int, exclude: list[str]) -> list[Candidate]:
        """Audible tracks whose intro is closest to vector."""
        res = self.client.query_points(
            self.tracks,
            query=vector.tolist(),
            using="intro",
            limit=limit,
            query_filter=models.Filter(
                must_not=[
                    models.HasIdCondition(has_id=[point_id(p) for p in exclude]),
                    models.FieldCondition(key="silent", match=models.MatchValue(value=True)),
                ]
            ),
            with_payload=True,
            with_vectors=["intro", "intro_dir", "mean"],
        )
        return self._to_candidates(res.points, vector)

    def intros_of(self, vector: np.ndarray, paths: list[str]) -> list[Candidate]:
        """Like intros_near, for the given tracks rather than the nearest ones."""
        points = self.client.retrieve(
            self.tracks,
            ids=[point_id(p) for p in paths],
            with_payload=True,
            with_vectors=["intro", "intro_dir", "mean"],
        )
        return self._to_candidates(points, vector)

    def sustained(self, vector: list[float], paths: list[str]) -> dict[str, float]:
        """search()'s score for each of the given tracks; tracks with no loud windows are left out."""
        if not paths:
            return {}
        only = models.FieldCondition(key="path", match=models.MatchAny(any=paths))
        return {h.path: h.score for h in self._groups(vector, len(paths), only)}

    def _groups(
        self, vector: list[float], limit: int, only: models.FieldCondition | None = None
    ) -> list[Hit]:
        res = self.client.query_points_groups(
            self.chunks,
            group_by="path",
            query=vector,
            query_filter=models.Filter(
                must=[only] if only else None,
                must_not=[models.FieldCondition(key="silent", match=models.MatchValue(value=True))],
            ),
            limit=limit,
            group_size=self.sustain_windows,
            with_payload=["path", "start"],
        )
        return [
            Hit(
                path=str((g.hits[0].payload or {}).get("path", "")),
                score=float(np.mean([h.score for h in g.hits])),
                duration=None,
                start=(g.hits[0].payload or {}).get("start"),
            )
            for g in res.groups
        ]

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        """Tracks ranked by the mean similarity of their best sustain_windows non-silent
        windows, so a track has to hold the vibe for a while rather than for one lucky
        moment. Tracks with fewer windows are averaged over all of them. Candidates come from
        each track's single best window, over-fetched to leave room for the re-rank."""
        hits = self._groups(vector, limit * SEARCH_OVERFETCH if self.sustain_windows > 1 else limit)
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]

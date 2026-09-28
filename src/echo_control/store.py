from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

from .config import QdrantConfig

log = logging.getLogger(__name__)

_NS = uuid.UUID("5f0c7c1e-6d7a-4f55-9d0e-6563686f6374")


class QdrantUnavailable(RuntimeError):
    pass


def point_id(path: str) -> str:
    return str(uuid.uuid5(_NS, path))


@dataclass
class Hit:
    path: str
    score: float
    duration: float | None


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


class Store:
    def __init__(self, cfg: QdrantConfig):
        self.cfg = cfg
        self.collection = cfg.collection
        self.client = QdrantClient(url=cfg.url, timeout=30)

    def ensure(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            vectors = self.client.get_collection(self.collection).config.params.vectors
            if getattr(vectors, "multivector_config", None) is not None:
                return
            log.warning(
                "collection %r uses the old single-vector layout; recreating it", self.collection
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

    def upsert(self, path: str, vectors: list[list[float]], payload: dict) -> None:
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

    def iter_payloads(self) -> Iterator[dict]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                yield p.payload or {}
            if offset is None:
                break

    def search(self, vector: list[float], limit: int, share_weight: float = 0.0) -> list[Hit]:
        """MaxSim over each track's vectors. With share_weight > 0, candidates are re-ranked
        by max(similarity * share ** share_weight), so a vector covering a sliver of a track
        (a bridge, one section of a DJ mix) counts for less than one covering most of it."""
        rerank = share_weight > 0
        res = self.client.query_points(
            self.collection,
            query=[vector],
            limit=limit * 3 if rerank else limit,
            with_payload=True,
            with_vectors=rerank,
        )
        hits = []
        q = np.asarray(vector, dtype=np.float32)
        for p in res.points:
            score = p.score
            if rerank:
                sims = np.asarray(p.vector, dtype=np.float32) @ q
                shares = np.asarray(p.payload.get("shares") or [1.0] * len(sims))
                if len(shares) == len(sims):
                    score = float(np.max(sims * shares**share_weight))
            hits.append(Hit(path=p.payload["path"], score=score, duration=p.payload.get("duration")))
        if rerank:
            hits.sort(key=lambda h: -h.score)
        return hits[:limit]

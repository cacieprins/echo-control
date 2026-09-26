from __future__ import annotations

import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .config import QdrantConfig

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
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(
                self.collection, "path", models.PayloadSchemaType.KEYWORD
            )

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def upsert(self, path: str, vector: list[float], payload: dict) -> None:
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=point_id(path), vector=vector, payload=payload)],
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

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        res = self.client.query_points(
            self.collection, query=vector, limit=limit, with_payload=True
        )
        return [
            Hit(path=p.payload["path"], score=p.score, duration=p.payload.get("duration"))
            for p in res.points
        ]

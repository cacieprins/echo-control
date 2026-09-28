from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import trajectory
from .clap import Clap
from .config import Config
from .segments import Windows
from .store import ChunkStore, Store

log = logging.getLogger(__name__)

DECODE_WORKERS = 4

Key = tuple[float | None, int | None, str | None]


def iter_audio_files(
    roots: Iterable[Path], extensions: Iterable[str], exclude: Iterable[Path] = ()
) -> Iterator[Path]:
    exts = {"." + e.lower() for e in extensions}
    excluded = set(exclude)
    for root in roots:
        if not root.is_dir():
            log.warning("library path %s does not exist", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and Path(dirpath, d) not in excluded
            ]
            for name in filenames:
                if not name.startswith(".") and os.path.splitext(name)[1].lower() in exts:
                    yield Path(dirpath) / name


def is_audio(path: Path, cfg: Config) -> bool:
    return (
        path.suffix.lower().lstrip(".") in cfg.library.extensions
        and any(path.is_relative_to(r) for r in cfg.library.paths)
        and not any(path.is_relative_to(x) for x in cfg.library.exclude)
    )


def state_key(payload: dict) -> Key:
    return (payload.get("mtime"), payload.get("size"), payload.get("embedding"))


def current_key(path: Path, signature: str) -> Key | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime, st.st_size, signature)


class Indexer:
    def __init__(self, cfg: Config, encoders: list[Clap], store: Store | ChunkStore):
        self.cfg = cfg
        self.encoders = encoders
        self.clap = encoders[0]
        self.store = store
        store.ensure(self.clap.dim)

    def current_key(self, path: Path) -> Key | None:
        return current_key(path, self.clap.signature)

    def index_file(
        self,
        path: Path,
        known: dict[str, Key] | None = None,
        decoded: Future | None = None,
        encoder: Clap | None = None,
    ) -> bool:
        """Embed and upsert one file. Returns True if it was (re)indexed."""
        try:
            st = path.stat()
        except FileNotFoundError:
            self.remove([path])
            return False
        key = str(path)
        if known is not None and known.get(key) == self.current_key(path):
            return False
        encoder = encoder or self.clap
        chunked = self.cfg.model.strategy == "chunks"
        try:
            clips, duration = decoded.result() if decoded else self.clap.decode(path)
            with encoder.lock:
                if chunked:
                    vectors = encoder.embed_windows(clips)
                else:
                    vectors, shares = encoder.embed_decoded(clips)
        except Exception as e:
            log.warning("skipping %s: %s", path, e)
            return False
        if chunked:
            self._store_chunks(key, st, clips, vectors, duration)
            return True
        self.store.upsert(
            key,
            vectors,
            {
                "path": key,
                "mtime": st.st_mtime,
                "size": st.st_size,
                "duration": duration,
                "shares": shares,
                "embedding": self.clap.signature,
            },
        )
        return True

    def _store_chunks(
        self, key: str, st: os.stat_result, windows: Windows, vectors: np.ndarray, duration: float
    ) -> None:
        spans = [windows.span(i) for i in range(len(windows))]
        starts = np.array([s for s, _ in spans])
        speed = trajectory.speeds(vectors, starts)
        chunk_payloads = [
            {
                "path": key,
                "idx": i,
                "total": len(windows),
                "start": round(start, 3),
                "end": round(end, 3),
                "db": round(windows.db[i], 2),
                "silent": windows.silent[i],
                "speed": None if np.isnan(speed[i]) else round(float(speed[i]), 5),
            }
            for i, (start, end) in enumerate(spans)
        ]
        loud = ~np.array(windows.silent)
        mean = vectors[loud].mean(axis=0) if loud.any() else vectors.mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-12)
        intro, outro = trajectory.edges(
            vectors, starts, windows.silent, windows.db, self.cfg.model.edge_seconds
        )
        track_vectors = {"mean": mean.tolist()}
        for name, edge in (("intro", intro), ("outro", outro)):
            track_vectors[name] = edge.position.tolist()
            if edge.direction is not None:
                track_vectors[f"{name}_dir"] = edge.direction.tolist()
        self.store.replace_track(
            key,
            vectors,
            chunk_payloads,
            track_vectors,
            {
                "intro_drift": round(intro.drift, 6),
                "outro_drift": round(outro.drift, 6),
                "path": key,
                "mtime": st.st_mtime,
                "size": st.st_size,
                "duration": duration,
                "embedding": self.clap.signature,
                "n_chunks": len(windows),
                "silent": not loud.any(),
                **trajectory.summarize(vectors, starts, self.cfg.model.window_seconds),
            },
        )

    def remove(self, paths: Iterable[Path]) -> None:
        self.store.delete_paths([str(p) for p in paths])

    def scan(self, stop: threading.Event | None = None) -> tuple[int, int, int]:
        """Full pass: index new/changed files, prune missing ones."""
        known = {p["path"]: state_key(p) for p in self.store.iter_payloads() if "path" in p}
        lib = self.cfg.library
        files = list(iter_audio_files(lib.paths, lib.extensions, lib.exclude))
        todo = [f for f in files if known.get(str(f)) != self.current_key(f)]
        log.info("scan: %d files, %d new or changed", len(files), len(todo))

        added = self.index_many(todo, stop)

        present = {str(f) for f in files}
        missing = [p for p in known if p not in present]
        self.store.delete_paths(missing)
        if missing:
            log.info("pruned %d missing files", len(missing))
        return len(files), added, len(missing)

    def index_many(self, todo: list[Path], stop: threading.Event | None = None) -> int:
        """Decode on a thread pool and embed on one worker thread per encoder (GPU)."""
        workers = len(self.encoders)
        work: queue.Queue[tuple[Path, Future] | None] = queue.Queue(maxsize=workers * 4)
        progress = {"done": 0, "added": 0}
        progress_lock = threading.Lock()
        def stopped() -> bool:
            return stop is not None and stop.is_set()

        def consume(encoder: Clap) -> None:
            while (item := work.get()) is not None:
                f, fut = item
                if stopped():
                    fut.cancel()
                    continue
                ok = self.index_file(f, decoded=fut, encoder=encoder)
                with progress_lock:
                    progress["done"] += 1
                    progress["added"] += ok
                    done = progress["done"]
                if done % 100 == 0 or done == len(todo):
                    log.info("indexed %d/%d", done, len(todo))

        consumers = [
            threading.Thread(target=consume, args=(e,), name=f"embed-{e.device}", daemon=True)
            for e in self.encoders
        ]
        for t in consumers:
            t.start()
        with ThreadPoolExecutor(DECODE_WORKERS * workers, thread_name_prefix="decode") as pool:
            for f in todo:
                if stopped():
                    break
                work.put((f, pool.submit(self.clap.decode, f)))
            for _ in consumers:
                work.put(None)
            for t in consumers:
                t.join()
        return progress["added"]

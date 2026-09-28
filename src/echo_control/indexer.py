from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .clap import Clap
from .config import Config
from .store import Store

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


class Indexer:
    def __init__(self, cfg: Config, encoders: list[Clap], store: Store):
        self.cfg = cfg
        self.encoders = encoders
        self.clap = encoders[0]
        self.store = store
        store.ensure(self.clap.dim)

    def state_key(self, payload: dict) -> Key:
        return (payload.get("mtime"), payload.get("size"), payload.get("embedding"))

    def current_key(self, path: Path) -> Key | None:
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime, st.st_size, self.clap.signature)

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
        try:
            clips, duration = decoded.result() if decoded else self.clap.decode(path)
            with encoder.lock:
                vectors, shares = encoder.embed_decoded(clips)
        except Exception as e:
            log.warning("skipping %s: %s", path, e)
            return False
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

    def remove(self, paths: Iterable[Path]) -> None:
        self.store.delete_paths([str(p) for p in paths])

    def scan(self, stop: threading.Event | None = None) -> tuple[int, int, int]:
        """Full pass: index new/changed files, prune missing ones."""
        known = {p["path"]: self.state_key(p) for p in self.store.iter_payloads() if "path" in p}
        lib = self.cfg.library
        files = list(iter_audio_files(lib.paths, lib.extensions, lib.exclude))
        todo = [f for f in files if known.get(str(f)) != self.current_key(f)]
        log.info("scan: %d files, %d new or changed", len(files), len(todo))

        added = self._index_many(todo, stop)

        present = {str(f) for f in files}
        missing = [p for p in known if p not in present]
        self.store.delete_paths(missing)
        if missing:
            log.info("pruned %d missing files", len(missing))
        return len(files), added, len(missing)

    def _index_many(self, todo: list[Path], stop: threading.Event | None) -> int:
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

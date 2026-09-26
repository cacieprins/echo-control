from __future__ import annotations

import logging
import os
import threading
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .clap import Clap
from .config import Config
from .store import Store

log = logging.getLogger(__name__)

DECODE_WORKERS = 4


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
    def __init__(self, cfg: Config, clap: Clap, store: Store):
        self.cfg = cfg
        self.clap = clap
        self.store = store
        self.lock = threading.Lock()
        store.ensure(clap.dim)

    def index_file(
        self,
        path: Path,
        known: dict[str, tuple[float, int]] | None = None,
        decoded: Future | None = None,
    ) -> bool:
        """Embed and upsert one file. Returns True if it was (re)indexed."""
        try:
            st = path.stat()
        except FileNotFoundError:
            self.remove([path])
            return False
        key = str(path)
        if known is not None and known.get(key) == (st.st_mtime, st.st_size):
            return False
        try:
            clips, duration = decoded.result() if decoded else self.clap.decode(path)
            with self.lock:
                vector = self.clap.embed_decoded(clips)
        except Exception as e:
            log.warning("skipping %s: %s", path, e)
            return False
        self.store.upsert(
            key,
            vector,
            {"path": key, "mtime": st.st_mtime, "size": st.st_size, "duration": duration},
        )
        return True

    def remove(self, paths: Iterable[Path]) -> None:
        self.store.delete_paths([str(p) for p in paths])

    def scan(self, stop: threading.Event | None = None) -> tuple[int, int, int]:
        """Full pass: index new/changed files, prune missing ones."""
        known = {
            p["path"]: (p.get("mtime"), p.get("size"))
            for p in self.store.iter_payloads()
            if "path" in p
        }
        lib = self.cfg.library
        files = list(iter_audio_files(lib.paths, lib.extensions, lib.exclude))
        todo = [f for f in files if known.get(str(f)) != _stat_key(f)]
        log.info("scan: %d files, %d new or changed", len(files), len(todo))

        added = 0
        with ThreadPoolExecutor(DECODE_WORKERS, thread_name_prefix="decode") as pool:
            queue: deque[tuple[Path, Future]] = deque()
            upcoming = iter(todo)
            for i in range(1, len(todo) + 1):
                while len(queue) < DECODE_WORKERS * 2 and (f := next(upcoming, None)):
                    queue.append((f, pool.submit(self.clap.decode, f)))
                if stop is not None and stop.is_set():
                    for _, fut in queue:
                        fut.cancel()
                    break
                f, fut = queue.popleft()
                if self.index_file(f, decoded=fut):
                    added += 1
                if i % 100 == 0 or i == len(todo):
                    log.info("indexed %d/%d", i, len(todo))

        present = {str(f) for f in files}
        missing = [p for p in known if p not in present]
        self.store.delete_paths(missing)
        if missing:
            log.info("pruned %d missing files", len(missing))
        return len(files), added, len(missing)


def _stat_key(path: Path) -> tuple[float, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime, st.st_size)

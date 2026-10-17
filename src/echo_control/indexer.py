from __future__ import annotations

import logging
import os
import queue
import threading
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Any

from .config import Config
from .records import ScanKey, ScanStats
from .store import VectorIndex
from .strategies import AudioEncoder, Decoded, Strategy, get_strategy

log = logging.getLogger(__name__)

DECODE_WORKERS = 4

Key = ScanKey
StopFlag = threading.Event | ProcessEvent


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


def state_key(payload: dict[str, Any]) -> ScanKey:
    return ScanKey(payload.get("mtime"), payload.get("size"), payload.get("embedding"))


def current_key(path: Path, signature: str) -> ScanKey | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return ScanKey(st.st_mtime, st.st_size, signature)


class Indexer:
    def __init__(self, cfg: Config, encoders: Sequence[AudioEncoder], store: VectorIndex):
        self.cfg: Config = cfg
        self.strategy: Strategy = get_strategy(cfg.model)
        self.signature: str = self.strategy.signature(cfg.model)
        self.encoders: Sequence[AudioEncoder] = encoders
        self.encoder: AudioEncoder = encoders[0]
        self.clap: AudioEncoder = encoders[0]
        self.store: VectorIndex = store
        store.ensure(self.encoder.dim)

    def current_key(self, path: Path) -> ScanKey | None:
        return current_key(path, self.signature)

    def decode(self, path: Path) -> Decoded:
        return self.strategy.decode(path, self.cfg.model)

    def index_file(
        self,
        path: Path,
        known: dict[str, ScanKey] | None = None,
        decoded: Future[Decoded] | None = None,
        encoder: AudioEncoder | None = None,
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
        encoder = encoder or self.encoder
        try:
            dec = decoded.result() if decoded else self.decode(path)
            self.strategy.embed_and_upsert(key, st, dec, self.cfg.model, encoder, self.store)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", path, e)
            return False
        return True

    def remove(self, paths: Iterable[Path]) -> None:
        self.store.delete_paths([str(p) for p in paths])

    def scan(self, stop: StopFlag | None = None) -> ScanStats:
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
        return ScanStats(len(files), added, len(missing))

    def index_many(self, todo: list[Path], stop: StopFlag | None = None) -> int:
        """Decode on a thread pool and embed on one worker thread per encoder (GPU)."""
        workers = len(self.encoders)
        work: queue.Queue[tuple[Path, Future[Decoded]] | None] = queue.Queue(maxsize=workers * 4)
        progress = {"done": 0, "added": 0}
        progress_lock = threading.Lock()
        def stopped() -> bool:
            return stop is not None and stop.is_set()

        def consume(encoder: AudioEncoder) -> None:
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
                work.put((f, pool.submit(self.decode, f)))
            for _ in consumers:
                work.put(None)
            for t in consumers:
                t.join()
        return progress["added"]

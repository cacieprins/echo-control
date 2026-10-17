from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import socketserver
import sys
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn, Protocol, override

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .clap import TextEncoder, signature
from .config import Config, socket_path
from .indexer import current_key, is_audio, state_key
from .records import ScanKey, ScanStats
from .store import VectorIndex, wait_ready
from .strategies import open_store

log = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 5.0
# Reads (including our own while embedding) emit "opened" and "closed_no_write"; ignore them.
CHANGE_EVENTS = {"created", "modified", "moved", "deleted", "closed"}


class TextEmbedder(Protocol):
    dim: int

    def embed_text(self, text: str) -> list[float]: ...


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "urllib3", "numba", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _hard_exit(code: int) -> NoReturn:
    """Skip normal interpreter teardown, which aborts inside ROCm/OpenBLAS worker threads
    (SIGABRT core dump); every upsert is already committed before exiting."""
    logging.shutdown()
    os._exit(code)


def _index_job(
    cfg: Config,
    paths: list[str] | None,
    stop: Event,
    conn: Connection[ScanStats | int, object],
) -> None:
    """Runs in a spawned child: loads the GPU encoders, indexes, and exits, which is the
    only way to get rid of ROCm's spinning runtime thread once the GPU has been used."""
    from .clap import load_encoders
    from .indexer import Indexer

    setup_logging()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    code = 0
    try:
        indexer = Indexer(cfg, load_encoders(cfg.model), open_store(cfg))
        if paths is None:
            conn.send(indexer.scan(stop))
        else:
            conn.send(indexer.index_many([Path(p) for p in paths], stop))
    except Exception:
        log.exception("indexing failed")
        code = 1
    finally:
        conn.close()
        _hard_exit(code)


class DaemonState:
    """Thread-safe daemon runtime status."""

    def __init__(self, started: float | None = None):
        self._lock: threading.Lock = threading.Lock()
        self.started: float = time.time() if started is None else started
        self.scanning: bool = False
        self.indexing: bool = False
        self.last_scan: dict[str, Any] | None = None

    def set_scanning(self, value: bool) -> None:
        with self._lock:
            self.scanning = value

    def set_indexing(self, value: bool) -> None:
        with self._lock:
            self.indexing = value

    def record_scan(self, stats: ScanStats, at: float | None = None) -> None:
        total, added, removed = stats
        with self._lock:
            self.last_scan = {
                "at": time.time() if at is None else at,
                "files": total,
                "indexed": added,
                "removed": removed,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self.started,
                "scanning": self.scanning,
                "indexing": self.indexing,
                "last_scan": dict(self.last_scan) if self.last_scan else None,
            }

    def __getitem__(self, key: str) -> Any:
        return self.snapshot()[key]


class PendingQueue:
    """Thread-safe debounce queue for changed library paths."""

    def __init__(
        self,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.debounce_seconds: float = debounce_seconds
        self.clock: Callable[[], float] = clock
        self._lock: threading.Lock = threading.Lock()
        self._pending: dict[Path, float] = {}

    def mark_changed(self, path: Path, now: float | None = None) -> None:
        ts = self.clock() if now is None else now
        with self._lock:
            self._pending[path] = ts

    def take_ready(self, now: float | None = None) -> list[Path]:
        ts = self.clock() if now is None else now
        with self._lock:
            ready = [
                p for p, t in self._pending.items() if ts - t >= self.debounce_seconds
            ]
            for p in ready:
                del self._pending[p]
            return ready

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)


class _Handler(FileSystemEventHandler):
    def __init__(self, daemon: Daemon):
        self.daemon: Daemon = daemon

    @override
    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in CHANGE_EVENTS:
            return
        if event.is_directory:
            if event.event_type in ("created", "moved", "deleted"):
                self.daemon.request_scan()
            return
        for p in (event.src_path, getattr(event, "dest_path", "")):
            if p:
                self.daemon.mark_changed(Path(os.fsdecode(p)))


class Daemon:
    def __init__(
        self,
        cfg: Config,
        *,
        text: TextEmbedder | None = None,
        store: VectorIndex | None = None,
        pending: PendingQueue | None = None,
    ):
        self.cfg: Config = cfg
        self.stop: threading.Event = threading.Event()
        self.scan_now: threading.Event = threading.Event()
        self.pending: PendingQueue = pending if pending is not None else PendingQueue()
        self.state: DaemonState = DaemonState()
        self.job_lock: threading.Lock = threading.Lock()
        self.signature: str = signature(cfg.model)

        if store is None:
            wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
        self.text: TextEmbedder = text or TextEncoder(cfg.model)
        self.store: VectorIndex = store or open_store(cfg)
        self.store.ensure(self.text.dim)

    def _run_job(self, paths: list[Path] | None) -> Any:
        """Index in a child process (a full scan when paths is None) and return its result,
        or None if it failed. Jobs run one at a time."""
        ctx = multiprocessing.get_context("spawn")
        with self.job_lock:
            if self.stop.is_set():
                return None
            recv, send = ctx.Pipe(duplex=False)
            stop = ctx.Event()
            args = (self.cfg, None if paths is None else [str(p) for p in paths], stop, send)
            proc = ctx.Process(target=_index_job, args=args, name="echo-control-index")
            self.state.set_indexing(True)
            try:
                proc.start()
                send.close()
                while proc.is_alive():
                    if self.stop.is_set():
                        stop.set()
                    proc.join(1.0)
                result = recv.recv() if recv.poll() else None
            finally:
                recv.close()
                self.state.set_indexing(False)
            if proc.exitcode != 0:
                log.error("indexing process exited with %s", proc.exitcode)
            return result

    def mark_changed(self, path: Path) -> None:
        if is_audio(path, self.cfg):
            self.pending.mark_changed(path)

    def touch(self, path: Path) -> None:
        self.mark_changed(path)

    def request_scan(self) -> None:
        self.scan_now.set()

    def _scan_loop(self) -> None:
        while not self.stop.is_set():
            self.state.set_scanning(True)
            result = None
            try:
                result = self._run_job(None)
            except Exception:
                log.exception("scan job failed")
            finally:
                self.state.set_scanning(False)
            if result is not None:
                self.state.record_scan(result)
            self.scan_now.wait(self.cfg.library.rescan_interval)
            self.scan_now.clear()

    def _pending_loop(self) -> None:
        while not self.stop.wait(1.0):
            self.flush_pending()

    def flush_pending(self, now: float | None = None) -> None:
        ready = self.pending.take_ready(now)
        if not ready:
            return
        gone = [p for p in ready if not p.exists()]
        if gone:
            self.store.delete_paths([str(p) for p in gone])
            for p in gone:
                log.info("removed %s", p)
        known = self._known(ready)
        changed = [
            p for p in ready
            if p.exists() and known.get(str(p)) != current_key(p, self.signature)
        ]
        if changed:
            added = self._run_job(changed)
            if added is not None:
                log.info("indexed %d of %d changed files", added, len(changed))

    def _known(self, paths: list[Path]) -> dict[str, ScanKey]:
        payloads = self.store.payloads([str(p) for p in paths])
        return {pl["path"]: state_key(pl) for pl in payloads}

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        op = req.get("op")
        if op == "embed":
            return {"vector": self.text.embed_text(str(req["text"]))}
        if op == "status":
            return {
                **self.state.snapshot(),
                "pending": len(self.pending),
                "device": self.cfg.model.device,
                "model": self.cfg.model.name,
                "strategy": str(self.cfg.model.strategy),
                "indexed": self.store.count(),
            }
        return {"error": f"unknown op {op!r}"}

    def serve(self) -> None:
        sock = socket_path()
        sock.unlink(missing_ok=True)
        daemon = self

        class RequestHandler(socketserver.StreamRequestHandler):
            @override
            def handle(self) -> None:
                line = self.rfile.readline()
                try:
                    resp = daemon.handle(json.loads(line))
                except Exception as e:
                    log.exception("request failed")
                    resp = {"error": str(e)}
                try:
                    self.wfile.write(json.dumps(resp).encode() + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads: bool = True

        server = Server(str(sock), RequestHandler)
        os.chmod(sock, 0o600)

        observer = Observer()
        handler = _Handler(self)
        for root in self.cfg.library.paths:
            if root.is_dir():
                observer.schedule(handler, str(root), recursive=True)
        observer.start()

        threads = [
            threading.Thread(target=self._scan_loop, name="scan", daemon=True),
            threading.Thread(target=self._pending_loop, name="pending", daemon=True),
            threading.Thread(target=server.serve_forever, name="socket", daemon=True),
        ]
        for t in threads:
            t.start()

        def shutdown(signum: int, _frame: FrameType | None) -> None:
            log.info("received signal %d, shutting down", signum)
            self.stop.set()
            self.scan_now.set()

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        log.info("listening on %s", sock)

        self.stop.wait()
        observer.stop()
        server.shutdown()
        server.server_close()
        sock.unlink(missing_ok=True)
        observer.join(timeout=5)
        for t in threads[:2]:
            t.join(timeout=30)
        _hard_exit(0)


def request(req: dict[str, Any], timeout: float = 30.0) -> dict[str, Any] | None:
    """Send one request to the running daemon; None if it isn't running."""
    import socket

    path = socket_path()
    if not path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(path))
            s.sendall(json.dumps(req).encode() + b"\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except (ConnectionRefusedError, FileNotFoundError, TimeoutError, OSError):
        return None
    if not buf.strip():
        return None
    resp = json.loads(buf)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp

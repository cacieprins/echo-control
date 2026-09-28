from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import socketserver
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .clap import TextEncoder, signature
from .config import Config, socket_path
from .indexer import current_key, is_audio, state_key
from .store import open_store, wait_ready

log = logging.getLogger(__name__)


def _index_job(cfg: Config, paths: list[str] | None, stop, conn) -> None:
    """Runs in a spawned child: loads the GPU encoders, indexes, and exits, which is the
    only way to get rid of ROCm's spinning runtime thread once the GPU has been used."""
    from .cli import _setup_logging
    from .clap import load_encoders
    from .indexer import Indexer

    _setup_logging()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    code = 0
    try:
        indexer = Indexer(cfg, load_encoders(cfg.model), open_store(cfg))
        if paths is None:
            conn.send(indexer.scan(stop))
        else:
            conn.send(indexer.index_many([Path(p) for p in paths], stop))
    except BaseException:
        log.exception("indexing failed")
        code = 1
    finally:
        conn.close()
        logging.shutdown()
        # Interpreter teardown aborts inside ROCm/OpenBLAS worker threads (SIGABRT core
        # dump); every upsert is already committed, so skip it.
        os._exit(code)

DEBOUNCE_SECONDS = 5.0
# Reads (including our own while embedding) emit "opened" and "closed_no_write"; ignore them.
CHANGE_EVENTS = {"created", "modified", "moved", "deleted", "closed"}


class _Handler(FileSystemEventHandler):
    def __init__(self, daemon: Daemon):
        self.daemon = daemon

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type not in CHANGE_EVENTS:
            return
        if event.is_directory:
            if event.event_type in ("created", "moved"):
                self.daemon.request_scan()
            return
        for p in (event.src_path, getattr(event, "dest_path", "")):
            if p:
                self.daemon.touch(Path(os.fsdecode(p)))


class Daemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stop = threading.Event()
        self.scan_now = threading.Event()
        self.pending: dict[Path, float] = {}
        self.pending_lock = threading.Lock()
        self.state = {"started": time.time(), "scanning": False, "indexing": False, "last_scan": None}
        self.job_lock = threading.Lock()
        self.signature = signature(cfg.model)

        wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
        self.text = TextEncoder(cfg.model)
        self.store = open_store(cfg)
        self.store.ensure(self.text.dim)

    def _run_job(self, paths: list[Path] | None):
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
            self.state["indexing"] = True
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
                self.state["indexing"] = False
            if proc.exitcode != 0:
                log.error("indexing process exited with %s", proc.exitcode)
            return result

    def touch(self, path: Path) -> None:
        if is_audio(path, self.cfg):
            with self.pending_lock:
                self.pending[path] = time.monotonic()

    def request_scan(self) -> None:
        self.scan_now.set()

    def _scan_loop(self) -> None:
        while not self.stop.is_set():
            self.state["scanning"] = True
            try:
                result = self._run_job(None)
            finally:
                self.state["scanning"] = False
            if result is not None:
                total, added, removed = result
                self.state["last_scan"] = {
                    "at": time.time(), "files": total, "indexed": added, "removed": removed
                }
            self.scan_now.wait(self.cfg.library.rescan_interval)
            self.scan_now.clear()

    def _pending_loop(self) -> None:
        while not self.stop.wait(1.0):
            now = time.monotonic()
            with self.pending_lock:
                ready = [p for p, t in self.pending.items() if now - t >= DEBOUNCE_SECONDS]
                for p in ready:
                    del self.pending[p]
            if not ready:
                continue
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

    def _known(self, paths: list[Path]) -> dict:
        payloads = self.store.payloads([str(p) for p in paths])
        return {pl["path"]: state_key(pl) for pl in payloads}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "embed":
            return {"vector": self.text.embed_text(str(req["text"]))}
        if op == "status":
            with self.pending_lock:
                pending = len(self.pending)
            return {
                **self.state,
                "pending": pending,
                "device": self.cfg.model.device,
                "model": self.cfg.model.name,
                "strategy": self.cfg.model.strategy,
                "indexed": self.store.count(),
            }
        return {"error": f"unknown op {op!r}"}

    def serve(self) -> None:
        sock = socket_path()
        sock.unlink(missing_ok=True)
        daemon = self

        class RequestHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                line = self.rfile.readline()
                try:
                    resp = daemon.handle(json.loads(line))
                except Exception as e:
                    log.exception("request failed")
                    resp = {"error": str(e)}
                self.wfile.write(json.dumps(resp).encode() + b"\n")

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

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

        def shutdown(signum, _frame):
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
        logging.shutdown()
        # Normal interpreter teardown aborts inside ROCm/OpenBLAS worker threads (SIGABRT
        # core dump); every upsert is already committed, so skip it.
        os._exit(0)


def request(req: dict, timeout: float = 30.0) -> dict | None:
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
    except (ConnectionRefusedError, FileNotFoundError, TimeoutError):
        return None
    resp = json.loads(buf)
    if "error" in resp:
        raise RuntimeError(resp["error"])
    return resp

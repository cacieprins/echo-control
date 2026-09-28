from __future__ import annotations

import json
import logging
import os
import signal
import socketserver
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .clap import load_encoders
from .config import Config, socket_path
from .indexer import Indexer, is_audio
from .store import open_store, wait_ready

log = logging.getLogger(__name__)

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
        self.state = {"started": time.time(), "scanning": False, "last_scan": None}

        wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
        self.encoders = load_encoders(cfg.model)
        self.clap = self.encoders[0]
        self.store = open_store(cfg)
        self.indexer = Indexer(cfg, self.encoders, self.store)

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
                total, added, removed = self.indexer.scan(self.stop)
                self.state["last_scan"] = {
                    "at": time.time(), "files": total, "indexed": added, "removed": removed
                }
            except Exception:
                log.exception("scan failed")
            finally:
                self.state["scanning"] = False
            self.scan_now.wait(self.cfg.library.rescan_interval)
            self.scan_now.clear()

    def _pending_loop(self) -> None:
        while not self.stop.wait(1.0):
            now = time.monotonic()
            with self.pending_lock:
                ready = [p for p, t in self.pending.items() if now - t >= DEBOUNCE_SECONDS]
                for p in ready:
                    del self.pending[p]
            if ready:
                known = self._known(ready)
            for p in ready:
                if p.exists():
                    if self.indexer.index_file(p, known):
                        log.info("indexed %s", p)
                else:
                    self.indexer.remove([p])
                    log.info("removed %s", p)

    def _known(self, paths: list[Path]) -> dict:
        payloads = self.store.payloads([str(p) for p in paths])
        return {pl["path"]: self.indexer.state_key(pl) for pl in payloads}

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        if op == "embed":
            return {"vector": self.clap.embed_text(str(req["text"]))}
        if op == "status":
            with self.pending_lock:
                pending = len(self.pending)
            return {
                **self.state,
                "pending": pending,
                "device": ", ".join(e.device for e in self.encoders),
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
        threads[0].join(timeout=30)
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

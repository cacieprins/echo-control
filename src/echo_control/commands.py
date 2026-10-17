# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as config_mod
from .config import Config
from .errors import EchoControlError, UsageError
from .players import play, play_mpv_continuous
from .store import ChunkIndex, is_ready, wait_ready
from .strategies import get_strategy, open_store

Notify = Callable[[str], None]


def _stderr_notify(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class VibeRequest:
    query: list[str] = field(default_factory=list)
    limit: int | None = None
    player: str | None = None
    shuffle: bool = False
    print_m3u: bool = False
    as_json: bool = False
    detach: bool = False
    play_path: list[str] | None = None
    wander: bool = False
    orbit: float | None = None
    seed: str | None = None
    continuous: bool = False

    @property
    def clean_query(self) -> str:
        return self.query[0].strip() if self.query else ""


def validate_vibe(req: VibeRequest, cfg: Config) -> None:
    if req.limit is not None and req.limit <= 0:
        raise UsageError("-n/--limit must be at least 1")
    if req.orbit is not None:
        if req.orbit < 0:
            raise UsageError("--orbit strength can't be negative")
        req.wander = True
    if req.continuous:
        req.wander = True
        if req.as_json or req.print_m3u:
            raise UsageError(
                "--continuous plays in mpv; it can't be combined with --json or --print"
            )
        if (req.player or cfg.player.backend) != "mpv":
            raise UsageError("--continuous needs the mpv player")
    if req.play_path:
        return
    if len(req.query) > 1:
        raise UsageError(f'put the query in quotes, e.g. vibe "{" ".join(req.query)}"')
    if req.seed and not req.wander:
        raise UsageError("--from only applies to --wander and --orbit")
    if req.wander and req.shuffle:
        raise UsageError("--shuffle would undo the walk's ordering")
    if req.wander and not get_strategy(cfg.model).supports_wander:
        raise UsageError('--wander and --orbit need [model] strategy = "chunks"')
    if not req.clean_query and not req.seed:
        raise UsageError(show_usage=True)


def choose_backend(req: VibeRequest, cfg: Config, *, is_tty: bool) -> str:
    if req.print_m3u or not (is_tty or req.detach or req.player):
        return "stdout"
    return req.player or cfg.player.backend


def relative_dir(path: Path, roots: list[Path]) -> str:
    for root in roots:
        if path.is_relative_to(root):
            rel = path.parent.relative_to(root)
            return str(rel) if str(rel) != "." else root.name
    return str(path.parent)


def format_tracks_json(scores: list[tuple[str, float]], roots: list[Path]) -> str:
    out = [
        {
            "path": path,
            "name": Path(path).stem,
            "dir": relative_dir(Path(path), roots),
            "score": round(score, 4),
        }
        for path, score in scores
    ]
    return json.dumps(out)


def embed_query(cfg: Config, text: str, notify: Notify = _stderr_notify) -> list[float]:
    from .daemon import request

    resp = request({"op": "embed", "text": text})
    if resp is not None:
        return resp["vector"]
    notify("daemon not running; loading CLAP locally...")
    from .clap import TextEncoder

    return TextEncoder(cfg.model).embed_text(text)


def run_vibe(
    req: VibeRequest,
    cfg: Config,
    *,
    is_tty: bool = False,
    notify: Notify = _stderr_notify,
) -> str | None:
    validate_vibe(req, cfg)
    if req.play_path:
        backend = req.player or cfg.player.backend
        play(backend, req.play_path, Path(req.play_path[0]).stem, cfg.player, req.detach)
        return None

    if not is_ready(cfg.qdrant.url):
        raise EchoControlError(
            f"Qdrant not reachable at {cfg.qdrant.url} (systemctl --user start echo-control)"
        )

    store = open_store(cfg)
    if not store.exists():
        raise EchoControlError("the index is empty; start the daemon or run `echo-control scan`")

    query = req.clean_query
    limit = req.limit if req.limit is not None else cfg.player.limit
    vector: list[float] | None = None
    if query:
        try:
            vector = embed_query(cfg, query, notify=notify)
        except RuntimeError as e:
            raise EchoControlError(f"embedding failed: {e}") from e

    scores: list[tuple[str, float]] = []
    if req.seed:
        seed = str(Path(req.seed).expanduser().resolve())
    else:
        assert vector is not None
        hits = store.search(vector, 1 if req.wander else limit)
        if req.wander and not hits:
            raise EchoControlError("nothing matched to start wandering from")
        seed = hits[0].path if hits else ""
        scores = [(h.path, h.score) for h in hits]

    if req.wander:
        from .wander import Orbit, walk, wander

        assert isinstance(store, ChunkIndex)
        found = store.track(seed)
        if found is None:
            raise EchoControlError(f"{seed} isn't in the chunk index")
        orbit = None
        if req.orbit is not None and req.orbit > 0:
            orbit = Orbit(vector or found[1]["mean"].tolist(), req.orbit)
        if req.continuous:
            steps = (t.path for t in walk(store, seed, cfg.wander, orbit))
            play_mpv_continuous(steps, query or f"wander from {Path(seed).stem}", req.detach)
            return None
        chain = wander(store, seed, limit, cfg.wander, orbit)
        scores = [(t.path, t.score) for t in chain]

    if req.as_json:
        return format_tracks_json(scores, cfg.library.paths)

    paths = [path for path, _ in scores]
    if req.shuffle:
        random.shuffle(paths)
    backend = choose_backend(req, cfg, is_tty=is_tty)
    play(backend, paths, query or f"wander from {Path(seed).stem}", cfg.player, req.detach)
    return None


def run_daemon(cfg: Config) -> None:
    from .daemon import Daemon, setup_logging

    setup_logging()
    Daemon(cfg).serve()


def run_scan(cfg: Config) -> str:
    from .clap import load_encoders
    from .daemon import setup_logging
    from .indexer import Indexer

    setup_logging()
    wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
    total, added, removed = Indexer(cfg, load_encoders(cfg.model), open_store(cfg)).scan()
    return f"{total} files, {added} indexed, {removed} removed"


def run_reindex(cfg: Config) -> Iterator[str]:
    from .daemon import request

    wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
    open_store(cfg).drop()
    yield f"dropped the {cfg.model.strategy} index for {cfg.qdrant.collection!r}"
    try:
        running = request({"op": "status"}, timeout=5) is not None
    except RuntimeError:
        running = False
    if running:
        yield "restart the daemon to rebuild: systemctl --user restart echo-control"
    else:
        yield run_scan(cfg)


@dataclass
class Status:
    config_path: Path | None
    library_paths: list[Path]
    qdrant_url: str
    qdrant_ready: bool
    daemon_status: dict[str, Any] | None = None
    daemon_error: str | None = None
    indexed_fallback: int | None = None


def collect_status(cfg: Config) -> Status:
    from .daemon import request

    ready = is_ready(cfg.qdrant.url)
    st: dict[str, Any] | None = None
    err: str | None = None
    try:
        st = request({"op": "status"}, timeout=5)
    except RuntimeError as e:
        err = str(e)
    fallback = open_store(cfg).count() if (st is None and ready) else None
    return Status(
        config_path=cfg.path,
        library_paths=cfg.library.paths,
        qdrant_url=cfg.qdrant.url,
        qdrant_ready=ready,
        daemon_status=st,
        daemon_error=err,
        indexed_fallback=fallback,
    )


def format_status(status: Status) -> str:
    lines = [
        f"config:  {status.config_path or '(defaults, no config file)'}",
        f"library: {', '.join(map(str, status.library_paths))}",
        f"qdrant:  {status.qdrant_url} {'ready' if status.qdrant_ready else 'NOT REACHABLE'}",
    ]
    if status.daemon_error is not None:
        lines.append(f"daemon:  error: {status.daemon_error}")
    st = status.daemon_status
    if st is None:
        lines.append("daemon:  not running")
        if status.indexed_fallback is not None:
            lines.append(f"indexed: {status.indexed_fallback} tracks")
        return "\n".join(lines)
    lines.append(
        f"daemon:  running ({st['model']} on {st['device']}, {st.get('strategy', 'mean')} strategy)"
    )
    lines.append(f"indexed: {st['indexed']} tracks")
    scan = st.get("last_scan")
    if st["scanning"]:
        lines.append("scan:    in progress (journalctl --user -fu echo-control)")
    elif st.get("indexing"):
        lines.append("scan:    indexing changed files")
    elif scan:
        lines.append(
            f"scan:    last found {scan['files']} files, indexed {scan['indexed']}, removed {scan['removed']}"
        )
    if st["pending"]:
        lines.append(f"pending: {st['pending']} changed files")
    return "\n".join(lines)


def _contrib_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "contrib", here.parent.parent / "contrib"):
        if candidate.is_dir():
            return candidate
    raise EchoControlError("contrib files not found in the installation")


def _run_cmd(argv: list[str]) -> None:
    print("$ " + " ".join(argv))
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    if r.returncode != 0:
        raise EchoControlError(f"{argv[0]} exited with {r.returncode}")


def install_service(_cfg: Config | None = None) -> None:
    contrib = _contrib_dir()
    exe = shutil.which("echo-control") or str(Path(sys.argv[0]).resolve())
    unit_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (config_mod.DATA_DIR / "qdrant").mkdir(parents=True, exist_ok=True)

    compose = config_mod.CONFIG_DIR / "docker-compose.yml"
    compose.write_text(
        (contrib / "docker-compose.yml")
        .read_text()
        .replace("@DATA_DIR@", str(config_mod.DATA_DIR))
    )
    print(f"wrote {compose}")
    for name in ("echo-control-qdrant.service", "echo-control.service"):
        text = (contrib / name).read_text()
        text = text.replace("@EXE@", exe).replace("@COMPOSE@", str(compose))
        text = text.replace("@DOCKER@", shutil.which("docker") or "/usr/bin/docker")
        (unit_dir / name).write_text(text)
        print(f"wrote {unit_dir / name}")
    if not config_mod.CONFIG_PATH.exists():
        example = contrib / "config.example.toml"
        if example.exists():
            shutil.copy(example, config_mod.CONFIG_PATH)
            print(f"wrote {config_mod.CONFIG_PATH} (edit library paths)")
    _run_cmd(["systemctl", "--user", "daemon-reload"])
    print("next: systemctl --user enable --now echo-control")


def install_noctalia(_cfg: Config | None = None) -> None:
    if not shutil.which("noctalia"):
        raise EchoControlError("noctalia is not installed")
    source = _contrib_dir() / "noctalia"
    r = subprocess.run(
        ["noctalia", "msg", "plugins", "source", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not any(line.split()[:1] == ["vibe"] for line in r.stdout.splitlines()):
        _run_cmd(["noctalia", "msg", "plugins", "source", "add", "vibe", "path", str(source)])
    _run_cmd(["noctalia", "msg", "plugins", "update", "vibe"])
    _run_cmd(["noctalia", "msg", "plugins", "enable", "case/vibe"])
    print("type /vibe <query> in the launcher")

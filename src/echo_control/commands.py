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
    auto: bool = False
    continuous: bool = False
    no_tui: bool = False

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
    if req.auto:
        req.wander = True
        if req.seed or req.clean_query:
            raise UsageError("--auto picks the starting track; don't combine it with --from or a query")
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
    if not req.clean_query and not req.seed and not req.auto:
        raise UsageError(show_usage=True)


def choose_backend(req: VibeRequest, cfg: Config, *, is_tty: bool) -> str:
    if req.print_m3u or not (is_tty or req.detach or req.player):
        return "stdout"
    return req.player or cfg.player.backend


def should_use_tui(req: VibeRequest, cfg: Config, *, is_tty: bool) -> bool:
    return (
        is_tty
        and not req.no_tui
        and not req.detach
        and not req.print_m3u
        and not req.as_json
        and req.wander
        and choose_backend(req, cfg, is_tty=is_tty) == "mpv"
    )


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
    elif req.auto:
        seed = ""
    else:
        assert vector is not None
        hits = store.search(vector, 1 if req.wander else limit)
        if req.wander and not hits:
            raise EchoControlError("nothing matched to start wandering from")
        seed = hits[0].path if hits else ""
        scores = [(h.path, h.score) for h in hits]

    if req.wander:
        from .art import ensure_latent_art
        from .atlas import LiveAtlasRenderer
        from .config import CACHE_DIR, STATE_DIR
        from .cooldown import CooldownScorer, HeatField, TrailLog, coldest_track
        from .manifold import CorpusManifold
        from .players import play_mpv_tui
        from .playlog import DailyPlayLog
        from .surfing import ChannelSurfer
        from .tui import QueueEntry, VibeSession
        from .wander import Orbit, Transition, walk, wander

        assert isinstance(store, ChunkIndex)
        manifold = CorpusManifold.from_store(
            store,
            neighbors=cfg.wander.csls_neighbors,
            cache_path=CACHE_DIR / f"manifold_{cfg.qdrant.collection}.npz",
        )
        trails = TrailLog.load(STATE_DIR / "trails.json", cfg.wander)
        if req.auto:
            if not manifold.paths:
                raise EchoControlError("the index is empty; start the daemon or run `echo-control scan`")
            seed = coldest_track(manifold, trails, cfg.wander)
            notify(f"auto seed (coldest): {Path(seed).stem}")
        found = store.track(seed)
        if found is None:
            raise EchoControlError(f"{seed} isn't in the chunk index")
        orbit = None
        if req.orbit is not None and req.orbit > 0:
            orbit = Orbit(vector or found[1]["mean"].tolist(), req.orbit)

        play_log = DailyPlayLog(STATE_DIR / "logs")
        heat_field = HeatField.from_config(cfg.wander)
        cooldown = CooldownScorer(
            manifold=manifold,
            heat_field=heat_field,
            trails=trails,
            cfg=cfg.wander,
        )
        surfer = ChannelSurfer(manifold=manifold, trails=trails, heat_field=heat_field)
        live_atlas = LiveAtlasRenderer(
            manifold=manifold,
            trails=trails,
            cfg=cfg,
            log_dir=STATE_DIR / "logs",
            cache_dir=CACHE_DIR,
        )

        use_tui = should_use_tui(req, cfg, is_tty=is_tty)
        title = query or f"wander from {Path(seed).stem}"

        if req.continuous:
            if use_tui:
                def _walk_factory(
                    from_path: str, strength: float | None, center_path: str | None
                ) -> Iterator[Transition]:
                    orb = None
                    if strength is not None and strength > 0:
                        if center_path is None and vector is not None:
                            center_vec = vector
                        elif center_path is not None:
                            tr = store.track(center_path)
                            center_vec = tr[1]["mean"].tolist() if tr is not None else found[1]["mean"].tolist()
                        else:
                            center_vec = vector or found[1]["mean"].tolist()
                        orb = Orbit(center_vec, strength)
                    return walk(store, from_path, cfg.wander, orb, cooldown=cooldown)

                stream = _walk_factory(seed, req.orbit, None if vector is not None else seed)
                session = VibeSession(
                    entries=[],
                    query=query,
                    continuous=True,
                    orbit_strength=req.orbit,
                    orbit_center_path=seed if (req.orbit is not None and vector is None) else None,
                    walk_factory=_walk_factory,
                    stream=stream,
                )
                play_mpv_tui(
                    session,
                    title,
                    art_provider=lambda p: ensure_latent_art(p, store),
                    map_provider=live_atlas.render,
                    on_commit=lambda p, prev: cooldown.commit_play(p, prev, persist=True),
                    play_log=play_log,
                    surfer=surfer,
                    commit_seconds=cfg.wander.commit_seconds,
                    commit_fraction=cfg.wander.commit_fraction,
                )
                return None

            steps = (
                t.path
                for t in walk(
                    store, seed, cfg.wander, orbit, cooldown=cooldown, update_heat_on_step=True
                )
            )
            play_mpv_continuous(steps, title, req.detach)
            return None

        chain = wander(
            store, seed, limit, cfg.wander, orbit, cooldown=cooldown, update_heat_on_step=True
        )
        if use_tui:
            session = VibeSession(
                entries=[QueueEntry.from_transition(t) for t in chain],
                query=query,
                continuous=False,
                orbit_strength=req.orbit,
                orbit_center_path=seed if req.orbit is not None else None,
            )
            play_mpv_tui(
                session,
                title,
                art_provider=lambda p: ensure_latent_art(p, store),
                map_provider=live_atlas.render,
                on_commit=lambda p, prev: cooldown.commit_play(p, prev, persist=True),
                play_log=play_log,
                surfer=surfer,
                commit_seconds=cfg.wander.commit_seconds,
                commit_fraction=cfg.wander.commit_fraction,
            )
            return None
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

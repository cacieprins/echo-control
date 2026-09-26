from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as config_mod
from .players import BACKENDS, PlayerError, play
from .store import QdrantUnavailable, is_ready


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _contrib_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "contrib", here.parent.parent / "contrib"):
        if candidate.is_dir():
            return candidate
    _die("contrib files not found in the installation")
    raise AssertionError


def _embed_text(cfg: config_mod.Config, text: str) -> list[float]:
    from .daemon import request

    resp = request({"op": "embed", "text": text})
    if resp is not None:
        return resp["vector"]
    print("daemon not running; loading CLAP locally (slow)...", file=sys.stderr)
    from .clap import Clap

    return Clap(cfg.model).embed_text(text)


def _relative_dir(path: Path, roots: list[Path]) -> str:
    for root in roots:
        if path.is_relative_to(root):
            rel = path.parent.relative_to(root)
            return str(rel) if str(rel) != "." else root.name
    return str(path.parent)


def vibe(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="vibe",
        description="Play music from your library that sounds like the description.",
        epilog='example: vibe "angsty lesbian music from the late 20th century"',
    )
    p.add_argument("query", nargs="*", help="description, in quotes")
    p.add_argument("-n", "--limit", type=int, help="number of tracks")
    p.add_argument("--player", choices=BACKENDS, help="playback backend")
    p.add_argument("--shuffle", action="store_true", help="shuffle the results")
    p.add_argument("--print", action="store_true", help="print an M3U playlist to stdout")
    p.add_argument("--json", action="store_true", help="print results as JSON, don't play")
    p.add_argument("--detach", action="store_true", help="start the player in the background")
    p.add_argument("--play-path", metavar="FILE", nargs="+", help="play these files, no search")
    args = p.parse_args(argv)
    cfg = config_mod.load()

    if args.play_path:
        backend = args.player or cfg.player.backend
        try:
            play(backend, args.play_path, Path(args.play_path[0]).stem, cfg.player, args.detach)
        except PlayerError as e:
            _die(str(e))
        return

    if len(args.query) > 1:
        _die(f'put the query in quotes, e.g. vibe "{" ".join(args.query)}"', 2)
    if not args.query or not args.query[0].strip():
        p.print_usage(sys.stderr)
        sys.exit(2)
    query = args.query[0].strip()

    if not is_ready(cfg.qdrant.url):
        _die(f"Qdrant not reachable at {cfg.qdrant.url} (systemctl --user start echo-control)")

    from .store import Store

    store = Store(cfg.qdrant)
    if not store.exists():
        _die("the index is empty; start the daemon or run `echo-control scan`")

    try:
        vector = _embed_text(cfg, query)
    except RuntimeError as e:
        _die(f"embedding failed: {e}")
    hits = store.search(vector, args.limit or cfg.player.limit)

    if args.json:
        out = [
            {
                "path": h.path,
                "name": Path(h.path).stem,
                "dir": _relative_dir(Path(h.path), cfg.library.paths),
                "score": round(h.score, 4),
            }
            for h in hits
        ]
        json.dump(out, sys.stdout)
        sys.stdout.write("\n")
        return

    paths = [h.path for h in hits]
    if args.shuffle:
        random.shuffle(paths)
    if args.print or not (sys.stdout.isatty() or args.detach or args.player):
        backend = "stdout"
    else:
        backend = args.player or cfg.player.backend
    try:
        play(backend, paths, query, cfg.player, args.detach)
    except PlayerError as e:
        _die(str(e))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "urllib3", "numba", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_daemon(cfg: config_mod.Config, _args) -> None:
    _setup_logging()
    from .daemon import Daemon

    Daemon(cfg).serve()


def cmd_scan(cfg: config_mod.Config, _args) -> None:
    _setup_logging()
    from .clap import Clap
    from .indexer import Indexer
    from .store import Store, wait_ready

    wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
    total, added, removed = Indexer(cfg, Clap(cfg.model), Store(cfg.qdrant)).scan()
    print(f"{total} files, {added} indexed, {removed} removed")


def cmd_reindex(cfg: config_mod.Config, args) -> None:
    from .store import Store, wait_ready

    wait_ready(cfg.qdrant.url, cfg.qdrant.ready_timeout)
    Store(cfg.qdrant).drop()
    print(f"dropped collection {cfg.qdrant.collection!r}")
    from .daemon import request

    try:
        running = request({"op": "status"}, timeout=5) is not None
    except RuntimeError:
        running = False
    if running:
        print("restart the daemon to rebuild: systemctl --user restart echo-control")
    else:
        cmd_scan(cfg, args)


def cmd_status(cfg: config_mod.Config, _args) -> None:
    from .daemon import request

    print(f"config:  {cfg.path or '(defaults, no config file)'}")
    print(f"library: {', '.join(map(str, cfg.library.paths))}")
    ready = is_ready(cfg.qdrant.url)
    print(f"qdrant:  {cfg.qdrant.url} {'ready' if ready else 'NOT REACHABLE'}")
    try:
        st = request({"op": "status"}, timeout=5)
    except RuntimeError as e:
        st = None
        print(f"daemon:  error: {e}")
    if st is None:
        print("daemon:  not running")
        if ready:
            from .store import Store

            print(f"indexed: {Store(cfg.qdrant).count()} tracks")
        return
    print(f"daemon:  running ({st['model']} on {st['device']})")
    print(f"indexed: {st['indexed']} tracks")
    scan = st.get("last_scan")
    if st["scanning"]:
        print("scan:    in progress (journalctl --user -fu echo-control)")
    elif scan:
        print(f"scan:    last found {scan['files']} files, indexed {scan['indexed']}, removed {scan['removed']}")
    if st["pending"]:
        print(f"pending: {st['pending']} changed files")


def _run(argv: list[str]) -> None:
    print("$ " + " ".join(argv))
    r = subprocess.run(argv, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    if r.returncode != 0:
        _die(f"{argv[0]} exited with {r.returncode}")


def cmd_install_service(_cfg: config_mod.Config, _args) -> None:
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
    _run(["systemctl", "--user", "daemon-reload"])
    print("next: systemctl --user enable --now echo-control")


def cmd_install_noctalia(_cfg: config_mod.Config, _args) -> None:
    if not shutil.which("noctalia"):
        _die("noctalia is not installed")
    source = _contrib_dir() / "noctalia"
    r = subprocess.run(
        ["noctalia", "msg", "plugins", "source", "list"], capture_output=True, text=True
    )
    if not any(line.split()[:1] == ["vibe"] for line in r.stdout.splitlines()):
        _run(["noctalia", "msg", "plugins", "source", "add", "vibe", "path", str(source)])
    _run(["noctalia", "msg", "plugins", "update", "vibe"])
    _run(["noctalia", "msg", "plugins", "enable", "case/vibe"])
    print("type /vibe <query> in the launcher")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="echo-control", description="CLAP music index daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon", help="run the indexer, file watcher and embedding socket")
    sub.add_parser("scan", help="index new and changed files once, without the daemon")
    sub.add_parser("reindex", help="drop the collection and rebuild it")
    sub.add_parser("status", help="show daemon, Qdrant and index state")
    sub.add_parser("install-service", help="install docker-compose and systemd user units")
    sub.add_parser("install-noctalia", help="register and enable the Noctalia launcher plugin")
    args = p.parse_args(argv)

    cfg = config_mod.load()
    handler = {
        "daemon": cmd_daemon,
        "scan": cmd_scan,
        "reindex": cmd_reindex,
        "status": cmd_status,
        "install-service": cmd_install_service,
        "install-noctalia": cmd_install_noctalia,
    }[args.cmd]
    try:
        handler(cfg, args)
    except QdrantUnavailable as e:
        _die(str(e))
    except KeyboardInterrupt:
        sys.exit(130)

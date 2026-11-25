# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from . import atlas, commands
from . import config as config_mod
from .errors import EchoControlError, UsageError
from .players import BACKENDS


def _fail(err: EchoControlError, parser: argparse.ArgumentParser | None = None) -> NoReturn:
    if isinstance(err, UsageError) and err.show_usage and parser is not None:
        parser.print_usage(sys.stderr)
    if str(err):
        print(f"error: {err}", file=sys.stderr)
    sys.exit(err.exit_code)


def vibe(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="vibe",
        description="Walk a local music library along smooth trajectories in audio embedding space.",
        epilog="example: vibe --orbit --from ~/Music/album/opener.flac",
    )
    p.add_argument("query", nargs="*", help="description, in quotes")
    p.add_argument("-n", "--limit", type=int, help="number of tracks")
    p.add_argument("--player", choices=BACKENDS, help="playback backend")
    p.add_argument("--shuffle", action="store_true", help="shuffle the results")
    p.add_argument("--print", action="store_true", help="print an M3U playlist to stdout")
    p.add_argument("--json", action="store_true", help="print results as JSON, don't play")
    p.add_argument("--detach", action="store_true", help="start the player in the background")
    p.add_argument("--play-path", metavar="FILE", nargs="+", help="play these files, no search")
    p.add_argument(
        "--wander",
        action="store_true",
        help="start from the best match (or --from) and chain tracks whose intros continue the previous outro (chunks strategy only)",
    )
    p.add_argument(
        "--orbit",
        nargs="?",
        const=1.0,
        type=float,
        metavar="N",
        help="--wander, pulled back toward the query (or the --from track) with strength N (default 1; 0 = plain --wander)",
    )
    p.add_argument("--from", dest="seed", metavar="FILE", help="seed track for --wander/--orbit")
    p.add_argument(
        "--auto",
        action="store_true",
        help="start wandering from the coldest track in the manifold (implies --wander)",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="keep wandering (or orbiting) forever in mpv, queueing tracks as they play",
    )
    p.add_argument(
        "--no-tui",
        action="store_true",
        help="disable the TUI display and use plain mpv output",
    )
    args = p.parse_args(argv)
    req = commands.VibeRequest(
        query=args.query,
        limit=args.limit,
        player=args.player,
        shuffle=args.shuffle,
        print_m3u=args.print,
        as_json=args.json,
        detach=args.detach,
        play_path=args.play_path,
        wander=args.wander,
        orbit=args.orbit,
        seed=args.seed,
        auto=args.auto,
        continuous=args.continuous,
        no_tui=args.no_tui,
    )
    try:
        cfg = config_mod.load()
        out = commands.run_vibe(req, cfg, is_tty=sys.stdout.isatty())
        if out is not None:
            print(out)
    except EchoControlError as e:
        _fail(e, p)
    except KeyboardInterrupt:
        sys.exit(130)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="echo-control", description="CLAP music index daemon")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon", help="run the indexer, file watcher and embedding socket")
    sub.add_parser("scan", help="index new and changed files once, without the daemon")
    sub.add_parser("reindex", help="drop the collection and rebuild it")
    sub.add_parser("status", help="show daemon, Qdrant and index state")
    sub.add_parser("install-service", help="install docker-compose and systemd user units")
    sub.add_parser("install-noctalia", help="register and enable the Noctalia launcher plugin")
    mp = sub.add_parser("map", help="render a 2D library atlas or local topographic heat map")
    mp.add_argument("-o", "--output", type=Path, help="output file (.png or .svg)")
    mp.add_argument(
        "-k", "--clusters", type=int, default=15, help="number of 512D spherical clusters (default 15)"
    )
    mp.add_argument("--highlight", metavar="TEXT", help="highlight matching tracks with pink rings")
    mp.add_argument(
        "--focus",
        metavar="TEXT",
        help="render a local 2D spherical topographic heat map around tracks matching TEXT",
    )
    mp.add_argument("--date", metavar="YYYY-MM-DD", help="play log date to overlay (default today)")
    mp.add_argument(
        "--recent",
        action="store_true",
        help="label the last 3 played tracks with callouts instead of cluster labels",
    )
    mp.add_argument("--open", action="store_true", help="open the rendered map with xdg-open")
    args = p.parse_args(argv)

    try:
        if args.cmd == "install-service":
            commands.install_service()
            return
        if args.cmd == "install-noctalia":
            commands.install_noctalia()
            return
        cfg = config_mod.load()
        if args.cmd == "daemon":
            commands.run_daemon(cfg)
        elif args.cmd == "scan":
            print(commands.run_scan(cfg))
        elif args.cmd == "reindex":
            for line in commands.run_reindex(cfg):
                print(line)
        elif args.cmd == "status":
            print(commands.format_status(commands.collect_status(cfg)))
        elif args.cmd == "map":
            written = atlas.generate_map(
                cfg,
                output=args.output,
                clusters=args.clusters,
                highlight=args.highlight,
                focus=args.focus,
                date=args.date,
                open_after=args.open,
                label_mode="recent" if args.recent else "clusters",
            )
            print(written)
    except EchoControlError as e:
        _fail(e, p)
    except KeyboardInterrupt:
        sys.exit(130)

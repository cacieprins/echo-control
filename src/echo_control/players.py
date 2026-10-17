from __future__ import annotations

import itertools
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any, Self

from .config import DATA_DIR, PlayerConfig, socket_path
from .errors import EchoControlError

BACKENDS = ("mpv", "spotatui", "command", "stdout")
CONTINUOUS_AHEAD = 3
M3U_KEEP = 20
# mpv only opens a window for a video track, and cover art counts as one; without this the
# window closes on every track that has no art.
MPV_WINDOW = ("--force-window=immediate",)


class PlayerError(EchoControlError):
    pass


def m3u(paths: list[str], title: str | None = None) -> str:
    lines = ["#EXTM3U"]
    if title:
        lines.append(f"#PLAYLIST:{title}")
    lines += paths
    return "\n".join(lines) + "\n"


def write_m3u(paths: list[str], title: str) -> Path:
    out = DATA_DIR / "playlists"
    out.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="vibe-", suffix=".m3u", dir=out)
    with os.fdopen(fd, "w") as f:
        f.write(m3u(paths, title))
    for old in sorted(out.glob("vibe-*.m3u"), key=lambda p: p.stat().st_mtime)[:-M3U_KEEP]:
        old.unlink(missing_ok=True)
    return Path(name)


def _spawn(argv: list[str], detach: bool) -> None:
    if detach:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        os.execvp(argv[0], argv)


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"\s+", " ", s)[:limit] or "vibe"


def _mpv_argv(
    paths: list[str], title: str, detach: bool, ipc_sock: Path | None = None
) -> list[str]:
    if not shutil.which("mpv"):
        raise PlayerError("mpv is not installed (pacman -S mpv), or pick another --player")
    argv = [
        "mpv",
        f"--playlist={write_m3u(paths, title)}",
        f"--title=vibe: {title}",
        *MPV_WINDOW,
    ]
    if ipc_sock is not None:
        argv.append(f"--input-ipc-server={ipc_sock}")
    if detach:
        argv.append("--player-operation-mode=pseudo-gui")
    return argv


def play_mpv(paths: list[str], title: str, detach: bool) -> None:
    _spawn(_mpv_argv(paths, title, detach), detach)


class _MpvIpc:
    """Just enough of mpv's JSON IPC to read properties and append files."""

    def __init__(self, path: Path, proc: subprocess.Popen[Any], timeout: float = 15.0):
        self.sock: socket.socket
        self.buf: bytes = b""
        self.next_id: int = 0
        deadline = time.monotonic() + timeout
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(str(path))
                self.sock = sock
                break
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise PlayerError("mpv exited before its IPC socket came up") from None
                time.sleep(0.1)

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def command(self, *args: Any) -> Any:
        self.next_id += 1
        rid = self.next_id
        self.sock.sendall(json.dumps({"command": list(args), "request_id": rid}).encode() + b"\n")
        while True:
            while b"\n" not in self.buf:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError("mpv closed its IPC socket")
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            msg = json.loads(line)
            if msg.get("request_id") == rid:
                if msg.get("error") != "success":
                    raise PlayerError(f"mpv {args[0]}: {msg.get('error')}")
                return msg.get("data")


def _interrupt(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def play_mpv_continuous(
    tracks: Iterator[str], title: str, detach: bool, ahead: int = CONTINUOUS_AHEAD
) -> None:
    """Play tracks as an endless mpv playlist, keeping `ahead` queued past the current one
    and pulling more from the iterator as they play. Returns once mpv quits."""
    if not shutil.which("mpv"):
        raise PlayerError("mpv is not installed (pacman -S mpv), or pick another --player")
    first = list(itertools.islice(tracks, ahead + 1))
    if not first:
        raise PlayerError("no results")
    if detach:
        if os.fork():
            return
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        if devnull > 2:
            os.close(devnull)
    sock = socket_path().with_name(f"vibe-mpv-{os.getpid()}.sock")
    proc = subprocess.Popen(_mpv_argv(first, title, detach, ipc_sock=sock))
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        with _MpvIpc(sock, proc) as ipc:
            exhausted = False
            while proc.poll() is None and not exhausted:
                pos = ipc.command("get_property", "playlist-pos") or 0
                count = ipc.command("get_property", "playlist-count") or 0
                while count - max(pos, 0) - 1 < ahead:
                    path = next(tracks, None)
                    if path is None:
                        exhausted = True
                        break
                    ipc.command("loadfile", path, "append")
                    count += 1
                time.sleep(1.0)
        proc.wait()
    except (ConnectionError, BrokenPipeError):
        proc.wait()
    except KeyboardInterrupt:
        # Launchers like uv forward the same signal again; don't die mid-cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        proc.terminate()
        proc.wait()
    finally:
        sock.unlink(missing_ok=True)
        if detach:
            os._exit(0)


def play_spotatui(paths: list[str], title: str, cfg: PlayerConfig, detach: bool) -> None:
    """spotatui can't take a playlist on its CLI; its Local Files source shows every folder
    of audio as a playlist, so build a folder of ordered symlinks for it to open."""
    root = cfg.spotatui_dir.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{time.strftime('%Y-%m-%d %H%M%S')} {_slug(title)}"
    folder.mkdir()
    width = len(str(len(paths)))
    for i, p in enumerate(paths, 1):
        src = Path(p)
        (folder / f"{i:0{width}d} - {src.name}").symlink_to(src)

    playlists = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    for old in playlists[: max(len(playlists) - cfg.spotatui_keep, 0)]:
        shutil.rmtree(old, ignore_errors=True)

    print(f"playlist folder: {folder}", file=sys.stderr)
    print(
        f"open it in spotatui: press d, choose Local Files, then pick the folder (spotatui's local_music_path must include {root})",
        file=sys.stderr,
    )
    if not detach and sys.stdout.isatty() and shutil.which("spotatui"):
        os.execvp("spotatui", ["spotatui"])


def play_command(paths: list[str], title: str, template: str, detach: bool) -> None:
    cmd = template.replace("{paths}", " ".join(shlex.quote(p) for p in paths))
    if "{m3u}" in cmd:
        cmd = cmd.replace("{m3u}", shlex.quote(str(write_m3u(paths, title))))
    _spawn(["sh", "-c", cmd], detach)


def play(
    backend: str, paths: list[str], title: str, cfg: PlayerConfig, detach: bool = False
) -> None:
    if not paths:
        raise PlayerError("no results")
    if backend == "stdout":
        sys.stdout.write(m3u(paths, title))
    elif backend == "mpv":
        play_mpv(paths, title, detach)
    elif backend == "spotatui":
        play_spotatui(paths, title, cfg, detach)
    elif backend == "command":
        play_command(paths, title, cfg.command, detach)
    else:
        raise PlayerError(f"unknown player backend {backend!r}; choose from {', '.join(BACKENDS)}")

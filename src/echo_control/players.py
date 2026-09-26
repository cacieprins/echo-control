from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import DATA_DIR, PlayerConfig

BACKENDS = ("mpv", "spotatui", "command", "stdout")


class PlayerError(RuntimeError):
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
    for old in sorted(out.glob("vibe-*.m3u"), key=lambda p: p.stat().st_mtime)[:-20]:
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


def play_mpv(paths: list[str], title: str, detach: bool) -> None:
    if not shutil.which("mpv"):
        raise PlayerError("mpv is not installed (pacman -S mpv), or pick another --player")
    argv = ["mpv", f"--playlist={write_m3u(paths, title)}", f"--title=vibe: {title}"]
    if detach:
        argv.append("--player-operation-mode=pseudo-gui")
    _spawn(argv, detach)


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
        "open it in spotatui: press d, choose Local Files, then pick the folder "
        f"(spotatui's local_music_path must include {root})",
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

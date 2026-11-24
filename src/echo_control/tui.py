# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Textual TUI for live `vibe` playback and trajectory monitoring."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast, final, override

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, ProgressBar, Static

from .errors import EchoControlError
from .wander import Transition

_log = logging.getLogger(__name__)
ORBIT_STEP = 0.25
CONTINUOUS_AHEAD = 3
_IPC_ERRORS = (OSError, RuntimeError, ValueError, EchoControlError)


class IpcClient(Protocol):
    def command(self, *args: Any) -> Any: ...


@dataclass
class QueueEntry:
    path: str
    score: float
    intro_sim: float
    center: float = 0.0
    direction: float = 0.0
    drift: float = 0.0

    @classmethod
    def from_transition(cls, t: Transition) -> QueueEntry:
        return cls(
            path=t.path,
            score=t.score,
            intro_sim=t.intro_sim,
            center=t.center,
            direction=t.direction,
            drift=t.drift,
        )


@dataclass(frozen=True)
class TrackMeta:
    title: str
    artist: str = ""
    album: str = ""

    @property
    def display(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.artist


@lru_cache(maxsize=512)
def read_track_meta(path: str) -> TrackMeta:
    """Read title/artist/album tags via ffprobe, falling back to filename stem."""
    stem = Path(path).stem
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=artist,title,album",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode != 0 or not res.stdout:
            return TrackMeta(title=stem)
        data = cast("dict[str, Any]", json.loads(res.stdout))
        fmt = cast("dict[str, Any]", data.get("format") or {})
        raw_tags = cast("dict[str, Any]", fmt.get("tags") or {})
        tags = {str(k).lower(): str(v).strip() for k, v in raw_tags.items() if v}
        return TrackMeta(
            title=tags.get("title") or stem,
            artist=tags.get("artist", ""),
            album=tags.get("album", ""),
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return TrackMeta(title=stem)


@dataclass
class PlaybackState:
    pos: int = 0
    time_pos: float = 0.0
    duration: float = 0.0
    paused: bool = False
    volume: float = 100.0
    media_title: str = ""
    has_video: bool = False


WalkFactory = Callable[[str, float | None, str | None], Iterator[Transition]]


@dataclass
class VibeSession:
    entries: list[QueueEntry] = field(default_factory=list)
    query: str = ""
    continuous: bool = False
    orbit_strength: float | None = None
    orbit_center_path: str | None = None
    walk_factory: WalkFactory | None = None
    stream: Iterator[Transition] | None = None
    exhausted: bool = False

    def pull_ahead(self, current_pos: int, ahead: int = CONTINUOUS_AHEAD) -> list[QueueEntry]:
        if not self.continuous or self.stream is None or self.exhausted:
            return []
        added: list[QueueEntry] = []
        pos = max(current_pos, 0)
        while len(self.entries) - pos - 1 < ahead:
            nxt = next(self.stream, None)
            if nxt is None:
                self.exhausted = True
                break
            entry = QueueEntry.from_transition(nxt)
            self.entries.append(entry)
            added.append(entry)
        return added

    def _reset_upcoming(
        self,
        current_pos: int,
        from_path: str,
        strength: float | None,
        center_path: str | None,
        ahead: int,
    ) -> list[QueueEntry]:
        if not self.continuous or self.walk_factory is None:
            return []
        pos = max(0, min(current_pos, len(self.entries) - 1))
        del self.entries[pos + 1 :]
        self.exhausted = False
        stream = self.walk_factory(from_path, strength, center_path)
        first = next(stream, None)
        if first is None:
            self.exhausted = True
            self.stream = stream
            return []
        if first.path != from_path:
            # Put back if factory didn't yield from_path first
            def _chained() -> Iterator[Transition]:
                yield first
                yield from stream

            self.stream = _chained()
        else:
            self.stream = stream
        return self.pull_ahead(pos, ahead=ahead)

    def recenter_on(self, current_pos: int, ahead: int = CONTINUOUS_AHEAD) -> list[QueueEntry]:
        if not self.entries or not self.continuous:
            return []
        pos = max(0, min(current_pos, len(self.entries) - 1))
        target = self.entries[pos].path
        self.orbit_center_path = target
        if self.orbit_strength is None or self.orbit_strength <= 0.0:
            self.orbit_strength = 1.0
        return self._reset_upcoming(pos, target, self.orbit_strength, target, ahead)

    def adjust_orbit(
        self, current_pos: int, delta: float, ahead: int = CONTINUOUS_AHEAD
    ) -> list[QueueEntry]:
        if not self.entries or not self.continuous:
            return []
        pos = max(0, min(current_pos, len(self.entries) - 1))
        from_path = self.entries[pos].path
        base = self.orbit_strength if self.orbit_strength is not None else 0.0
        self.orbit_strength = round(max(0.0, base + delta), 4)
        center = self.orbit_center_path or from_path
        self.orbit_center_path = center
        return self._reset_upcoming(pos, from_path, self.orbit_strength, center, ahead)


class MpvController:
    def __init__(self, ipc: IpcClient) -> None:
        self.ipc: IpcClient = ipc
        self._last_art_path: str | None = None

    def _get(self, prop: str, default: Any = None) -> Any:
        try:
            val = self.ipc.command("get_property", prop)
            return default if val is None else val
        except _IPC_ERRORS:
            return default

    def poll_state(self) -> PlaybackState:
        vid = self._get("vid", False)
        return PlaybackState(
            pos=int(self._get("playlist-pos", 0) or 0),
            time_pos=float(self._get("time-pos", 0.0) or 0.0),
            duration=float(self._get("duration", 0.0) or 0.0),
            paused=bool(self._get("pause", False)),
            volume=float(self._get("volume", 100.0) or 100.0),
            media_title=str(self._get("media-title", "") or ""),
            has_video=bool(vid),
        )

    def toggle_pause(self) -> None:
        self.ipc.command("cycle", "pause")

    def next_track(self) -> None:
        self.ipc.command("playlist-next", "weak")

    def prev_track(self) -> None:
        self.ipc.command("playlist-prev", "weak")

    def seek(self, seconds: float) -> None:
        self.ipc.command("seek", seconds, "relative")

    def change_volume(self, delta: float) -> None:
        self.ipc.command("add", "volume", delta)

    def append_track(self, path: str) -> None:
        self.ipc.command("loadfile", path, "append")

    def replace_upcoming(self, current_pos: int, new_paths: list[str]) -> None:
        count = int(self._get("playlist-count", 0) or 0)
        for idx in range(count - 1, max(current_pos, 0), -1):
            self.ipc.command("playlist-remove", idx)
        for path in new_paths:
            self.ipc.command("loadfile", path, "append")

    def ensure_cover_art(
        self,
        path: str,
        has_video: bool,
        art_provider: Callable[[str], Path | None] | None,
    ) -> None:
        if has_video or art_provider is None or self._last_art_path == path:
            return
        self._last_art_path = path
        art_path = art_provider(path)
        if art_path is not None:
            self.ipc.command("video-add", str(art_path), "select")


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


@final
class VibeApp(App[None]):
    TITLE: str | None = "vibe"
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False
    CSS: ClassVar[str] = """
    Screen {
        background: transparent;
    }
    Footer {
        background: transparent;
    }
    #now-playing {
        height: auto;
        padding: 0 1;
        border: solid $secondary;
        border-title-color: $secondary;
        background: transparent;
        margin: 0 1;
    }
    #np-title {
        text-style: bold;
        color: $accent;
    }
    #np-meta {
        color: $text-muted;
    }
    #np-progress {
        width: 1fr;
        margin: 1 0 0 0;
    }
    #np-progress > Bar > .bar--bar {
        color: $accent;
    }
    #np-status {
        color: $secondary;
    }
    #queue {
        height: 1fr;
        border: solid $primary;
        border-title-color: $primary;
        background: transparent;
        margin: 0 1;
    }
    #queue > .datatable--header {
        background: transparent;
        color: $secondary;
        text-style: bold;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("space", "toggle_pause", "Pause/Play"),
        Binding("n", "next_track", "Next"),
        Binding("p", "prev_track", "Prev"),
        Binding("right", "seek_forward", "+5s"),
        Binding("left", "seek_backward", "-5s"),
        Binding("plus,equals", "volume_up", "Vol +"),
        Binding("minus", "volume_down", "Vol -"),
        Binding("o", "recenter_orbit", "Orbit here"),
        Binding("left_square_bracket", "orbit_weaker", "Orbit -"),
        Binding("right_square_bracket", "orbit_stronger", "Orbit +"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        session: VibeSession,
        controller: MpvController,
        is_alive: Callable[[], bool],
        art_provider: Callable[[str], Path | None] | None = None,
        meta_provider: Callable[[str], TrackMeta] = read_track_meta,
        ahead: int = CONTINUOUS_AHEAD,
        on_commit: Callable[[str, str | None], None] | None = None,
        commit_seconds: float = 30.0,
        commit_fraction: float = 0.40,
    ) -> None:
        super().__init__()
        self.theme = "ansi-dark"
        self.session: VibeSession = session
        self.controller: MpvController = controller
        self.is_alive: Callable[[], bool] = is_alive
        self.art_provider: Callable[[str], Path | None] | None = art_provider
        self.meta_provider: Callable[[str], TrackMeta] = meta_provider
        self.ahead: int = ahead
        self.on_commit: Callable[[str, str | None], None] | None = on_commit
        self.commit_seconds: float = commit_seconds
        self.commit_fraction: float = commit_fraction
        self.state: PlaybackState = PlaybackState()
        self._committed_indices: set[int] = set()
        self._last_committed_path: str | None = None

    @override
    def compose(self) -> ComposeResult:
        with Vertical(id="now-playing"):
            yield Static("Loading…", id="np-title")
            yield Static("", id="np-meta")
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="np-progress")
            yield Static("00:00 / 00:00", id="np-status")
        yield DataTable(id="queue")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#now-playing", Vertical).border_title = "Now Playing"
        table = cast("DataTable[str]", self.query_one("#queue", DataTable))
        table.border_title = "Trajectory"
        table.cursor_type = "row"
        table.add_columns("", "Track", "Score", "Intro", "Orbit")
        self._refresh_from_mpv()
        self.set_interval(0.25, self._refresh_from_mpv)

    def _refresh_from_mpv(self) -> None:
        if not self.is_alive():
            self.exit()
            return
        try:
            self.state = self.controller.poll_state()
        except _IPC_ERRORS:
            if not self.is_alive():
                self.exit()
            return

        pos = max(0, self.state.pos)
        if self.session.continuous:
            added = self.session.pull_ahead(pos, ahead=self.ahead)
            for entry in added:
                try:
                    self.controller.append_track(entry.path)
                except _IPC_ERRORS:
                    break

        if 0 <= pos < len(self.session.entries) and (
            self.state.duration > 0 or bool(self.state.media_title)
        ):
            current = self.session.entries[pos]
            try:
                self.controller.ensure_cover_art(
                    current.path, self.state.has_video, self.art_provider
                )
            except _IPC_ERRORS as err:
                _log.debug("cover art attach failed for %s: %s", current.path, err)

            if self.on_commit is not None and pos not in self._committed_indices:
                dur = self.state.duration
                played_enough = self.state.time_pos >= self.commit_seconds or (
                    dur > 0.0 and (self.state.time_pos / dur) >= self.commit_fraction
                )
                if played_enough:
                    self._committed_indices.add(pos)
                    prev_committed = self._last_committed_path
                    self._last_committed_path = current.path
                    self.on_commit(current.path, prev_committed)

        self._render_view()

    def _render_view(self) -> None:
        pos = max(0, self.state.pos)
        entry = self.session.entries[pos] if pos < len(self.session.entries) else None
        if entry is not None:
            meta = self.meta_provider(entry.path)
            has_tags = bool(meta.artist or meta.title != Path(entry.path).stem)
            title = meta.display if has_tags else (self.state.media_title or meta.display)
            location = meta.album or str(Path(entry.path).parent)
        else:
            title = self.state.media_title or "—"
            location = ""

        badges: list[str] = []
        if self.session.query:
            badges.append(f"vibe: {self.session.query}")
        if self.session.orbit_strength is not None:
            badges.append(f"orbit: {self.session.orbit_strength:.2f}")
        if self.session.orbit_center_path:
            center_name = self.meta_provider(self.session.orbit_center_path).display
            badges.append(f"center: {center_name}")
        if self.session.continuous:
            badges.append("continuous")

        self.query_one("#np-title", Static).update(title)
        meta_line = "  •  ".join([b for b in [location, *badges] if b])
        self.query_one("#np-meta", Static).update(meta_line)

        bar = self.query_one("#np-progress", ProgressBar)
        dur = max(self.state.duration, 0.0)
        bar.update(total=max(dur, 1.0), progress=min(self.state.time_pos, dur) if dur > 0 else 0.0)

        pause_icon = "⏸ Paused" if self.state.paused else "▶ Playing"
        status_text = (
            f"{pause_icon}   {_fmt_time(self.state.time_pos)} / {_fmt_time(dur)}"
            f"   vol {round(self.state.volume)}%"
        )
        self.query_one("#np-status", Static).update(status_text)

        table = cast("DataTable[str]", self.query_one("#queue", DataTable))
        table.clear()
        for idx, item in enumerate(self.session.entries):
            marker = "▶" if idx == pos else ("✓" if idx < pos else "·")
            orbit_col = f"{item.center:.2f}" if self.session.orbit_strength else "—"
            table.add_row(
                marker,
                self.meta_provider(item.path).display,
                f"{item.score:.2f}",
                f"{item.intro_sim:.2f}",
                orbit_col,
                key=str(idx),
            )
        if self.session.entries and 0 <= pos < len(self.session.entries):
            table.move_cursor(row=pos)

    def action_toggle_pause(self) -> None:
        self.controller.toggle_pause()

    def action_next_track(self) -> None:
        self.controller.next_track()

    def action_prev_track(self) -> None:
        self.controller.prev_track()

    def action_seek_forward(self) -> None:
        self.controller.seek(5.0)

    def action_seek_backward(self) -> None:
        self.controller.seek(-5.0)

    def action_volume_up(self) -> None:
        self.controller.change_volume(5.0)

    def action_volume_down(self) -> None:
        self.controller.change_volume(-5.0)

    def action_recenter_orbit(self) -> None:
        pos = max(0, self.state.pos)
        added = self.session.recenter_on(pos, ahead=self.ahead)
        if added:
            self.controller.replace_upcoming(pos, [e.path for e in added])
            self._render_view()

    def action_orbit_weaker(self) -> None:
        pos = max(0, self.state.pos)
        added = self.session.adjust_orbit(pos, -ORBIT_STEP, ahead=self.ahead)
        if added:
            self.controller.replace_upcoming(pos, [e.path for e in added])
            self._render_view()

    def action_orbit_stronger(self) -> None:
        pos = max(0, self.state.pos)
        added = self.session.adjust_orbit(pos, ORBIT_STEP, ahead=self.ahead)
        if added:
            self.controller.replace_upcoming(pos, [e.path for e in added])
            self._render_view()

# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Textual TUI for live `vibe` playback and trajectory monitoring."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast, final, override

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, ProgressBar, Static

from .errors import EchoControlError
from .playlog import DailyPlayLog
from .surfing import ChannelSurfer, SurfResult
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
    heat: float = 0.0
    csls: float = 0.0

    @classmethod
    def from_transition(cls, t: Transition) -> QueueEntry:
        return cls(
            path=t.path,
            score=t.score,
            intro_sim=t.intro_sim,
            center=t.center,
            direction=t.direction,
            drift=t.drift,
            heat=t.heat,
            csls=t.csls,
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

    def hard_cut_to(self, path: str, ahead: int = CONTINUOUS_AHEAD) -> list[QueueEntry]:
        """Reset playlist to a single hard-cut track at pos 0 and regenerate lookahead."""
        self.entries = [QueueEntry(path=path, score=0.0, intro_sim=0.0)]
        self.exhausted = False
        if self.orbit_strength is not None and self.orbit_strength > 0.0:
            self.orbit_center_path = path
        if not self.continuous or self.walk_factory is None:
            return []
        return self._reset_upcoming(0, path, self.orbit_strength, self.orbit_center_path, ahead)


class MpvController:
    def __init__(self, ipc: IpcClient) -> None:
        self.ipc: IpcClient = ipc
        self._last_art_path: str | None = None
        self._overlay_image_path: str | None = None

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

    def _send(self, *args: Any) -> bool:
        try:
            self.ipc.command(*args)
            return True
        except _IPC_ERRORS as err:
            _log.debug("mpv command %s failed: %s", args[0] if args else "", err)
            return False

    def toggle_pause(self) -> None:
        self._send("cycle", "pause")

    def next_track(self) -> None:
        self._send("playlist-next", "weak")

    def prev_track(self) -> None:
        self._send("playlist-prev", "weak")

    def seek(self, seconds: float) -> None:
        self._send("seek", seconds, "relative")

    def change_volume(self, delta: float) -> None:
        self._send("add", "volume", delta)

    def append_track(self, path: str) -> None:
        self._send("loadfile", path, "append")

    def hard_cut(self, path: str) -> None:
        """Instant 0 ms tuner cut: replace mpv playlist with `path`."""
        self._last_art_path = None
        self._send("loadfile", path, "replace")

    def replace_upcoming(self, current_pos: int, new_paths: list[str]) -> None:
        count = int(self._get("playlist-count", 0) or 0)
        for idx in range(count - 1, max(current_pos, 0), -1):
            self._send("playlist-remove", idx)
        for path in new_paths:
            self._send("loadfile", path, "append")

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
            self._send("video-add", str(art_path), "select")

    def _track_list(self) -> list[dict[str, Any]]:
        raw = self._get("track-list", [])
        if not isinstance(raw, list):
            return []
        return [cast("dict[str, Any]", item) for item in cast("list[Any]", raw) if isinstance(item, dict)]

    def _remove_external_video(self, filename: str) -> None:
        for tr in self._track_list():
            if (
                tr.get("type") == "video"
                and tr.get("external") is True
                and tr.get("external-filename") == filename
            ):
                tid: Any = tr.get("id")
                if tid is not None:
                    self._send("video-remove", tid)

    def show_overlay_image(self, image_path: Path) -> None:
        """Display `image_path` as mpv's active video track, swapping seamlessly without flashing cover art."""
        prev = self._overlay_image_path
        target = str(image_path)
        self._overlay_image_path = target
        self._send("video-add", target, "select")
        if prev is not None and prev != target:
            self._remove_external_video(prev)

    def clear_overlay_image(
        self,
        path: str,
        art_provider: Callable[[str], Path | None] | None,
    ) -> None:
        """Remove any active overlay image and restore embedded or latent cover art."""
        prev = self._overlay_image_path
        self._overlay_image_path = None
        self._last_art_path = None
        if prev is not None:
            self._remove_external_video(prev)
        for tr in self._track_list():
            if tr.get("type") == "video" and not tr.get("external"):
                tid: Any = tr.get("id")
                if tid is not None:
                    self._send("set_property", "vid", tid)
                    return
        self.ensure_cover_art(path, False, art_provider)


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
    #up-next {
        height: auto;
        padding: 0 1;
        border: solid $primary;
        border-title-color: $primary;
        background: transparent;
        margin: 1 1 0 1;
    }
    #next-title {
        text-style: bold;
        color: $primary;
    }
    #next-meta {
        color: $text-muted;
    }
    #log-strip {
        height: auto;
        padding: 0 2;
        margin: 1 0 0 0;
        color: $text-muted;
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
        Binding("b", "bail", "Bail"),
        Binding("v", "warmer", "Midpoint"),
        Binding("m", "toggle_map", "Map"),
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
        map_provider: Callable[[str, str | None], Path | None] | None = None,
        meta_provider: Callable[[str], TrackMeta] = read_track_meta,
        ahead: int = CONTINUOUS_AHEAD,
        on_commit: Callable[[str, str | None], None] | None = None,
        play_log: DailyPlayLog | None = None,
        surfer: ChannelSurfer | None = None,
        commit_seconds: float = 30.0,
        commit_fraction: float = 0.40,
    ) -> None:
        super().__init__()
        self.theme = "ansi-dark"
        self.session: VibeSession = session
        self.controller: MpvController = controller
        self.is_alive: Callable[[], bool] = is_alive
        self.art_provider: Callable[[str], Path | None] | None = art_provider
        self.map_provider: Callable[[str, str | None], Path | None] | None = map_provider
        self.meta_provider: Callable[[str], TrackMeta] = meta_provider
        self.ahead: int = ahead
        self.on_commit: Callable[[str, str | None], None] | None = on_commit
        self.play_log: DailyPlayLog | None = play_log
        self.surfer: ChannelSurfer | None = surfer
        self.commit_seconds: float = commit_seconds
        self.commit_fraction: float = commit_fraction
        self.state: PlaybackState = PlaybackState()
        self._committed_indices: set[int] = set()
        self._last_committed_path: str | None = None
        self._logged_today: int = play_log.count_today() if play_log is not None else 0
        self._map_mode: bool = False
        self._map_key: tuple[str, str | None, int] | None = None
        self._map_rendering: bool = False
        self._last_map_image: Path | None = None
        self._last_map_track: str | None = None

    @override
    def compose(self) -> ComposeResult:
        with Vertical(id="now-playing"):
            yield Static("Loading…", id="np-title")
            yield Static("", id="np-meta")
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="np-progress")
            yield Static("00:00 / 00:00", id="np-status")
        with Vertical(id="up-next"):
            yield Static("—", id="next-title")
            yield Static("", id="next-meta")
        yield Static("", id="log-strip")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#now-playing", Vertical).border_title = "Now Playing"
        self.query_one("#up-next", Vertical).border_title = "Up Next"
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
            if self._map_mode and self.map_provider is not None:
                if current.path != self._last_map_track and self._last_map_image is not None:
                    # Immediately re-attach the last map frame so mpv doesn't flash embedded cover art
                    self._last_map_track = current.path
                    try:
                        self.controller.show_overlay_image(self._last_map_image)
                    except _IPC_ERRORS as err:
                        _log.debug("immediate map re-attach failed: %s", err)
                next_path = (
                    self.session.entries[pos + 1].path
                    if (pos + 1) < len(self.session.entries)
                    else None
                )
                key = (current.path, next_path, self._logged_today)
                if key != self._map_key and not self._map_rendering:
                    self._request_map_render(current.path, next_path, key)
            else:
                try:
                    self.controller.ensure_cover_art(
                        current.path, self.state.has_video, self.art_provider
                    )
                except _IPC_ERRORS as err:
                    _log.debug("cover art attach failed for %s: %s", current.path, err)

            if (self.on_commit is not None or self.play_log is not None) and (
                pos not in self._committed_indices
            ):
                dur = self.state.duration
                played_enough = self.state.time_pos >= self.commit_seconds or (
                    dur > 0.0 and (self.state.time_pos / dur) >= self.commit_fraction
                )
                if played_enough:
                    self._committed_indices.add(pos)
                    prev_committed = self._last_committed_path
                    self._last_committed_path = current.path
                    if self.on_commit is not None:
                        self.on_commit(current.path, prev_committed)
                    if self.play_log is not None:
                        meta = self.meta_provider(current.path)
                        try:
                            self.play_log.append(
                                current.path,
                                meta.display,
                                score=current.score,
                                intro_sim=current.intro_sim,
                                heat=current.heat,
                            )
                            self._logged_today += 1
                        except OSError as err:
                            _log.debug("daily play log write failed: %s", err)

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
        if self.surfer is not None:
            surf_badge = self.surfer.status_badge()
            if surf_badge:
                badges.append(surf_badge)
        if self._map_mode:
            badges.append("map…" if self._map_rendering else "map")

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

        next_entry = (
            self.session.entries[pos + 1] if (pos + 1) < len(self.session.entries) else None
        )
        if next_entry is not None:
            next_meta = self.meta_provider(next_entry.path)
            next_loc = next_meta.album or str(Path(next_entry.path).parent)
            metrics = [
                f"score {next_entry.score:.2f}",
                f"intro {next_entry.intro_sim:.2f}",
                f"heat {next_entry.heat:.2f}",
            ]
            if self.session.orbit_strength:
                metrics.append(f"orbit {next_entry.center:.2f}")
            self.query_one("#next-title", Static).update(next_meta.display)
            self.query_one("#next-meta", Static).update(
                "  •  ".join([p for p in [next_loc, *metrics] if p])
            )
        else:
            self.query_one("#next-title", Static).update("End of queue")
            self.query_one("#next-meta", Static).update("")

        if self.play_log is not None:
            log_path = self.play_log.path_for()
            self.query_one("#log-strip", Static).update(
                f"log: {log_path}  ({self._logged_today} today)"
            )
        else:
            self.query_one("#log-strip", Static).update("")

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

    def _apply_surf(self, result: SurfResult | None) -> None:
        if result is None:
            return
        self.controller.hard_cut(result.path)
        added = self.session.hard_cut_to(result.path, ahead=self.ahead)
        for entry in added:
            try:
                self.controller.append_track(entry.path)
            except _IPC_ERRORS:
                break
        self._committed_indices.clear()
        self._last_committed_path = None
        self.state.pos = 0
        self.state.time_pos = 0.0
        self._render_view()

    def action_bail(self) -> None:
        if self.surfer is None or not self.session.entries:
            return
        pos = max(0, min(self.state.pos, len(self.session.entries) - 1))
        self._apply_surf(self.surfer.bail(self.session.entries[pos].path))

    def action_warmer(self) -> None:
        if self.surfer is None or not self.session.entries:
            return
        pos = max(0, min(self.state.pos, len(self.session.entries) - 1))
        self._apply_surf(self.surfer.warmer(self.session.entries[pos].path))

    def _request_map_render(
        self, track_path: str, next_path: str | None, key: tuple[str, str | None, int]
    ) -> None:
        provider = self.map_provider
        if provider is None:
            return
        self._map_rendering = True

        def _worker() -> None:
            img: Path | None = None
            try:
                img = provider(track_path, next_path)
            except _IPC_ERRORS as err:
                _log.debug("live atlas render failed: %s", err)
            try:
                self.call_from_thread(self._apply_map_image, img, key)
            except RuntimeError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_map_image(
        self, image_path: Path | None, key: tuple[str, str | None, int]
    ) -> None:
        self._map_rendering = False
        if not self._map_mode or image_path is None:
            self._render_view()
            return
        self._map_key = key
        self._last_map_image = image_path
        self._last_map_track = key[0]
        try:
            self.controller.show_overlay_image(image_path)
        except _IPC_ERRORS as err:
            _log.debug("map overlay attach failed: %s", err)
        self._render_view()

    def action_toggle_map(self) -> None:
        if self.map_provider is None or not self.session.entries:
            return
        pos = max(0, min(self.state.pos, len(self.session.entries) - 1))
        current_path = self.session.entries[pos].path
        next_path = (
            self.session.entries[pos + 1].path
            if (pos + 1) < len(self.session.entries)
            else None
        )
        self._map_mode = not self._map_mode
        if self._map_mode:
            self._map_key = None
            if not self._map_rendering:
                self._request_map_render(
                    current_path, next_path, (current_path, next_path, self._logged_today)
                )
        else:
            self._map_key = None
            try:
                self.controller.clear_overlay_image(current_path, self.art_provider)
            except _IPC_ERRORS as err:
                _log.debug("restore cover art failed: %s", err)
        self._render_view()

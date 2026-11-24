# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast, override

from echo_control import tui
from echo_control.wander import Transition


class FakeIpc:
    def __init__(self) -> None:
        self.props: dict[str, Any] = {
            "playlist-pos": 0,
            "time-pos": 12.5,
            "duration": 180.0,
            "pause": False,
            "volume": 80.0,
            "media-title": "Track A",
            "vid": False,
            "playlist-count": 1,
        }
        self.commands: list[tuple[Any, ...]] = []

    def command(self, *args: Any) -> Any:
        self.commands.append(args)
        if args[0] == "get_property":
            return self.props.get(args[1])
        if args[0] == "set_property":
            self.props[args[1]] = args[2]
            return None
        if args[0] == "playlist-remove":
            self.props["playlist-count"] = max(0, int(self.props["playlist-count"]) - 1)
            return None
        if args[0] == "loadfile" and len(args) >= 3 and args[2] == "append":
            self.props["playlist-count"] = int(self.props["playlist-count"]) + 1
            return None
        return None


def _t(name: str, score: float = 0.85, intro: float = 0.82, center: float = 0.75) -> Transition:
    return Transition(path=f"/music/{name}.flac", score=score, intro_sim=intro, direction=0.5, drift=0.0, center=center)


def test_vibe_session_pull_ahead_and_recenter() -> None:
    calls: list[tuple[str, float | None, str | None]] = []

    def walk_factory(
        seed_path: str, strength: float | None, center_path: str | None
    ) -> Iterator[Transition]:
        calls.append((seed_path, strength, center_path))
        tag = Path(seed_path).stem
        for i in range(1, 10):
            yield _t(f"{tag}-{i}", score=0.9 - i * 0.01)

    initial = [_t("seed"), _t("old-1"), _t("old-2"), _t("old-3")]
    session = tui.VibeSession(
        entries=[tui.QueueEntry.from_transition(x) for x in initial],
        query="night drive",
        continuous=True,
        orbit_strength=1.0,
        orbit_center_path="/music/seed.flac",
        walk_factory=walk_factory,
    )

    # At pos 0 with 3 ahead, nothing new needed
    assert session.pull_ahead(0, ahead=3) == []

    # Recenter on pos 1 ("/music/old-1.flac")
    added = session.recenter_on(1, ahead=3)
    assert len(added) == 3
    assert [e.path for e in session.entries] == [
        "/music/seed.flac",
        "/music/old-1.flac",
        "/music/old-1-1.flac",
        "/music/old-1-2.flac",
        "/music/old-1-3.flac",
    ]
    assert session.orbit_center_path == "/music/old-1.flac"
    assert calls == [("/music/old-1.flac", 1.0, "/music/old-1.flac")]


def test_vibe_session_adjust_orbit_clamps_and_regenerates() -> None:
    calls: list[tuple[str, float | None, str | None]] = []

    def walk_factory(
        seed_path: str, strength: float | None, center_path: str | None
    ) -> Iterator[Transition]:
        calls.append((seed_path, strength, center_path))
        for i in range(1, 6):
            yield _t(f"orb-{strength}-{i}")

    session = tui.VibeSession(
        entries=[tui.QueueEntry.from_transition(_t("seed")), tui.QueueEntry.from_transition(_t("old-1"))],
        query="ambient",
        continuous=True,
        orbit_strength=0.2,
        orbit_center_path="/music/seed.flac",
        walk_factory=walk_factory,
    )

    added = session.adjust_orbit(0, -0.5, ahead=2)
    assert session.orbit_strength == 0.0
    assert len(added) == 2
    assert calls == [("/music/seed.flac", 0.0, "/music/seed.flac")]

    added2 = session.adjust_orbit(0, 0.25, ahead=2)
    assert session.orbit_strength == 0.25
    assert len(added2) == 2
    assert calls[-1] == ("/music/seed.flac", 0.25, "/music/seed.flac")


def test_mpv_controller_poll_replace_and_cover_art(tmp_path: Path) -> None:
    ipc = FakeIpc()
    ipc.props["playlist-count"] = 4
    ctrl = tui.MpvController(ipc)

    state = ctrl.poll_state()
    assert state.pos == 0
    assert state.time_pos == 12.5
    assert state.duration == 180.0
    assert state.paused is False
    assert state.volume == 80.0
    assert state.media_title == "Track A"
    assert state.has_video is False

    # Cover art attached when has_video is False, and cached per track
    art_file = tmp_path / "cover.png"
    art_file.write_bytes(b"png")
    art_calls: list[str] = []

    def art_provider(path: str) -> Path | None:
        art_calls.append(path)
        return art_file

    ctrl.ensure_cover_art("/music/a.flac", state.has_video, art_provider)
    ctrl.ensure_cover_art("/music/a.flac", state.has_video, art_provider)
    assert art_calls == ["/music/a.flac"]
    assert ("video-add", str(art_file), "select") in ipc.commands

    # When track has video/embedded art, art_provider is not called
    ctrl.ensure_cover_art("/music/b.flac", True, art_provider)
    assert art_calls == ["/music/a.flac"]

    # Replace upcoming removes indices 3, 2, 1 and appends new paths
    ipc.commands.clear()
    ctrl.replace_upcoming(0, ["/music/x.flac", "/music/y.flac"])
    assert ipc.commands == [
        ("get_property", "playlist-count"),
        ("playlist-remove", 3),
        ("playlist-remove", 2),
        ("playlist-remove", 1),
        ("loadfile", "/music/x.flac", "append"),
        ("loadfile", "/music/y.flac", "append"),
    ]


def test_textual_vibe_app_keybindings() -> None:
    ipc = FakeIpc()
    ipc.props["playlist-count"] = 3

    def walk_factory(
        _seed_path: str, _strength: float | None, _center_path: str | None
    ) -> Iterator[Transition]:
        for i in range(1, 6):
            yield _t(f"next-{i}")

    session = tui.VibeSession(
        entries=[
            tui.QueueEntry.from_transition(_t("seed")),
            tui.QueueEntry.from_transition(_t("two")),
            tui.QueueEntry.from_transition(_t("three")),
        ],
        query="dreamy",
        continuous=True,
        orbit_strength=1.0,
        orbit_center_path="/music/seed.flac",
        walk_factory=walk_factory,
    )
    ctrl = tui.MpvController(ipc)
    app = tui.VibeApp(session=session, controller=ctrl, is_alive=lambda: True)

    async def _run() -> None:
        from textual.widgets import Header

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.theme == "ansi-dark"
            assert app.ENABLE_COMMAND_PALETTE is False
            assert len(app.query(Header)) == 0
            await pilot.press("space")
            await pilot.press("n")
            await pilot.press("right_square_bracket")
            await pilot.press("o")
            await pilot.press("q")

    asyncio.run(_run())
    assert ("cycle", "pause") in ipc.commands
    assert ("playlist-next", "weak") in ipc.commands
    assert session.orbit_strength == 1.25


def test_tui_activation_and_no_tui_flag(monkeypatch: Any) -> None:
    from echo_control import cli, commands
    from echo_control.config import Config, ModelConfig, PlayerConfig, StrategyName

    cfg = Config(
        model=ModelConfig(strategy=StrategyName.CHUNKS),
        player=PlayerConfig(backend="mpv"),
    )

    # should_use_tui matrix
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=True), cfg, is_tty=True
    ) is True
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=True, no_tui=True), cfg, is_tty=True
    ) is False
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=False), cfg, is_tty=True
    ) is False
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=True), cfg, is_tty=False
    ) is False
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=True, detach=True), cfg, is_tty=True
    ) is False
    assert commands.should_use_tui(
        commands.VibeRequest(query=["rain"], wander=True, player="spotatui"), cfg, is_tty=True
    ) is False

    # CLI --no-tui flag passes no_tui=True into run_vibe
    seen: list[commands.VibeRequest] = []

    def fake_run_vibe(req: commands.VibeRequest, _cfg: Config, *, is_tty: bool = False) -> None:
        _ = is_tty
        seen.append(req)

    monkeypatch.setattr(commands, "run_vibe", fake_run_vibe)
    cli.vibe(["--wander", "--no-tui", "rain"])
    assert len(seen) == 1
    assert seen[0].no_tui is True
    assert seen[0].wander is True


def test_textual_vibe_app_ctrl_c_quits() -> None:
    ipc = FakeIpc()
    ctrl = tui.MpvController(ipc)
    app = tui.VibeApp(
        session=tui.VibeSession(entries=[tui.QueueEntry.from_transition(_t("seed"))]),
        controller=ctrl,
        is_alive=lambda: True,
    )

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+c")
            assert app._exit is True  # pyright: ignore[reportPrivateUsage]

    asyncio.run(_run())


def test_read_track_meta_and_app_display(monkeypatch: Any) -> None:
    import json
    import subprocess

    def fake_run(cmd: list[str], **_kwargs: Any) -> Any:
        path = cmd[-1]
        if "no-tags" in path:
            payload: dict[str, Any] = {"format": {"tags": {}}}
        else:
            payload = {
                "format": {
                    "tags": {
                        "ARTIST": "VNV Nation",
                        "TITLE": "Saviour (Vox)",
                        "ALBUM": "Burning Empires",
                    }
                }
            }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    tui.read_track_meta.cache_clear()
    monkeypatch.setattr(subprocess, "run", fake_run)

    meta = tui.read_track_meta("/music/01-saviour.flac")
    assert meta.artist == "VNV Nation"
    assert meta.title == "Saviour (Vox)"
    assert meta.album == "Burning Empires"
    assert meta.display == "VNV Nation — Saviour (Vox)"

    fallback = tui.read_track_meta("/music/no-tags.flac")
    assert fallback.title == "no-tags"
    assert fallback.display == "no-tags"

    # Verify VibeApp uses meta_provider in Now Playing and Trajectory table
    ipc = FakeIpc()
    ipc.props["media-title"] = ""
    session = tui.VibeSession(
        entries=[
            tui.QueueEntry.from_transition(_t("01-saviour")),
            tui.QueueEntry.from_transition(_t("no-tags")),
        ],
        orbit_strength=1.0,
        orbit_center_path="/music/01-saviour.flac",
    )
    app = tui.VibeApp(
        session=session,
        controller=tui.MpvController(ipc),
        is_alive=lambda: True,
        meta_provider=tui.read_track_meta,
    )

    async def _check_ui() -> None:
        from textual.widgets import DataTable, Static

        async with app.run_test() as pilot:
            await pilot.pause()
            np_title = str(app.query_one("#np-title", Static).render())
            np_meta = str(app.query_one("#np-meta", Static).render())
            assert "VNV Nation — Saviour (Vox)" in np_title
            assert "Burning Empires" in np_meta
            table = cast("DataTable[str]", app.query_one("#queue", DataTable))
            assert table.get_row("0")[1] == "VNV Nation — Saviour (Vox)"
            assert table.get_row("1")[1] == "no-tags"

    asyncio.run(_check_ui())


def test_mpv_controller_handles_property_unavailable() -> None:
    from echo_control.players import PlayerError

    class UnavailableIpc(FakeIpc):
        @override
        def command(self, *args: Any) -> Any:
            if args[0] == "get_property" and args[1] in ("time-pos", "duration", "vid"):
                raise PlayerError("mpv get_property: property unavailable")
            return super().command(*args)

    ctrl = tui.MpvController(UnavailableIpc())
    state = ctrl.poll_state()
    assert state.time_pos == 0.0
    assert state.duration == 0.0
    assert state.has_video is False


def test_vibe_app_listen_time_commit_gate() -> None:
    ipc = FakeIpc()
    ipc.props["playlist-pos"] = 0
    ipc.props["duration"] = 200.0
    ipc.props["time-pos"] = 10.0  # < 30s and < 40%

    committed: list[tuple[str, str | None]] = []
    session = tui.VibeSession(
        entries=[
            tui.QueueEntry.from_transition(_t("track-a")),
            tui.QueueEntry.from_transition(_t("track-b")),
        ]
    )
    app = tui.VibeApp(
        session=session,
        controller=tui.MpvController(ipc),
        is_alive=lambda: True,
        on_commit=lambda path, prev: committed.append((path, prev)),
        commit_seconds=30.0,
        commit_fraction=0.40,
    )

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            # Fast skip at 10s must NOT commit
            assert committed == []

            # Crossing 30s commits track-a once
            ipc.props["time-pos"] = 31.0
            app._refresh_from_mpv()  # pyright: ignore[reportPrivateUsage]
            app._refresh_from_mpv()  # pyright: ignore[reportPrivateUsage]
            assert committed == [("/music/track-a.flac", None)]

            # Advance to short track-b (50s) and cross 40% (22s >= 20s)
            ipc.props["playlist-pos"] = 1
            ipc.props["duration"] = 50.0
            ipc.props["time-pos"] = 22.0
            app._refresh_from_mpv()  # pyright: ignore[reportPrivateUsage]
            assert committed == [
                ("/music/track-a.flac", None),
                ("/music/track-b.flac", "/music/track-a.flac"),
            ]

    asyncio.run(_run())



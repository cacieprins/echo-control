from __future__ import annotations

from pathlib import Path

import pytest

from echo_control import cli, config, players
from echo_control.clap import prompt, signature
from echo_control.config import Config, ModelConfig, PlayerConfig


def test_parse_duration() -> None:
    assert config.parse_duration(42) == 42.0
    assert config.parse_duration(3.5) == 3.5
    assert config.parse_duration("10") == 10.0
    assert config.parse_duration("10s") == 10.0
    assert config.parse_duration("2.5m") == 150.0
    assert config.parse_duration("6h") == 21600.0
    assert config.parse_duration("1d") == 86400.0
    with pytest.raises(ValueError, match="invalid duration"):
        config.parse_duration("5w")


def test_config_load_defaults_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.toml"
    cfg = config.load(missing)
    assert cfg.path is None
    assert cfg.model.strategy == "clusters"
    assert cfg.player.spotatui_dir.expanduser().resolve() in cfg.library.exclude


def test_config_load_custom_toml(tmp_path: Path) -> None:
    lib_dir = tmp_path / "music"
    excl_dir = tmp_path / "music" / "ignore"
    spota_dir = tmp_path / "spota"
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        f"""
[library]
paths = ["{lib_dir}"]
extensions = [".MP3", "Flac"]
rescan_interval = "2h"
exclude = ["{excl_dir}"]

[model]
strategy = "chunks"
window_seconds = 8.0
hop_seconds = 2.0

[player]
backend = "spotatui"
limit = 10
spotatui_dir = "{spota_dir}"

[search]
sustain_seconds = 30.0
"""
    )
    cfg = config.load(cfg_file)
    assert cfg.path == cfg_file
    assert cfg.library.paths == [lib_dir.resolve()]
    assert cfg.library.extensions == ["mp3", "flac"]
    assert cfg.library.rescan_interval == 7200.0
    assert cfg.library.exclude == [excl_dir.resolve(), spota_dir.resolve()]
    assert cfg.model.strategy == "chunks"
    assert cfg.model.window_seconds == 8.0
    assert cfg.model.hop_seconds == 2.0
    assert cfg.player.backend == "spotatui"
    assert cfg.player.limit == 10
    assert cfg.search.sustain_seconds == 30.0


def test_config_load_invalid_strategy_and_hop(tmp_path: Path) -> None:
    bad_strat = tmp_path / "bad_strat.toml"
    bad_strat.write_text('[model]\nstrategy = "magic"\n')
    with pytest.raises(ValueError, match="strategy must be"):
        config.load(bad_strat)

    bad_hop = tmp_path / "bad_hop.toml"
    bad_hop.write_text("[model]\nwindow_seconds = 5.0\nhop_seconds = 10.0\n")
    with pytest.raises(ValueError, match="hop_seconds"):
        config.load(bad_hop)

    zero_hop = tmp_path / "zero_hop.toml"
    zero_hop.write_text("[model]\nhop_seconds = 0.0\n")
    with pytest.raises(ValueError, match="hop_seconds"):
        config.load(zero_hop)


@pytest.mark.parametrize(
    ("argv", "strategy", "expected_msg"),
    [
        (["-n", "0", "rock"], "clusters", "-n/--limit must be at least 1"),
        (["-n", "-5", "rock"], "clusters", "-n/--limit must be at least 1"),
        (["--orbit", "-1", "rock"], "chunks", "--orbit strength can't be negative"),
        (
            ["--continuous", "--json", "rock"],
            "chunks",
            "--continuous plays in mpv; it can't be combined with --json or --print",
        ),
        (
            ["--continuous", "--print", "rock"],
            "chunks",
            "--continuous plays in mpv; it can't be combined with --json or --print",
        ),
        (
            ["--continuous", "--player", "spotatui", "rock"],
            "chunks",
            "--continuous needs the mpv player",
        ),
        (["two", "words"], "clusters", 'put the query in quotes, e.g. vibe "two words"'),
        (["--from", "song.mp3", "rock"], "chunks", "--from only applies to --wander and --orbit"),
        (["--wander", "--shuffle", "rock"], "chunks", "--shuffle would undo the walk's ordering"),
        (
            ["--wander", "rock"],
            "clusters",
            '--wander and --orbit need [model] strategy = "chunks"',
        ),
    ],
)
def test_vibe_invalid_flag_combinations(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    strategy: str,
    expected_msg: str,
) -> None:
    cfg = Config(model=ModelConfig(strategy=strategy))
    monkeypatch.setattr(config, "load", lambda: cfg)
    with pytest.raises(SystemExit) as exc:
        cli.vibe(argv)
    assert exc.value.code == 2
    assert expected_msg in capsys.readouterr().err


def test_vibe_missing_query_and_seed_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "load", Config)
    with pytest.raises(SystemExit) as exc:
        cli.vibe([])
    assert exc.value.code == 2
    assert "usage: vibe" in capsys.readouterr().err


def test_vibe_play_path_bypasses_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(config, "load", Config)
    cli.vibe(["--player", "stdout", "--play-path", "/music/first.flac", "/music/second.flac"])
    out = capsys.readouterr().out
    assert out == "#EXTM3U\n#PLAYLIST:first\n/music/first.flac\n/music/second.flac\n"


def test_clap_signature_varies_by_strategy() -> None:
    m_mean = ModelConfig(strategy="mean", windows=4, window_seconds=8.0)
    assert signature(m_mean) == f"{m_mean.name}|mean|4x8"

    m_chunks = ModelConfig(
        strategy="chunks", window_seconds=10.0, hop_seconds=2.5, silence_db=-45.0, edge_seconds=30.0
    )
    assert signature(m_chunks) == f"{m_chunks.name}|chunks|10|2.5|-45|30"

    m_clusters = ModelConfig(
        strategy="clusters",
        window_seconds=10.0,
        max_chunks=60,
        silence_db=-45.0,
        max_clusters=4,
        min_cluster_share=0.12,
        merge_threshold=0.85,
    )
    assert signature(m_clusters) == f"{m_clusters.name}|clusters|10|60|-45|4|0.12|0.85"


def test_clap_prompt_normalizes_prefix() -> None:
    cfg = ModelConfig(text_prefix="A recording of ")
    assert prompt(cfg, "sad synth") == "A recording of sad synth"
    assert prompt(cfg, "  a recording of   sad synth ") == "A recording of sad synth"
    assert prompt(cfg, "A RECORDING OF jazz") == "A recording of jazz"

    no_prefix = ModelConfig(text_prefix="")
    assert prompt(no_prefix, "  ambient  ") == "ambient"


def test_players_m3u() -> None:
    assert players.m3u(["/a.mp3", "/b.mp3"], "chill") == "#EXTM3U\n#PLAYLIST:chill\n/a.mp3\n/b.mp3\n"
    assert players.m3u(["/a.mp3"]) == "#EXTM3U\n/a.mp3\n"


def test_players_spotatui(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spota_dir = tmp_path / "playlists"
    p_cfg = PlayerConfig(spotatui_dir=spota_dir, spotatui_keep=2)
    (spota_dir / "2020-01-01 000001 old1").mkdir(parents=True)
    (spota_dir / "2020-01-01 000002 old2").mkdir(parents=True)
    players.play("spotatui", ["/music/a.mp3", "/music/b.mp3"], "  teen / angst!!  2026 ", p_cfg, detach=True)
    folders = sorted(p.name for p in spota_dir.iterdir())
    assert len(folders) == 2
    assert folders[0] == "2020-01-01 000002 old2"
    assert folders[1].endswith(" teen angst 2026")
    links = sorted(p.name for p in (spota_dir / folders[1]).iterdir())
    assert links == ["1 - a.mp3", "2 - b.mp3"]
    assert "playlist folder:" in capsys.readouterr().err

    spota_empty = tmp_path / "playlists_empty"
    players.play("spotatui", ["/music/a.mp3"], "!!!", PlayerConfig(spotatui_dir=spota_empty), detach=True)
    assert next(spota_empty.iterdir()).name.endswith(" vibe")


def test_players_write_m3u_prunes_old_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    monkeypatch.setattr(players, "DATA_DIR", tmp_path)
    out_dir = tmp_path / "playlists"
    out_dir.mkdir(parents=True)
    for i in range(22):
        f = out_dir / f"vibe-old-{i:02d}.m3u"
        f.write_text("#EXTM3U\n")
        os.utime(f, (1000 + i, 1000 + i))
    written = players.write_m3u(["/track.flac"], "my vibe")
    assert written.exists()
    assert written.read_text() == "#EXTM3U\n#PLAYLIST:my vibe\n/track.flac\n"
    remaining = {p.name for p in out_dir.glob("vibe-*.m3u")}
    assert len(remaining) == 20
    assert "vibe-old-00.m3u" not in remaining
    assert "vibe-old-01.m3u" not in remaining
    assert "vibe-old-02.m3u" not in remaining


def test_players_play_stdout_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    p_cfg = PlayerConfig()
    players.play("stdout", ["/x.mp3"], "test", p_cfg)
    assert capsys.readouterr().out == "#EXTM3U\n#PLAYLIST:test\n/x.mp3\n"

    with pytest.raises(players.PlayerError, match="no results"):
        players.play("stdout", [], "test", p_cfg)

    with pytest.raises(players.PlayerError, match="unknown player backend"):
        players.play("winamp", ["/x.mp3"], "test", p_cfg)


def test_store_and_chunk_store_ensure_recreate_on_dim_mismatch() -> None:
    from types import SimpleNamespace
    from typing import Any

    from qdrant_client import models

    from echo_control.config import QdrantConfig
    from echo_control.store import TRACK_VECTORS, ChunkStore, Store

    class FakeClient:
        def __init__(self) -> None:
            self.collections: dict[str, Any] = {}
            self.deleted: list[str] = []

        def collection_exists(self, name: str) -> bool:
            return name in self.collections

        def get_collection(self, name: str) -> Any:
            return SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors=self.collections[name]))
            )

        def delete_collection(self, name: str) -> None:
            self.deleted.append(name)
            self.collections.pop(name, None)

        def create_collection(self, name: str, vectors_config: Any, **_kwargs: Any) -> None:
            self.collections[name] = vectors_config

        def create_payload_index(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    s = Store.__new__(Store)
    s.cfg = QdrantConfig(collection="col")
    s.collection = "col"
    fake = FakeClient()
    s.client = fake  # pyright: ignore[reportAttributeAccessIssue]
    s.ensure(512)
    assert fake.deleted == []
    s.ensure(512)
    assert fake.deleted == []
    s.ensure(256)
    assert fake.deleted == ["col"]
    assert fake.collections["col"].size == 256

    cs = ChunkStore.__new__(ChunkStore)
    cs.cfg = QdrantConfig(collection="col")
    cs.collection = "col"
    cs.tracks = "col_tracks"
    cs.chunks = "col_chunks"
    cs.sustain_windows = 1
    fake_c = FakeClient()
    cs.client = fake_c  # pyright: ignore[reportAttributeAccessIssue]
    cs.ensure(512)
    assert fake_c.deleted == []
    cs.ensure(512)
    assert fake_c.deleted == []
    # Mismatch on chunks collection triggers recreation of both
    fake_c.collections["col_chunks"] = models.VectorParams(size=128, distance=models.Distance.COSINE)
    cs.ensure(512)
    assert set(fake_c.deleted) == {"col_chunks", "col_tracks"}
    assert all(fake_c.collections["col_tracks"][k].size == 512 for k in TRACK_VECTORS)
    assert fake_c.collections["col_chunks"].size == 512


def test_play_mpv_continuous_handles_none_playlist_pos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil
    import subprocess
    import time
    from typing import Any, Self

    def fake_which(_cmd: str) -> str:
        return "/usr/bin/mpv"

    def fake_sleep(_s: float) -> None:
        pass

    monkeypatch.setattr(players, "DATA_DIR", tmp_path)
    monkeypatch.setattr(players, "socket_path", lambda: tmp_path / "daemon.sock")
    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    appended: list[str] = []
    closed: list[bool] = []

    class FakeProc:
        polls: int

        def __init__(self, *_a: Any, **_kw: Any) -> None:
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls <= 2 else 0

        def wait(self) -> int:
            return 0

    class FakeIpc:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            closed.append(True)

        def command(self, *args: Any) -> Any:
            if args == ("get_property", "playlist-pos"):
                return None
            if args == ("get_property", "playlist-count"):
                return 1
            if args[0] == "loadfile":
                appended.append(args[1])
            return None

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    monkeypatch.setattr(players, "_MpvIpc", FakeIpc)

    players.play_mpv_continuous(
        iter(["/1.mp3", "/2.mp3", "/3.mp3", "/4.mp3"]), "walk", detach=False, ahead=2
    )
    assert appended == ["/4.mp3"]
    assert closed == [True]


def test_daemon_handler_and_request_edge_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socket
    import threading

    from watchdog.events import DirDeletedEvent

    from echo_control import daemon

    scans: list[bool] = []

    class FakeDaemon:
        def request_scan(self) -> None:
            scans.append(True)

    handler = daemon._Handler(FakeDaemon())  # pyright: ignore[reportArgumentType, reportPrivateUsage]
    handler.on_any_event(DirDeletedEvent("/music/album"))
    assert scans == [True]

    # Test request() when server closes socket immediately without sending newline
    sock_file = tmp_path / "test.sock"
    monkeypatch.setattr(daemon, "socket_path", lambda: sock_file)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_file))
    srv.listen(1)

    def accept_and_close() -> None:
        conn, _ = srv.accept()
        conn.close()
        srv.close()

    t = threading.Thread(target=accept_and_close)
    t.start()
    assert daemon.request({"op": "status"}, timeout=2.0) is None
    t.join()


def test_errors_hierarchy_and_cli_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from echo_control import store
    from echo_control.errors import ConfigError, EchoControlError, UsageError
    from echo_control.segments import SAMPLE_RATE, ffmpeg_decode_cmd

    assert issubclass(UsageError, EchoControlError)
    assert issubclass(ConfigError, EchoControlError)
    assert issubclass(players.PlayerError, EchoControlError)
    assert issubclass(store.QdrantUnavailable, EchoControlError)
    assert UsageError("bad flag").exit_code == 2
    assert ConfigError("bad config").exit_code == 1
    assert players.PlayerError("no mpv").exit_code == 1
    assert store.QdrantUnavailable("down").exit_code == 1

    assert ffmpeg_decode_cmd(Path("/x.mp3"), "s16le") == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        "/x.mp3",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]

    def raise_config_error() -> Config:
        raise ConfigError("broken toml")

    monkeypatch.setattr(config, "load", raise_config_error)
    with pytest.raises(SystemExit) as exc:
        cli.main(["status"])
    assert exc.value.code == 1
    assert "error: broken toml" in capsys.readouterr().err


def test_commands_choose_backend_and_json_formatting() -> None:
    import json

    from echo_control.commands import (
        VibeRequest,
        choose_backend,
        format_tracks_json,
        validate_vibe,
    )
    from echo_control.errors import UsageError

    cfg = Config(player=PlayerConfig(backend="mpv"), model=ModelConfig(strategy="chunks"))

    # When not a tty and not detached and no explicit player, falls back to stdout
    assert choose_backend(VibeRequest(query=["rock"]), cfg, is_tty=False) == "stdout"
    assert choose_backend(VibeRequest(query=["rock"]), cfg, is_tty=True) == "mpv"
    assert choose_backend(VibeRequest(query=["rock"], detach=True), cfg, is_tty=False) == "mpv"
    assert (
        choose_backend(VibeRequest(query=["rock"], player="spotatui"), cfg, is_tty=False)
        == "spotatui"
    )
    assert choose_backend(VibeRequest(query=["rock"], print_m3u=True), cfg, is_tty=True) == "stdout"

    # validate_vibe sets wander=True when orbit or continuous is provided
    req_orbit = VibeRequest(query=["rock"], orbit=1.5)
    validate_vibe(req_orbit, cfg)
    assert req_orbit.wander is True

    with pytest.raises(UsageError) as exc:
        validate_vibe(VibeRequest(), cfg)
    assert exc.value.show_usage is True

    roots = [Path("/music")]
    formatted = json.loads(
        format_tracks_json(
            [("/music/album/song.flac", 0.87654), ("/music/root.mp3", 0.5), ("/other/x.ogg", 0.1)],
            roots,
        )
    )
    assert formatted == [
        {"path": "/music/album/song.flac", "name": "song", "dir": "album", "score": 0.8765},
        {"path": "/music/root.mp3", "name": "root", "dir": "music", "score": 0.5},
        {"path": "/other/x.ogg", "name": "x", "dir": "/other", "score": 0.1},
    ]


def test_commands_format_status() -> None:
    from echo_control.commands import Status, format_status

    st_stopped = Status(
        config_path=None,
        library_paths=[Path("/music")],
        qdrant_url="http://localhost:6333",
        qdrant_ready=True,
        daemon_status=None,
        daemon_error="boom",
        indexed_fallback=42,
    )
    text_stopped = format_status(st_stopped)
    assert "config:  (defaults, no config file)" in text_stopped
    assert "qdrant:  http://localhost:6333 ready" in text_stopped
    assert "daemon:  error: boom" in text_stopped
    assert "daemon:  not running" in text_stopped
    assert "indexed: 42 tracks" in text_stopped

    st_running = Status(
        config_path=Path("/etc/echo.toml"),
        library_paths=[Path("/music")],
        qdrant_url="http://localhost:6333",
        qdrant_ready=True,
        daemon_status={
            "model": "clap",
            "device": "cpu",
            "strategy": "chunks",
            "indexed": 100,
            "scanning": False,
            "indexing": False,
            "last_scan": {"files": 100, "indexed": 5, "removed": 1},
            "pending": 3,
        },
    )
    text_running = format_status(st_running)
    assert "daemon:  running (clap on cpu, chunks strategy)" in text_running
    assert "scan:    last found 100 files, indexed 5, removed 1" in text_running
    assert "pending: 3 changed files" in text_running


def test_records_and_index_protocols() -> None:
    import os
    from types import SimpleNamespace
    from typing import Any

    import numpy as np

    from echo_control.config import QdrantConfig
    from echo_control.indexer import state_key
    from echo_control.records import (
        ScanKey,
        ScanStats,
        chunk_payload,
        chunk_track_payload,
        multivector_payload,
    )
    from echo_control.store import Candidate, ChunkIndex, ChunkStore, Store, VectorIndex

    st = os.stat_result((0o644, 1, 1, 1, 1000, 1000, 4096, 10.0, 20.0, 30.0))
    mv_pl = multivector_payload("/a.mp3", st, 120.5, [0.6, 0.4], "sig1")
    assert state_key(mv_pl) == ScanKey(20.0, 4096, "sig1")
    assert mv_pl["shares"] == [0.6, 0.4]

    cp = chunk_payload("/a.mp3", 2, 10, 5.0004, 15.0004, -12.345, False, 0.123456)
    assert cp == {
        "path": "/a.mp3",
        "idx": 2,
        "total": 10,
        "start": 5.0,
        "end": 15.0,
        "db": -12.35,
        "silent": False,
        "speed": 0.12346,
    }

    tp = chunk_track_payload("/a.mp3", st, 120.5, "sig1", 10, False, 0.1, 0.2, {"spikes": 2})
    assert state_key(tp) == ScanKey(20.0, 4096, "sig1")
    assert tp["n_chunks"] == 10 and tp["spikes"] == 2

    stats = ScanStats(10, 3, 1)
    total, indexed, removed = stats
    assert (total, indexed, removed) == (10, 3, 1)

    s = Store.__new__(Store)
    cs = ChunkStore.__new__(ChunkStore)
    assert isinstance(s, VectorIndex)
    assert isinstance(cs, VectorIndex)
    assert isinstance(cs, ChunkIndex)

    cand = Candidate.of(
        {"path": "/a.mp3"},
        0.75,
        {"mean": [1.0, 0.0], "intro": [0.0, 1.0], "intro_dir": [1.0, 0.0]},
    )
    assert cand.intro_sim == 0.75
    assert cand.position == 0.75

    # Verify chunk_vectors paginates across multiple scroll batches and sorts by idx
    cs.cfg = QdrantConfig(collection="col")
    cs.chunks = "col_chunks"

    class PaginatingClient:
        def scroll(self, *_a: Any, offset: int | None = None, **_kw: Any) -> tuple[list[Any], int | None]:
            if offset is None:
                return [SimpleNamespace(payload={"idx": 1}, vector=[0.0, 1.0])], 1
            return [SimpleNamespace(payload={"idx": 0}, vector=[1.0, 0.0])], None

    cs.client = PaginatingClient()  # pyright: ignore[reportAttributeAccessIssue]
    vecs = cs.chunk_vectors("/a.mp3")
    assert np.allclose(vecs, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))




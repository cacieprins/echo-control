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



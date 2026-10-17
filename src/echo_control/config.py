# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import ConfigError

PRECISIONS = ("fp32", "bf16")
PLAYER_BACKENDS = ("mpv", "spotatui", "command", "stdout")


class StrategyName(StrEnum):
    MEAN = "mean"
    CLUSTERS = "clusters"
    CHUNKS = "chunks"


CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "echo-control"
CONFIG_PATH = CONFIG_DIR / "config.toml"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "echo-control"


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "echo-control.sock"


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    m = _DURATION_RE.match(value)
    if not m:
        raise ConfigError(f"invalid duration: {value!r}")
    return float(m.group(1)) * _UNITS[m.group(2)]


@dataclass
class LibraryConfig:
    paths: list[Path] = field(default_factory=lambda: [Path.home() / "Music"])
    extensions: list[str] = field(
        default_factory=lambda: ["mp3", "flac", "ogg", "opus", "m4a", "wav", "aac", "wma", "aiff"]
    )
    rescan_interval: float = 6 * 3600
    exclude: list[Path] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str = "laion/larger_clap_music_and_speech"
    device: str = "auto"
    precision: str = "bf16"
    batch_size: int = 8
    windows: int = 3
    window_seconds: float = 10.0
    text_prefix: str = "A recording of "
    strategy: str = StrategyName.CHUNKS
    max_clusters: int = 4
    min_cluster_share: float = 0.12
    merge_threshold: float = 0.85
    max_chunks: int = 60
    silence_db: float = -45.0
    hop_seconds: float = 2.5
    edge_seconds: float = 30.0


@dataclass
class SearchConfig:
    share_weight: float = 0.0
    sustain_seconds: float = 60.0


@dataclass
class WanderConfig:
    position: float = 1.0
    direction: float = 0.5
    drift: float = 0.25
    candidates: int = 100
    duplicate_similarity: float = 0.95
    duplicate_alignment: float = 0.92


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection: str = "echo_control"
    ready_timeout: float = 60.0


@dataclass
class PlayerConfig:
    backend: str = "mpv"
    limit: int = 25
    command: str = "mpv --playlist={m3u}"
    spotatui_dir: Path = field(default_factory=lambda: Path.home() / "Music" / "vibe-playlists")
    spotatui_keep: int = 10


@dataclass
class Config:
    library: LibraryConfig = field(default_factory=LibraryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    player: PlayerConfig = field(default_factory=PlayerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    wander: WanderConfig = field(default_factory=WanderConfig)
    path: Path | None = None


_SECTION_TYPES: dict[str, type[Any]] = {
    "library": LibraryConfig,
    "model": ModelConfig,
    "qdrant": QdrantConfig,
    "player": PlayerConfig,
    "search": SearchConfig,
    "wander": WanderConfig,
}


def _check_section_keys(section: str, data: dict[str, Any], cls: type[Any]) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"[{section}] unknown key {unknown[0]!r}")


def load(path: Path | None = None) -> Config:
    path = path or Path(os.environ.get("ECHO_CONTROL_CONFIG", CONFIG_PATH))
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

    unknown_sections = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown_sections:
        raise ConfigError(f"unknown config section [{unknown_sections[0]}]")
    for sec, cls in _SECTION_TYPES.items():
        if sec in raw:
            if not isinstance(raw[sec], dict):
                raise ConfigError(f"[{sec}] must be a table")
            _check_section_keys(sec, raw[sec], cls)

    lib = raw.get("library", {})
    library = LibraryConfig()
    if "paths" in lib:
        library.paths = [Path(p).expanduser().resolve() for p in lib["paths"]]
    if "extensions" in lib:
        library.extensions = [e.lower().lstrip(".") for e in lib["extensions"]]
    if "rescan_interval" in lib:
        library.rescan_interval = parse_duration(lib["rescan_interval"])

    player_raw = dict(raw.get("player", {}))
    if "spotatui_dir" in player_raw:
        player_raw["spotatui_dir"] = Path(player_raw["spotatui_dir"]).expanduser().resolve()
    player = PlayerConfig(**player_raw)
    if player.backend not in PLAYER_BACKENDS:
        raise ConfigError(
            f"[player] backend must be one of {', '.join(PLAYER_BACKENDS)}, not {player.backend!r}"
        )
    if player.limit <= 0:
        raise ConfigError("[player] limit must be > 0")

    # Generated spotatui playlists are symlinks into the library; never index them.
    configured_exclude = [Path(p).expanduser().resolve() for p in lib.get("exclude", [])]
    library.exclude = [*configured_exclude, player.spotatui_dir.expanduser().resolve()]

    model = ModelConfig(**raw.get("model", {}))
    if model.strategy not in StrategyName:
        raise ConfigError(
            f"[model] strategy must be 'mean', 'clusters' or 'chunks', not {model.strategy!r}"
        )
    if model.precision not in PRECISIONS:
        raise ConfigError(
            f"[model] precision must be 'fp32' or 'bf16', not {model.precision!r}"
        )
    if not 0 < model.hop_seconds <= model.window_seconds:
        raise ConfigError("[model] hop_seconds must be > 0 and <= window_seconds")

    search = SearchConfig(**raw.get("search", {}))
    if search.sustain_seconds <= 0:
        raise ConfigError("[search] sustain_seconds must be > 0")

    return Config(
        library=library,
        model=model,
        qdrant=QdrantConfig(**raw.get("qdrant", {})),
        player=player,
        search=search,
        wander=WanderConfig(**raw.get("wander", {})),
        path=path if path.exists() else None,
    )

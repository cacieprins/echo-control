from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "echo-control"
CONFIG_PATH = CONFIG_DIR / "config.toml"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "echo-control"


def socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "echo-control.sock"


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")
_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(f"invalid duration: {value!r}")
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
    batch_size: int = 8
    windows: int = 3
    window_seconds: float = 10.0
    text_prefix: str = "A recording of "


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
    path: Path | None = None


def load(path: Path | None = None) -> Config:
    path = path or Path(os.environ.get("ECHO_CONTROL_CONFIG", CONFIG_PATH))
    raw: dict = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)

    lib = raw.get("library", {})
    library = LibraryConfig()
    if "paths" in lib:
        library.paths = [Path(p).expanduser().resolve() for p in lib["paths"]]
    if "extensions" in lib:
        library.extensions = [e.lower().lstrip(".") for e in lib["extensions"]]
    if "rescan_interval" in lib:
        library.rescan_interval = parse_duration(lib["rescan_interval"])
    if "exclude" in lib:
        library.exclude = [Path(p).expanduser().resolve() for p in lib["exclude"]]

    player_raw = dict(raw.get("player", {}))
    if "spotatui_dir" in player_raw:
        player_raw["spotatui_dir"] = Path(player_raw["spotatui_dir"]).expanduser().resolve()
    player = PlayerConfig(**player_raw)
    # Generated spotatui playlists are symlinks into the library; never index them.
    library.exclude.append(player.spotatui_dir.expanduser().resolve())

    return Config(
        library=library,
        model=ModelConfig(**raw.get("model", {})),
        qdrant=QdrantConfig(**raw.get("qdrant", {})),
        player=player,
        path=path if path.exists() else None,
    )

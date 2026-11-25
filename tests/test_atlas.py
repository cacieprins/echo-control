# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echo_control import atlas, cli
from echo_control.config import Config, WanderConfig
from echo_control.cooldown import TrailLog
from echo_control.errors import EchoControlError
from echo_control.manifold import CorpusManifold


def _synthetic_manifold(n: int = 24, dim: int = 16) -> CorpusManifold:
    rng = np.random.default_rng(7)
    artists = ["Alpha", "Beta", "Gamma"]
    centers = rng.normal(size=(len(artists), dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    paths: list[str] = []
    vecs = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        a = artists[i % len(artists)]
        paths.append(f"/music/{a}/Album/{i:02d} Track {i}.mp3")
        v = centers[i % len(artists)] + 0.15 * rng.normal(size=dim).astype(np.float32)
        vecs[i] = v / np.linalg.norm(v)
    return CorpusManifold.from_arrays(paths, vecs, vecs, neighbors=5)


def test_render_library_atlas_and_cache(tmp_path: Path) -> None:
    manifold = _synthetic_manifold()
    trails = TrailLog()
    trails.commit(manifold.paths[0], now=1000.0, save=False)
    session = [manifold.paths[0], manifold.paths[1], manifold.paths[2]]
    cache = tmp_path / "tsne.npz"

    svg = atlas.render_library_atlas_svg(
        manifold,
        trails,
        session,
        WanderConfig(),
        clusters=3,
        highlight="Alpha",
        library_roots=[Path("/music")],
        cache_path=cache,
        now=1010.0,
    )
    assert svg.startswith("<svg ")
    assert "echo-control Library Atlas" in svg
    assert "Alpha" in svg
    assert "NOW: 02 Track 2" in svg
    assert cache.exists()

    # Second call reuses the cached t-SNE coordinates identically
    c1 = atlas.compute_tsne_layout(manifold, cache_path=cache)
    c2 = atlas.compute_tsne_layout(manifold, cache_path=cache)
    assert np.array_equal(c1, c2)


def test_render_focus_topo_svg() -> None:
    manifold = _synthetic_manifold()
    trails = TrailLog()
    session = [manifold.paths[0], manifold.paths[3]]
    svg = atlas.render_focus_topo_svg(
        manifold, trails, session, WanderConfig(), "Alpha", now=1000.0
    )
    assert svg.startswith("<svg ")
    assert 'Topographic Heat Map Around "Alpha"' in svg
    assert "Radial Kernel Profile" in svg

    with pytest.raises(EchoControlError, match="no indexed tracks matched"):
        atlas.render_focus_topo_svg(manifold, trails, session, WanderConfig(), "NoSuchArtist")


def test_load_session_paths(tmp_path: Path) -> None:
    log = tmp_path / "2026-11-25.log"
    log.write_text(
        """\
12:00:00  A — S1  [score=0.00 intro=0.00 heat=0.00]  /music/A/01.mp3
12:03:00  A — S2  [score=0.50 intro=0.80 heat=0.01]  /music/A/02.mp3
12:06:00  B — S3  [score=0.00 intro=0.00 heat=0.00]  /music/B/03.mp3
12:09:00  B — S4  [score=0.60 intro=0.85 heat=0.01]  /music/B/04.mp3
""",
        encoding="utf-8",
    )
    assert atlas.load_session_runs(tmp_path, "2026-11-25") == [
        ["/music/A/01.mp3", "/music/A/02.mp3"],
        ["/music/B/03.mp3", "/music/B/04.mp3"],
    ]
    assert atlas.load_session_paths(tmp_path, "2026-11-25") == [
        "/music/A/01.mp3",
        "/music/A/02.mp3",
        "/music/B/03.mp3",
        "/music/B/04.mp3",
    ]
    assert atlas.load_session_paths(tmp_path, "2026-11-24") == []


def test_cli_map_dispatches_to_atlas(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}
    out_file = tmp_path / "out.svg"

    def fake_generate(_cfg: Config, **kwargs: Any) -> Path:
        seen.update(kwargs)
        return out_file

    monkeypatch.setattr("echo_control.config.load", lambda: Config())
    monkeypatch.setattr(atlas, "generate_map", fake_generate)

    cli.main(["map", "-o", str(out_file), "-k", "12", "--highlight", "kagerou", "--date", "2026-11-25"])
    assert seen == {
        "output": out_file,
        "clusters": 12,
        "highlight": "kagerou",
        "focus": None,
        "date": "2026-11-25",
        "open_after": False,
        "label_mode": "clusters",
    }
    assert capsys.readouterr().out.strip() == str(out_file)


def test_render_library_atlas_recent_callouts() -> None:
    manifold = _synthetic_manifold()
    trails = TrailLog()
    session = [manifold.paths[0], manifold.paths[1], manifold.paths[2], manifold.paths[3]]
    svg = atlas.render_library_atlas_svg(
        manifold,
        trails,
        session,
        WanderConfig(),
        clusters=3,
        library_roots=[Path("/music")],
        label_mode="recent",
        next_path=manifold.paths[4],
        now=1010.0,
    )
    assert "Recent Trail (4 played)" in svg
    assert "03 Track 3" in svg
    assert "02 Track 2" in svg
    assert "01 Track 1" in svg
    assert "Earlier in session" in svg
    assert "00 Track 0" in svg
    assert 'marker-end="url(#nextArrow)"' in svg

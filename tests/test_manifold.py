# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np

from echo_control.manifold import CorpusManifold
from tests.fakes import InMemoryChunkIndex


def _unit(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _add_track(
    store: InMemoryChunkIndex,
    path: str,
    mean: np.ndarray,
    intro: np.ndarray | None = None,
    outro: np.ndarray | None = None,
    *,
    silent: bool = False,
) -> None:
    intro_v = mean if intro is None else intro
    outro_v = mean if outro is None else outro
    store.replace_track(
        path,
        mean[None, :],
        [{"path": path, "idx": 0, "start": 0.0, "silent": silent}],
        {"mean": mean, "intro": intro_v, "outro": outro_v},
        {"path": path, "n": 1, "silent": silent},
    )


def test_corpus_manifold_centering_and_csls_hubness() -> None:
    rng = np.random.default_rng(42)
    # Tight dense cluster of 15 tracks around [1, 0, 0, ...] + 2 peripheral bridge tracks
    dim = 16
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    paths: list[str] = []
    vecs: list[np.ndarray] = []
    for i in range(15):
        v = base + 0.05 * rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        paths.append(f"/music/hub_{i:02d}.flac")
        vecs.append(v)

    bridge_a = np.zeros(dim, dtype=np.float32)
    bridge_a[1] = 1.0
    bridge_b = np.zeros(dim, dtype=np.float32)
    bridge_b[2] = 1.0
    paths.extend(["/music/bridge_a.flac", "/music/bridge_b.flac"])
    vecs.extend([bridge_a, bridge_b])

    mat = np.stack(vecs, axis=0)
    m = CorpusManifold.from_arrays(paths, mat, mat, neighbors=5)

    assert m.means.shape == (17, dim)
    assert m.centered.shape == (17, dim)
    assert np.allclose(np.linalg.norm(m.centered, axis=1), 1.0, atol=1e-5)
    assert m.median_radius > 0.0

    # Hub tracks inside the dense cluster must have much higher CSLS than peripheral bridges
    hub_csls = m.csls_for("/music/hub_00.flac")
    bridge_csls = m.csls_for("/music/bridge_a.flac")
    assert hub_csls > 0.90
    assert bridge_csls < 0.25
    assert hub_csls > bridge_csls + 0.65

    # raw_probe maps centered direction back into the raw unit cone aligned with bridge_a
    c_bridge = m.centered_for("/music/bridge_a.flac")
    assert c_bridge is not None
    raw = m.raw_probe(c_bridge)
    assert abs(float(np.linalg.norm(raw)) - 1.0) < 1e-5
    assert float(raw @ bridge_a) > float(raw @ bridge_b)
    assert float(raw @ bridge_a) > 0.20


def test_corpus_manifold_cache_hit_and_invalidation(tmp_path: Path) -> None:
    store = InMemoryChunkIndex()
    store.ensure(4)
    _add_track(store, "/a.flac", _unit([1.0, 0.0, 0.0, 0.0]))
    _add_track(store, "/b.flac", _unit([0.0, 1.0, 0.0, 0.0]))
    _add_track(store, "/silent.flac", _unit([0.0, 0.0, 1.0, 0.0]), silent=True)

    cache = tmp_path / "manifold.npz"
    m1 = CorpusManifold.from_store(store, neighbors=2, cache_path=cache)
    assert cache.exists()
    assert m1.paths == ["/a.flac", "/b.flac"]

    # Second load hits cache and matches identically
    m2 = CorpusManifold.from_store(store, neighbors=2, cache_path=cache)
    assert m2.paths == m1.paths
    assert np.allclose(m2.csls, m1.csls)
    assert np.allclose(m2.mu, m1.mu)

    # Adding a track invalidates cache and recomputes
    _add_track(store, "/c.flac", _unit([0.0, 0.0, 1.0, 0.0]))
    m3 = CorpusManifold.from_store(store, neighbors=2, cache_path=cache)
    assert m3.paths == ["/a.flac", "/b.flac", "/c.flac"]
    assert m3.means.shape == (3, 4)

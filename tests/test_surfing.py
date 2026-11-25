# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np

from echo_control.cooldown import HeatField
from echo_control.manifold import CorpusManifold
from echo_control.surfing import ChannelSurfer


def _make_sphere_manifold(n: int = 128, dim: int = 16, seed: int = 123) -> CorpusManifold:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    paths = [f"/music/t_{i:03d}.flac" for i in range(n)]
    return CorpusManifold.from_arrays(paths, mat, mat, neighbors=5)


def test_bail_antipode_and_cold_halving() -> None:
    manifold = _make_sphere_manifold(n=128)
    heat = HeatField()
    surfer = ChannelSurfer(manifold=manifold, heat_field=heat)

    start = manifold.paths[0]
    p0 = manifold.centered[0]

    # Tap 1: antipodal opening gambit + repulsive heat spike on rejected track
    r1 = surfer.bail(start, now=100.0)
    assert r1 is not None
    assert r1.mode == "antipode"
    assert r1.path != start
    p1 = manifold.centered_for(r1.path)
    assert p1 is not None
    assert float(p1 @ p0) < 0.0
    assert heat.heat(manifold.means[0]) > 1.0

    # Tap 2 (dt = 1.0s < 3.0s): cold median hyperplane cut halves remaining viable pool
    v1 = r1.viable_count
    r2 = surfer.bail(r1.path, now=101.0)
    assert r2 is not None
    assert r2.mode == "cold"
    assert r2.path not in {start, r1.path}
    assert r2.viable_count <= (v1 // 2) + 1

    # Tap 3 (dt = 1.5s < 3.0s): halves again
    v2 = r2.viable_count
    r3 = surfer.bail(r2.path, now=102.5)
    assert r3 is not None
    assert r3.mode == "cold"
    assert r3.viable_count <= (v2 // 2) + 1


def test_warm_linger_and_settle_reset() -> None:
    manifold = _make_sphere_manifold(n=128)
    surfer = ChannelSurfer(manifold=manifold)

    r1 = surfer.bail(manifold.paths[0], now=10.0)
    assert r1 is not None
    assert surfer.active is True

    # Linger 6s (3s <= dt < 15s) -> warm principal-axis refinement
    r2 = surfer.bail(r1.path, now=16.0)
    assert r2 is not None
    assert r2.mode == "warm"
    assert r2.path != r1.path
    assert "warm" in surfer.status_badge(now=17.0)

    # Settle after >= 15s resets search state
    assert surfer.status_badge(now=32.0) == ""
    assert surfer.active is False


def test_warmer_geodesic_midpoint_bracket() -> None:
    # Construct 3 clear directions: pole A (+e0), pole B (-e0 + eps*e1), midpoint (+e1)
    dim = 8
    eye = np.eye(dim, dtype=np.float32)
    vecs = np.stack(
        [
            eye[0],
            -eye[0] + 0.05 * eye[1],
            eye[1],
            eye[2],
            eye[3],
            eye[4],
        ],
        axis=0,
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    paths = [f"/music/{name}.flac" for name in ("pole_a", "pole_b", "mid", "x1", "x2", "x3")]
    manifold = CorpusManifold.from_arrays(paths, vecs, vecs, neighbors=2)
    surfer = ChannelSurfer(manifold=manifold, opening_cut=0.5)

    r1 = surfer.bail("/music/pole_a.flac", now=50.0)
    assert r1 is not None
    assert r1.path == "/music/pole_b.flac"

    # Pressing [v] finds the geodesic midpoint between pole_a and pole_b
    r2 = surfer.warmer(r1.path, now=52.0)
    assert r2 is not None
    assert r2.mode == "midpoint"
    assert r2.path == "/music/mid.flac"


def test_safety_valve_wraps_when_viable_exhausted() -> None:
    manifold = _make_sphere_manifold(n=24)
    surfer = ChannelSurfer(manifold=manifold, min_viable=8)

    cur = manifold.paths[0]
    t = 10.0
    wrapped_seen = False
    for _ in range(8):
        res = surfer.bail(cur, now=t)
        assert res is not None
        if res.wrapped:
            wrapped_seen = True
            break
        cur = res.path
        t += 1.0
    assert wrapped_seen is True

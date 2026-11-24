# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np

from echo_control.config import WanderConfig
from echo_control.cooldown import CooldownScorer, HeatField, TrailLog, sample_next
from echo_control.manifold import CorpusManifold
from echo_control.wander import Transition, rank_next, wander
from tests.fakes import InMemoryChunkIndex


def _unit(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_heat_field_micro_decay_and_macro_basin() -> None:
    hf = HeatField(
        micro_sigma=0.10,
        micro_tau=10.0,
        macro_sigma=0.30,
        macro_beta=0.25,
        macro_alpha=0.80,
    )
    a = _unit([1.0, 0.0, 0.0, 0.0])
    near = _unit([0.99, 0.14, 0.0, 0.0])
    far = _unit([0.0, 0.0, 1.0, 0.0])

    hf.record(a)
    h_same_1 = hf.heat(a)
    h_near_1 = hf.heat(near)
    h_far_1 = hf.heat(far)
    assert h_same_1 > h_near_1 > h_far_1
    assert h_far_1 < 0.01

    # Injecting a spike repels immediately without moving macro centroid
    before_spike = hf.heat(far)
    hf.inject_spike(far, weight=1.5)
    assert hf.heat(far) > before_spike + 1.0

    # Advancing many steps elsewhere decays micro heat on `a`
    other = _unit([0.0, 1.0, 0.0, 0.0])
    for _ in range(30):
        hf.record(other)
    assert hf.heat(a) < h_same_1 * 0.15
    # Meanwhile the macro basin around `other` is now hot
    assert hf.heat(other) > 1.0


def test_trail_log_persistence_and_wall_clock_decay(tmp_path: Path) -> None:
    path = tmp_path / "trails.json"
    t0 = 1_700_000_000.0
    log = TrailLog.load(path)
    log.commit("/a.flac", None, now=t0)
    log.commit("/b.flac", "/a.flac", now=t0)

    assert path.exists()
    reloaded = TrailLog.load(path)
    assert abs(reloaded.node_heat("/b.flac", now=t0) - 0.15) < 1e-5
    assert abs(reloaded.edge_wear("/a.flac", "/b.flac", now=t0) - 0.25) < 1e-5
    # Directed edge is asymmetric
    assert reloaded.edge_wear("/b.flac", "/a.flac", now=t0) == 0.0

    # Half-life checks: 60h for node heat, 14d for edge wear
    assert abs(reloaded.node_heat("/b.flac", now=t0 + 60 * 3600) - 0.075) < 1e-5
    assert abs(reloaded.edge_wear("/a.flac", "/b.flac", now=t0 + 14 * 86400) - 0.125) < 1e-5


def test_sample_next_margin_and_close_call_branching() -> None:
    cfg = WanderConfig(softmax_k=6, softmax_margin=0.06, softmax_temp=0.02)
    # Clear outlier winner (> 0.06 ahead): always deterministic
    clear = [
        Transition("/win.flac", 0.90, 0.9, 0.0, 0.0),
        Transition("/lose.flac", 0.80, 0.8, 0.0, 0.0),
    ]
    rng = np.random.default_rng(7)
    for _ in range(25):
        assert sample_next(clear, cfg, rng).path == "/win.flac"

    # Close tie within margin: branches across candidates
    close = [
        Transition("/t1.flac", 0.90, 0.9, 0.0, 0.0),
        Transition("/t2.flac", 0.89, 0.89, 0.0, 0.0),
        Transition("/t3.flac", 0.88, 0.88, 0.0, 0.0),
    ]
    picks = {sample_next(close, cfg, rng).path for _ in range(40)}
    assert len(picks) >= 2


def test_cooldown_breaks_hub_dominance_and_edge_ruts(tmp_path: Path) -> None:
    store = InMemoryChunkIndex()
    store.ensure(8)
    eye = np.eye(8, dtype=np.float32)

    def put(path: str, mean: np.ndarray, intro: np.ndarray, outro: np.ndarray) -> None:
        store.replace_track(
            path,
            mean[None, :],
            [{"path": path, "idx": 0, "start": 0.0, "silent": False}],
            {"mean": mean, "intro": intro, "outro": outro},
            {"path": path, "n": 1},
        )

    # Seed ends between axis 1 (dense hub) and axis 5 (peripheral bridge), slightly closer to hub.
    seed_outro = _unit((0.73 * eye[1] + 0.68 * eye[5]).tolist())
    put("/seed", eye[0], eye[0], seed_outro)
    # Dense hub cluster around axis 1: /hub0 has slightly higher raw intro sim (~0.73)
    # than /bridge (~0.68), plus 5 tight neighbors right next to /hub0's intro (CSLS ~0.99).
    for i in range(6):
        v = _unit((eye[1] + 0.02 * (i + 1) * eye[2]).tolist())
        put(f"/hub{i}", v, v, eye[3])

    # Peripheral bridge sits alone on axis 5: slightly lower raw intro sim (~0.68), CSLS ~0.0.
    put("/bridge", eye[5], eye[5], eye[6])

    cfg = WanderConfig(csls_gamma=0.40, csls_neighbors=5, softmax_temp=0.0)
    manifold = CorpusManifold.from_store(store, neighbors=cfg.csls_neighbors)
    trails = TrailLog.load(tmp_path / "trails.json", cfg)
    scorer = CooldownScorer(manifold, HeatField.from_config(cfg), trails, cfg)

    # Without cooldown, hub wins on raw intro_sim
    raw_ranked = rank_next(store, [Transition("/seed", 0.0, 0.0, 0.0, 0.0, mean=eye[0], intro=eye[0])], cfg)
    assert raw_ranked[0].path.startswith("/hub")

    # With cooldown CSLS penalty, peripheral bridge outscores the dense hub!
    cd_ranked = rank_next(
        store,
        [Transition("/seed", 0.0, 0.0, 0.0, 0.0, mean=eye[0], intro=eye[0])],
        cfg,
        cooldown=scorer,
    )
    assert cd_ranked[0].path == "/bridge"
    assert cd_ranked[0].csls < cd_ranked[1].csls

    # Edge wear erodes a previously traversed /seed -> /bridge rut on the next session
    trails.commit("/bridge", "/seed")
    trails.commit("/bridge", "/seed")
    chain = wander(store, "/seed", 2, cfg, cooldown=scorer, update_heat_on_step=True)
    assert chain[0].path == "/seed"
    assert chain[1].path != "/bridge"

"""Chain tracks so each one picks up where the last left off in embedding space: the next
track's intro should sit near the previous outro, head the same way, and move at a similar
pace. Orbiting adds a pull toward a fixed center (the query, or the seed track), so the walk
circles a vibe instead of drifting off."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from .config import WanderConfig
from .store import Candidate, ChunkStore

DRIFT_EPS = 1e-4


@dataclass
class Transition:
    path: str
    score: float
    position: float
    direction: float
    drift: float
    center: float = 0.0
    mean: np.ndarray | None = field(default=None, repr=False)
    duration: float = 0.0


@dataclass
class Orbit:
    center: list[float]
    strength: float


def drift_match(a: float, b: float) -> float:
    """0 for equal drift, more negative the further apart they are in ratio."""
    return -abs(math.log((a + DRIFT_EPS) / (b + DRIFT_EPS)))


def is_duplicate(cand: Candidate, played: list[Transition], cfg: WanderConfig) -> bool:
    """Another file of a track already played: near-identical overall sound and length.
    Similar tracks from one album can sound as alike, but rarely run the same length."""
    duration = float(cand.payload.get("duration") or 0.0)
    return any(
        float(cand.mean @ t.mean) >= cfg.duplicate_similarity
        and abs(duration - t.duration) <= cfg.duplicate_seconds
        for t in played
    )


def rank_next(
    store: ChunkStore,
    played: list[Transition],
    cfg: WanderConfig,
    orbit: Orbit | None = None,
) -> list[Transition]:
    """Candidate follow-ups to the last played track, best first, leaving out anything
    played already or a duplicate of it. When orbiting, the tracks closest to the center
    are candidates too, so the walk can come back even if every intro near the current
    outro has drifted away."""
    path = played[-1].path
    found = store.track(path)
    if found is None:
        raise KeyError(path)
    payload, vecs = found
    outro, outro_dir = vecs["outro"], vecs.get("outro_dir")
    outro_drift = float(payload.get("outro_drift") or 0.0)
    skip = {t.path for t in played}
    cands = store.intros_near(outro, cfg.candidates, list(skip))
    near_center: dict[str, float] = {}
    if orbit:
        seen = skip | {c.payload["path"] for c in cands}
        extra = [h.path for h in store.search(orbit.center, cfg.candidates) if h.path not in seen]
        cands += store.intros_of(outro, extra)
        near_center = store.sustained(orbit.center, [c.payload["path"] for c in cands])

    out = []
    for c in cands:
        if is_duplicate(c, played, cfg):
            continue
        direction = (
            float(outro_dir @ c.intro_dir)
            if outro_dir is not None and c.intro_dir is not None
            else 0.0
        )
        drift = drift_match(outro_drift, float(c.payload.get("intro_drift") or 0.0))
        center = near_center.get(c.payload["path"], 0.0)
        score = cfg.position * c.position + cfg.direction * direction + cfg.drift * drift
        if orbit:
            score += orbit.strength * center
        out.append(Transition(
            c.payload["path"], score, c.position, direction, drift, center,
            c.mean, float(c.payload.get("duration") or 0.0),
        ))
    out.sort(key=lambda t: -t.score)
    return out


def walk(
    store: ChunkStore, seed: str, cfg: WanderConfig, orbit: Orbit | None = None
) -> Iterator[Transition]:
    """Greedy walk from seed, one step at a time, until nothing unplayed is left; the seed
    comes first, with a score of 0."""
    found = store.track(seed)
    if found is None:
        raise KeyError(seed)
    payload, vecs = found
    chain = [Transition(
        seed, 0.0, 0.0, 0.0, 0.0, mean=vecs["mean"], duration=float(payload.get("duration") or 0.0)
    )]
    yield chain[0]
    while ranked := rank_next(store, chain, cfg, orbit):
        chain.append(ranked[0])
        yield ranked[0]


def wander(
    store: ChunkStore, seed: str, length: int, cfg: WanderConfig, orbit: Orbit | None = None
) -> list[Transition]:
    return list(itertools.islice(walk(store, seed, cfg, orbit), length))

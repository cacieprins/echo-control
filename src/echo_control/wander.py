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

from . import trajectory
from .config import WanderConfig
from .store import Candidate, ChunkIndex

DRIFT_EPS = 1e-4


@dataclass
class Transition:
    path: str
    score: float
    intro_sim: float
    direction: float
    drift: float
    center: float = 0.0
    mean: np.ndarray | None = field(default=None, repr=False)
    intro: np.ndarray | None = field(default=None, repr=False)

    @property
    def position(self) -> float:
        return self.intro_sim


@dataclass
class Orbit:
    center: list[float]
    strength: float


def drift_match(a: float, b: float) -> float:
    """0 for equal drift, more negative the further apart they are in ratio."""
    return -abs(math.log((a + DRIFT_EPS) / (b + DRIFT_EPS)))


class Duplicates:
    """Spots another file of a track already played. Pairs that are close overall or at
    the intro get their windows compared in time order over the stretch they share:
    copies of one recording (other encodes, remasters, compilation versions, truncated
    files) line up window for window, while similar tracks from one album only match on
    average. Window vectors are fetched once per track and cached for the walk."""

    def __init__(self, store: ChunkIndex, cfg: WanderConfig):
        self.store: ChunkIndex = store
        self.cfg: WanderConfig = cfg
        self.windows: dict[str, np.ndarray] = {}

    def _windows(self, path: str) -> np.ndarray:
        if path not in self.windows:
            self.windows[path] = self.store.chunk_vectors(path)
        return self.windows[path]

    def __call__(self, cand: Candidate, played: list[Transition]) -> bool:
        cfg = self.cfg
        path = cand.payload["path"]
        for t in played:
            close = float(cand.mean @ t.mean) >= cfg.duplicate_similarity or (
                t.intro is not None and float(cand.intro @ t.intro) >= cfg.duplicate_similarity
            )
            if close and (
                trajectory.aligned_similarity(self._windows(path), self._windows(t.path))
                >= cfg.duplicate_alignment
            ):
                return True
        return False


def rank_next(
    store: ChunkIndex,
    played: list[Transition],
    cfg: WanderConfig,
    orbit: Orbit | None = None,
    duplicates: Duplicates | None = None,
) -> list[Transition]:
    """Candidate follow-ups to the last played track, best first, leaving out anything
    played already or a duplicate of it. When orbiting, the tracks closest to the center
    are candidates too, so the walk can come back even if every intro near the current
    outro has drifted away."""
    duplicates = duplicates or Duplicates(store, cfg)
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

    out: list[Transition] = []
    for c in cands:
        if duplicates(c, played):
            continue
        direction = (
            float(outro_dir @ c.intro_dir)
            if outro_dir is not None and c.intro_dir is not None
            else 0.0
        )
        drift = drift_match(outro_drift, float(c.payload.get("intro_drift") or 0.0))
        center = near_center.get(c.payload["path"], 0.0)
        score = cfg.position * c.intro_sim + cfg.direction * direction + cfg.drift * drift
        if orbit:
            score += orbit.strength * center
        out.append(Transition(
            c.payload["path"], score, c.intro_sim, direction, drift, center, c.mean, c.intro,
        ))
    out.sort(key=lambda t: -t.score)
    return out


def walk(
    store: ChunkIndex, seed: str, cfg: WanderConfig, orbit: Orbit | None = None
) -> Iterator[Transition]:
    """Greedy walk from seed, one step at a time, until nothing unplayed is left; the seed
    comes first, with a score of 0."""
    found = store.track(seed)
    if found is None:
        raise KeyError(seed)
    _, vecs = found
    chain = [Transition(seed, 0.0, 0.0, 0.0, 0.0, mean=vecs["mean"], intro=vecs["intro"])]
    yield chain[0]
    duplicates = Duplicates(store, cfg)
    while ranked := rank_next(store, chain, cfg, orbit, duplicates):
        chain.append(ranked[0])
        yield ranked[0]


def wander(
    store: ChunkIndex, seed: str, length: int, cfg: WanderConfig, orbit: Orbit | None = None
) -> list[Transition]:
    return list(itertools.islice(walk(store, seed, cfg, orbit), length))

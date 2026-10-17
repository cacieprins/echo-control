from __future__ import annotations

import os
from typing import Any, NamedTuple


class ScanKey(NamedTuple):
    mtime: float | None
    size: int | None
    signature: str | None


class ScanStats(NamedTuple):
    total: int
    indexed: int
    removed: int


def track_scan_payload(
    path: str, st: os.stat_result, duration: float, signature: str
) -> dict[str, Any]:
    return {
        "path": path,
        "mtime": st.st_mtime,
        "size": st.st_size,
        "duration": duration,
        "embedding": signature,
    }


def multivector_payload(
    path: str, st: os.stat_result, duration: float, shares: list[float], signature: str
) -> dict[str, Any]:
    return {
        **track_scan_payload(path, st, duration, signature),
        "shares": shares,
    }


def chunk_payload(
    path: str,
    idx: int,
    total: int,
    start: float,
    end: float,
    db: float,
    silent: bool,
    speed: float | None,
) -> dict[str, Any]:
    return {
        "path": path,
        "idx": idx,
        "total": total,
        "start": round(start, 3),
        "end": round(end, 3),
        "db": round(db, 2),
        "silent": silent,
        "speed": None if speed is None else round(speed, 5),
    }


def chunk_track_payload(
    path: str,
    st: os.stat_result,
    duration: float,
    signature: str,
    n_chunks: int,
    silent: bool,
    intro_drift: float,
    outro_drift: float,
    trajectory_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        **track_scan_payload(path, st, duration, signature),
        "intro_drift": round(intro_drift, 6),
        "outro_drift": round(outro_drift, 6),
        "n_chunks": n_chunks,
        "silent": silent,
        **trajectory_summary,
    }

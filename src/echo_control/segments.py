from __future__ import annotations

import math
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
MIN_CHUNK_SECONDS = 3.0


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _db(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
    return 20 * math.log10(max(rms, 1e-10))


def _ffmpeg_stream(path: Path, n: int) -> Iterator[np.ndarray]:
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    try:
        while buf := proc.stdout.read(n * 4):
            yield np.frombuffer(buf, dtype=np.float32)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _librosa_full(path: Path) -> np.ndarray:
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return np.zeros(0, dtype=np.float32)
    return audio


def _select(chunks: Iterable[np.ndarray], stride: int) -> tuple[list[np.ndarray], int]:
    kept, samples = [], 0
    for idx, chunk in enumerate(chunks):
        samples += len(chunk)
        if idx % stride == 0 and len(chunk) >= MIN_CHUNK_SECONDS * SAMPLE_RATE:
            kept.append(chunk)
    return kept, samples


def dense_chunks(
    path: Path, window_seconds: float, max_chunks: int, silence_db: float
) -> tuple[list[np.ndarray], float]:
    """Consecutive fixed-length mono chunks covering the whole track, streamed from ffmpeg so
    long mixes never sit fully decoded in memory. Tracks longer than max_chunks windows are
    subsampled with an even stride. Near-silent chunks are dropped."""
    n = int(window_seconds * SAMPLE_RATE)
    duration = probe_duration(path)
    total = max(math.ceil(duration / window_seconds), 1) if duration else 0
    stride = max(math.ceil(total / max_chunks), 1) if total else 1

    chunks, samples = _select(_ffmpeg_stream(path, n), stride)
    if not chunks:
        # ffmpeg rejects some files libsndfile reads (e.g. WAV data behind an ID3 header).
        audio = _librosa_full(path)
        duration = len(audio) / SAMPLE_RATE
        stride = max(math.ceil(math.ceil(duration / window_seconds) / max_chunks), 1)
        chunks, samples = _select((audio[i : i + n] for i in range(0, len(audio), n)), stride)
    if not duration:
        duration = samples / SAMPLE_RATE
    if not chunks:
        return [], duration

    loudness = [_db(c) for c in chunks]
    keep = [c for c, db in zip(chunks, loudness) if db >= silence_db]
    if not keep:
        keep = [chunks[int(np.argmax(loudness))]]
    return keep, duration


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def spherical_kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Cluster unit vectors by cosine similarity; returns a label per row."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        dist = 1 - np.max(x @ np.array(centers).T, axis=1)
        p = np.clip(dist, 0, None)
        if p.sum() <= 0:
            break
        centers.append(x[rng.choice(len(x), p=p / p.sum())])
    c = np.array(centers)
    labels = np.full(len(x), -1)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(len(c)):
            members = x[labels == j]
            if len(members):
                c[j] = _normalize(members.mean(axis=0))
    return labels


def summarize(
    embs: np.ndarray,
    max_clusters: int,
    min_share: float,
    merge_threshold: float,
    chunks_per_cluster: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-chunk embeddings to a few normalized centroids plus the share of the track
    each one covers. Small clusters (intros, applause, hidden tracks) are dropped and
    near-duplicate centroids merged, so uniform tracks end up with a single vector."""
    embs = _normalize(np.asarray(embs, dtype=np.float32))
    k = max(1, min(max_clusters, math.ceil(len(embs) / chunks_per_cluster), len(embs)))
    labels = spherical_kmeans(embs, k) if k > 1 else np.zeros(len(embs), dtype=int)

    clusters = []
    for j in np.unique(labels):
        members = embs[labels == j]
        clusters.append([members.mean(axis=0), len(members) / len(embs)])
    clusters.sort(key=lambda c: -c[1])

    kept = [c for c in clusters if c[1] >= min_share] or clusters[:1]
    merged: list[list] = []
    for vec, share in kept:
        unit = _normalize(vec)
        for m in merged:
            if float(_normalize(m[0]) @ unit) >= merge_threshold:
                m[0] = m[0] * m[1] + vec * share
                m[1] += share
                m[0] /= m[1]
                break
        else:
            merged.append([vec, share])

    centroids = _normalize(np.array([m[0] for m in merged]))
    shares = np.array([m[1] for m in merged])
    return centroids, shares / shares.sum()

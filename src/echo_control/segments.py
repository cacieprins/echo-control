from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
MIN_CHUNK_SECONDS = 3.0
INTRO_START_FRAC = 0.10
OUTRO_END_FRAC = 0.85


def _window_starts(duration: float, windows: int, win: float) -> np.ndarray:
    # Skip intros/outros: sample evenly between 10% and 85% of the track.
    return np.linspace(duration * INTRO_START_FRAC, duration * OUTRO_END_FRAC - win, windows)


def _split_short(audio: np.ndarray, windows: int, win: float) -> list[np.ndarray]:
    n = int(win * SAMPLE_RATE)
    clips = [audio[i : i + n] for i in range(0, max(len(audio), 1), n)][:windows]
    return [c for c in clips if len(c) >= SAMPLE_RATE] or [audio]


def _ffmpeg_windows(path: Path, windows: int, win: float) -> tuple[list[np.ndarray], float]:
    """Fallback for files libsndfile rejects (broken MP3 headers, odd containers)."""
    out = subprocess.run(
        ffmpeg_decode_cmd(path, "f32le"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    ).stdout
    audio = np.frombuffer(out, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE
    if duration <= win * windows:
        return _split_short(audio, windows, win), duration
    n = int(win * SAMPLE_RATE)
    starts = _window_starts(duration, windows, win)
    return [audio[int(s * SAMPLE_RATE) : int(s * SAMPLE_RATE) + n] for s in starts], duration


def _librosa_windows(path: Path, windows: int, win: float) -> tuple[list[np.ndarray], float]:
    import librosa

    duration = float(librosa.get_duration(path=str(path)))
    if duration <= win * windows:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=win * windows)
        return _split_short(audio, windows, win), duration

    starts = _window_starts(duration, windows, win)
    clips: list[np.ndarray] = []
    for start in starts:
        audio, _ = librosa.load(
            str(path), sr=SAMPLE_RATE, mono=True, offset=float(start), duration=win
        )
        if len(audio):
            clips.append(audio)
    return clips, duration


def load_windows(path: Path, windows: int, window_seconds: float) -> tuple[list[np.ndarray], float]:
    """Decode a few fixed-length mono clips spread across the track, plus its duration."""
    try:
        return _librosa_windows(path, windows, window_seconds)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
        return _ffmpeg_windows(path, windows, window_seconds)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def rms_db(chunk: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
    return 20 * math.log10(max(rms, 1e-10))


def ffmpeg_decode_cmd(path: Path, fmt: str = "f32le") -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        fmt,
        "-",
    ]


class _DecodeFailed(RuntimeError):
    pass


def _ffmpeg_stream(path: Path, n: int) -> Iterator[np.ndarray]:
    proc = subprocess.Popen(
        ffmpeg_decode_cmd(path, "f32le"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    eof = False
    try:
        while buf := proc.stdout.read(n * 4):
            yield np.frombuffer(buf, dtype=np.float32)
        eof = True
    finally:
        proc.stdout.close()
        if proc.poll() is None and not eof:
            proc.kill()
        rc = proc.wait()
        if eof and rc != 0:
            raise _DecodeFailed(f"ffmpeg exited with {rc}")


def _librosa_full(path: Path) -> np.ndarray:
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype=np.float32)
    return audio


def _select(chunks: Iterable[np.ndarray], stride: int) -> tuple[list[np.ndarray], int]:
    kept: list[np.ndarray] = []
    samples = 0
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

    try:
        chunks, samples = _select(_ffmpeg_stream(path, n), stride)
    except _DecodeFailed:
        chunks, samples = [], 0
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

    loudness = [rms_db(c) for c in chunks]
    keep = [c for c, db in zip(chunks, loudness) if db >= silence_db]
    if not keep:
        keep = [chunks[int(np.argmax(loudness))]]
    return keep, duration


@dataclass
class Windows:
    """Overlapping fixed-length windows over one decoded track. The audio is held once as
    int16 and windows are sliced from it on demand, so a long mix isn't copied per window."""

    audio: np.ndarray
    starts: list[int]
    size: int
    db: list[float]
    silent: list[bool]

    def __len__(self) -> int:
        return len(self.starts)

    def clip(self, i: int) -> np.ndarray:
        s = self.starts[i]
        return self.audio[s : s + self.size].astype(np.float32) / 32768.0

    def span(self, i: int) -> tuple[float, float]:
        s = self.starts[i]
        return s / SAMPLE_RATE, min(s + self.size, len(self.audio)) / SAMPLE_RATE


def window_starts(length: int, size: int, hop: int) -> list[int]:
    """Start offsets of size-long windows every hop samples. A final shorter window covers
    whatever audio the last full one missed; tracks shorter than one window get one."""
    if length <= 0:
        return []
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, hop))
    if starts[-1] + size < length:
        starts.append(starts[-1] + hop)
    return starts


def to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def _ffmpeg_s16(path: Path) -> np.ndarray:
    proc = subprocess.Popen(
        ffmpeg_decode_cmd(path, "s16le"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    blocks: list[np.ndarray] = []
    eof = False
    try:
        while buf := proc.stdout.read(1 << 22):
            blocks.append(np.frombuffer(buf, dtype=np.int16))
        eof = True
    finally:
        proc.stdout.close()
        if proc.poll() is None and not eof:
            proc.kill()
        rc = proc.wait()
    if not eof or rc != 0 or not blocks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(blocks)


def sliding_chunks(
    path: Path, window_seconds: float, hop_seconds: float, silence_db: float
) -> tuple[Windows, float]:
    """Windows of window_seconds every hop_seconds across the whole track. Near-silent
    windows are kept and flagged rather than dropped, so positions stay evenly spaced."""
    audio = _ffmpeg_s16(path)
    if not len(audio):
        # ffmpeg rejects some files libsndfile reads (e.g. WAV data behind an ID3 header).
        audio = to_int16(_librosa_full(path))
    size = int(window_seconds * SAMPLE_RATE)
    starts = window_starts(len(audio), size, int(hop_seconds * SAMPLE_RATE))
    win = Windows(audio, starts, size, [], [])
    win.db = [rms_db(win.clip(i)) for i in range(len(starts))]
    win.silent = [d < silence_db for d in win.db]
    return win, len(audio) / SAMPLE_RATE


def normalize(v: np.ndarray) -> np.ndarray:
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
                c[j] = normalize(members.mean(axis=0))
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
    embs = normalize(np.asarray(embs, dtype=np.float32))
    k = max(1, min(max_clusters, math.ceil(len(embs) / chunks_per_cluster), len(embs)))
    labels = spherical_kmeans(embs, k) if k > 1 else np.zeros(len(embs), dtype=int)

    clusters: list[tuple[np.ndarray, float]] = []
    for j in np.unique(labels):
        members = embs[labels == j]
        clusters.append((members.mean(axis=0), len(members) / len(embs)))
    clusters.sort(key=lambda c: -c[1])

    kept = [c for c in clusters if c[1] >= min_share] or clusters[:1]
    merged: list[tuple[np.ndarray, float]] = []
    for vec, share in kept:
        unit = normalize(vec)
        for i, (m_vec, m_share) in enumerate(merged):
            if float(normalize(m_vec) @ unit) >= merge_threshold:
                total = m_share + share
                merged[i] = ((m_vec * m_share + vec * share) / total, total)
                break
        else:
            merged.append((vec, share))

    centroids = normalize(np.array([m[0] for m in merged]))
    shares = np.array([m[1] for m in merged])
    return centroids, shares / shares.sum()

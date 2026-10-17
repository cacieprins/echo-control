import numpy as np
import pytest

from echo_control.segments import SAMPLE_RATE, Windows, rms_db, to_int16, window_starts


def test_window_starts_cover_track_with_one_tail():
    # 23 samples, windows of 10 every 3: full windows at 0..12, then a tail from 15.
    assert window_starts(23, 10, 3) == [0, 3, 6, 9, 12, 15]
    assert window_starts(22, 10, 3) == [0, 3, 6, 9, 12]
    assert window_starts(5, 10, 3) == [0]
    assert window_starts(0, 10, 3) == []


def test_every_sample_is_covered():
    for length in range(1, 60):
        starts = window_starts(length, 10, 3)
        covered = np.zeros(length, dtype=bool)
        for s in starts:
            covered[s : s + 10] = True
        assert covered.all(), length


def test_windows_flag_silence_and_report_spans():
    sr = SAMPLE_RATE
    loud = 0.5 * np.sin(np.linspace(0, 2000 * np.pi, 10 * sr))
    audio = to_int16(np.concatenate([loud, np.zeros(10 * sr)]))
    starts = window_starts(len(audio), 5 * sr, 5 * sr)
    win = Windows(audio, starts, 5 * sr, [], [])
    win.db = [rms_db(win.clip(i)) for i in range(len(win))]
    win.silent = [d < -45 for d in win.db]
    assert win.silent == [False, False, True, True]
    assert win.span(1) == (5.0, 10.0)
    assert win.clip(0).dtype == np.float32


def test_ffmpeg_midstream_failure_discards_partial_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import subprocess
    from pathlib import Path
    from typing import Any

    from echo_control import segments

    # Simulate ffmpeg outputting 5 seconds of valid audio and then exiting with code 1.
    partial_f32 = (0.5 * np.ones(5 * SAMPLE_RATE, dtype=np.float32)).tobytes()
    partial_s16 = (10000 * np.ones(5 * SAMPLE_RATE, dtype=np.int16)).tobytes()

    class FailingProc:
        stdout: io.BytesIO
        killed: bool

        def __init__(self, cmd: list[str], **_kw: Any) -> None:
            buf = partial_f32 if "f32le" in cmd else partial_s16
            self.stdout = io.BytesIO(buf)
            self.killed = False

        def poll(self) -> int:
            return 1

        def kill(self) -> None:
            self.killed = True

        def wait(self) -> int:
            return 1

    def fake_probe(_p: Path) -> float:
        return 20.0

    fallback_audio = 0.25 * np.ones(6 * SAMPLE_RATE, dtype=np.float32)

    def fake_librosa(_p: Path) -> np.ndarray:
        return fallback_audio

    monkeypatch.setattr(subprocess, "Popen", FailingProc)
    monkeypatch.setattr(segments, "probe_duration", fake_probe)
    monkeypatch.setattr(segments, "_librosa_full", fake_librosa)

    chunks, dur = segments.dense_chunks(Path("/corrupt.mp3"), 5.0, 10, -45.0)
    assert len(chunks) == 1
    assert np.allclose(chunks[0], 0.25)
    assert dur == 6.0

    win, w_dur = segments.sliding_chunks(Path("/corrupt.mp3"), 5.0, 2.5, -45.0)
    assert w_dur == 6.0
    assert len(win) == 2


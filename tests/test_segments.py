import numpy as np

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

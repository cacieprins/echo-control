import numpy as np

from echo_control import trajectory


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _track(sections, per_section, noise=0.02, dim=64, seed=0):
    """Chunk vectors that jitter around one direction per section."""
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(sections, dim)))
    vecs = np.concatenate([
        centers[s] + noise * rng.normal(size=(per_section, dim)) for s in range(sections)
    ])
    return _unit(vecs), np.arange(len(vecs)) * 2.5


def test_speeds_are_angle_per_second():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    speed = trajectory.speeds(vecs, np.array([0.0, 2.0, 4.0]))
    assert np.isnan(speed[0])
    assert np.allclose(speed[1:], [np.pi / 4, 0.0])


def test_steady_track_has_no_spikes():
    vecs, starts = _track(1, 80)
    out = trajectory.summarize(vecs, starts, 10.0)
    assert out["spikes"] == 0
    assert out["intro_len"] is None


def test_section_changes_are_spikes_at_the_boundaries():
    vecs, starts = _track(3, 40)
    out = trajectory.summarize(vecs, starts, 10.0)
    assert out["spikes"] == 2
    # Boundaries between chunks 39/40 and 79/80, i.e. starts 97.5-100 and 197.5-200 s.
    for t, boundary in zip(out["spike_times"], (100.0, 200.0), strict=True):
        assert abs(t - boundary) <= 10.0


def test_gradual_transition_is_one_spike():
    # Overlapping windows blend a boundary across ~window/hop chunks.
    rng = np.random.default_rng(1)
    a, b = _unit(rng.normal(size=(2, 64)))
    mix = np.concatenate([np.zeros(40), np.linspace(0, 1, 6)[1:-1], np.ones(40)])
    vecs = _unit((1 - mix)[:, None] * a + mix[:, None] * b + 0.02 * rng.normal(size=(len(mix), 64)))
    out = trajectory.summarize(vecs, np.arange(len(vecs)) * 2.5, 10.0)
    assert out["spikes"] == 1
    assert abs(out["spike_times"][0] - (42 * 2.5 + 5)) <= 5.0


def test_edges_skip_silence_and_fades_and_find_drift():
    rng = np.random.default_rng(2)
    a, b, fade = _unit(rng.normal(size=(3, 64)))
    n = 60
    # Drifts steadily from a toward b, then fades out over the last 8 windows and goes silent.
    mix = np.linspace(0, 1, n)
    vecs = _unit((1 - mix)[:, None] * a + mix[:, None] * b)
    vecs[-8:] = fade
    starts = np.arange(n) * 2.5
    db = [-10.0] * (n - 8) + [-30.0] * 5 + [-60.0] * 3
    silent = [d < -45 for d in db]
    intro, outro = trajectory.edges(vecs, starts, silent, db, 30.0)
    assert outro.position @ fade < 0.5
    assert outro.position @ vecs[n - 9] > 0.9
    # Moving toward b: the direction points at b's side of the tangent plane.
    assert outro.direction @ b > 0 and intro.direction @ b > 0
    assert intro.drift > 0 and outro.drift > 0


def test_edges_of_a_single_window():
    v = _unit(np.ones((1, 8)))
    intro, outro = trajectory.edges(v, np.array([0.0]), [False], [-10.0], 30.0)
    assert intro.direction is None and outro.drift == 0.0
    assert np.allclose(intro.position, v[0])


def test_aligned_similarity_finds_offset_copies_and_prefixes():
    vecs, _ = _track(4, 10, noise=0.05)
    assert trajectory.aligned_similarity(vecs, vecs) > 0.999
    # Same recording with 5 extra windows of lead-in, and a truncated copy.
    rng = np.random.default_rng(3)
    lead_in = _unit(rng.normal(size=(5, vecs.shape[1])))
    assert trajectory.aligned_similarity(np.concatenate([lead_in, vecs]), vecs) > 0.999
    assert trajectory.aligned_similarity(vecs[:12], vecs) > 0.999
    # The same windows shuffled: same average, but they don't line up.
    shuffled = vecs[rng.permutation(len(vecs))]
    assert trajectory.aligned_similarity(shuffled, vecs) < 0.6


def test_wandering_path_is_more_tortuous_than_a_straight_one():
    straight = _unit(np.stack([np.cos(np.linspace(0, 1, 50)), np.sin(np.linspace(0, 1, 50))], 1))
    starts = np.arange(50) * 2.5
    wander, _ = _track(1, 50, noise=0.3)
    assert trajectory.summarize(straight, starts, 10.0)["tortuosity"] < 1.01
    t = trajectory.summarize(wander, starts, 10.0)["tortuosity"]
    assert t is None or t > 5

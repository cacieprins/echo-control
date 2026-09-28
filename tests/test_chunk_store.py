import uuid

import numpy as np
import pytest

from echo_control.config import QdrantConfig
from echo_control.store import ChunkStore, is_ready

URL = "http://localhost:6333"

pytestmark = pytest.mark.skipif(not is_ready(URL), reason="no Qdrant on localhost:6333")


@pytest.fixture
def store():
    s = ChunkStore(QdrantConfig(url=URL, collection=f"test_{uuid.uuid4().hex[:8]}"))
    s.ensure(8)
    yield s
    s.drop()


def _put(store, path, vecs, silent=None, intro=None, outro=None, payload=None):
    silent = silent or [False] * len(vecs)
    payloads = [
        {"path": path, "idx": i, "start": i * 2.5, "silent": s} for i, s in enumerate(silent)
    ]
    mean = vecs.mean(axis=0)
    mean = (mean / np.linalg.norm(mean)).tolist()
    track_vectors = {"mean": mean, "intro": intro or mean, "outro": outro or mean}
    store.replace_track(
        path, vecs, payloads, track_vectors, {"path": path, "n": len(vecs), **(payload or {})}
    )


def test_wander_follows_outros_into_intros(store):
    from echo_control.config import WanderConfig
    from echo_control.wander import wander

    eye = np.eye(8, dtype=np.float32)
    # a ends on 1; b starts on 1 and ends on 2; c starts on 2; d starts on 3.
    _put(store, "/a", eye[[4]], intro=eye[0].tolist(), outro=eye[1].tolist())
    _put(store, "/b", eye[[5]], intro=eye[1].tolist(), outro=eye[2].tolist())
    _put(store, "/c", eye[[6]], intro=eye[2].tolist(), outro=eye[0].tolist())
    _put(store, "/d", eye[[7]], intro=eye[3].tolist(), outro=eye[3].tolist())
    chain = wander(store, "/a", 4, WanderConfig())
    assert [t.path for t in chain] == ["/a", "/b", "/c", "/d"]
    assert chain[1].position > 0.99


def test_wander_skips_duplicates_and_silent_tracks(store):
    from echo_control.config import WanderConfig
    from echo_control.wander import wander

    eye = np.eye(8, dtype=np.float32)
    song = eye[[2, 3, 4, 5, 6, 7]]
    e1 = eye[1].tolist()
    _put(store, "/a", song, intro=e1, outro=e1)
    # Each of these continues /a perfectly, but is another file of it or silent.
    _put(store, "/a (1)", song, intro=e1, outro=e1)
    _put(store, "/a (partial)", song[:3], intro=e1, outro=e1)
    _put(store, "/quiet", eye[[0]], intro=e1, outro=e1, payload={"silent": True})
    # The same windows in another order: identical on average, but not the same recording.
    _put(store, "/same-album", song[::-1], intro=(0.9 * eye[1] + 0.44 * eye[2]).tolist(),
         outro=eye[2].tolist())
    _put(store, "/b", eye[[0, 1]], intro=eye[2].tolist(), outro=eye[3].tolist())
    chain = wander(store, "/a", 3, WanderConfig())
    assert [t.path for t in chain] == ["/a", "/same-album", "/b"]


def test_search_groups_by_track_and_skips_silence(store):
    eye = np.eye(8, dtype=np.float32)
    _put(store, "/a", eye[[0, 1, 2]])
    _put(store, "/b", eye[[3, 4]], silent=[False, True])
    hits = store.search(eye[4].tolist(), 10)
    assert not any(h.path == "/b" and h.score > 0.5 for h in hits)
    hits = store.search(eye[2].tolist(), 10)
    assert hits[0].path == "/a" and hits[0].start == 5.0
    assert len({h.path for h in hits}) == len(hits)


def test_orbit_pulls_back_toward_the_center(store):
    from echo_control.config import WanderConfig
    from echo_control.wander import Orbit, wander

    eye = np.eye(8, dtype=np.float32)
    center = eye[5]
    _put(store, "/a", eye[[0]], intro=eye[0].tolist(), outro=eye[1].tolist())
    # /b continues /a's outro but sounds nothing like the center; /c is the reverse.
    _put(store, "/b", eye[[3]], intro=eye[1].tolist(), outro=eye[3].tolist())
    _put(store, "/c", eye[[5]], intro=eye[2].tolist(), outro=eye[2].tolist())
    # With one intro candidate per step, /c is only reachable through the center search.
    cfg = WanderConfig(candidates=1)
    assert [t.path for t in wander(store, "/a", 2, cfg)] == ["/a", "/b"]
    chain = wander(store, "/a", 2, cfg, Orbit(center.tolist(), 2.0))
    assert [t.path for t in chain] == ["/a", "/c"]
    assert chain[1].center > 0.99


def test_sustained_match_beats_one_lucky_window(store):
    q = np.eye(8, dtype=np.float32)[0]
    other = np.eye(8, dtype=np.float32)[1]
    near = (0.8 * q + 0.6 * other)[None]
    _put(store, "/spike", np.stack([q, other, other, other]))
    _put(store, "/steady", np.repeat(near, 4, axis=0))
    store.sustain_windows = 1
    assert store.search(q.tolist(), 2)[0].path == "/spike"
    store.sustain_windows = 4
    hits = store.search(q.tolist(), 2)
    assert hits[0].path == "/steady"
    assert abs(hits[0].score - 0.8) < 0.02 and abs(hits[1].score - 0.25) < 0.02


def test_replace_shrinks_and_delete_removes(store):
    eye = np.eye(8, dtype=np.float32)
    _put(store, "/a", eye[[0, 1, 2]])
    _put(store, "/a", eye[[5]])
    assert store.client.count(store.chunks, exact=True).count == 1
    assert [p["n"] for p in store.payloads(["/a", "/missing"])] == [1]
    store.delete_paths(["/a"])
    assert store.count() == 0
    assert store.client.count(store.chunks, exact=True).count == 0

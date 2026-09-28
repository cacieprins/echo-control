# echo-control

Search a local music library by vibe. A daemon embeds every track with
[CLAP](https://huggingface.co/laion/larger_clap_music_and_speech) and stores the vectors in
Qdrant; `vibe` embeds a text prompt and plays the closest matches.

```bash
vibe "angsty lesbian music from the late 20th century"
vibe "sad synth" -n 40 | mpv --playlist=-
```

Tags are never read; matching is purely on how the audio sounds.

## Setup

```bash
uv tool install --editable .          # installs `vibe` and `echo-control` into ~/.local/bin
mkdir -p ~/.config/echo-control
cp contrib/config.example.toml ~/.config/echo-control/config.toml   # edit library paths
echo-control install-service          # docker-compose + systemd user units
systemctl --user enable --now echo-control
echo-control status
```

Requires Docker (system `docker.service` enabled, user in the `docker` group).
The first scan is slow on CPU; watch it with `journalctl --user -fu echo-control`.

## Commands

- `vibe "<query>"` plays results with the configured backend. Options: `-n/--limit`,
  `--player {mpv,spotatui,command,stdout}`, `--shuffle`, `--print` (M3U to stdout), `--json`.
  When stdout is not a terminal, an M3U is printed instead of playing.
- `vibe --wander "<query>"` / `vibe --wander --from FILE` starts at the best match (or FILE)
  and chains tracks whose intro picks up where the previous outro left off. Needs the
  `chunks` strategy.
- `vibe --orbit[=N] "<query>"` is `--wander` with a pull back toward the query (or, with
  `--from FILE` and no query, toward that track). N is how tight the orbit is: 1 by
  default, 0 is plain `--wander`, higher stays closer to the vibe.
- `echo-control daemon` runs the indexer/watcher and the text-embedding socket.
- `echo-control scan` runs a one-off index pass without the daemon.
- `echo-control reindex` drops the current strategy's index and rebuilds it.
- `echo-control status` shows daemon, Qdrant and index state.
- `echo-control install-service` / `install-noctalia` install integrations.

## Strategies

`[model] strategy` decides how a track becomes vectors. Changing it, or any setting it
uses, re-embeds the library on the next scan. `clusters` and `mean` share `<collection>`;
`chunks` has its own collections, so switching to and from it leaves the other index intact.

| strategy | stored per track | good for |
| --- | --- | --- |
| `clusters` (default) | up to `max_clusters` centroids of its 10s chunks | best-section search |
| `mean` | one vector averaged from `windows` clips | fast indexing, blurrier matches |
| `chunks` | every window, with its position | positions, trajectories, `--wander` |

Settings shared by all strategies: `window_seconds` (10), and `silence_db` (-45: clips
quieter than this are dropped, or only flagged under `chunks`).

**`clusters`** embeds consecutive `window_seconds` chunks across the track and k-means
them into centroids, stored as a Qdrant multivector in `<collection>`. A query matches a
track's best centroid.

- `max_chunks` (60): longer tracks are subsampled evenly to this many chunks.
- `max_clusters` (4), `min_cluster_share` (0.12): drop sections covering less of the
  track than this, such as intros and applause.
- `merge_threshold` (0.85): merge centroids at least this similar, so uniform tracks end
  up with one.
- `[search] share_weight` (0): re-rank by `similarity * share ** share_weight`.

**`mean`** averages `windows` (3) clips taken evenly between 10% and 85% of the track into
a single vector, in `<collection>`.

**`chunks`** stores a `window_seconds` window every `hop_seconds` (2.5) as its own point in
`<collection>_chunks`, with no cap. Expect about 4× the embedding work of `clusters` for
typical tracks, and far more for long mixes. Silent windows are kept and flagged so
positions stay evenly spaced. Search ranks each track by the mean similarity of its best
`[search] sustain_seconds` (60) of non-silent windows, so a track has to hold the vibe
rather than match it for one moment. 10 means the single best window; tracks shorter than
the setting are averaged over their whole length.

- Each chunk's payload has `idx`/`total`, `start`/`end` (seconds), `db`, `silent`, and
  `speed`: the angle travelled through embedding space since the previous window, per second.
- Each track in `<collection>_tracks` holds:
  - the scan state;
  - vectors for the track's `mean` and its `intro`/`outro` positions and directions;
  - `intro_drift`/`outro_drift`;
  - trajectory features: `speed_mean`, `speed_std`, `spikes` and `spike_times` (likely
    section changes), `tortuosity`, `intro_len`.
- `edge_seconds` (30) sets how much of each end counts as the intro and outro. Leading and
  trailing silence and fades are skipped.

`[wander]` weights `vibe --wander`'s choice of next track:
`position * cos(outro, intro) + direction * cos(outro heading, intro heading)
+ drift * -|log(outro drift / intro drift)|`, with defaults `position = 1.0`,
`direction = 0.5`, `drift = 0.25`. `candidates` (100) is how many of the nearest intros
are rescored at each step. `--orbit=N` adds `N * similarity to the center`, where the
similarity is the same sustained score search uses. The `candidates` tracks closest to the
center are considered too, so the walk can return even after every nearby intro has
drifted away.

Walks never pick a track with no audible windows, or another file of a track already in
the walk. A duplicate is one whose `mean` vector is at least `duplicate_similarity` (0.99)
similar and whose duration is within `duplicate_seconds` (3) of it. Lowering the
similarity to about 0.975 also catches remasters and compilation copies, but starts to
catch similar same-length tracks from the same album.

## Noctalia

`echo-control install-noctalia` registers `contrib/noctalia` as a local plugin source
and enables `case/vibe`. In the launcher, type `/vibe <query>`.

## GPU

torch comes from PyTorch's ROCm 7.2 index (see `[tool.uv.sources]` in `pyproject.toml`),
so AMD GPUs are used automatically with `device = "auto"` (ROCm devices report as `cuda`).
`triton-rocm` is listed as a direct dependency because uv only applies custom indexes to
direct dependencies. For CPU-only machines, point the index at
`https://download.pytorch.org/whl/cpu` and drop `triton-rocm`.

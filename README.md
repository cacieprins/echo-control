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
cp config.example.toml ~/.config/echo-control/config.toml   # edit library paths
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
- `echo-control daemon` runs the indexer/watcher and the text-embedding socket.
- `echo-control scan` runs a one-off index pass without the daemon.
- `echo-control reindex` drops the collection and rebuilds it.
- `echo-control status` shows daemon, Qdrant and index state.
- `echo-control install-service` / `install-noctalia` install integrations.

## Noctalia

`echo-control install-noctalia` registers `contrib/noctalia` as a local plugin source
and enables `case/vibe`. In the launcher, type `/vibe <query>`.

## GPU

torch comes from PyTorch's ROCm 7.2 index (see `[tool.uv.sources]` in `pyproject.toml`),
so AMD GPUs are used automatically with `device = "auto"` (ROCm devices report as `cuda`).
`triton-rocm` is listed as a direct dependency because uv only applies custom indexes to
direct dependencies. For CPU-only machines, point the index at
`https://download.pytorch.org/whl/cpu` and drop `triton-rocm`.

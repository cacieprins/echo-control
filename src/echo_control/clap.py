from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import ModelConfig
from .segments import (
    SAMPLE_RATE,
    Windows,
    dense_chunks,
    ffmpeg_decode_cmd,
    normalize,
    sliding_chunks,
    summarize,
)

log = logging.getLogger(__name__)

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


def load_windows(path: Path, windows: int, window_seconds: float) -> tuple[list[np.ndarray], float]:
    """Decode a few fixed-length mono clips spread across the track, plus its duration."""
    try:
        return _librosa_windows(path, windows, window_seconds)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
        return _ffmpeg_windows(path, windows, window_seconds)


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


def resolve_devices(setting: str) -> list[str]:
    """Map the device setting to torch devices: auto means every visible GPU (ROCm reports
    as cuda) or else the CPU; a comma-separated list like "cuda:0,cuda:1" is used as given."""
    import torch

    if setting == "auto":
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return [f"cuda:{i}" for i in range(n)] or ["cpu"]
    return [d.strip() for d in setting.split(",") if d.strip()]


def load_encoders(cfg: ModelConfig) -> list[Clap]:
    """One model replica per device."""
    return [Clap(cfg, device) for device in resolve_devices(cfg.device)]


def signature(c: ModelConfig) -> str:
    """Identifies the embedding settings; tracks indexed under another one get redone."""
    if c.strategy == "mean":
        return f"{c.name}|mean|{c.windows}x{c.window_seconds:g}"
    if c.strategy == "chunks":
        return (
            f"{c.name}|chunks|{c.window_seconds:g}|{c.hop_seconds:g}|{c.silence_db:g}"
            f"|{c.edge_seconds:g}"
        )
    return (
        f"{c.name}|clusters|{c.window_seconds:g}|{c.max_chunks}|{c.silence_db:g}"
        f"|{c.max_clusters}|{c.min_cluster_share:g}|{c.merge_threshold:g}"
    )


def prompt(cfg: ModelConfig, text: str) -> str:
    # CLAP's tokenizer is case-sensitive and "a recording of" embeds far from
    # "A recording of", so a user-typed prefix is replaced with the configured one.
    prefix = cfg.text_prefix
    text = text.strip()
    if not prefix:
        return text
    head = prefix.strip()
    if text.lower().startswith(head.lower()):
        text = text[len(head) :].lstrip()
    return prefix + text


class TextEncoder:
    """CLAP's text tower alone, on the CPU. Touching the GPU starts a ROCm runtime
    thread that busy-spins for the life of the process, so query serving stays off it."""

    def __init__(self, cfg: ModelConfig):
        import torch
        import transformers
        from transformers import ClapProcessor, ClapTextModelWithProjection

        self.cfg: ModelConfig = cfg
        self.torch: Any = torch
        verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()  # the unused audio weights get a report
        try:
            self.processor: ClapProcessor = ClapProcessor.from_pretrained(cfg.name)  # pyright: ignore[reportUnknownMemberType]
            self.model: ClapTextModelWithProjection = (
                ClapTextModelWithProjection.from_pretrained(cfg.name).eval()  # pyright: ignore[reportUnknownMemberType]
            )
        finally:
            transformers.logging.set_verbosity(verbosity)
        self.dim: int = int(self.model.config.projection_dim)

    def embed_text(self, text: str) -> list[float]:
        inputs = cast(Mapping[str, Any], self.processor(
            text=[prompt(self.cfg, text)], return_tensors="pt", padding=True  # pyright: ignore[reportCallIssue]
        ))
        with self.torch.inference_mode():
            emb = self.model(**inputs).text_embeds
        return normalize(emb.float().numpy())[0].tolist()


class Clap:
    def __init__(self, cfg: ModelConfig, device: str | None = None):
        import threading

        import torch
        from transformers import ClapModel, ClapProcessor

        self.cfg: ModelConfig = cfg
        self.device: str = device or resolve_devices(cfg.device)[0]
        self.lock: threading.Lock = threading.Lock()
        log.info("loading %s on %s", cfg.name, self.device)
        self.torch: Any = torch
        self.processor: ClapProcessor = ClapProcessor.from_pretrained(cfg.name)  # pyright: ignore[reportUnknownMemberType]
        self.model: ClapModel = ClapModel.from_pretrained(cfg.name).to(self.device).eval()  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
        self.dim: int = int(self.model.config.projection_dim)
        self._mel_fb: Any = None
        self._window: Any = None

    @staticmethod
    def _features(out: Any) -> np.ndarray:
        # Some transformers versions wrap projected embeddings in a model output object.
        tensor = out if hasattr(out, "cpu") else out.pooler_output
        return tensor.detach().float().cpu().numpy()

    def _gpu_features_supported(self) -> bool:
        fe = self.processor.feature_extractor
        return fe.truncation == "rand_trunc" and fe.padding == "repeatpad"

    def _log_mel(self, clips: list[np.ndarray]) -> dict[str, Any]:
        """Same features as ClapFeatureExtractor (rand_trunc + repeatpad), computed with
        torch.stft on the model's device instead of numpy on the CPU."""
        torch = self.torch
        fe = self.processor.feature_extractor
        max_len = fe.nb_max_samples
        batch = np.zeros((len(clips), max_len), dtype=np.float32)
        for i, clip in enumerate(clips):
            clip = clip[:max_len]
            if 0 < len(clip) < max_len:
                clip = np.tile(clip, max_len // len(clip))
            batch[i, : len(clip)] = clip
        if self._mel_fb is None:
            self._mel_fb = torch.from_numpy(fe.mel_filters_slaney.astype(np.float32)).to(self.device)
            self._window = torch.hann_window(fe.fft_window_size, periodic=True, device=self.device)
        x = torch.from_numpy(batch).to(self.device)
        spec = torch.stft(
            x,
            n_fft=fe.fft_window_size,
            hop_length=fe.hop_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs().pow(2)
        mel = torch.einsum("fm,bft->btm", self._mel_fb, spec)
        db = 10 * torch.log10(mel.clamp(min=1e-10))
        return {
            "input_features": db[:, None],
            "is_longer": torch.zeros((len(clips), 1), dtype=torch.bool, device=self.device),
        }

    def embed_clips(self, clips: list[np.ndarray]) -> np.ndarray:
        out: list[np.ndarray] = []
        bs = max(self.cfg.batch_size, 1)
        gpu_features = self._gpu_features_supported()
        # fp16 overflows to NaN in this model; bf16 matches fp32 to cosine > 0.999.
        autocast = self.torch.autocast(
            "cuda",
            dtype=self.torch.bfloat16,
            enabled=self.device.startswith("cuda") and self.cfg.precision == "bf16",
        )
        for i in range(0, len(clips), bs):
            batch = clips[i : i + bs]
            with self.torch.inference_mode(), autocast:
                if gpu_features:
                    inputs = self._log_mel(batch)
                else:
                    inputs = cast(Mapping[str, Any], self.processor(  # pyright: ignore[reportUnknownMemberType]
                        audio=batch, sampling_rate=SAMPLE_RATE, return_tensors="pt"  # pyright: ignore[reportCallIssue]
                    ).to(self.device))
                emb = cast(object, self.model.get_audio_features(**inputs))  # pyright: ignore[reportUnknownMemberType]
            out.append(self._features(emb))
        return normalize(np.concatenate(out))

    @property
    def signature(self) -> str:
        return signature(self.cfg)

    def decode(self, path: Path) -> tuple[list[np.ndarray] | Windows, float]:
        c = self.cfg
        if c.strategy == "mean":
            return load_windows(path, c.windows, c.window_seconds)
        if c.strategy == "chunks":
            return sliding_chunks(path, c.window_seconds, c.hop_seconds, c.silence_db)
        return dense_chunks(path, c.window_seconds, c.max_chunks, c.silence_db)

    def embed_windows(self, windows: Windows) -> np.ndarray:
        """One unit vector per window, in order. Windows are converted a batch at a time."""
        if not len(windows):
            raise ValueError("no decodable audio")
        bs = max(self.cfg.batch_size, 1) * 16
        return np.concatenate([
            self.embed_clips([windows.clip(j) for j in range(i, min(i + bs, len(windows)))])
            for i in range(0, len(windows), bs)
        ])

    def embed_decoded(self, clips: list[np.ndarray]) -> tuple[list[list[float]], list[float]]:
        """Returns the track's vectors and the share of the track each one represents."""
        if not clips:
            raise ValueError("no decodable audio")
        embs = self.embed_clips(clips)
        c = self.cfg
        if c.strategy == "mean":
            return [normalize(embs.mean(axis=0)).tolist()], [1.0]
        centroids, shares = summarize(
            embs, c.max_clusters, c.min_cluster_share, c.merge_threshold
        )
        return centroids.tolist(), [round(float(s), 4) for s in shares]

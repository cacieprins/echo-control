from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import ModelConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 48_000


def _window_starts(duration: float, windows: int, win: float) -> np.ndarray:
    # Skip intros/outros: sample evenly between 10% and 85% of the track.
    return np.linspace(duration * 0.10, duration * 0.85 - win, windows)


def _split_short(audio: np.ndarray, windows: int, win: float) -> list[np.ndarray]:
    n = int(win * SAMPLE_RATE)
    clips = [audio[i : i + n] for i in range(0, max(len(audio), 1), n)][:windows]
    return [c for c in clips if len(c) >= SAMPLE_RATE] or [audio]


def _ffmpeg_windows(path: Path, windows: int, win: float) -> tuple[list[np.ndarray], float]:
    """Fallback for files libsndfile rejects (broken MP3 headers, odd containers)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "f32le", "-"],
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
    clips = []
    for start in starts:
        audio, _ = librosa.load(
            str(path), sr=SAMPLE_RATE, mono=True, offset=float(start), duration=win
        )
        if len(audio):
            clips.append(audio)
    return clips, duration


class Clap:
    def __init__(self, cfg: ModelConfig):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.cfg = cfg
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        log.info("loading %s on %s", cfg.name, self.device)
        self.torch = torch
        self.processor = ClapProcessor.from_pretrained(cfg.name)
        self.model = ClapModel.from_pretrained(cfg.name).to(self.device).eval()
        self.dim = int(self.model.config.projection_dim)

    @staticmethod
    def _features(out) -> np.ndarray:
        # transformers >= 5 wraps projected embeddings in a model output
        tensor = out if hasattr(out, "cpu") else out.pooler_output
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    def prompt(self, text: str) -> str:
        # CLAP's tokenizer is case-sensitive and "a recording of" embeds far from
        # "A recording of", so a user-typed prefix is replaced with the configured one.
        prefix = self.cfg.text_prefix
        text = text.strip()
        if not prefix:
            return text
        head = prefix.strip()
        if text.lower().startswith(head.lower()):
            text = text[len(head) :].lstrip()
        return prefix + text

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(
            text=[self.prompt(text)], return_tensors="pt", padding=True
        ).to(self.device)
        with self.torch.inference_mode():
            emb = self.model.get_text_features(**inputs)
        return self._normalize(self._features(emb))[0].tolist()

    def embed_clips(self, clips: list[np.ndarray]) -> np.ndarray:
        out = []
        bs = max(self.cfg.batch_size, 1)
        for i in range(0, len(clips), bs):
            batch = clips[i : i + bs]
            inputs = self.processor(
                audio=batch, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).to(self.device)
            with self.torch.inference_mode():
                emb = self.model.get_audio_features(**inputs)
            out.append(self._features(emb))
        return self._normalize(np.concatenate(out))

    def decode(self, path: Path) -> tuple[list[np.ndarray], float]:
        return load_windows(path, self.cfg.windows, self.cfg.window_seconds)

    def embed_decoded(self, clips: list[np.ndarray]) -> list[float]:
        if not clips:
            raise ValueError("no decodable audio")
        return self._normalize(self.embed_clips(clips).mean(axis=0)).tolist()

    def embed_audio(self, path: Path) -> tuple[list[float], float]:
        clips, duration = self.decode(path)
        return self.embed_decoded(clips), duration

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import ModelConfig
from .segments import (
    INTRO_START_FRAC,
    OUTRO_END_FRAC,
    SAMPLE_RATE,
    Windows,
    ffmpeg_decode_cmd,
    load_clips,
    load_windows,
    normalize,
)
from .strategies import get_strategy

log = logging.getLogger(__name__)

__all__ = [
    "INTRO_START_FRAC",
    "OUTRO_END_FRAC",
    "Clap",
    "TextEncoder",
    "ffmpeg_decode_cmd",
    "load_clips",
    "load_encoders",
    "load_windows",
    "prompt",
    "resolve_devices",
    "signature",
]


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
    return get_strategy(c).signature(c)


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
        self._lock: threading.RLock = threading.RLock()
        log.info("loading %s on %s", cfg.name, self.device)
        self.torch: Any = torch
        self.processor: ClapProcessor = ClapProcessor.from_pretrained(cfg.name)  # pyright: ignore[reportUnknownMemberType]
        self.model: ClapModel = ClapModel.from_pretrained(cfg.name).to(self.device).eval()  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
        self.dim: int = int(self.model.config.projection_dim)
        self._mel_fb: Any = None
        self._window: Any = None
        if self._gpu_features_supported():
            fe = self.processor.feature_extractor
            self._mel_fb = self.torch.from_numpy(fe.mel_filters_slaney.astype(np.float32)).to(
                self.device
            )
            self._window = self.torch.hann_window(
                fe.fft_window_size, periodic=True, device=self.device
            )

    @property
    def lock(self) -> Any:
        return self._lock

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
        with self._lock:
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
        return get_strategy(self.cfg).decode(path, self.cfg)

    def embed_windows(self, windows: Windows) -> np.ndarray:
        """One unit vector per window, in order. Windows are converted a batch at a time."""
        if not len(windows):
            raise ValueError("no decodable audio")
        bs = max(self.cfg.batch_size, 1) * 16
        return np.concatenate([
            self.embed_clips([windows.clip(j) for j in range(i, min(i + bs, len(windows)))])
            for i in range(0, len(windows), bs)
        ])

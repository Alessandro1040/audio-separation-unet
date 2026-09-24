"""Signal-processing helpers: audio <-> spectrogram (the "camera" of this project).

The whole project treats source separation as image segmentation: audio is turned
into a 2-D image (magnitude/phase spectrogram) with `stft`, the U-Net segments it
into per-source masks, the masks are applied back to the complex mixture spectrum
and `istft` turns the result into audio again.

Everything here keeps the same time/frequency convention as `torch.stft`:
    waveform  (..., channels, samples)
    spectrogram (..., channels, freq_bins, frames)   # complex when return_complex
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import STFTConfig


def get_window(cfg: STFTConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    win_length = cfg.win_length or cfg.n_fft
    return torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)


def stft_padding(n_samples: int, cfg: STFTConfig) -> tuple[int, int]:
    """(front, back) zero padding so the frames exactly cover the padded signal.

    `torch.stft(center=True)` would use reflection padding, which is not invertible at
    the edges and refuses signals shorter than the window. We pad with zeros instead
    and keep the framing aligned, so that `istft(stft(x)) == x` bit-for-bit.
    """
    front = cfg.n_fft // 2 if cfg.center else 0
    hop = cfg.hop_length
    total = n_samples + 2 * front
    if total < cfg.n_fft:                      # a single, zero-padded frame
        back = cfg.n_fft - total
    else:                                      # round up to the next full hop
        back = (-(total - cfg.n_fft)) % hop
    return front, back


def stft(x: torch.Tensor, cfg: STFTConfig) -> torch.Tensor:
    """Complex spectrogram of `x` -> (..., n_bins, frames).

    Hand-rolled instead of `torch.stft` because (a) with `center=False` torch only
    accepts 1-D/2-D input, and (b) `torch.stft(center=True)` pads by *reflection*,
    which is not invertible at the edges. Here the signal is zero-padded and framed
    exactly, so `istft(stft(x)) == x` holds to floating point precision.
    """
    front, back = stft_padding(x.shape[-1], cfg)
    padded = F.pad(x, (front, back))
    frames = padded.unfold(-1, cfg.n_fft, cfg.hop_length)         # (..., frames, n_fft)
    frames = frames * get_window(cfg, x.device, x.dtype)
    return torch.fft.rfft(frames, dim=-1).transpose(-1, -2)       # (..., n_bins, frames)


def _overlap_add(frames: torch.Tensor, total: int, hop: int) -> torch.Tensor:
    """Overlap-add frames (..., frames, n_fft) -> (..., total) using `fold`."""
    lead = frames.shape[:-2]
    n_fft = frames.shape[-1]
    flat = frames.reshape(-1, frames.shape[-2], n_fft).transpose(1, 2)
    out = F.fold(
        flat, output_size=(1, total), kernel_size=(1, n_fft), stride=(1, hop)
    )
    return out.reshape(*lead, total)


def istft(spec: torch.Tensor, cfg: STFTConfig, length: int) -> torch.Tensor:
    """Inverse of `stft`: weighted overlap-add, normalised by the window envelope."""
    front = cfg.n_fft // 2 if cfg.center else 0
    n_frames = spec.shape[-1]
    total = cfg.n_fft + (n_frames - 1) * cfg.hop_length
    window = get_window(cfg, spec.real.device, spec.real.dtype)
    frames = torch.fft.irfft(spec.transpose(-1, -2), n=cfg.n_fft, dim=-1) * window
    signal = _overlap_add(frames, total, cfg.hop_length)
    win_sq = (window * window).reshape(1, 1, cfg.n_fft).expand(1, n_frames, cfg.n_fft)
    envelope = _overlap_add(win_sq, total, cfg.hop_length).reshape(total)
    signal = signal / envelope.clamp_min(1e-8)
    available = total - 2 * front
    if length > available:                    # caller asks for more than the STFT covers
        signal = F.pad(signal, (0, length - available))
    return signal[..., front : front + length]


def split_complex_maps(x: torch.Tensor, n_sources: int, channels: int) -> torch.Tensor:
    """(B, S*ch*2, F, T) real (interleaved re/im) -> (B, S, ch, F, T) complex."""
    b, total, f, t = x.shape
    x = x.reshape(b, n_sources, channels, 2, f, t)
    return torch.complex(x[:, :, :, 0].contiguous(), x[:, :, :, 1].contiguous())


def stack_complex_maps(spec: torch.Tensor) -> torch.Tensor:
    """(B, S, ch, F, T) complex -> (B, S*ch*2, F, T) real (interleaved re/im)."""
    b, s, ch, f, t = spec.shape
    return torch.stack([spec.real, spec.imag], dim=3).reshape(b, s * ch * 2, f, t)


def log_magnitude(spec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-channel log-magnitude, the actual "image" fed to the U-Net."""
    return torch.log(spec.abs().clamp_min(eps))


def normalize_features(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-example, per-channel standardisation (mean 0, std 1) of the input image."""
    dims = tuple(range(1, feat.dim()))
    mean = feat.mean(dim=dims, keepdim=True)
    std = feat.std(dim=dims, keepdim=True).clamp_min(eps)
    return (feat - mean) / std


def apply_masks(masks: torch.Tensor, mix_spec: torch.Tensor) -> torch.Tensor:
    """Multiply complex masks (B, S, ch, F, T) with the mixture (B, ch, F, T)."""
    return masks * mix_spec.unsqueeze(1)


def mixture_consistency(
    estimates: torch.Tensor, mixture_wave: torch.Tensor
) -> torch.Tensor:
    """Project the estimates onto the "sum = mixture" subspace (Wavesplit trick)."""
    n_sources = estimates.shape[1]
    residual = mixture_wave.unsqueeze(1) - estimates.sum(dim=1, keepdim=True)
    return estimates + residual / n_sources


def waveform_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Scale-invariant SDR in dB, mean over all but the last (time) axis."""
    scale = (estimate * target).sum(dim=-1, keepdim=True) / (
        target.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    e_target = scale * target
    e_res = estimate - e_target
    num = e_target.pow(2).sum(dim=-1)
    den = e_res.pow(2).sum(dim=-1) + eps
    return 10.0 * torch.log10(num / den + eps)


def load_audio(path, sample_rate: int, channels: int) -> torch.Tensor:
    """Decode any audio file to (channels, samples) at `sample_rate`.

    Tries libsndfile first (fast, no subprocess) and falls back to ffmpeg, so mp3/m4a/
    ogg files uploaded by a user work even when the installed libsndfile is old.
    """
    import numpy as np

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        wav = torch.from_numpy(np.ascontiguousarray(data.T))  # (ch, samples)
    except Exception:                                          # pragma: no cover
        from .data.musdb import _decode_ffmpeg

        wav = torch.from_numpy(_decode_ffmpeg(Path(path), sample_rate, channels))
        sr = sample_rate
    if sr != sample_rate:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    if channels == 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif channels == 2 and wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav

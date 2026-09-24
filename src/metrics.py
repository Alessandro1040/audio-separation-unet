"""Objective metrics: SI-SDR (training/validation) and BSS-Eval v4 (final report).

BSS-Eval (Vincent et al.) decomposes each estimate into
`estimate = target + interference + artifacts` and reports
    SDR = source-to-distortion  (overall quality, higher is better)
    SIR = source-to-interference (how much of the *other* sources leaks in)
    SAR = source-to-artifacts   (how much artificial signal the model added)
We use `mir_eval`'s frame-wise implementation on 1-second windows and take the median
over frames, which is the protocol used by the official `museval` package (so numbers
are comparable to the MUSDB18 literature).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

from .dsp import waveform_sdr


@dataclass
class BssMetrics:
    sdr: float
    sir: float
    sar: float
    n_frames: int


def si_sdr_per_source(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB, averaged over batch/channels: (B, S) -> (S,)."""
    sdr = waveform_sdr(estimate, target)          # (B, S, ch)
    return sdr.mean(dim=(0, 2))


def _bss_eval_images(reference: np.ndarray, estimate: np.ndarray) -> tuple:
    """Call mir_eval's BSS-Eval v4 implementation, tolerating its deprecation.

    mir_eval 0.8 deprecated (and 0.9 will remove) the whole `bss_eval_*` family; the
    recommended successor is the `museval` package. The 0.8 implementation is the one
    everyone reports MUSDB18 numbers with, so it is used here, with the deprecation
    warning suppressed and a clear error if a future mir_eval removes it.
    """
    import mir_eval

    func = getattr(mir_eval.separation, "bss_eval_images", None)
    if func is None:                                     # mir_eval >= 0.9
        raise RuntimeError(
            "mir_eval>=0.9 removed bss_eval_images; install 'mir_eval<0.9' or 'museval'"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return func(reference, estimate, compute_permutation=False)


def bss_eval_frames(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    window: float = 1.0,
    hop: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame BSS-Eval -> three (n_frames, n_sources) arrays (SDR, SIR, SAR) in dB.

    `reference` / `estimate`: (n_sources, channels, samples); mir_eval wants
    (n_sources, samples, channels), so the axes are swapped here. mir_eval deprecated its
    own `*_framewise` helpers in 0.8 (removed in 0.9), hence the explicit frame loop: the
    per-source medians over 1-second frames are what `museval` reports for MUSDB18.
    """
    ref = np.ascontiguousarray(np.transpose(reference, (0, 2, 1)), dtype=np.float64)
    est = np.ascontiguousarray(np.transpose(estimate, (0, 2, 1)), dtype=np.float64)
    n_samples = ref.shape[1]
    win = int(round(window * sample_rate))
    step = int(round(hop * sample_rate))
    starts = list(range(0, max(n_samples - win + 1, 1), max(step, 1))) or [0]

    sdrs, sirs, sars = [], [], []
    for start in starts:
        ref_slice = ref[:, start : start + win, :]
        est_slice = est[:, start : start + win, :]
        if ref_slice.shape[1] < win or est_slice.shape[1] < win:
            break  # ignore the ragged tail, like museval does
        try:
            sdr, _isr, sir, sar, _perm = _bss_eval_images(ref_slice, est_slice)
        except ValueError:
            continue  # silent frame: BSS-Eval is undefined
        sdrs.append(sdr)
        sirs.append(sir)
        sars.append(sar)
    if not sdrs:
        empty = np.zeros((0, reference.shape[0]))
        return empty, empty.copy(), empty.copy()
    return np.stack(sdrs), np.stack(sirs), np.stack(sars)


def bss_eval_track(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    window: float = 1.0,
    hop: float = 1.0,
) -> BssMetrics:
    """BSS-Eval of one track: median over frames and over the four sources."""
    sdr, sir, sar = bss_eval_frames(reference, estimate, sample_rate, window, hop)
    if sdr.size == 0:
        nan = float("nan")
        return BssMetrics(sdr=nan, sir=nan, sar=nan, n_frames=0)
    return BssMetrics(
        sdr=float(np.nanmedian(sdr)),
        sir=float(np.nanmedian(sir)),
        sar=float(np.nanmedian(sar)),
        n_frames=int(sdr.shape[0]),
    )


def bss_eval_track_per_source(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    sources: tuple[str, ...],
    window: float = 1.0,
    hop: float = 1.0,
) -> dict[str, BssMetrics]:
    """Same as `bss_eval_track`, but keeping the sources separate (median per source)."""
    sdr, sir, sar = bss_eval_frames(reference, estimate, sample_rate, window, hop)
    n_frames = int(sdr.shape[0])
    out: dict[str, BssMetrics] = {}
    for j, name in enumerate(sources):
        if n_frames == 0 or j >= sdr.shape[1]:
            break
        out[name] = BssMetrics(
            sdr=float(np.nanmedian(sdr[:, j])),
            sir=float(np.nanmedian(sir[:, j])),
            sar=float(np.nanmedian(sar[:, j])),
            n_frames=n_frames,
        )
    return out


def aggregate_bss(per_track: list[dict[str, BssMetrics]], sources: tuple[str, ...]) -> dict:
    """Median over tracks, per source (the numbers reported in MUSDB18 papers)."""
    summary: dict[str, dict[str, float]] = {}
    for name in sources:
        entry: dict[str, float] = {}
        for metric in ("sdr", "sir", "sar"):
            values = [
                getattr(track[name], metric)
                for track in per_track
                if name in track and not np.isnan(getattr(track[name], metric))
            ]
            entry[metric] = float(np.median(values)) if values else float("nan")
        summary[name] = entry
    all_entry: dict[str, float] = {}
    for metric in ("sdr", "sir", "sar"):
        values = [summary[name][metric] for name in sources if name in summary]
        values = [v for v in values if not np.isnan(v)]
        all_entry[metric] = float(np.mean(values)) if values else float("nan")
    summary["all"] = all_entry
    return summary

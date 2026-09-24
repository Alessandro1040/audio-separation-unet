"""Objective metrics: SI-SDR sanity and BSS-Eval behaviour."""
from __future__ import annotations

import numpy as np
import torch

from src.data.synthetic import STEMS, make_song
from src.metrics import aggregate_bss, bss_eval_track, si_sdr_per_source


def _signals(seed: int = 0, seconds: float = 2.0, sr: int = 22050):
    song = make_song(seconds, sr, seed)
    reference = np.stack([song[s].T for s in STEMS])           # (S, ch, samples)
    return reference, sr


def test_si_sdr_perfect_and_silent() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 4, 2, 4000)
    perfect = si_sdr_per_source(target.clone(), target)
    assert float(perfect.min()) > 60.0            # identical signals -> huge SI-SDR
    silent = si_sdr_per_source(torch.zeros_like(target), target)
    assert float(silent.max()) < -60.0


def test_bss_eval_identical_estimates() -> None:
    reference, sr = _signals()
    metrics = bss_eval_track(reference, reference.copy(), sr, window=1.0, hop=1.0)
    assert metrics.sdr > 50.0
    assert metrics.n_frames > 0


def test_bss_eval_mixture_as_estimate_is_a_pure_interference_case() -> None:
    """Using the mixture as the estimate of every source is *exactly* target+interference.

    So the artifacts term must be ~perfect (huge SAR) while SDR is poor and SIR low:
    this checks the decomposition, not just the ranking.
    """
    reference, sr = _signals(seed=3)
    mixture = reference.sum(axis=0)
    estimate = np.stack([mixture] * len(STEMS))
    metrics = bss_eval_track(reference, estimate, sr)
    assert metrics.sdr < 0.0
    assert metrics.sir < 5.0
    assert metrics.sar > 30.0


def test_bss_eval_worse_does_not_mean_better() -> None:
    reference, sr = _signals(seed=5)
    good = reference + 0.05 * np.random.default_rng(0).standard_normal(reference.shape)
    bad = reference + 0.5 * np.random.default_rng(0).standard_normal(reference.shape)
    sdr_good = bss_eval_track(reference, good, sr).sdr
    sdr_bad = bss_eval_track(reference, bad, sr).sdr
    assert sdr_good > sdr_bad


def test_aggregate_bss_medians() -> None:
    reference, sr = _signals(seed=7)
    metrics = bss_eval_track(reference, reference.copy(), sr)
    per_track = [{name: metrics for name in STEMS}, {name: metrics for name in STEMS}]
    summary = aggregate_bss(per_track, STEMS)
    assert set(summary) == set(STEMS) | {"all"}
    assert summary["all"]["sdr"] > 50.0

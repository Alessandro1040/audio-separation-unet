"""The Mel-band RoFormer: bands, mask algebra, gradients and the end-to-end pipeline.

These tests are the RoFormer counterpart of `tests/test_model.py` +
`tests/test_pipeline.py`: the U-Net files stay untouched, and this file pins down the
guarantees the new architecture has to keep to be usable in this repository.

The invariants that matter most:

  * `test_bands_partition_the_spectrum` - every FFT bin belongs to at least one band, so no
    frequency is dropped for every source at once;
  * `test_bands_are_mel_spaced_and_bounded` - bands stay manageable (memory) while the low
    frequencies keep their resolution (quality);
  * `test_roformer_matches_separator_contract` - the wrapper is a drop-in for
    `SpectrogramSeparator`, including the `sum(estimates) == mixture` projection, which is
    what lets `separate_long` and the shared loss work unchanged.
"""
from __future__ import annotations

import pytest
import torch

from src.config import STFTConfig
from src.config_roformer import RoFormerConfig, RoFormerModelConfig, load_roformer_config
from src.data.synthetic import generate_dataset
from src.dsp import stft
from src.losses_mrstft import MRSTFTLoss
from src.models.roformer import MelBandRoFormer, build_bands
from src.models.roformer_separation import RoFormerSeparator


def _cfg_kwargs() -> dict:
    return dict(dim=32, depth=1, heads=4, ff_mult=2, n_bands=12, max_band_bins=32,
                per_channel_mask=False)


def test_bands_partition_the_spectrum() -> None:
    for n_fft in (256, 1024, 2048):
        n_bins = n_fft // 2 + 1
        bands = build_bands(n_bins, 22050, n_fft, n_bands=16, max_band_bins=32)
        covered = torch.zeros(n_bins, dtype=torch.long)
        for start, end in bands:
            assert 0 <= start < end <= n_bins
            covered[start:end] += 1
        assert int(covered.min()) >= 1, f"n_fft={n_fft}: some bins have no band"


def test_bands_are_mel_spaced_and_bounded() -> None:
    n_fft, sr = 2048, 44100
    bands = build_bands(n_fft // 2 + 1, sr, n_fft, n_bands=64, max_band_bins=32)
    widths = [end - start for start, end in bands]
    assert max(widths) <= 32, "wide bands would blow up the padded band tensors"
    low = [end - start for start, end in bands
           if (end + start) / 2 < 500 / sr * n_fft]
    assert max(low) <= min(widths) + 4, "low bands must stay narrow (pitch resolution)"
    starts = [start for start, _ in bands]
    assert starts == sorted(starts)


def test_roformer_shapes_and_bounded_masks() -> None:
    stft_cfg = STFTConfig(n_fft=1024, hop_length=256)
    model = MelBandRoFormer(n_bins=stft_cfg.n_bins, channels=2, n_sources=4,
                            sample_rate=22050, n_fft=1024, n_fft_extra=0,
                            **_cfg_kwargs())
    mixture = torch.randn(2, 2, 22050)
    mix_spec = stft(mixture, stft_cfg)
    masks = model(mix_spec)
    assert masks.shape == (2, 4, 2, stft_cfg.n_bins, mix_spec.shape[-1])
    assert masks.dtype == torch.complex64
    assert float(masks.real.abs().max()) <= 1.0 + 1e-6
    assert float(masks.imag.abs().max()) <= 1.0 + 1e-6


def test_second_resolution_changes_the_result_and_is_required() -> None:
    stft_cfg = STFTConfig(n_fft=1024, hop_length=256)
    model = MelBandRoFormer(n_bins=stft_cfg.n_bins, channels=2, n_sources=4,
                            sample_rate=22050, n_fft=1024, n_fft_extra=512,
                            **_cfg_kwargs())
    mixture = torch.randn(1, 2, 22050)
    primary = stft(mixture, stft_cfg)
    extra = stft(mixture, STFTConfig(n_fft=512, hop_length=256))
    with_extra = model(primary, extra)
    silent_extra = model(primary, torch.zeros_like(extra))
    assert with_extra.shape == silent_extra.shape
    # the second view must actually reach the masks, not be dropped on the way
    assert not torch.allclose(with_extra, silent_extra)
    with pytest.raises(ValueError):
        model(primary)          # built with n_fft_extra: the second view is mandatory


def test_gradients_are_finite() -> None:
    stft_cfg = STFTConfig(n_fft=512, hop_length=256)
    model = RoFormerSeparator(stft_cfg, RoFormerModelConfig(**_cfg_kwargs()),
                              channels=2, n_sources=4, sample_rate=22050)
    out = model(torch.randn(1, 2, 22050))
    out.masks.real.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_roformer_matches_separator_contract() -> None:
    """Same members and same algebra as `SpectrogramSeparator` (drop-in compatibility)."""
    stft_cfg = STFTConfig(n_fft=512, hop_length=256)
    model = RoFormerSeparator(stft_cfg, RoFormerModelConfig(**_cfg_kwargs()),
                              channels=2, n_sources=4, sample_rate=22050)
    mixture = torch.randn(2, 2, 22050)
    out = model(mixture)
    assert out.masks.shape[:3] == (2, 4, 2)
    assert out.spectra.shape[1] == 4
    assert out.waveforms.shape == (2, 4, 2, 22050)
    # mixture consistency: the four estimates must sum back to the mixture
    assert torch.allclose(out.waveforms.sum(dim=1), mixture, atol=1e-4)
    masks, mix_spec = model.predict_masks(mixture)
    assert masks.shape == out.masks.shape
    assert torch.allclose(mix_spec, out.mixture_spec)


def test_mrstft_loss_is_zero_on_a_perfect_estimate_and_positive_otherwise() -> None:
    cfg = load_roformer_config(None)
    cfg.loss.magnitude = 0.5
    cfg.loss.mrstft_ffts = [512, 256]
    loss_fn = MRSTFTLoss(cfg.loss, cfg.stft)
    target = torch.randn(1, 4, 2, 4096)
    perfect = loss_fn(target, target)
    assert float(perfect.total) == pytest.approx(0.0, abs=1e-6)
    worse = loss_fn(target * 0.5, target)
    assert float(worse.total) > 0.0
    assert float(worse.magnitude) > 0.0        # both spectral terms are active


def test_roformer_trains_and_separates_end_to_end(tmp_path) -> None:
    """Train a tiny RoFormer on the procedural dataset, then separate through the CLI."""
    import subprocess
    import sys

    import soundfile as sf

    from src.data.synthetic import STEMS
    from src.dsp import load_audio
    from src.models.separation import separate_long
    from src.separate_roformer import separate_file
    from src.train import pick_device
    from src.train_roformer import load_roformer_checkpoint, train

    root = generate_dataset(tmp_path / "synthetic", n_train=4, n_test=1, seconds=3.0,
                            sample_rate=22050)
    cfg = load_roformer_config("configs/roformer_smoke.yaml")
    assert cfg.model.n_fft_extra == 0, "the smoke config must stay single-resolution"
    cfg.data.root = str(root)
    cfg.data.cache_tracks = 2
    cfg.train.out_dir = str(tmp_path / "runs")
    cfg.train.ckpt_dir = str(tmp_path / "ckpt")
    cfg.train.device = "cpu"
    cfg.train.epochs = 1
    cfg.train.steps_per_epoch = 4
    cfg.train.warmup_steps = 2
    cfg.train.val_every = 4
    cfg.train.val_chunks = 1
    cfg.train.log_every = 2
    cfg.train.ema_decay = 0.9

    best = train(cfg)
    assert best.exists() and (tmp_path / "ckpt" / "last.pt").exists()

    device = pick_device("cpu")
    model, loaded = load_roformer_checkpoint(best, device)
    assert isinstance(loaded, RoFormerConfig)
    mixture_path = next((root / "test").iterdir()) / "mixture.wav"
    estimates = separate_file(model, loaded, mixture_path, device, chunk_seconds=1.0)
    mixture = load_audio(mixture_path, loaded.data.sample_rate, loaded.data.channels)
    assert estimates.shape[0] == len(STEMS)
    assert estimates.shape[-1] == mixture.shape[-1]
    assert torch.isfinite(estimates).all()
    assert float(estimates.abs().max()) < 10.0
    direct = separate_long(model, mixture, loaded.data.sample_rate, chunk_seconds=1.0)
    assert torch.allclose(estimates, direct, atol=1e-5)

    out_dir = tmp_path / "stems"
    subprocess.run(
        [sys.executable, "-m", "src.separate_roformer", "--checkpoint", str(best),
         "--input", str(mixture_path), "--out", str(out_dir), "--device", "cpu",
         "--chunk-seconds", "1.0", "--weights", "live"],
        check=True, cwd=".", capture_output=True,
    )
    for stem in STEMS:
        audio, sr = sf.read(out_dir / f"{stem}.wav")
        assert sr == loaded.data.sample_rate
        assert audio.shape[1] == 2

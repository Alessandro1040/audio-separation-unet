#!/usr/bin/env python3
"""The app: separate any audio file and show *in detail* what the model did.

    python -m src.analyze --checkpoint models/unet_musdb18_ema.pt --input song.mp3 --out outputs/song

It writes

    <out>/vocals.wav, drums.wav, bass.wav, other.wav, mixture.wav   separated audio
    <out>/masks.png        mixture spectrogram + the four predicted masks
    <out>/stems.png        magnitude spectrograms of the four estimates
    <out>/report.json/.txt duration, levels, energy share, mask statistics,
                           mixture-consistency error, and (if references are given)
                           per-stem SI-SDR / BSS-Eval

Everything is also available as a Python API: `analyze_file()` returns a dict with the
waveforms, the masks and all the numbers, which is what the Colab notebook uses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import STEMS
from .dsp import istft, load_audio, log_magnitude, stft
from .metrics import bss_eval_track_per_source, si_sdr_per_source
from .models.separation import separate_long
from .separate import load_model


def _spectral_centroid(magnitude: np.ndarray, sample_rate: int) -> float:
    """Energy-weighted mean frequency of a spectrogram (Hz)."""
    freqs = np.linspace(0, sample_rate / 2, magnitude.shape[0])[:, None]
    total = magnitude.sum() + 1e-12
    return float((freqs * magnitude).sum() / total)


def analyze_file(
    checkpoint: str | Path,
    audio_path: str | Path,
    out_dir: str | Path,
    device_name: str = "auto",
    chunk_seconds: float = 10.0,
    overlap: float = 0.5,
    figure_seconds: float = 24.0,
    reference_dir: str | Path | None = None,
    make_figures: bool = True,
    bss_seconds: float = 30.0,
) -> dict:
    """Separate one file and collect every diagnostic number about the process."""
    from .train import pick_device

    device = pick_device(device_name)
    model, cfg = load_model(checkpoint, device)
    sample_rate = cfg.data.sample_rate
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mixture = load_audio(audio_path, sample_rate, cfg.data.channels)
    duration = mixture.shape[-1] / sample_rate
    print(f"[analyze] {audio_path}: {duration:.1f} s, {mixture.shape[0]} channel(s) "
          f"-> {sample_rate} Hz, device={device}")

    estimates = separate_long(model, mixture.to(device), sample_rate, chunk_seconds,
                              overlap).detach().cpu()          # (S, ch, samples)

    # --- deterministic comparison with ground truth, when the stems are available
    reference = None
    ref_dir = Path(reference_dir) if reference_dir else Path(audio_path).parent
    if reference_dir or all((ref_dir / f"{s}.wav").exists() for s in STEMS):
        tracks = []
        for stem in STEMS:
            wav = load_audio(ref_dir / f"{stem}.wav", sample_rate, cfg.data.channels)
            tracks.append(wav[..., : estimates.shape[-1]])
        reference = torch.stack(tracks)

    # --- figures: what the network "sees" and what it predicted
    if make_figures:
        _plot_masks(model, cfg, mixture, sample_rate, figure_seconds,
                    out / "masks.png", device_name)
        _plot_stems(estimates, sample_rate, figure_seconds, out / "stems.png")

    # --- write the audio
    for i, stem in enumerate(STEMS):
        sf.write(out / f"{stem}.wav", estimates[i].numpy().T, sample_rate,
                 subtype="PCM_16")
    sf.write(out / "mixture.wav", mixture.numpy().T, sample_rate, subtype="PCM_16")

    report = _report(model, cfg, mixture, estimates, reference, sample_rate, {
        "input": str(audio_path), "checkpoint": str(checkpoint),
        "duration_s": duration, "device": str(device),
        "stems": list(STEMS), "out_dir": str(out),
    }, bss_seconds)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.txt").write_text(_format_report(report))
    print(_format_report(report))
    return {"mixture": mixture, "estimates": estimates, "reference": reference,
            "report": report, "model": model, "config": cfg}



def _plot_masks(model, cfg, mixture: torch.Tensor, sample_rate: int,
                seconds: float, path: Path, device_name: str) -> None:
    """Mixture spectrogram + the four predicted masks, as images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .train import pick_device

    device = pick_device(device_name)
    seg = mixture[..., : int(seconds * sample_rate)].to(device)
    with torch.no_grad():
        masks, mix_spec = model.predict_masks(seg.unsqueeze(0))
    mix_mag = mix_spec[0].abs().mean(dim=0).cpu().numpy()
    mask_mag = masks[0].abs().mean(dim=1).cpu().numpy()          # (S, F, T)

    fig, axes = plt.subplots(len(STEMS) + 1, 1, figsize=(11, 2.1 * (len(STEMS) + 1)),
                             sharex=True)
    extent = [0, mix_mag.shape[1] * cfg.stft.hop_length / sample_rate,
              0, sample_rate / 2000]
    im = axes[0].imshow(20 * np.log10(mix_mag.clip(1e-8)), origin="lower",
                        aspect="auto", extent=extent, cmap="magma")
    axes[0].set_title("mixture spectrogram (what the U-Net sees)")
    fig.colorbar(im, ax=axes[0], format="%d dB", pad=0.01)
    for i, stem in enumerate(STEMS):
        ax = axes[i + 1]
        im = ax.imshow(mask_mag[i], origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"mask: {stem}  (mean |mask| = {mask_mag[i].mean():.2f})")
        fig.colorbar(im, ax=ax, pad=0.01)
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.set_ylabel("freq (kHz)")
    fig.suptitle("Step 1-2: the network segments the spectrogram into per-instrument masks",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[analyze] wrote {path}")


def _plot_stems(estimates: torch.Tensor, sample_rate: int, seconds: float,
                path: Path) -> None:
    """Magnitude spectrograms of the four separated sources."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .config import STFTConfig

    cfg = STFTConfig()
    seg = estimates[..., : int(seconds * sample_rate)]
    with torch.no_grad():
        spec = stft(seg, cfg).abs().mean(dim=1).numpy()           # (S, F, T)
    fig, axes = plt.subplots(len(STEMS), 1, figsize=(11, 6.4), sharex=True)
    extent = [0, spec.shape[-1] * cfg.hop_length / sample_rate, 0.0, 22.05]
    for i, stem in enumerate(STEMS):
        im = axes[i].imshow(20 * np.log10(spec[i].clip(1e-8)), origin="lower",
                            aspect="auto", extent=extent, cmap="magma")
        axes[i].set_title(f"{stem}", loc="left", fontsize=10)
        fig.colorbar(im, ax=axes[i], format="%d dB", pad=0.01)
    axes[-1].set_xlabel("time (s)")
    for ax in axes:
        ax.set_ylabel("freq (kHz)")
    fig.suptitle("Step 3: masks applied to the mixture spectrum (iSTFT gives the audio)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[analyze] wrote {path}")


def _report(model, cfg, mixture: torch.Tensor, estimates: torch.Tensor,
            reference: torch.Tensor | None, sample_rate: int, base: dict,
            bss_seconds: float = 30.0) -> dict:
    """Every number that describes the separation, ground truth or not."""
    report: dict = dict(base)

    def stats(signal: torch.Tensor) -> dict:
        per_channel = signal.pow(2).sum(dim=-1).sqrt() / np.sqrt(signal.shape[-1])
        return {
            "rms": float(per_channel.mean()),
            "peak": float(signal.abs().max()),
            "spectral_centroid_hz": _spectral_centroid(
                stft(signal, cfg.stft).abs().mean(dim=0).numpy(), sample_rate),
        }

    per_stem = {name: stats(estimates[i]) for i, name in enumerate(STEMS)}
    total_energy = sum(v["rms"] ** 2 for v in per_stem.values()) + 1e-12
    for value in per_stem.values():
        value["energy_share"] = value["rms"] ** 2 / total_energy
    report["per_stem"] = per_stem

    # How well the four estimates add back up to the input (it should be exact: the
    # model enforces mixture consistency, so this is a free self-check of the pipeline)
    residual = mixture - estimates.sum(dim=0)
    residual_rms = float(residual.pow(2).mean().sqrt())
    mixture_rms = float(mixture.pow(2).mean().sqrt())
    report["mixture_consistency"] = {
        "residual_rms_db": 20.0 * float(np.log10((residual_rms + 1e-12)
                                                 / (mixture_rms + 1e-12))),
        "residual_rms": residual_rms,
        "mixture_rms": mixture_rms,
    }

    # Mask behaviour: how selective the network is, on the first 24 s
    seg = mixture[..., : int(24 * sample_rate)].unsqueeze(0)
    device = next(model.parameters()).device
    with torch.no_grad():
        masks, _ = model.predict_masks(seg.to(device))
    mask_mag = masks[0].abs()                                    # (S, ch, F, T)
    sum_masks = mask_mag.sum(dim=0)                              # (ch, F, T)
    n_bins = mask_mag.shape[-2]
    freqs = torch.linspace(0, 22050, n_bins)
    bands = {"low_<200Hz": freqs < 200, "mid_200_2000Hz": (freqs >= 200) & (freqs < 2000),
             "high_>2kHz": freqs >= 2000}
    band_profile: dict[str, dict[str, float]] = {}
    for i, name in enumerate(STEMS):
        total = float(mask_mag[i].sum()) + 1e-12
        band_profile[name] = {
            band: float(mask_mag[i][..., mask, :].sum()) / total
            for band, mask in bands.items()
        }
    report["mask_statistics"] = {
        "mean_abs_mask_per_stem": {name: float(mask_mag[i].mean())
                                   for i, name in enumerate(STEMS)},
        "fraction_of_bins_above_0.5": {name: float((mask_mag[i] > 0.5).float().mean())
                                       for i, name in enumerate(STEMS)},
        "mask_mass_per_frequency_band": band_profile,
        "mean_sum_of_masks": float(sum_masks.mean()),
        "std_sum_of_masks": float(sum_masks.std()),
        "note": "a well calibrated 4-source mask set sums to ~1 per bin; this model is "
                "not trained for that explicitly, mixture consistency is enforced on the "
                "waveform instead. The band profile shows where each mask puts its mass: "
                "a bass mask should be concentrated in low_<200Hz",
    }

    if reference is not None:
        report["reference_per_stem"] = {name: stats(reference[i])
                                        for i, name in enumerate(STEMS)}
        # si_sdr_per_source expects (batch, sources, channels, samples) and returns (S,)
        sdr = si_sdr_per_source(estimates.unsqueeze(0), reference.unsqueeze(0))
        report["metrics_vs_reference"] = {
            "si_sdr_db": {name: float(sdr[i]) for i, name in enumerate(STEMS)},
            "si_sdr_mean_db": float(sdr.mean()),
        }
        per_source = bss_eval_track_per_source(
            reference[..., : int(bss_seconds * sample_rate)].numpy().astype(np.float64),
            estimates[..., : int(bss_seconds * sample_rate)].numpy().astype(np.float64),
            sample_rate, STEMS,
        )
        if per_source:
            report["metrics_vs_reference"]["bss_eval"] = {
                name: {"sdr": m.sdr, "sir": m.sir, "sar": m.sar}
                for name, m in per_source.items()
            }
            report["metrics_vs_reference"]["bss_eval_window_s"] = bss_seconds
            report["metrics_vs_reference"]["bss_eval_n_frames"] = next(
                iter(per_source.values())).n_frames
    else:
        report["metrics_vs_reference"] = None
        report["reference_hint"] = ("no ground-truth stems found next to the input; pass "
                                    "--reference-dir DIR containing vocals.wav, "
                                    "drums.wav, bass.wav, other.wav to get SI-SDR and "
                                    "BSS-Eval numbers")
    return report


def _format_report(report: dict) -> str:
    lines = [
        "",
        "separation report",
        "=" * 60,
        f"input      : {report['input']}",
        f"duration   : {report['duration_s']:.1f} s at 44100 Hz stereo",
        f"checkpoint : {report['checkpoint']}",
        "",
        f"{'stem':<8} {'rms':>8} {'peak':>8} {'energy %':>9} {'centroid':>10} {'|mask|':>8}",
    ]
    for name in STEMS:
        s = report["per_stem"][name]
        mask = report["mask_statistics"]["mean_abs_mask_per_stem"][name]
        lines.append(f"{name:<8} {s['rms']:8.4f} {s['peak']:8.3f} "
                     f"{100 * s['energy_share']:8.1f}% {s['spectral_centroid_hz']:9.0f}Hz "
                     f"{mask:8.2f}")
    cons = report["mixture_consistency"]
    lines += [
        "",
        f"mixture consistency : residual of sum(stems) vs mixture = "
        f"{cons['residual_rms_db']:.1f} dB relative to the input",
        f"mean sum of masks   : "
        f"{report['mask_statistics']['mean_sum_of_masks']:.2f} +- "
        f"{report['mask_statistics']['std_sum_of_masks']:.2f} (ideally ~1.00)",
    ]
    bands = report["mask_statistics"].get("mask_mass_per_frequency_band")
    if bands:
        lines += ["", "where each mask puts its energy (a bass mask should be low-heavy)"]
        lines.append(f"{'stem':<8} {'low<200Hz':>10} {'200-2kHz':>10} {'>2kHz':>10}")
        for name in STEMS:
            b = bands[name]
            lines.append(f"{name:<8} {b['low_<200Hz']:9.1%} {b['mid_200_2000Hz']:9.1%} "
                         f"{b['high_>2kHz']:9.1%}")
    if report.get("reference_per_stem"):
        lines += ["", "estimates vs ground truth (a large centroid mismatch = leakage of "
                      "other instruments)"]
        lines.append(f"{'stem':<8} {'rms est':>8} {'rms ref':>8} "
                     f"{'centroid est':>13} {'centroid ref':>13}")
        for name in STEMS:
            est = report["per_stem"][name]
            ref = report["reference_per_stem"][name]
            lines.append(f"{name:<8} {est['rms']:8.4f} {ref['rms']:8.4f} "
                         f"{est['spectral_centroid_hz']:10.0f} Hz "
                         f"{ref['spectral_centroid_hz']:10.0f} Hz")
    metrics = report.get("metrics_vs_reference")
    if metrics:
        lines += ["", "against the reference stems "
                      f"(mean SI-SDR {metrics['si_sdr_mean_db']:.2f} dB)"]
        if "bss_eval" in metrics:
            lines.append(f"{'stem':<8} {'SI-SDR':>9} {'SDR':>8} {'SIR':>8} {'SAR':>8}")
            for name in STEMS:
                b = metrics["bss_eval"].get(name)
                cell = (f"{b['sdr']:8.2f} {b['sir']:8.2f} {b['sar']:8.2f}"
                        if b else f"{'-':>8} {'-':>8} {'-':>8}")
                lines.append(f"{name:<8} {metrics['si_sdr_db'][name]:9.2f} {cell}")
    else:
        lines += ["", report.get("reference_hint", "")]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", default="models/unet_musdb18_ema.pt",
                   help="inference checkpoint (scripts/export_inference_checkpoint.py)")
    p.add_argument("--input", required=True, help="any audio file: wav/mp3/flac/m4a/...")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--figure-seconds", type=float, default=24.0)
    p.add_argument("--reference-dir", default=None,
                   help="directory with the ground-truth stems (optional)")
    p.add_argument("--bss-seconds", type=float, default=30.0,
                   help="how much audio to use for the (CPU-bound) BSS-Eval")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    analyze_file(args.checkpoint, args.input, args.out, args.device,
                 args.chunk_seconds, args.overlap, args.figure_seconds,
                 args.reference_dir, not args.no_figures, args.bss_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


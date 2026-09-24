# Model card — `unet_musdb18_ema.pt`

Spectrogram **U-Net** for 4-stem music source separation (`vocals`, `drums`, `bass`,
`other`), trained in this repository. File: `models/unet_musdb18_ema.pt` (42 MB,
float32, exponential-moving-average weights of the best validation checkpoint).

## What it is

| | |
| --- | --- |
| Architecture | U-Net, 5 levels, 32→64→128→256→512 base-2 channels, two 3×3 convs per level, BatchNorm, bilinear upsampling, skip connections (`src/models/unet.py`) |
| Parameters | 10.55 M |
| Input | log-magnitude spectrogram of the mixture, 2 channels (L/R), STFT 2048 / hop 1024 @ 44.1 kHz → 1025 × T |
| Output | complex ideal-ratio masks (real + imag) for 4 sources × 2 channels = 16 maps, `tanh` bounded |
| Reconstruction | mask ⊙ mixture spectrum → iSTFT (exact, 50 % overlap) → mixture consistency projection |
| Training data | MUSDB18 (compressed), 86 songs from the `train` split; 14 songs held out for validation; never trained on the `test` split |
| Training budget | 2500 optimization steps (10 epochs × 250), batch 4 × 5 s crops, AdamW, peak LR 3e-4 with a 250-step linear warm-up and cosine decay to 5 %, hybrid loss (waveform L1 + L1 on `|X|^0.3` magnitudes, mixture consistency), heavy augmentation (cross-song swap of the **same** stem, full remix, gains, channel swap, polarity, source dropout) |
| Hardware | Apple M5, 16 GB, MPS (~1.2 s/step) |

## History: why the previous export was withdrawn

The first version of this file (step 1200 of `runs/unet`) did not separate anything: all four
estimates were the mixture with a per-source gain. `scripts/diagnose_masks.py` measured it on
validation chunks - the model reached **−3.94 dB** mean SI-SDR while the best per-source
*fader* (the optimal 50 ms time-varying gain on the mixture, i.e. pure volume automation)
reached **−2.55 dB**. Inside the masks, 76-95 % of each mask's variance was a fixed frequency
profile and only 0.01-0.05 of it varied with time: an EQ, not a separator.

The cause was in the training *data*, not in the architecture: `_augment` swapped a stem with
a *random* stem of the donor song instead of the same one, so three targets out of four
carried a different instrument (measured: 36.5 % of all targets mislabelled; 0.0 % after the
fix). The mixture does not reveal which instrument was substituted, so the loss-minimising
answer for every head was the conditional mean - a per-source gain on the mixture. A second
bug compounded it: `lr_at` restarted a resumed run at the schedule floor (~5e-6), so "just
train longer" did nothing.

Both are fixed; the README section *Why the first trained model only changed the volumes*
has the controlled A/B (same seed, same hyper-parameters, only the swap line differs:
−6.74 dB → −1.53 dB after 600 steps on the procedural dataset). **The weights in this file
are the retrained model.**

## Measured quality

| metric | value | notes |
| --- | --- | --- |
| validation SI-SDR (14 held-out songs) | **−5.76 dB** | chunk SI-SDR, EMA weights, model-selection metric (best step 2250) |
| test-set SI-SDR, 10 songs | **−6.40 dB** | vs **−9.99 dB** for the trivial "mixture as every source" baseline → **+3.59 dB**, and +1.89 dB better than the step-1200 checkpoint, which it beats on 9/10 songs |
| vs the strongest "volume only" baseline | **+1.22 dB** | best per-source fader on 4 validation chunks: −2.55 dB vs −1.33 dB for this model |
| test-set BSS-Eval (2 songs × 60 s) | in progress | the ~15 min `mir_eval` run for this checkpoint was still in flight when this table was written; `results/bss_eval_2songs.json` currently holds the step-1200 numbers (SDR +1.16 dB, SIR −2.84 dB, SAR +19.32 dB) |
| oracle ideal-ratio mask (ceiling for this STFT/architecture) | +8.05 dB SI-SDR | measured with `scripts/oracle_mask_baseline.py` |

Per source (10 test songs): `drums` −3.69 dB, `other` −2.67 dB, `bass` −2.64 dB,
**`vocals` −14.35 dB** (the hard one: it overlaps everything in time and frequency).

## What the model actually learned (diagnostics from `src/analyze.py`)

* The pipeline is exact: `sum(estimates) − mixture` residual ≈ **−147 dB**, and the four
  masks now sum to **1.02 ± 0.24** per bin - a calibrated mask set. The step-1200 model was
  at 0.74 ± 0.28, i.e. it was re-levelling rather than separating.
* The masks have the tilts they should: `bass` is three times as heavy below 200 Hz as above
  2 kHz (0.32 vs 0.11), `drums` is by far the most active mask (mean gain 0.47) and `other` /
  `vocals` are more selective than before (cv 0.38 → 0.71 and 0.26 → 0.50). Most mask variance
  is still the *frequency* profile (share 0.75-0.96), so this is not a perfect time-frequency
  segmenter yet.
* The `bass` estimate is still too bright on the example song (spectral centroid ~5 kHz
  against 222 Hz for the reference) even though its SI-SDR now beats the fader: harmonic-rich
  low content and leakage both push that number up.
* Interference, not artifacts, is the limiting factor: the app reports SAR ≈ +13 dB against
  SIR between −3.4 and +2.6 dB, i.e. the estimates follow the target structure but still
  contain other sources.

## Intended use

* Research/teaching: a complete, small, reproducible example of "audio separation as
  image segmentation", trainable on a laptop.
* Demos: separating a song you own into four stems, with a full diagnostic report.

**Not** intended for: production separation quality, speech/podcast separation (out of
domain), or anything requiring state-of-the-art results. Expect noticeably weaker
separation on modern loud commercial masters than on MUSDB18 material (measured on an
out-of-domain mp3: vocals/other partially separated, bass heavily leaking).

## How to load

```python
from src.separate import load_model
from src.train import pick_device

model, cfg = load_model("models/unet_musdb18_ema.pt", pick_device("cpu"))
```

or use the app:

```bash
python -m src.analyze --checkpoint models/unet_musdb18_ema.pt --input song.mp3 --out outputs/song
```

## Licence

Code: MIT (see `LICENSE`). Weights: derived from **MUSDB18**, which is distributed under
**CC BY-NC-SA 4.0** — so these weights inherit the non-commercial restriction and must be
attributed to the MUSDB18 dataset (Rafii et al., *MUSDB18 — a corpus for music
separation*, 2017; and the Slakh/MedleyDB sources it is built from).

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
| Training budget | ~1200 optimization steps, batch 4 × 5 s crops, AdamW, hybrid loss (waveform L1 + L1 on `|X|^0.3` magnitudes, SI-SDR term in the last 300 steps), heavy augmentation (cross-song stem swap/remix, gains, channel swap, polarity, source dropout) |
| Hardware | Apple M5, 16 GB, MPS (~1.2 s/step) |

## Measured quality

| metric | value | notes |
| --- | --- | --- |
| validation SI-SDR (14 held-out songs) | **−7.85 dB** | chunk SI-SDR, EMA weights, model-selection metric |
| test-set SI-SDR, 10 songs | **−8.29 dB** | vs **−9.99 dB** for the trivial "mixture as every source" baseline → **+1.70 dB**, wins 9/10 songs |
| test-set BSS-Eval (2 songs × 60 s) | SDR **+1.16 dB**, SIR −2.84 dB, SAR **+19.32 dB** | 1 s frames, median over frames |
| oracle ideal-ratio mask (ceiling for this STFT/architecture) | +8.05 dB SI-SDR | measured with `scripts/oracle_mask_baseline.py` |

Per source (10 test songs): `other` −3.56 dB, `drums` −4.70 dB, `bass` −5.43 dB,
**`vocals` −16.76 dB** (the hard one).

## What the model actually learned (diagnostics from `src/analyze.py`)

* The pipeline is exact: `sum(estimates) − mixture` residual ≈ **−148 dB**.
* The masks are still close to frequency-agnostic: ~90 % of the mask mass sits above
  2 kHz and the mean sum of the four masks is 0.74 ± 0.28 instead of ~1.0. So the
  network separates mostly *temporally*, and instruments whose energy is concentrated in
  a narrow band (bass) come out worst unless the song is bass-heavy in a way it saw in
  training.
* Interference, not artifacts, is the limiting factor (SAR +19 dB vs SIR −2.8 dB):
  the estimates follow the target structure but still contain other sources.

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

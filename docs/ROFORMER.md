# The second architecture: Mel-band RoFormer (band split + RoPE attention)

This document describes the model added in `src/models/roformer*.py`, *why* it is shaped the
way it is, what it costs on a laptop, and - most importantly - what the evidence in this
repository does and does not prove about it.

**Nothing that existed before was modified.** The U-Net path (`src/models/unet.py`,
`src/models/separation.py`, `src/train.py`, `src/separate.py`, `src/analyze.py`,
`src/config.py`, `configs/unet*.yaml`, `models/unet_musdb18_ema.pt`) is untouched and still
the default. Everything below is *additional* files, and the two architectures share the
data pipeline, the loss interface, the EMA, the checkpoint format and the inference
sliding window.

---

## 1. Why a second architecture at all

The U-Net in this repository treats separation as image segmentation: it convolves over a
spectrogram and predicts one mask per source. That is the ISMIR-2017/2018 school (Jansson
et al.; Stoller et al.) and it is what the shipped checkpoint implements: 10.55 M
parameters, −5.76 dB validation SI-SDR, −6.40 dB on the MUSDB18 test set
(`models/MODEL_CARD.md`).

The state of the art moved elsewhere. The family that wins the Music Demixing Challenge
and the MVSep leaderboards is *band-split + attention*:

| Reference | Idea taken here |
| --- | --- |
| Wang et al., **BS-RoFormer** (*Music Source Separation with Band-split RoPE Transformer*, 2023/24) | cut the frequency axis into bands, one token per band, rotary position embeddings over the band index |
| Lu et al., **Mel-Band RoFormer** (2023) | make the bands mel-spaced: narrow where pitch matters, wide at the top |
| the MDX23 winners' recipe | a second STFT resolution as extra input, multi-resolution spectral loss, per-stem specialists, ensembles |
| Su et al., **RoFormer** (2021) | the rotary embedding itself |

Two things a U-Net cannot do, that this family can:

1. **Relate distant frequencies cheaply.** A bass note's fundamental and its 8th partial are
   ~1 kHz apart; for a stack of 3×3 convolutions with a limited receptive field that is a
   long way, while attention over bands compares *any* two bands in one step. That is what
   RoPE-on-the-band-index makes learnable.
2. **Keep a memory of timbre over seconds.** Attention over time sees the whole chunk at
   once, which is what a sustained instrument (a held organ chord, a bowed string) needs.

## 2. The architecture, piece by piece

`src/models/roformer.py`:

```
mixture spectrum (B, ch, F, T) complex
  -> gather the bins of every mel band                (bands, B, T, 2*ch*Wmax)
  -> per-band linear + band embedding                 (bands, B, T, dim)
  + second-resolution log-magnitude band features     (bands, B, T, ch)
  -> depth x (band attention + time attention + FFN)  (bands, B, T, dim)
  -> per-band mask head + tanh                        (bands, Wmax, B, S, ch, 2, T)
  -> scatter back onto the FFT grid, averaging overlaps
  -> masks (B, S, ch, F, T) complex  == the same cIRM convention as the U-Net
```

* **`build_bands` / `band_hz_ranges`** - mel-spaced bands with 25 % overlap; bands wider
  than `max_band_bins` are split in half (the band tensors are padded to the widest band, so
  keeping widths comparable is a straight 2× memory saving). Two guarantees that are
  actually *tested* (`tests/test_roformer.py`): the bands form a partition of
  `[0, n_bins)` - every bin belongs to at least one band, so no frequency is silently
  zeroed for every source at once - and the low bands stay narrow (pitch resolution).
* **`RoPESelfAttention`** - the exact same attention module is used twice per block: once
  over the band axis, once over the time axis, with rotary embeddings on whichever axis is
  the sequence. Because RoPE is relative, a model trained on 3 s chunks runs on the
  430-frame windows of a whole song without any positional table to resize.
* **`MelBandRoFormer`** - per-band linear layers (one weight per band, `einsum`), a learned
  band embedding, and a mask head that predicts a *full-width* mask per band (a mask value
  per bin, not a single gain per band). Overlapping bands are **averaged** when scattered
  back, and every bin is covered.
* **Multi-resolution input** (`model.n_fft_extra`) - the same waveform through a 1024-point
  STFT (sharper attacks) is band-pooled, standardised per band and added to the band tokens
  before the blocks. The two views' frame grids do not line up exactly, so the coarse one is
  resampled by nearest neighbour; both cover the same time span.
* **Masks** - `tanh` complex ratio masks, i.e. literally the same algebra as the U-Net
  (`|mask| <= 1`, mask × mixture spectrum, iSTFT, mixture consistency). That is why
  `src/models/roformer_separation.py::RoFormerSeparator` can be swapped into
  `separate_long` and into the loss with no changes at all - the contract is verified by
  `test_roformer_matches_separator_contract`.

## 3. Files added (all additive)

| File | What it is |
| --- | --- |
| `src/models/roformer.py` | the architecture: band construction, RoPE attention, `MelBandRoFormer` |
| `src/models/roformer_separation.py` | `RoFormerSeparator`: STFT wrapper with the *same* interface as `SpectrogramSeparator` |
| `src/config_roformer.py` | `RoFormerModelConfig`, `RoFormerLossConfig`, `RoFormerConfig` (reuses `STFTConfig`/`DataConfig`/`TrainConfig` from `src/config.py`) |
| `src/losses_mrstft.py` | `MRSTFTLoss`: waveform L1 + 4-resolution STFT loss (and the U-Net's compressed-magnitude term, for A/B runs) |
| `src/train_roformer.py` | trainer + `load_roformer_checkpoint` (reuses `lr_at`, `EMA`, `evaluate_chunks`, `build_loaders`, `pick_device` from `src/train.py`) |
| `src/separate_roformer.py` | inference CLI, same output contract as `python -m src.separate` |
| `configs/roformer_smoke.yaml` | tiny end-to-end config on the procedural dataset |
| `configs/roformer_musdb18.yaml` | the full recipe measured on the M5 (see §4) |
| `configs/roformer_ab.yaml` | the controlled comparison config (loss switched to the U-Net's) |
| `scripts/benchmark_roformer.py` | step time and memory per batch size |
| `scripts/check_roformer.py` | "is this checkpoint actually separating?" diagnostics |
| `scripts/ab_roformer_vs_unet.sh` | the A/B in §6, one command |
| `tests/test_roformer.py` | 8 tests: band partition, band bounds, mask algebra, drop-in contract, gradients, loss, end-to-end |
| `docs/ROFORMER.md` | this document |

## 4. What it costs on a laptop (Apple M5, 16 GB, MPS)

Measured with `python scripts/benchmark_roformer.py --batches 1 2 --steps 5` on
`configs/roformer_musdb18.yaml` (44.1 kHz, `n_fft` 2048 / hop 512, 82 mel bands,
dim 128, depth 4, 2.75 M parameters):

```
device=mps  model=2.75M params  chunk=3.0s  bins=1025 frames=259  bands=82 band_width=32
  batch= 1    632.7 ms/step    4.74 s_audio/s  peak_alloc= 3.40 GB
  batch= 2   1226.4 ms/step    4.89 s_audio/s  peak_alloc= 5.91 GB
```

| | U-Net (`configs/unet_musdb18.yaml`) | Mel-band RoFormer |
| --- | --- | --- |
| parameters | 10.55 M | 2.75 M |
| seconds of audio processed per second of compute | ~17-20 s/s | ~4.9 s/s |
| one training step | ~1.2 s (batch 4 × 5 s) | ~1.2 s (batch 2 × 3 s) |
| allocator peak | ~0.5 GB | ~5.9 GB |

In other words: **the RoFormer costs ~3-4× more compute per second of audio** (mostly the
time-axis attention, which is quadratic in the number of frames, plus the second STFT). It
is cheaper in parameters, not in FLOPs. That is the price of the family it belongs to.

The knobs, in order of effect:

| Knob | Effect |
| --- | --- |
| `data.chunk_seconds` | sets T, hence the time attention: halving it costs 4× less |
| `stft.hop_length` | same thing from the other side (1024 halves T vs 512) |
| `model.max_band_bins` | width of the padded band tensors (32 is a good trade-off) |
| `model.depth`, `model.dim` | linear in parameters, visible in both time and memory |
| `model.n_fft_extra` | 0 disables the multi-resolution input (cheaper, slightly worse) |

Measured candidates on the M5, for reference:

```
dim=192 depth=6  3 s  B=2  T=258  6.58 M params  1.96 s/step  ~9.8 GB   (too big for 16 GB)
dim=128 depth=6  3 s  B=2  T=258  3.41 M params  1.40 s/step  ~7.1 GB   <- configs/roformer_musdb18.yaml
dim=128 depth=4  3 s  B=1  T=258  2.75 M params  0.62 s/step  ~2.8 GB
dim=64  depth=2  2 s  B=2  T=172  0.43 M params  0.24 s/step  ~1.2 GB   <- configs/roformer_smoke.yaml
```

## 5. How to run it

```bash
# 1. does it work at all? (~1 minute, procedural dataset, no MUSDB18 needed)
python scripts/make_synthetic_data.py --out data/synthetic --train 6 --test 2
python -m src.train_roformer --config configs/roformer_smoke.yaml
python -m src.separate_roformer --checkpoint checkpoints/roformer_smoke/best.pt \
    --input data/synthetic/test/test00/mixture.wav --out outputs/roformer_smoke --also-mixture
python scripts/check_roformer.py --checkpoint checkpoints/roformer_smoke/best.pt \
    --input data/synthetic/test/test00/mixture.wav --seconds 5

# 2. the real run (MUSDB18; ~1.4 s/step on an M5)
python -m src.train_roformer --config configs/roformer_musdb18.yaml
python -m src.separate_roformer --checkpoint checkpoints_roformer/best.pt \
    --input song.mp3 --out outputs/song_roformer --also-mixture

# 3. costs and diagnostics
python scripts/benchmark_roformer.py --batches 1 2 4 --chunk-seconds 3
python scripts/check_roformer.py --checkpoint checkpoints_roformer/best.pt --input song.mp3
```

`check_roformer.py` prints the same kind of verdict as the U-Net's
`check_separation.py`: the mean sum of the masks (≈1.0 means a calibrated mask set), the
per-stem mask energy in three frequency bands, and the similarity between the four stems
(≈0.6 means they are different audio; ≈0.9 would mean the model re-levelled the mixture).
Using the U-Net's diagnostics script on a RoFormer checkpoint does **not** work - they are
different architectures on purpose, which is why this one exists.

## 6. The evidence (a short, controlled A/B)

`scripts/ab_roformer_vs_unet.sh` runs both architectures with *everything except the
architecture* held fixed: same STFT (2048/1024), same augmentation, same loss (waveform L1 +
compressed-magnitude L1 - the RoFormer's multi-resolution term is switched off so nobody can
attribute a difference to the loss), same optimizer, same warm-up, same EMA, same 14
held-out validation songs, same number of steps, same batch size, same chunk length, same
seed. Both sides therefore see exactly the same amount of audio.

```
DEV=mps STEPS=150 ./scripts/ab_roformer_vs_unet.sh
DEV=mps STEPS=600 TAG=600 ./scripts/ab_roformer_vs_unet.sh
```

| steps | U-Net (10.55 M) | Mel-band RoFormer (2.75 M) | delta | wall-clock (U-Net / RoFormer) |
| --- | --- | --- | --- | --- |
| 150 | −7.84 dB | −8.44 dB | **−0.60 dB** (U-Net better) | 0.016 h / 0.023 h |
| 600 | −6.04 dB | **−5.49 dB** | **+0.55 dB** (RoFormer better) | 0.068 h / 0.095 h |

At 600 steps the RoFormer is better on **all four** stems (vocals −11.35 vs −11.51, drums
−6.97 vs −8.67, bass −0.82 vs −0.94, other −2.79 vs −3.04 dB), and over the 150 → 600
interval it gained **2.95 dB against the U-Net's 1.80 dB** at the same cost ratio.

Raw numbers on disk: `runs/ab_unet*/summary.json`, `runs/ab_roformer*/summary.json`,
logs in `runs/ab_*/log.csv`.

### How to read this honestly

1. **It is not a proof.** 600 steps on 3 s chunks is ~1 hour of audio and ~6 minutes of
   compute per side; the shipped U-Net checkpoint saw 2500 steps of 5 s × 4 (≈14 hours of
   audio) and reaches −5.76 dB on validation. Neither model here is converged, and the
   published numbers of this family (−9 dB and beyond) come from days of training on far
   more data.
2. **What it does show:** the RoFormer learns *faster per step* once past the initial
   phase, which is the expected signature of an attention model, and it does so with 3.8×
   fewer parameters - but at 3-4× the FLOPs per second of audio.
3. **What it does not show:** that the RoFormer's *ceiling* here is higher. To claim that
   you need a long run on both sides, e.g.

   ```bash
   python -m src.train_roformer --config configs/roformer_musdb18.yaml \
       --set train.epochs=20 --set train.steps_per_epoch=1000   # ~20k steps, ~7 h on an M5
   ```

   and then the same for the U-Net (`configs/unet_musdb18.yaml`, unchanged).

## 7. Verdict: is it better than what is currently shipped?

**Not proven, so the shipped checkpoint is unchanged.** `models/unet_musdb18_ema.pt` and all
of `src/separate.py`, `src/analyze.py` and `src/evaluate.py` still run the U-Net exactly as
before, and the Colab still clones and separates with it.

What would make the RoFormer the better default: a run of ≥20k steps on both, compared on
the MUSDB18 *test* split with the same BSS-Eval protocol (`src/metrics.py` is
architecture-agnostic - `si_sdr_per_source`, the BSS-Eval wrapper - but `src/evaluate.py`
loads the U-Net only, so RoFormer stems go through `src/separate_roformer.py` and then
`src/metrics.py`). Until such a run exists, the
honest statement is the one in the table above: *at equal short budgets the two are within
±0.6 dB of each other, each winning one point, with the RoFormer improving faster and
paying ~3-4× more compute per second of audio.*

For **state-of-the-art quality today**, without training anything, the right move is not
either of these two models: it is a pretrained member of this family (BS-RoFormer /
Mel-Band RoFormer checkpoints, e.g. via `nomadkaraoke/python-audio-separator` or the
original `lucidrains/BS-RoFormer` weights). This repository's value is that both
architectures are small, readable and trainable from scratch on a laptop.

## 8. Reproducing everything in this document

```bash
python -m pytest tests/ -q                                   # 39 tests (31 U-Net + 8 RoFormer)
python scripts/make_synthetic_data.py --out data/synthetic --train 6 --test 2
python -m src.train_roformer --config configs/roformer_smoke.yaml   # 0.4 M params, ~5 s
python scripts/benchmark_roformer.py --batches 1 2 --steps 5        # the numbers in §4
DEV=mps STEPS=150 ./scripts/ab_roformer_vs_unet.sh                  # the first row of §6
DEV=mps STEPS=600 TAG=600 ./scripts/ab_roformer_vs_unet.sh          # the second row of §6
```

## 9. Licence

Code: MIT, as the rest of the repository. Any weights trained with the recipe above are
derived from **MUSDB18** (CC BY-NC-SA 4.0) and inherit the non-commercial restriction, the
same as `models/MODEL_CARD.md` states for the U-Net checkpoint.

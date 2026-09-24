# Music source separation as image segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/audio-separation-unet/blob/main/notebooks/separate_audio_with_unet.ipynb)
[![tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)](#5-tests)

**Audio → spectrogram → U-Net → masks → separated audio.**
Upload any song to the Colab above, or run the app locally:
`python -m src.analyze --checkpoint models/unet_musdb18_ema.pt --input song.mp3 --out outputs/song`.

This project treats source separation as a **computer-vision segmentation problem**:
a song is turned into a 2-D image (a magnitude/phase spectrogram with an STFT), a
**U-Net** segments that image into per-instrument masks, the masks cut the sources out
of the original spectrogram, and an inverse STFT turns the result back into audio.

```
mixture waveform ──STFT──► spectrogram image (1025 × T per channel)
                                   │
                              U-Net (encoder/decoder + skip connections)
                                   │
                    complex masks per source: vocals / drums / bass / other
                                   │
                  mask ⊙ mixture spectrum ──iSTFT──► 4 separated waveforms
```

Trained on **MUSDB18** (100 training songs / 50 test songs, 4 stems each), evaluated
with the standard **BSS-Eval** metrics (SDR / SIR / SAR).

---

## 1. Install

```bash
python -m pip install -r requirements.txt   # torch, torchaudio, numpy, mir_eval, ...
brew install ffmpeg                          # only needed for the compressed MUSDB18
```

Everything runs on CPU, CUDA or Apple **MPS** (this was developed on an M5 / 16 GB).

## 2. Get the dataset

```bash
# MUSDB18 (4.7 GB, AAC-compressed) - the reference dataset for this task
python scripts/download_musdb.py --out data/musdb18.zip --extract-to data/musdb18

# MUSDB18-HQ (22.7 GB, uncompressed wav) if you have the disk space
python scripts/download_musdb.py --hq --out data/musdb18hq.zip --extract-to data/musdb18hq
```

The downloader is threaded and **resumable**: re-run the same command and it picks up
where it stopped (Zenodo is heavily throttled, expect ~1–2 MB/s).

Sanity-check what you downloaded:

```bash
python scripts/inspect_dataset.py --root data/musdb18 --tracks 3
python scripts/check_dataloader.py --verify-alignment   # stem/seek alignment check
```

`inspect_dataset.py` prints a per-stem fingerprint (RMS, spectral centroid, low-frequency
ratio). It is how you confirm the stream→instrument mapping: `bass` must be the stream
with ~60-70% of its energy below 200 Hz, `drums` the one with the highest centroid.

## 3. Train

```bash
python -m src.train --config configs/unet_musdb18.yaml          # ~6 h on an M5
python -m src.train --config configs/unet_musdb18.yaml --resume checkpoints/last.pt
python -m src.train --config configs/unet_musdb18.yaml --set train.batch_size=2 --set data.chunk_seconds=4
```

* validation runs every 250 steps on 14 held-out songs (SI-SDR per stem)
* checkpoints: `checkpoints/last.pt` and `checkpoints/best.pt` (best mean validation SI-SDR)
* logs: `runs/unet/log.csv`, summary in `runs/unet/summary.json`

Sizing helpers:

```bash
python scripts/benchmark_model.py --batches 2 4 6      # step time + memory per batch size
python scripts/benchmark_model.py --set stft.hop_length=512 --batches 4
```

## 4. Separate a song / evaluate

```bash
# run the model on any audio file -> outputs/<song>/{vocals,drums,bass,other}.wav
python -m src.separate --checkpoint checkpoints/best.pt --input song.mp3 --out outputs/song

# BSS-Eval (SDR/SIR/SAR) on the MUSDB18 test set
python -m src.evaluate --checkpoint checkpoints/best.pt --limit 10
python -m src.evaluate --checkpoint checkpoints/best.pt --save-stems outputs/test
```

## 4. The app: separate a song and inspect *what the model did*

```bash
python -m src.analyze --checkpoint models/unet_musdb18_ema.pt \
    --input song.mp3 --out outputs/song            # any wav/mp3/flac/m4a/ogg
python -m src.analyze --checkpoint models/unet_musdb18_ema.pt \
    --input song.mp3 --out outputs/song --reference-dir /path/to/stems   # + metrics
```

It writes `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`, `mixture.wav`, two figures
(`masks.png`: mixture spectrogram + the four predicted masks; `stems.png`: magnitude
spectrograms of the estimates) and a report. Real output (MUSDB18 test song, 200 s,
checkpoint `models/unet_musdb18_ema.pt`, see `results/example_report.txt` and `figures/`):

```
stem          rms     peak  energy %   centroid   |mask|
vocals     0.0169    0.219      5.4%      6070Hz     0.15
drums      0.0394    0.438     29.5%      4805Hz     0.55
bass       0.0185    0.195      6.5%      5149Hz     0.09
other      0.0556    0.624     58.5%      2854Hz     0.22

mixture consistency : residual of sum(stems) vs mixture = -146.8 dB relative to the input
mean sum of masks   : 1.02 +- 0.24 (ideally ~1.00)

mean |mask| per band (a bass mask should be heavier at low<200Hz)
stem      low<200Hz   200-2kHz      >2kHz
vocals         0.10       0.10       0.16
drums          0.42       0.38       0.57
bass           0.15       0.06       0.09
other          0.26       0.40       0.21

estimates vs ground truth (a large centroid mismatch = leakage of other instruments)
stem      rms est  rms ref  centroid est  centroid ref
vocals     0.0169   0.0725       6070 Hz       3404 Hz
drums      0.0394   0.0433       4805 Hz       5692 Hz
bass       0.0185   0.0527       5149 Hz        222 Hz
other      0.0556   0.0480       2854 Hz       3224 Hz

against the reference stems (mean SI-SDR -3.73 dB)
stem        SI-SDR      SDR      SIR      SAR
vocals       -1.76     1.57     2.64    12.82
drums        -6.03     0.11    -2.69    13.95
bass         -2.10     1.26    -1.27    13.00
other        -5.03    -0.39    -3.39    14.65
```

The DSP is exact (consistency residual **−147 dB**), and compared with the step-1200
checkpoint on the same song the numbers that describe *what the model does* have changed
sign: the four masks now sum to **1.02 ± 0.24** instead of 0.74 ± 0.28 (a calibrated mask
set, i.e. the network is not just re-levelling), the `drums` mask is much more active
(mean |mask| 0.55 vs 0.21) and the `bass` mask is heavier below 200 Hz (0.15) than above
2 kHz (0.09) instead of the other way round. The per-stem SI-SDR on this one 30 s excerpt is
a wash (vocals worse, `bass`/`other` better) - it is the song where the retrained model gains
least (+0.13 dB of the 10 songs in the table below), and the `centroid est vs ref` row still
says the `bass` estimate is far too bright, so there is real room left. The aggregate numbers
are the ones to trust.

**Listen**: `samples/` contains 24 s excerpts (regenerated with the shipped checkpoint) of the
mixture, the four estimates *and* the ground truth for the same test song, as mp3.

## 4b. Pretrained weights and Colab

* `models/unet_musdb18_ema.pt` (42 MB) — EMA weights of the best validation checkpoint,
  float32, loaded with `src.separate.load_model` or the `--checkpoint` flag everywhere.
  Details, measured metrics and limitations: **`models/MODEL_CARD.md`**.
* `notebooks/separate_audio_with_unet.ipynb` — the Colab: clones this repo (weights
  included), self-tests the pipeline, lets you upload a song, separates it, displays
  spectrograms/masks, plays the stems, prints the report and lets you download them.
* Regenerate the small inference checkpoint from a training checkpoint:

  ```bash
  python scripts/export_inference_checkpoint.py checkpoints/best.pt \
      --out models/unet_musdb18_ema.pt --half     # --half for a 21 MB float16 export
  ```

  By default the EMA weights are exported; `--weights live` takes the raw ones instead,
  which is worth checking on short runs (`ema_decay: 0.999` only averages away the
  initialisation after a few thousand steps). Compare the two with
  `python scripts/diagnose_checkpoint.py checkpoints/best.pt`.

## 5. Tests

```bash
python -m pytest tests/ -q          # 31 tests, ~1 min on CPU
```

They cover the STFT round-trip (`istft(stft(x)) == x`), U-Net shapes/gradients, the
mask/consistency algebra, the dataset and augmentation layer (including
`test_stem_swap_never_relabels_the_instruments`, the regression test for the stem-swap bug
that turned the first model into a volume knob), BSS-Eval behaviour on
known cases, and a full end-to-end pipeline (train 6 steps → checkpoint → separate →
evaluate) on a procedural dataset, so the whole path is testable **without** the 4.7 GB
download:

```bash
python scripts/make_synthetic_data.py            # tiny procedural dataset
python -m src.train --config configs/unet_smoke.yaml
```


## 6. Design decisions (and why)

| Decision | Reason |
| --- | --- |
| **U-Net, 5 levels, 32 base channels** | classic segmentation network: the encoder captures context (which instruments are playing), the skip connections carry the fine detail (note onsets, harmonics) that downsampling destroys |
| **Complex ratio masks (cIRM), tanh output** | predicting (real, imag) per source lets the model fix *phase* too, not just magnitude; `tanh` bounds the mask to [-1, 1] exactly like the definition of cIRM. `model.mask: sigmoid` gives the purely real magnitude-mask variant |
| **Magnitude-only input** | the network sees `log|X|` of the mixture, standardised per example; the phase is not an input but is carried through the mask multiplication |
| **STFT 2048 / hop 1024 (50% overlap)** | a 46 ms window resolves pitch while the hop decides the cost. At 44.1 kHz the U-Net sees 216 frames per 5 s crop: measured ~4x faster than hop 512 with no real quality loss. Zero padding + exact framing makes `iSTFT(STFT(x)) = x` bit-exact (tested) |
| **Hybrid loss: waveform L1 + compressed magnitude L1** | the waveform term is what Demucs-style systems use and correlates with perceptual quality; the `|X|^0.3` term stops quiet material (reverb tails, cymbals, breath) from being ignored |
| **Mixture consistency** | the four estimates are projected so they sum back to the mixture (`est += (mix - sum(est))/4`). Free SDR, no training instability |
| **Aggressive augmentation** | MUSDB18 train has only 100 songs: random gains, channel swaps, polarity flips, silencing a source, swapping one stem with the *same* stem of another song (never a different instrument - that would relabel the targets), and full remixing. This is the most important regulariser here |
| **Segment decoding + chunk reuse** | songs are AAC-compressed; decoding a whole song per sample made the loader the bottleneck (5 s per batch!). Now a random 45 s slice is decoded once and 6 crops are taken from it (plus a 3-song "recent" pool for cross-song augmentation) → **103 s of audio/s**, 5x faster than the GPU |
| **EMA weights + selection on validation SI-SDR** | the moving average is a free stability/quality win; the checkpoint is chosen on 14 held-out songs, never on test |
| **BSS-Eval with 1 s frames, median** | the `museval` protocol used in the MUSDB18 literature, so numbers are comparable |

## 7. Project layout

```
configs/            YAML configs (full MUSDB18 recipe + tiny smoke recipe)
notebooks/          Colab notebook: upload a song -> stems + masks + report
models/             inference checkpoint (42 MB) + MODEL_CARD.md
samples/            example input/estimates/references (24 s mp3 excerpts)
figures/            example masks.png / stems.png from the app
results/            the evidence: training log/curve, test-set metrics, example report,
                    ab_augment_fix/ (the controlled A/B behind the augmentation fix)
scripts/
  download_musdb.py       resumable threaded MUSDB18 downloader (Zenodo)
  inspect_dataset.py      verify a download: stem fingerprints, order, mixture
  check_dataloader.py     dataloader throughput + stem/seek alignment verification
  diagnose_checkpoint.py  live vs EMA vs oracle SI-SDR of a checkpoint
  diagnose_masks.py       is the model separating, or only changing the volume?
  benchmark_model.py      step time & memory for batch sizes / model settings
  profile_step.py         per-stage timing of a training step
  make_synthetic_data.py  procedural dataset, so tests need no download
src/
  config.py         dataclass configs, YAML loader, dotted --set overrides
  dsp.py            STFT/iSTFT (exact), masks, mixture consistency, SI-SDR
  losses.py         hybrid waveform + compressed-magnitude loss
  metrics.py        SI-SDR and BSS-Eval (SDR/SIR/SAR) aggregation
  train.py          training loop: AdamW + warm-up/cosine, EMA, checkpoint/resume
  separate.py       CLI: audio file -> 4 stem wavs (sliding-window overlap-add)
  evaluate.py       CLI: BSS-Eval over a dataset subset, CSV + JSON reports
  data/
    musdb.py        both MUSDB18 flavours, ffmpeg stream decoding, LRU cache
    chunks.py       training sampler (segments + augmentation) & fixed eval chunks
    synthetic.py    procedural 4-stem dataset (tests / smoke runs)
  models/
    unet.py         the segmentation U-Net (BN, bilinear upsampling, skip fusion)
    separation.py   STFT wrapper: waveform -> masks -> masked spectra -> waveform
tests/              31 tests: dsp, model, data, metrics, training, end-to-end pipeline
```

## 8. Hardware notes (measured on Apple M5, 16 GB)

| Configuration | step time (batch 4) | audio throughput | peak memory |
| --- | --- | --- | --- |
| hop 512, chunk 5 s, base 32 | ~2.6 s | ~7 s/s | > 8 GB |
| **hop 1024, chunk 5 s, base 32** | **1.0 s** | **20 s/s** | **5.7 GB** |
| hop 1024, chunk 5 s, base 24 | 0.8 s | 24 s/s | 4.6 GB |
| data pipeline (4 workers, segment decoding) | — | 103 s/s | ~1.5 GB |

One "epoch" of MUSDB18 (1000 steps) is ~17 minutes here, so the reference recipe
(20k steps) is a ~6 hour run. The weights shipped in `models/` come from a shorter one -
2500 steps (10 epochs × 250) in ~55 minutes, which is what one evening on this laptop
buys. For a quicker, weaker model reduce `train.epochs`/`steps_per_epoch`, or try
`--set model.base_channels=24 --set data.chunk_seconds=4`.

## 9. Limitations / honest notes

* This is a **mask-based spectrogram U-Net trained from scratch** on 100 songs, not a
  large pretrained system. It will not match Demucs/HTDemucs (waveform models trained
  for weeks, SDR ~9 dB), but it is the exact architecture family the "CV view" of the
  problem describes, and it is small enough to train on a laptop.
* The 44.1 kHz stereo model does not explicitly model phase *prediction* beyond the
  complex mask, so percussion transients are the hardest part.
* The compressed MUSDB18 release is lossy AAC: `sum(stems)` differs from the container's
  mixture stream by ~2-4 % (measured), which is why the mixture is defined as the sum of
  the stems throughout this project.
* `mir_eval`'s framewise BSS-Eval helpers are deprecated in 0.8 and removed in 0.9, so
  `src/metrics.py` implements the 1-second frame loop itself on the stable API.


## 10. Results

All numbers are **SI-SDR in dB on 5-second validation chunks** (14 songs held out of the
MUSDB18 train split) unless stated otherwise, plus the standard **BSS-Eval** protocol
(1 s frames, median over frames and songs) on the official test split.

### Reference points (what the model has to beat)

Measured with `scripts/diagnose_checkpoint.py` and `scripts/oracle_mask_baseline.py`:

| estimator | mean SI-SDR | hybrid loss |
| --- | --- | --- |
| oracle ideal ratio mask (computed from the ground truth) | **+8.05 dB** | 0.041 |
| mixture used as the estimate of every source | −6.43 dB | 0.220 |
| silence (zero output) | −80 dB (undefined) | 0.233 |

The oracle mask is the *upper bound for any magnitude-mask model* with this STFT, so it
is the honest ceiling for this architecture. It doubles as an end-to-end self-test: if the
STFT framing, the stem/seek alignment or the BSS-Eval call were wrong, that number would
collapse instead of landing at a plausible 8 dB.

A companion measurement lives in *Why the first trained model only changed the volumes*
below: `scripts/diagnose_masks.py` adds a per-source **best fader** (the optimal 50 ms
time-varying gain on the mixture - the strongest thing "volume automation" can do) and
reports what is inside the predicted masks. The step-1200 checkpoint does not clear it.

### Training curve (validation, EMA weights)

The shipped checkpoint is the **retrained** run (`runs/unet_fixed/log.csv`, 2500 steps,
0.5 h), and unlike the first one its validation numbers actually move:

```
   step    loss    wave     mag  s/step   vocal   drums    bass   other    mean
-------------------------------------------------------------------------------
    250  0.1150  0.0392  0.1517    1.38  -17.90  -12.03   -8.98   -7.78  -11.67
    500  0.1039  0.0339  0.1400    1.34  -17.50  -10.66   -8.25   -9.54  -11.49
    750  0.1003  0.0332  0.1342    1.33  -17.61  -10.32   -9.11   -7.14  -11.04
   1000  0.0940  0.0294  0.1292    1.30  -17.12   -9.79   -8.99   -6.33  -10.56
   1250  0.0944  0.0290  0.1308    1.16  -16.92   -9.34   -7.98   -6.00  -10.06
   1500  0.0941  0.0299  0.1284    1.26  -16.32   -9.22   -7.33   -6.24   -9.78
   1750  0.0885  0.0283  0.1205    1.25  -14.80   -7.48   -4.35   -4.07   -7.67
   2000  0.0823  0.0268  0.1110    1.24  -12.96   -6.35   -2.56   -2.66   -6.13
   2250  0.0905  0.0297  0.1217    1.27  -12.24   -6.28   -2.40   -2.11   -5.76
   2500  0.0878  0.0279  0.1198    1.35  -12.05   -6.80   -2.37   -1.90   -5.78
```

For comparison the first run (mislabelled targets, `runs/unet/log.csv`) went from −11.82 dB at
step 250 to **−7.85 dB at step 1200** and then stopped improving; the retrained model is
**2.1 dB better** than that on the same 14 held-out songs.

One detail worth knowing about this curve: with `ema_decay: 0.999` the evaluated EMA weights
lag the live ones, badly at the start (at step 600 of the first run the EMA was ~6 dB worse
than the weights that were actually training) and barely at the end (by step 2250 the EMA is
0.4 dB *better*). Short runs should therefore use `ema_decay: 0.995`; the full 20k-step
recipe can afford 0.999. Full curve in `runs/unet_fixed/log.csv`, or:

```bash
python scripts/summarize_run.py runs/unet_fixed/log.csv
python scripts/diagnose_checkpoint.py checkpoints/fixed/best.pt --chunks 8 --device mps
```

### Live-vs-EMA weights on the same chunks (step 2250)

`python scripts/diagnose_checkpoint.py checkpoints/fixed/best.pt --chunks 8`:

| estimator | vocals | drums | bass | other | mean | loss |
| --- | --- | --- | --- | --- | --- | --- |
| EMA weights (**shipped**) | **+1.35** | −7.61 | +4.23 | −3.30 | **−1.33** | 0.114 |
| live weights | −3.06 | **−6.67** | **+5.55** | **−2.72** | −1.72 | 0.107 |
| oracle ideal mask | +10.48 | +3.48 | +12.05 | +6.20 | **+8.05** | 0.041 |
| mixture as estimate | −6.67 | −10.05 | −0.45 | −8.54 | −6.43 | 0.220 |
| zero output | −80 | −80 | −80 | −80 | −80 | 0.233 |

The EMA wins on the mean - it is smoother on `vocals`, the source where the live weights are
noisiest - and that is what `models/` ships; the live weights are ahead on `bass` and `drums`
(which is why `scripts/export_inference_checkpoint.py` grew a `--weights live` flag). Bass is
learned first, drums last - the usual pattern for mask-based models, which have no explicit
transient model.

### MUSDB18 test split — chunk-average SI-SDR (10 songs)

5-second chunk SI-SDR averaged over each whole song (`src/evaluate.py --fast`), same metric
for both checkpoints and for the trivial baseline (`scripts/baseline_bss_eval.py --fast`), on
the first 10 songs of the official test split (`runs/fast_fixed/`, `results/test_si_sdr_summary.txt`):

| system | median SI-SDR | vocals | drums | bass | other |
| --- | --- | --- | --- | --- | --- |
| **U-Net, retrained (shipped, step 2250)** | **−6.40 dB** | −14.35 | −3.69 | −2.64 | −2.67 |
| U-Net, step 1200 (mislabelled targets) | −8.29 dB | −16.76 | −4.70 | −5.43 | −3.56 |
| trivial "mixture" estimator | −9.99 dB | — | — | — | — |
| **improvement over the baseline** | **+3.59 dB** | | | | |

The shipped model beats the trivial baseline on every song and the previous checkpoint on 9 of
the 10 (+0.13 to +2.55 dB, one song −0.50 dB), and it improves all four sources. Its worst
case is still a quiet, reverb-heavy track (−22.6 dB). Reference point from the same pipeline:
the oracle ideal-ratio mask reaches **+8.05 dB** SI-SDR, the ceiling for this architecture and
STFT.

### MUSDB18 test split — standard BSS-Eval (SDR / SIR / SAR)

First 60 s of the first two test songs, 1-second frames, median over frames. These numbers are
the **step-1200** checkpoint's (`runs/final/eval/summary.json`, produced before the fix); the
retrained model's BSS-Eval is a ~15 minute `mir_eval` run and is recomputed separately into
`results/bss_eval_2songs.json`:

| system | SDR | SIR | SAR |
| --- | --- | --- | --- |
| U-Net, step 1200 | **+1.16 dB** | −2.84 dB | +19.32 dB |
| trivial mixture baseline | −5.65 dB | — | — |

SDR is *positive* here while the SI-SDR above is negative, and that is not a
contradiction: BSS-Eval allows a 512-tap time-invariant filter between estimate and
target, so it forgives spectral/temporal distortion but not leakage. The decomposition
says exactly what is happening: almost no artifacts (SAR +19 dB) and the estimates
follow the target structure, but **interference is still the limiting factor
(SIR −2.8 dB)** - the masks are in the right place, just not selective enough. That
is precisely what more training buys, and the retrained model's numbers go here when the
slow run lands.

> **Runtime note.** `mir_eval`'s BSS-Eval is CPU-bound (~8 minutes per minute of audio
> here: it solves a 512-tap filter decomposition per 1-second frame *and* per source), so
> the standard metric is reported on a subset. The full 50-song evaluation is the same
> command with `--limit 50`, given the machine time.

> **Note on the per-source block** in `runs/final/eval/summary.json`: per-source
> aggregation was added to `src/metrics.py` (`bss_eval_track_per_source`) *after* that
> run, so the JSON from that snapshot repeats the track-level values; re-running the
> command produces the correct per-source numbers.

Separated audio to listen to: `outputs/test_eval2/<song>/{vocals,drums,bass,other}.wav`
(`mix2` / `mixture.wav` next to them for an A/B comparison).



### Why the first trained model only changed the volumes (and what was fixed)

The first checkpoint this repository shipped (step 1200 of `runs/unet`, the one whose
numbers are quoted in the tables above) separated nothing: its four estimates sound like the
same song at four different levels. `scripts/diagnose_masks.py` (new) puts that network next
to the strongest things a pure "volume knob" can do, on deterministic validation chunks (the
first four of the 14 held-out songs, one 5-second chunk each,
`--chunks 4`):

```
estimator               vocals   drums    bass   other    mean
model                    -3.96   -9.76    3.34   -5.37   -3.94
mixture as estimate      -6.67  -10.05   -0.45   -8.54   -6.43
best static gain         -6.67  -10.05   -0.45   -8.54   -6.43
best fader               -1.84   -3.50    1.68   -6.53   -2.55   <- a fader beats it
oracle ideal mask        10.48    3.48   12.05    6.20    8.05
```

and it shows what is inside the masks (F/T/FxT are shares of the mask's variance, `low`/`high`
are the mean |mask| per bin below 200 Hz and above 2 kHz):

```
stem      mean|m|     cv  F share  T share    FxT    low   high
vocals      0.177   0.26     0.75     0.03   0.21   0.11   0.18
drums       0.196   0.35     0.76     0.05   0.18   0.17   0.21
bass        0.118   1.02     0.95     0.01   0.04   0.21   0.12
other       0.240   0.38     0.81     0.05   0.14   0.16   0.24
```

The masks have a plausible spectral tilt - the `bass` mask *is* heavier below 200 Hz than
above 2 kHz - but 75-95 % of each mask's variance is that fixed frequency profile and only
0.01-0.05 of it varies with *time*: the network is applying an EQ and barely looking at when
something happens. That is exactly the shape of the SI-SDR table above, where the model loses
to a per-source fader. `figures/masks_before_after.png` shows the same validation chunk
through both checkpoints: on the left the four masks are flat bands of colour, on the right
the retrained `drums` mask shows the vertical striations of percussive onsets and all four
masks visibly vary in time and frequency. Two defects produced this:

1. **The stem-swap augmentation relabelled the targets.** `_augment` is documented as
   "swapping a stem for the *same* stem of another song" (that is what teaches robustness
   to unseen instrument combinations), but it drew the donor with
   `j = rng.randrange(n_stems)`. With four sources that put a **different instrument in the
   slot 3 times out of 4**: on 400 augmented batches, **36.5 % of all targets were the wrong
   instrument** (`drums` content in the `vocals` slot, and so on). Nothing in the mixture
   reveals that substitution, so the target of a head became partly unpredictable and the
   loss-minimising answer is the conditional mean - a per-source gain on the mixture. The
   swap now takes the same stem index, and the same measurement reports **0.0 %**.
   Locked by `tests/test_data.py::test_stem_swap_never_relabels_the_instruments`.
2. **A resumed run inherited a dead learning rate.** `lr_at(step, total, cfg)` measured the
   cosine schedule from step 0, so resuming at step 900 of a 1200-step schedule restarted at
   ~5e-6 (`runs/train_sdr.log`) and the "continue training" chain learned nothing while
   looking like it ran for 300 more steps. The schedule is now measured from the resume
   step, with a short re-warm-up (`tests/test_train.py`).

Controlled A/B on the procedural dataset (`src/data/synthetic.py`, 600 steps, batch 4,
identical seed/hyper-parameters, only `_augment` differs; validation SI-SDR in dB, EMA
weights, the same held-out chunk):

| step | old `_augment`, mean | fixed `_augment`, mean |
| --- | --- | --- |
| 100 | −10.02 dB | **−7.36 dB** |
| 300 | −7.57 dB | **−5.63 dB** |
| 600 | −6.74 dB | **−1.53 dB** |

and per source at step 600 (the relabelled run never gets a source above 1 dB):

| source | old `_augment` | fixed `_augment` |
| --- | --- | --- |
| vocals | −5.69 dB | **+1.34 dB** |
| drums | −18.88 dB | −19.77 dB |
| bass | +0.87 dB | **+11.65 dB** |
| other | −3.26 dB | **+0.64 dB** |

Both arms are dominated by `drums` (the procedural drums are noise bursts) and both are far
from the oracle mask of the same chunks (≈ +13 dB mean), but the fixed arm wins on three
sources from step 300 onwards and the gap on the mean grows to 5.2 dB. The raw logs ship in
`results/ab_augment_fix/` (`ab_a_old_augment_log.csv`, `ab_b_fixed_augment_log.csv`); note
that a rerun measures this on a single 2 s validation chunk, so treat the numbers as a
trend, not as a benchmark.

The run itself is 600 steps on CPU, so it is cheap to repeat (`arm A` = the same command with
the old `j = rng.randrange(other_stems.shape[0])` line restored):

```bash
python -m src.train --config configs/unet_smoke.yaml \
    --set data.chunk_seconds=2.0 --set data.chunks_per_load=2 --set data.valid_tracks=1 \
    --set train.batch_size=4 --set train.steps_per_epoch=200 --set train.epochs=3 \
    --set train.warmup_steps=50 --set train.lr=0.001 --set train.val_every=100 \
    --set train.val_chunks=2 --set train.log_every=100 --set train.ema_decay=0.99 \
    --set train.device=cpu --set train.out_dir=runs/ab_b --set train.ckpt_dir=checkpoints/ab_b
```

### The same measurement after the fix

The retrained checkpoint, on the same four validation chunks
(`python scripts/diagnose_masks.py checkpoints/fixed/best.pt --chunks 4`):

```
estimator               vocals   drums    bass   other    mean
model                     1.35   -7.61    4.23   -3.30   -1.33
mixture as estimate      -6.67  -10.05   -0.45   -8.54   -6.43
best static gain         -6.67  -10.05   -0.45   -8.54   -6.43
best fader               -1.84   -3.50    1.68   -6.53   -2.55
oracle ideal mask        10.48    3.48   12.05    6.20    8.05
```

| | step-1200 model | retrained model |
| --- | --- | --- |
| mean SI-SDR (4 validation chunks) | −3.94 dB | **−1.33 dB** |
| vs the best fader on the same chunks | **1.39 dB worse** | **1.22 dB better** |
| `vocals` | −3.96 dB | **+1.35 dB** |
| `bass` | +3.34 dB | +4.23 dB |
| `drums` | −9.76 dB | −7.61 dB |
| `other` | −5.37 dB | −3.30 dB |

The masks changed in the directions the SI-SDR rows predict (mean gain, coefficient of
variation and the mean |mask| per bin in the two bands, before → after):

| stem | mean gain | cv | low<200Hz | high>2kHz |
| --- | --- | --- | --- | --- |
| vocals | 0.177 → 0.163 | 0.26 → 0.50 | 0.11 → 0.09 | 0.18 → 0.16 |
| drums | 0.196 → **0.469** | 0.35 → 0.49 | 0.17 → 0.33 | 0.21 → 0.49 |
| bass | 0.118 → 0.104 | 1.02 → **1.31** | 0.21 → **0.32** | 0.12 → 0.11 |
| other | 0.240 → 0.230 | 0.38 → 0.71 | 0.16 → 0.19 | 0.24 → 0.21 |

The `bass` mask is now nearly three times heavier below 200 Hz than above 2 kHz (0.32 vs
0.11) and the `drums` mask is both much more active and more selective - the network is using
the time-frequency plane instead of only a static tilt.

Two things are worth reading carefully here. The retrained model is the first checkpoint in
this repository that **beats the strongest volume-only baseline** (a per-source fader) - on
three sources and by 1.2 dB on the mean - and its `vocals` estimate has *positive* SI-SDR,
which a gain on the mixture can never reach (the "mixture" row is the exact score of every
static gain, by scale invariance). `drums` is the exception: a fader still beats it, which is
the transient/phase problem of a mask model, not a labelling problem.

The remaining distance to the oracle mask (+8.05 dB on those chunks) is the honest ceiling of
what more training can buy here, and the `--resume` path that makes that affordable is fixed
(see above).

## 12. Reproducing the state of this repository

```bash
python -m pytest tests/ -q                       # 31 tests
python scripts/check_separation.py               # which checkpoint, and is it separating?
python scripts/inspect_dataset.py --tracks 2     # dataset sanity check
python scripts/check_dataloader.py --verify-alignment
python scripts/diagnose_checkpoint.py checkpoints/best.pt --chunks 8   # live/EMA/oracle
python scripts/diagnose_masks.py checkpoints/best.pt --chunks 4        # masks vs "volume knobs"
python -m src.evaluate --checkpoint checkpoints/best.pt --limit 5
```

The shipped checkpoint was produced by exactly this run (~1 h on an M5), and its full config
is in `results/training_config.yaml`:

```bash
python -m src.train --config configs/unet_musdb18.yaml \
    --set train.epochs=10 --set train.steps_per_epoch=250 --set train.warmup_steps=250 \
    --set train.max_hours=2.0 --set train.log_every=50 \
    --set train.out_dir=runs/unet_fixed --set train.ckpt_dir=checkpoints/fixed
python scripts/compare_checkpoints.py checkpoints/best_step1200.pt checkpoints/fixed/best.pt \
    --out figures/masks_before_after.png          # the before/after figure
```

`docs/PROPOSAL.md` frames the whole thing as a Computer Vision course project
(what maps to which CV concept, realistic expectations, possible extensions).
## 13. Going further (how to get a *strong* model on this laptop)

The shipped recipe targets ~20k steps (~6 h). If you want a genuinely good separator,
the cheapest wins, in order:

1. **Train longer.** The loss was still descending when this snapshot was taken. `--resume`
   now re-measures the learning-rate schedule from the step you resume at (see
   `lr_at(..., start_step)`), so a continued run anneals over the steps it is actually
   going to take instead of starting at its floor:

   ```bash
   python -m src.train --config configs/unet_musdb18.yaml --resume checkpoints/last.pt \
       --set train.steps_per_epoch=400 --set train.epochs=25 --set train.max_hours=10
   ```

   `scripts/continue_training_chain.sh` is exactly this, chained after an evaluation.
2. **Keep the validation honest.** `train.val_every` controls checkpoint selection: the
   checkpoint is picked by validation SI-SDR on 14 held-out songs, never on the test set.
   Use `ema_decay: 0.995` for short runs and `0.999` for long ones (with 0.999 the EMA
   lags by ~1000 steps, which is invisible at 100k steps but misleading at 1k).
3. **Use MUSDB18-HQ** (uncompressed) if you have 23 GB free: the compressed release is
   lossy AAC, which costs a few tenths of a dB and adds ~2-4 % mixture inconsistency.
4. **Bigger model / more context** once the GPU is faster than the data loader
   (`scripts/benchmark_model.py` tells you): increase `model.base_channels` or
   `data.chunk_seconds`; the pipeline is fully convolutional, so both are safe changes.
5. **Then** experiment with the interesting research knobs: magnitude vs complex masks,
   waveform vs multi-resolution spectral loss, STFT resolution, attention or dilated
   blocks in the bottleneck (see `docs/PROPOSAL.md` for a project-shaped list).



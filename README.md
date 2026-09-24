# Music source separation as image segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/audio-separation-unet/blob/main/notebooks/separate_audio_with_unet.ipynb)
[![tests](https://img.shields.io/badge/tests-28%20passed-brightgreen)](#5-tests)

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
checkpoint `models/unet_musdb18_ema.pt`, see `results/example_report.txt` and
`figures/`):

```
stem          rms     peak  energy %   centroid   |mask|
vocals     0.0287    0.278     23.1%      3826Hz     0.18
drums      0.0212    0.192     12.6%      4324Hz     0.21
bass       0.0256    0.208     18.4%      3954Hz     0.11
other      0.0405    0.394     45.9%      3455Hz     0.24

mixture consistency : residual of sum(stems) vs mixture = -147.7 dB relative to the input
mean sum of masks   : 0.74 +- 0.28 (ideally ~1.00)

where each mask puts its energy (a bass mask should be low-heavy)
stem      low<200Hz   200-2kHz      >2kHz
vocals        0.7%      7.5%     91.8%
drums         0.7%      5.4%     93.9%
bass          1.7%      4.7%     93.7%
other         0.8%      9.2%     90.0%

estimates vs ground truth (a large centroid mismatch = leakage of other instruments)
stem      rms est  rms ref  centroid est  centroid ref
vocals     0.0287   0.0725       3826 Hz       3404 Hz
drums      0.0212   0.0433       4324 Hz       5692 Hz
bass       0.0256   0.0527       3954 Hz        222 Hz
other      0.0405   0.0480       3455 Hz       3224 Hz

against the reference stems (mean SI-SDR -3.38 dB)
stem        SI-SDR      SDR      SIR      SAR
vocals        0.15     2.14     0.10    23.05
drums        -5.95     1.10    -4.01    18.64
bass         -2.33     1.86    -1.14    20.46
other        -5.38    -0.05    -3.88    21.01
```

This is the honest picture of the model at this training budget: the DSP is exact
(consistency residual **−148 dB**), `vocals` and `other` land close to their references,
while `bass` still leaks everything (centroid 3954 Hz vs 222 Hz) because the masks are
still nearly frequency-agnostic (90 % of their mass above 2 kHz). Those are precisely the
diagnostics a longer training run has to move, and the app prints them.

**Listen**: `samples/` contains 24 s excerpts of the mixture, the four estimates *and* the
ground truth for one test song, as mp3.

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

## 5. Tests

```bash
python -m pytest tests/ -q          # 28 tests, ~40 s on CPU
```

They cover the STFT round-trip (`istft(stft(x)) == x`), U-Net shapes/gradients, the
mask/consistency algebra, the dataset and augmentation layer, BSS-Eval behaviour on
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
| **Aggressive augmentation** | MUSDB18 train has only 100 songs: random gains, channel swaps, polarity flips, silencing a source, swapping one stem with a *different song's* stem, and full remixing. This is the most important regulariser here |
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
results/            the evidence: training log/curve, test-set metrics, example report
scripts/
  download_musdb.py       resumable threaded MUSDB18 downloader (Zenodo)
  inspect_dataset.py      verify a download: stem fingerprints, order, mixture
  check_dataloader.py     dataloader throughput + stem/seek alignment verification
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
tests/              28 tests: dsp, model, data, metrics, end-to-end pipeline
```

## 8. Hardware notes (measured on Apple M5, 16 GB)

| Configuration | step time (batch 4) | audio throughput | peak memory |
| --- | --- | --- | --- |
| hop 512, chunk 5 s, base 32 | ~2.6 s | ~7 s/s | > 8 GB |
| **hop 1024, chunk 5 s, base 32** | **1.0 s** | **20 s/s** | **5.7 GB** |
| hop 1024, chunk 5 s, base 24 | 0.8 s | 24 s/s | 4.6 GB |
| data pipeline (4 workers, segment decoding) | — | 103 s/s | ~1.5 GB |

One "epoch" of MUSDB18 (1000 steps) is ~17 minutes here, so the shipped recipe
(20k steps) is a ~6 hour run. For a quicker, weaker model reduce
`train.epochs`/`steps_per_epoch`, or try `--set model.base_channels=24
--set data.chunk_seconds=4`.

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

### Training curve (validation, EMA weights)

```
   step    loss    wave     mag  s/step   vocal   drums    bass   other    mean
-------------------------------------------------------------------------------
    250  0.1357  0.0451  0.1812    1.25  -17.52  -11.42  -10.56   -7.79  -11.82
    300  0.1357  0.0451  0.1812    1.25  -17.50  -11.00   -8.66   -7.48  -11.16
    600  0.1357  0.0451  0.1812    1.25  -17.49  -11.48   -9.49   -6.36  -11.21
```

One detail worth knowing about this early part of the curve: with `ema_decay: 0.999` the
evaluated EMA weights are still ~50 % initial weights after 600 steps, so the validation
SI-SDR looks flat while the **live** weights were already at **−4.91 dB** - i.e. better
than the trivial baseline. Short runs should therefore use `ema_decay: 0.995`; the full
20k-step recipe can afford 0.999. Full curve in `runs/unet/log.csv`, or:

```bash
python scripts/summarize_run.py runs/unet/log.csv
python scripts/diagnose_checkpoint.py checkpoints/best.pt --chunks 8 --device mps
```

### Live-vs-EMA model on the same chunks (step 600)

| estimator | vocals | drums | bass | other | mean | loss |
| --- | --- | --- | --- | --- | --- | --- |
| EMA weights | −7.23 | −11.39 | −3.48 | −9.04 | −7.78 | 0.135 |
| live weights | −5.89 | −10.38 | **+1.95** | −5.34 | −4.91 | 0.125 |
| oracle ideal mask | +10.48 | +3.48 | +12.05 | +6.20 | **+8.05** | 0.041 |
| mixture as estimate | −6.67 | −10.05 | −0.45 | −8.54 | −6.43 | 0.220 |
| zero output | −80 | −80 | −80 | −80 | −80 | 0.233 |

Bass is learned first (low-frequency energy is easy to mask), drums last - the usual
pattern for mask-based models, which have no explicit transient model.

### MUSDB18 test split — chunk-average SI-SDR (10 songs)

5-second chunk SI-SDR averaged over each whole song (`src/evaluate.py --fast`), same
metric for the model and for the trivial baseline (`scripts/baseline_bss_eval.py --fast`),
on the first 10 songs of the official test split (`runs/fast2/SUMMARY.txt`):

| system | median SI-SDR | vocals | drums | bass | other |
| --- | --- | --- | --- | --- | --- |
| **U-Net (best checkpoint)** | **−8.29 dB** | −16.76 | −4.70 | −5.43 | −3.56 |
| trivial "mixture" estimator | −9.99 dB | — | — | — | — |
| **improvement** | **+1.70 dB** | | | | |

The model beats the trivial baseline on 9 of the 10 songs (+0.6 to +2.5 dB per song), with
one failure mode on a quiet, reverb-heavy track (−1.4 dB). Reference points from the same
pipeline: the pre-fine-tune checkpoint scored −8.84 dB, and the oracle ideal-ratio mask
reaches **+8.05 dB** SI-SDR, which is the ceiling for this architecture and STFT.

### MUSDB18 test split — standard BSS-Eval (SDR / SIR / SAR)

Same checkpoint, first 60 s of the first two test songs, 1-second frames, median over
frames (`runs/final/eval/summary.json`):

| system | SDR | SIR | SAR |
| --- | --- | --- | --- |
| U-Net (best checkpoint) | **+1.16 dB** | −2.84 dB | +19.32 dB |
| trivial mixture baseline | `runs/final/baseline_bss.json` | | |

SDR is *positive* here while the SI-SDR above is negative, and that is not a
contradiction: BSS-Eval allows a 512-tap time-invariant filter between estimate and
target, so it forgives spectral/temporal distortion but not leakage. The decomposition
says exactly what is happening: almost no artifacts (SAR +19 dB) and the estimates
follow the target structure, but **interference is still the limiting factor
(SIR −2.8 dB)** - the masks are in the right place, just not selective enough yet. That
is precisely what more training fixes, and it is the standard diagnosis for an
under-trained mask model.

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



## 11. Reproducing the state of this repository

```bash
python -m pytest tests/ -q                       # 28 tests
python scripts/inspect_dataset.py --tracks 2     # dataset sanity check
python scripts/check_dataloader.py --verify-alignment
python scripts/diagnose_checkpoint.py checkpoints/best.pt --chunks 8   # live/EMA/oracle
python -m src.evaluate --checkpoint checkpoints/best.pt --limit 5
```

`docs/PROPOSAL.md` frames the whole thing as a Computer Vision course project
(what maps to which CV concept, realistic expectations, possible extensions).
## 12. Going further (how to get a *strong* model on this laptop)

The shipped recipe targets ~20k steps (~6 h). If you want a genuinely good separator,
the cheapest wins, in order:

1. **Train longer.** The loss was still descending when this snapshot was taken. Resume
   with a fresh, shorter cosine horizon so the learning rate anneals to its floor at the
   new end of training:

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



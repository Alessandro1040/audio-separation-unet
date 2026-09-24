# A/B: Mel-band RoFormer vs U-Net (short, controlled)

Raw evidence behind §6 of `docs/ROFORMER.md`. Produced by

```bash
DEV=mps STEPS=150 ./scripts/ab_roformer_vs_unet.sh
DEV=mps STEPS=600 TAG=600 ./scripts/ab_roformer_vs_unet.sh
```

on an Apple M5 / 16 GB (MPS), 24/09/2026.

## Protocol

Everything except the architecture is held fixed: the STFT (2048 / 1024, 50 % overlap), the
augmentation, the loss (waveform L1 + L1 on `|X|^0.3` magnitudes - `loss.mrstft: 0.0` on the
RoFormer side, so no difference can be blamed on the multi-resolution term), AdamW with the
same warm-up and cosine decay, the same EMA decay, the same 14 held-out validation songs,
the same seed, the same number of steps, batch size (2) and chunk length (3 s). Both sides
therefore process exactly the same amount of audio: `steps x batch x 3 s`.

`configs/unet_musdb18.yaml` (U-Net, 10.55 M params) vs `configs/roformer_ab.yaml`
(Mel-band RoFormer, 2.75 M params, 82 mel bands).

## Result

| steps | U-Net | Mel-band RoFormer | delta | wall-clock (U-Net / RoFormer) |
| --- | --- | --- | --- | --- |
| 150 | −7.84 dB | −8.44 dB | −0.60 dB (U-Net better) | 0.016 h / 0.023 h |
| 600 | −6.04 dB | −5.49 dB | **+0.55 dB (RoFormer better)** | 0.068 h / 0.095 h |

Per stem at 600 steps (dB SI-SDR):

| stem | U-Net | RoFormer |
| --- | --- | --- |
| vocals | −11.51 | −11.35 |
| drums | −8.67 | **−6.97** |
| bass | −0.94 | **−0.82** |
| other | −3.04 | **−2.79** |
| mean | −6.04 | **−5.49** |

**Interpretation.** Short runs measure how fast an architecture learns, not how good it can
get. At 150 steps the U-Net is ahead; by 600 steps the RoFormer is ahead on every stem
(+0.55 dB) while costing ~1.4x more wall-clock per step and ~3-4x more FLOPs per second of
audio (see §4 of the doc). Neither side is converged, and the shipped checkpoint
(`models/unet_musdb18_ema.pt`, 2500 steps of 5 s x 4 batches, −5.76 dB validation) was
therefore **not** replaced: the evidence does not justify it.

## Files

| File | Content |
| --- | --- |
| `summary_{150,600}_{unet,roformer}.json` | steps, hours, params, best validation SI-SDR |
| `log_{150,600}_{unet,roformer}.csv` | per-step training log (loss, lr, s/step) + validation rows |
| `console_{150,600}.txt` | full console output of both sides, including the printed summary |

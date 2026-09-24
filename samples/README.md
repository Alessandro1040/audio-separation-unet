# Sample output

24-second excerpts of **"Al James — Schoolboy Facination"** (first song of the MUSDB18
*test* split, never used for training), separated with
`models/unet_musdb18_ema.pt`:

| file | what it is |
| --- | --- |
| `... - mixture (input).mp3` | what the model was given |
| `... - vocals / drums / bass / other (U-Net estimate).mp3` | what it produced |
| `... - vocals / drums / bass / other (ground truth).mp3` | the reference stems from MUSDB18 |

Listening tip: compare `bass (U-Net estimate)` with `bass (ground truth)` — as the report
in `runs/example_report.txt` says, the bass estimate has a spectral centroid of ~4 kHz
against ~0.2 kHz for the reference, i.e. it still leaks other instruments. `vocals` and
`other` are closer to their references. That is exactly what an under-trained mask model
looks like, and it is why the app prints those numbers.

The corresponding figures are `figures/masks_example.png` (mixture spectrogram + the four
predicted masks) and `figures/stems_example.png` (magnitude spectrograms of the estimates).

Produced with:

```bash
python -m src.analyze --checkpoint models/unet_musdb18_ema.pt \
    --input mixture.wav --reference-dir /path/to/stems --out outputs/example
```

## Licence of these files

The excerpts are derived from **MUSDB18** (CC BY-NC-SA 4.0, attribution required,
non-commercial). They are included here only to make the repository self-explanatory;
delete them if you need a purely code-only repository.

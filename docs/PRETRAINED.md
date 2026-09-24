# Un modello che funziona bene davvero (modello preaddestrato)

Se lo scopo è **separare bene**, addestrare da zero su un laptop non è la strada: la strada
è usare un modello **già addestrato**. Questo documento è la ricetta che funziona, provata
su questa macchina, più i numeri misurati sullo stesso estratto di 24 s di MUSDB18 che il
repository usa in `samples/`.

## 1. Installazione (una volta)

```bash
python3 -m pip install -U audio-separator
python3 -m pip install -U beartype          # <- necessario su Python 3.14, vedi §4
```

`audio-separator` (MIT, `nomadkaraoke/python-audio-separator`) scarica i pesi da Hugging
Face alla prima esecuzione in `/tmp/audio-separator-models/`: nessun training, nessuna
configurazione.

## 2. Separare un brano

```bash
# voce + instrumental (BS-RoFormer, il default: il modello più forte sulla voce)
audio-separator "samples/Al James - Schoolboy Facination - mixture (input).mp3" \
    --output_dir outputs/pretrained --output_format MP3 --output_bitrate 192k

# 4 stem (vocals/drums/bass/other) con HTDemucs: veloce e completo
audio-separator "song.mp3" -m htdemucs.yaml --output_dir outputs/pretrained \
    --output_format MP3 --output_bitrate 192k
```

Oppure tutto insieme, compreso il set di ascolto con la ground truth:

```bash
./scripts/make_listening_set.sh --input song.mp3 --out outputs/ascolto/song \
    --pretrained htdemucs.yaml --roformer checkpoints/ab_roformer600/best.pt
```

## 3. Perché: i numeri (stesso estratto di 24 s, ground truth di MUSDB18)

SI-SDR per stem, misurata con `src/dsp.py::waveform_sdr` (scale-invariante, quindi
insensibile alle normalizzazioni diverse dei due mondi):

| stem | mix come stima | U-Net del repo | BS-RoFormer ep368 | BS-RoFormer ep317 | HTDemucs-ft | HTDemucs |
| --- | --- | --- | --- | --- | --- | --- |
| vocals | −1.93 dB | −4.08 dB | **+11.82 dB** | **+11.82 dB** | +9.61 dB | +9.46 dB |
| drums | −5.54 dB | −5.70 dB | – (2 stem) | – (2 stem) | **+6.38 dB** | +6.01 dB |
| bass | −6.00 dB | −4.14 dB | – (2 stem) | – (2 stem) | **+11.01 dB** | +10.53 dB |
| other | −7.16 dB | −6.33 dB | – (2 stem) | – (2 stem) | **+4.68 dB** | +4.30 dB |

Sulla voce sono **~16 dB** di differenza: non è "un po' meglio", è un altro universo (e
spiega perché i SOTA dichiarano SDR ~9-12 dB mentre `models/MODEL_CARD.md` riporta −6.4 dB).
Costo: 7 s (HTDemucs), ~30 s (HTDemucs-ft, quattro modelli) e 57 s (BS-RoFormer) su 24 s di
audio, su M5 via MPS.

## 3b. Quale modello scegliere (misurato, non per sentito dire)

| Serve | Modello | Perché |
| --- | --- | --- |
| la voce migliore | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (o ep317) | sono a **2 stem** (voce/instrumental) ed è lì che eccellono: +11.8 dB |
| i 4 stem tutti | `htdemucs_ft.yaml` | il migliore a 4 stem: voce +9.6, drums +6.4, bass +11.0, other +4.7 |
| solo basso al top | `hdemucs_mmi.yaml` | nella tabella della libreria ha il SDR più alto sul basso (12.23 vs 12.02 di htdemucs_ft) — non misurato qui |
| chitarra e piano | `htdemucs_6s.yaml` | l'unico con stem guitar/piano oltre ai 4 |

Trucco pratico: la cartella `outputs/ascolto/pretrained_best/` contiene **il meglio per ogni
stem** (voce da BS-RoFormer, il resto da HTDemucs-ft), costruita misurando ogni coppia
modello/stem contro la ground truth. Per un uso generale, "4 stem con htdemucs_ft" è la
scelta che non sbaglia mai.

Attenzione a due cose che ho sbagliato io stesso la prima volta: il checkpoint BS-RoFormer di
default **non** è il migliore (ep368/ep317 lo battono) e i 4 stem non arrivano da lui, perché
è a 2 stem; `--single_stem Drums` su un modello a 2 stem non produce nulla.

## 4. L'insidia su Python 3.14

`audio-separator` 0.47.0 dichiara `beartype>=0.18.5,<0.19.0`, e con quella versione fallisce
così:

```
RuntimeError: Failed to instantiate Roformer model: Method ...BSRoformer.__init__()
parameter "stft_window_fn" type hint collections.abc.Callable | None either
PEP-noncompliant or currently unsupported by @beartype
```

Causa: `beartype` 0.18 non sa risolvere `collections.abc.Callable | None` su Python 3.14.
Rimedio: `python3 -m pip install -U beartype` (0.22.9 funziona; pip segnala un conflitto di
versione dichiarata, che in pratica è innocuo). Fatto questo, l'inferenza gira.

## 5. Come si incastra con questo repository

| | U-Net / Mel-band RoFormer di questo repo | modello preaddestrato |
| --- | --- | --- |
| qualità | −6 dB / −5 dB SI-SDR (validazione) | +6 … +12 dB per stem |
| costo per brano | 1-3 s | 7-57 s |
| cosa ci fai | capire e addestrare un modello da zero su un laptop | separare musica sul serio |
| dove sta il codice | `src/`, `configs/`, `docs/ROFORMER.md` | dipendenza esterna, pesi scaricati a runtime |

Sono complementari: il valore di questo repository è la trasparenza (tutto il percorso
addestrabile e ispezionabile), il valore del modello preaddestrato è il risultato. Per
ascoltare la differenza: `outputs/ascolto/pretrained/`.

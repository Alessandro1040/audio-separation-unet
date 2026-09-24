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

## 3c. Cosa c'è davvero dentro i pesi (il test del silenzio)

"159 milioni di parametri" non significa "159 milioni di numeri che descrivono un pianoforte
o una chitarra": sono i parametri di una **funzione**. Se il suono di uno strumento fosse
*contenuto* nel file, allora dandogli in pasto qualcosa che non contiene strumenti il modello
dovrebbe comunque "tirarlo fuori". Test riproducibile (`bash /tmp/input_test.sh`):

| ingresso al modello | uscita Vocals (rms) | uscita Instrumental (rms) |
| --- | --- | --- |
| silenzio digitale | **0.000000** | **0.000000** |
| rumore bianco a −80 dBFS | 0.000000 | 0.000085 |
| rumore bianco a −14 dBFS | 0.001391 | 0.180071 |

E l'impronta del checkpoint è identica prima e dopo (`40780dd7…`): il modello **non impara
nulla** da quello che gli dai, e **non conserva** nulla.

Come si legge:

* silenzio in ingresso → silenzio in uscita: **nessun pianoforte, nessuna chitarra
  "sbuca"**. Se fossero immagazzinati, uscirebbero anche senza ingresso;
* rumore in ingresso → rumore in uscita, con ampiezza proporzionale a quella dell'ingresso:
  l'uscita è **calcolata dall'ingresso**, non pescata da un archivio;
* l'md5 dei pesi non cambia: il file è di sola lettura, non è una memoria di ciò che ha visto.

La parte in cui l'intuizione "il modello sa com'è fatta una chitarra" è **giusta**: i pesi
codificano le *statistiche* che permettono di riconoscerla - "uno spettro con questa serie
armonica, questo attacco, questa posizione stereo è probabilmente quella sorgente". Ma è
conoscenza *condizionata all'ingresso* (una regola, una manopola), non una copia del suono:

* non è localizzata: non esiste una "zona chitarra" del file da cui estrarre un timbro.
  Metà dei parametri sono teste di maschera (decisioni), il resto MLP e attention (calcolo);
* è distribuita e si aggiorna tutta insieme quando il modello impara: per migliorare un
  timbro si riaddestra (o si fa fine-tuning di) tutta la rete, non si "riscrive" una parte;
* proprio perché sono statistiche, funzionano su brani mai visti - è il motivo per cui il
  modello ha separato il nostro estratto senza averlo mai ascoltato - e peggiorano fuori
  dominio (master moderni, generi lontani da MUSDB18).

## 3d. Costi reali misurati (perché un modello grande gira su un portatile)

| | BS-RoFormer ep368 | HTDemucs | Mel-band RoFormer del repo |
| --- | --- | --- | --- |
| parametri | **159.8 M** (misurati: 159 758 796) | ~21 M | 2.75 M |
| peso su disco | 639 MB (fp32) | 84 MB (fp32) | 11 MB |
| RAM di picco (inferenza) | **1.00 GB** | **1.13 GB** | 0.3 GB |
| 6 s di audio | 14.0 s (12 s di separazione) | **4.1 s (2 s di separazione)** | 0.2 s |
| 24 s di audio | 57 s | 7 s | 0.5 s |

Comandi con cui sono state prese (riproducibili):

```bash
/usr/bin/time -l audio-separator /tmp/clip.wav -m <modello> --output_dir /tmp/bench --output_format WAV
python3 -c "import torch; s=torch.load('<ckpt>', weights_only=False); print(sum(v.numel() for v in s.values() if hasattr(v,'numel')))"
```

Come si legge:

* **la memoria è parametri x byte per parametro**: 160 M in fp32 = 639 MB, cioè il 4 % dei
  16 GB della macchina. In inferenza non servono né i gradienti né lo stato di Adam (che
  triplicherebbe, come succede invece in training), quindi il modello "entra" senza problemi;
* **la velocità non viene dal numero di parametri ma dai FLOP per secondo di audio**: il
  modello da 21 M (convoluzioni, HTDemucs) è 3x più veloce del modello da 160 M (attention,
  che è quadratica nella lunghezza del segmento). Il mio modello da 2.75 M è il più veloce di
  tutti (0.2 s per 6 s), semplicemente perché è minuscolo;
* **il lavoro viene fatto a segmenti**: il log di `audio-separator` mostra `0/9` su un brano
  di 24 s, cioè il brano è tagliato in pezzi corti processati uno alla volta. Per questo la
  RAM resta costante e non cresce con la durata del brano;
* il costo è quasi tutto in **training**, non in inferenza: servono gradienti, attivazioni di
  tutto il grafo e giorni di GPU. È esattamente il motivo per cui il modello del repo (6
  minuti di addestramento, 2.75 M parametri) sta a −4 dB e quello preaddestrato a +11.8 dB:
  stessa famiglia di architettura, 58 volte i parametri e mesi di calcolo di differenza.

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

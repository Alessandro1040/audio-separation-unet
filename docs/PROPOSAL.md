# Proposta di progetto — "Separazione delle sorgenti audio come problema di Computer Vision"

*(documento pensato per essere consegnato/allegato a una proposta di progetto per un
corso di Computer Vision; il codice corrispondente è in questo repository)*

## 1. Idea

Trattare la separazione delle sorgenti audio (music source separation) come un problema
di **segmentazione di immagini**, e risolverlo con architetture di Computer Vision note
(prima fra tutte la **U-Net**). L'audio viene "fotografato": lo *spettrogramma* è
un'immagine 2-D tempo-frequenza, e separare gli strumenti equivale a **classificare/
segmentare i pixel** di quell'immagine.

Non è una forzatura didattica: è esattamente il metodo di una famiglia di lavori
scientifici sulla separazione (Jansson et al., *Singing Voice Separation with Deep U-Net
Convolutional Networks*, ISMIR 2017; Stoller et al., *U-Nets with various intermediate
blocks for spectrogram-based singing voice separation*, ISMIR 2018), ed è quindi un
ottimo modo per applicare gli strumenti di CV a un dominio diverso e "mostrare" come
i concetti del corso (convoluzioni, encoder/decoder, skip connections, maschere,
loss, metriche) si trasferiscano fuori dal mondo delle immagini naturali.

## 2. Pipeline

```
audio del mix ──STFT──► immagine spettrogramma (tempo × frequenza)
                             │
                        U-Net (encoder + decoder + skip connections)
                             │
              maschera per sorgente (voce / batteria / basso / altro)
                             │
        maschera ⊙ spettrogramma complesso del mix ──iSTFT──► 4 segnali separati
```

## 3. Elementi di Computer Vision coinvolti (mappa corso → progetto)

| Concetto del corso | Applicazione nel progetto |
| --- | --- |
| Convoluzioni 2-D su immagini | lo spettrogramma è l'immagine; il kernel 3×3 vede pattern tempo-frequenza (armoniche, transienti) |
| Segmentazione semantica | ogni pixel (bin frequenza, frame) viene assegnato a uno strumento |
| U-Net e skip connections | l'encoder cattura il contesto (quali strumenti suonano), le skip portano il dettaglio fine (attacco delle note) |
| Maschere e loss | la rete predice maschere (ratio mask complessa, `tanh`), la loss confronta il segnale ricostruito con lo stem di riferimento |
| Data augmentation | mix casuale di stem di brani diversi, guadagni, inversioni di canale/polarità: il dataset ha solo 100 brani, l'augmentation è il vero regolarizzatore |
| Metriche di valutazione | SDR / SIR / SAR (BSS-Eval), cioè quanto la sorgente stimata è pulita, quanto le altre sorgenti "sporcano", quanti artefatti introduce il modello |

## 4. Dataset

**MUSDB18** (Zenodo), il dataset di riferimento per la separazione musicale:
- 150 brani, 4 stem ciascuno (vocals, drums, bass, other), 44.1 kHz stereo
- 100 brani di training, 50 di test (split ufficiale)
- due versioni: `.mp4` AAC (4.7 GB) e `.wav` (MUSDB18-HQ, 22.7 GB)

## 5. Risultati ottenibili e stato attuale

Il repository contiene l'intera catena funzionante: download dei dati, pipeline
STFT/maschere/iSTFT, U-Net, training con validazione su 14 brani tenuti fuori,
checkpoint/ripartenza, inferenza su un brano qualsiasi e valutazione BSS-Eval sul test
set. I numeri misurati sono in `README.md` (sezione *Results*).

Aspettative realistiche: un U-Net addestrato da zero su 100 brani non raggiunge i sistemi
allo stato dell'arte (~9 dB SDR con modelli waveform tipo Demucs, addestrati per settimane),
ma supera in modo netto i baseline banali (il mix stesso come stima: SDR ≈ 0 dB) e
permette tutti gli esperimenti interessanti:

* loss diverse (solo magnitudine vs waveform + magnitudine, `L1` vs `L2`, loss multi-risoluzione)
* maschera reale (IRM) vs maschera complessa (cIRM) vs fase stimata esplicitamente
* iperparametri della STFT (finestra/hop, come cambia la "risoluzione dell'immagine")
* augmentation e sua ablazione
* architettura: profondità della U-Net, GroupNorm vs BatchNorm, dilation, attention

## 6. Possibili estensioni per un progetto d'esame

1. **Ablazione sistematica** delle scelte di progettazione (maschera, loss, augmentation,
   risoluzione STFT) con la stessa pipeline e la stessa valutazione.
2. **Confronto con un secondo paradigma**: stessa U-Net ma su immagini spettrogramma
   *multi-canale* (magnitudine + differenza tra i canali/stereo), oppure confronto con un
   modello waveform (Demucs-lite) addestrato con lo stesso budget di calcolo.
3. **Valutazione percettiva**: oltre a SDR/SIR/SAR, un piccolo test d'ascolto (MUSHRA-like)
   su esempi estratti automaticamente dal test set (il repository scrive già gli stem).
4. **Generalizzazione**: valutare su un altro dataset (es. MedleyDB/Slakh) senza
   riaddestrare, per discutere il domain shift — un tema molto "CV".
5. **Interpretabilità**: visualizzare le maschere predette per capire *cosa* ha imparato
   la rete (percussioni → strutture impulsive, basso → armoniche basse).

## 7. Budget di calcolo

Tutto è dimensionato per girare su un portatile (Apple M5, 16 GB): ~1 s per passo di
training, un'epoca ≈ 17 minuti, modello da 10.5 M parametri. Un run "serio" è
dell'ordine di qualche ora; il repository include i comandi misurati per scegliere
dimensione del modello e del batch (`scripts/benchmark_model.py`).

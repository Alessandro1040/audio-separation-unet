#!/usr/bin/env bash
# Build a listening set from any song: the mixture, the four stems from each model you have
# a checkpoint for, the ground truth if you have it, everything as mp3 - so you can hear
# what "SDR -6 dB" actually sounds like.
#
#     ./scripts/make_listening_set.sh --input song.mp3 --out outputs/ascolto/song
#     ./scripts/make_listening_set.sh --input song.mp3 --out outputs/ascolto/song \
#         --reference-dir /path/to/musdb18_song --roformer checkpoints_roformer/best.pt \
#         --start 60 --seconds 30
#
# Options
#   --input PATH       audio file to separate (anything ffmpeg reads: mp3/wav/m4a/flac/...)
#   --out DIR          where the listening set is written
#   --reference-dir D  folder holding vocals/drums/bass/other.(wav|flac|mp3|m4a) = ground truth
#   --unet CKPT        U-Net checkpoint            (default models/unet_musdb18_ema.pt)
#   --roformer CKPT    RoFormer checkpoint         (default: skipped, use "-" to skip explicitly)
#   --start SECONDS    excerpt start               (default 0)
#   --seconds SECONDS  excerpt length, 0 = whole file (default 30: long enough to judge,
#                      short enough that a 5-minute song does not take 10 minutes)
#   --chunk-seconds S  inference window             (default 10 for the U-Net, 5 for the RoFormer)
#   --device NAME      auto | mps | cuda | cpu     (default auto)
#   --bitrate RATE     mp3 bitrate                 (default 192k)
#
# Listening protocol that actually tells you something (do it in this order):
#   1. ground_truth_vocals vs mixture          how loud the "problem" is in the first place
#   2. unet/vocals vs roformer/vocals          what each model does with it
#   3. the same for drums, then bass, then other
#   4. sum two estimates against the mixture   leakage is easiest to hear in the low end
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

INPUT=""
OUT=""
REFERENCE=""
UNET="models/unet_musdb18_ema.pt"
ROFORMER=""
START=0
LENGTH=30
CHUNK_UNET=10
CHUNK_ROFORMER=5
DEVICE="auto"
BITRATE="192k"

while [ $# -gt 0 ]; do
    case "$1" in
        --input) INPUT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --reference-dir) REFERENCE="$2"; shift 2 ;;
        --unet) UNET="$2"; shift 2 ;;
        --roformer) ROFORMER="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --seconds) LENGTH="$2"; shift 2 ;;
        --chunk-seconds) CHUNK_UNET="$2"; CHUNK_ROFORMER="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --bitrate) BITRATE="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$INPUT" ] || [ -z "$OUT" ]; then
    echo "usage: $0 --input SONG --out DIR [--reference-dir DIR] [--roformer CKPT]" >&2
    exit 2
fi
[ -f "$INPUT" ] || { echo "no such input file: $INPUT" >&2; exit 2; }

encode() {  # wav/flac -> mp3, then drop the original (the wav is only an intermediate)
    local f="$1"
    ffmpeg -y -loglevel error -i "$f" -b:a "$BITRATE" "${f%.*}.mp3"
    rm -f "$f"
}

mkdir -p "$OUT"
DUR=()
[ "$LENGTH" != "0" ] && DUR=(-t "$LENGTH")

echo "== excerpt =="
ffmpeg -y -loglevel error -ss "$START" -i "$INPUT" "${DUR[@]}" -ar 44100 -ac 2 "$OUT/mixture.wav"
echo "  $OUT/mixture.wav  (start ${START}s, ${LENGTH}s)"

if [ -n "$UNET" ] && [ "$UNET" != "-" ]; then
    echo "== U-Net: $UNET =="
    "$PYTHON" -m src.separate --checkpoint "$UNET" --input "$OUT/mixture.wav" \
        --out "$OUT/unet" --device "$DEVICE" --chunk-seconds "$CHUNK_UNET"
fi

if [ -n "$ROFORMER" ] && [ "$ROFORMER" != "-" ]; then
    echo "== Mel-band RoFormer: $ROFORMER =="
    "$PYTHON" -m src.separate_roformer --checkpoint "$ROFORMER" --input "$OUT/mixture.wav" \
        --out "$OUT/roformer" --device "$DEVICE" --chunk-seconds "$CHUNK_ROFORMER"
fi

if [ -n "$REFERENCE" ]; then
    echo "== ground truth: $REFERENCE =="
    for stem in vocals drums bass other; do
        for ext in wav flac mp3 m4a; do
            if [ -f "$REFERENCE/$stem.$ext" ]; then
                ffmpeg -y -loglevel error -i "$REFERENCE/$stem.$ext" "${DUR[@]}" \
                    -b:a "$BITRATE" "$OUT/ground_truth_$stem.mp3"
                echo "  ground_truth_$stem.mp3"
                break
            fi
        done
    done
fi

echo "== mp3 =="
for f in "$OUT"/mixture.wav "$OUT"/unet/*.wav "$OUT"/roformer/*.wav; do
    [ -f "$f" ] && encode "$f"
done

echo
echo "listening set ready in $OUT"
echo "  play everything:   for f in $OUT/*.mp3 $OUT/*/*.mp3; do open \"\$f\"; done"
echo "  one file:          afplay \"$OUT/unet/vocals.mp3\""
echo "  open the folder:   open $OUT"

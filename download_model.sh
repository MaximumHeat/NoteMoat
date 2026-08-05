#!/usr/bin/env bash
# Download a Qwen2.5-VL GGUF model + vision projector for local journal digitization.
# Everything is licensed Apache-2.0 and stays on your machine.
#
# Usage:  ./download_model.sh [7b|3b] [output_dir]
#   default: ./download_model.sh 7b ./models

set -euo pipefail

SIZE="${1:-7b}"
DIR="${2:-./models}"

case "$SIZE" in
  7b)
    REPO="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
    MODEL="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    MMPROJ="mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
    ;;
  3b)
    REPO="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"
    MODEL="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    MMPROJ="mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    ;;
  *)
    echo "usage: $0 [7b|3b] [output_dir]" >&2
    exit 1
    ;;
esac

# Requires the `hf` CLI from huggingface_hub (pip install -U huggingface_hub).
if ! command -v hf >/dev/null 2>&1; then
  echo "error: 'hf' not found. Install it with:  pip install -U huggingface_hub" >&2
  exit 1
fi

mkdir -p "$DIR"
echo "Downloading $MODEL + $MMPROJ from $REPO into $DIR ..."
hf download "$REPO" \
  --include "$MODEL" \
  --include "$MMPROJ" \
  --local-dir "$DIR"

echo
echo "Done. Files in: $DIR"
echo "  model:       $DIR/$MODEL"
echo "  projector:   $DIR/$MMPROJ"

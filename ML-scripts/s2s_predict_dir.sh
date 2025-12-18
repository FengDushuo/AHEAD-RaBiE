#!/usr/bin/env bash
set -euo pipefail

# Example usage:
#   bash s2s_predict_dir.sh input/ output_pred/ runs/s2s_vac/best.pt
# Optional 4th arg: remove mode ("topk" or "threshold"), default: "topk"
# Optional 5th arg: value for topk (int) or threshold (float). Defaults: 1 (for topk) or 0.5 (for threshold)

IN_DIR="${1:-input}"
OUT_DIR="${2:-output_pred}"
CKPT="${3:-runs/s2s_vac/best.pt}"
MODE="${4:-topk}"
VAL="${5:-}"

mkdir -p "$OUT_DIR"

ARGS=( --input-dir "$IN_DIR" --output-dir "$OUT_DIR" --ckpt "$CKPT" --cutoff 6.0 --remove-element O --remove-mode "$MODE" )

if [[ "$MODE" == "topk" ]]; then
  ARGS+=( --remove-topk "${VAL:-1}" )
else
  ARGS+=( --remove-threshold "${VAL:-0.5}" )
fi

python s2s_batch_predict.py "${ARGS[@]}"

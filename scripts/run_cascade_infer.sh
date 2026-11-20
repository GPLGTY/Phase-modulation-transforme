#!/usr/bin/env bash
# RAG cascade eval (USMLE + MedMCQA by default).
# Writes results/result_summary or result_summaryN (never overwrites existing).
# CSV does not include upstream intermediates.
#
#   bash scripts/run_cascade_infer.sh
#   DATASETS=USMLE bash scripts/run_cascade_infer.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASETS="${DATASETS:-USMLE,MEDMCQA}"
SIDECAR_BACKEND="${SIDECAR_BACKEND:-transformers}"

EXTRA=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  EXTRA+=(--out_dir "$OUT_DIR")
fi
if [[ -n "${RESULT_DIR:-}" ]]; then
  EXTRA+=(--result_dir "$RESULT_DIR")
fi
if [[ -n "${DATA:-}" ]]; then
  EXTRA+=(--data_path "$DATA")
fi

python3 "$ROOT/inference_cascade_pmt.py" \
  --datasets "$DATASETS" \
  --sidecar_backend "$SIDECAR_BACKEND" \
  --sidecar_tp "${SIDECAR_TP:-1}" \
  --sidecar_mem "${SIDECAR_MEM:-0.85}" \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --num_key_value_heads 4 \
  --max_seq_len 4096 \
  "${EXTRA[@]}"

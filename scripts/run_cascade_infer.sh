#!/usr/bin/env bash
# Protocol evaluation (letter accuracy only; no upstream intermediates in CSV).
#   PROTOCOL=no_pmt|pmt_norag|pmt_rag|all
#   PhaseModulationTransforme checkpoints auto-resolved from out/run_cascade_{norag|rag}/
#
#   PROTOCOL=all bash scripts/run_cascade_infer.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROTOCOL="${PROTOCOL:-all}"
DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_test_1273.json}"
OUT_CSV="${OUT_CSV:-$ROOT/results/protocol_eval.csv}"
SIDECAR_BACKEND="${SIDECAR_BACKEND:-transformers}"

EXTRA=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${OUT_DIR:-}" ]]; then
  EXTRA+=(--out_dir "$OUT_DIR")
fi

python3 "$ROOT/inference_cascade_pmt.py" \
  --protocol "$PROTOCOL" \
  --data_path "$DATA" \
  --output_csv "$OUT_CSV" \
  --sidecar_backend "$SIDECAR_BACKEND" \
  --sidecar_tp0 "${SIDECAR_TP0:-1}" \
  --sidecar_tp1 "${SIDECAR_TP1:-1}" \
  --sidecar_tp2 "${SIDECAR_TP2:-1}" \
  --sidecar_mem0 "${SIDECAR_MEM0:-0.35}" \
  --sidecar_mem1 "${SIDECAR_MEM1:-0.35}" \
  --sidecar_mem2 "${SIDECAR_MEM2:-0.85}" \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --num_key_value_heads 4 \
  --max_seq_len 4096 \
  "${EXTRA[@]}"

echo "csv: $OUT_CSV"

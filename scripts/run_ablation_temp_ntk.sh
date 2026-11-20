#!/usr/bin/env bash
# temperature × ntk_alpha 消融（检索增强 JSON → 一体流程 → result_summary[N]）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_test_1273.json}"
DATASET_NAME="${DATASET_NAME:-USMLE}"
TRAIN_NTK="${TRAIN_NTK:-TRUE}"
INFER_NTK="${INFER_NTK:-TRUE}"

EXTRA=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "${RESULT_DIR:-}" ]]; then
  EXTRA+=(--result_dir "$RESULT_DIR")
fi

python3 "$ROOT/scripts/run_ablation_temp_ntk.py" \
  --data_path "$DATA" \
  --dataset_name "$DATASET_NAME" \
  --train_ntk "$TRAIN_NTK" \
  --infer_ntk "$INFER_NTK" \
  --max_seq_len 4096 \
  --ntk_max_train_len 4096 \
  --dtype bfloat16 \
  "${EXTRA[@]}"

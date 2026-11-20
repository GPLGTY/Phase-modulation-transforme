#!/usr/bin/env bash
# RAG Phase-modulation-transformer SFT (ckpt_0 + PMT as one stack).
#   Always retrieval-augmented. Outputs → out/run_pmt/
#
#   bash scripts/run_cascade_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_train_10178.json}"
INIT="${INIT:-$ROOT/parameters/sft_PMT_en_1024.pth}"
OUT="${OUT:-$ROOT/out/run_pmt}"

NPROC="${NPROC:-1}"
SIDECAR_BACKEND="${SIDECAR_BACKEND:-transformers}"
mkdir -p "$OUT"

COMMON=(
  --protocol pmt_rag
  --data_path "$DATA"
  --init_ckpt "$INIT"
  --out_dir "$OUT"
  --sidecar_backend "$SIDECAR_BACKEND"
  --sidecar_tp1 "${SIDECAR_TP:-1}"
  --sidecar_mem1 "${SIDECAR_MEM:-0.85}"
  --batch_size 1
  --accumulation_steps "${ACCUM:-8}"
  --epochs "${EPOCHS:-1}"
  --hidden_size 1024
  --num_hidden_layers 24
  --num_attention_heads 16
  --num_key_value_heads 4
  --max_seq_len 4096
)

if [[ "$NPROC" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/trainer/train_cascade_pmt.py" --ddp "${COMMON[@]}"
else
  python3 "$ROOT/trainer/train_cascade_pmt.py" \
    --device "${DEVICE:-cuda:0}" "${COMMON[@]}"
fi

echo "done: $OUT/cascade_PMT_en_1024_best.pth"

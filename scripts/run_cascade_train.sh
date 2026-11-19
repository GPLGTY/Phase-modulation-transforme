#!/usr/bin/env bash
# Protocol-specific PhaseModulationTransformer SFT.
#   PROTOCOL=pmt_norag|pmt_rag (required)
#   Train always loads released pretrained PhaseModulationTransformer (INIT / pack 01).
#   Outputs go to out/run_pmt_norag or out/run_pmt_rag.
#
#   PROTOCOL=pmt_norag bash scripts/run_cascade_train.sh
#   PROTOCOL=pmt_rag   bash scripts/run_cascade_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROTOCOL="${PROTOCOL:-}"
if [[ -z "$PROTOCOL" ]]; then
  echo "USAGE: PROTOCOL=pmt_norag|pmt_rag bash scripts/run_cascade_train.sh" >&2
  exit 1
fi

DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_train_10178.json}"
INIT="${INIT:-$ROOT/parameters/sft_PMT_en_1024.pth}"
if [[ "$PROTOCOL" == "pmt_norag" ]]; then
  OUT="${OUT:-$ROOT/out/run_pmt_norag}"
elif [[ "$PROTOCOL" == "pmt_rag" ]]; then
  OUT="${OUT:-$ROOT/out/run_pmt_rag}"
else
  echo "PROTOCOL must be pmt_norag or pmt_rag" >&2
  exit 1
fi

NPROC="${NPROC:-1}"
SIDECAR_BACKEND="${SIDECAR_BACKEND:-transformers}"
mkdir -p "$OUT"

COMMON=(
  --protocol "$PROTOCOL"
  --data_path "$DATA"
  --init_ckpt "$INIT"
  --out_dir "$OUT"
  --sidecar_backend "$SIDECAR_BACKEND"
  --sidecar_tp0 "${SIDECAR_TP0:-1}"
  --sidecar_tp1 "${SIDECAR_TP1:-1}"
  --sidecar_mem0 "${SIDECAR_MEM0:-0.4}"
  --sidecar_mem1 "${SIDECAR_MEM1:-0.4}"
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

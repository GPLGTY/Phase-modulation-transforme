#!/usr/bin/env bash
# Protocol no_pmt: standalone upstream letter prediction (no PhaseModulationTransforme).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROTOCOL=no_pmt \
DATA="${DATA:-$ROOT/dataset/RAG_MedQA_US_test_1273.json}" \
OUT_CSV="${OUT_CSV:-$ROOT/results/protocol_no_pmt.csv}" \
SIDECAR_BACKEND="${SIDECAR_BACKEND:-transformers}" \
SIDECAR_TP2="${SIDECAR_TP2:-2}" \
SIDECAR_MEM2="${SIDECAR_MEM2:-0.85}" \
  bash "$ROOT/scripts/run_cascade_infer.sh"

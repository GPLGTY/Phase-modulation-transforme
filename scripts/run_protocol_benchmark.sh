#!/usr/bin/env bash
# One-click: train (if needed) → eval all protocols on all test sets → comparison table.
#
#   bash scripts/run_protocol_benchmark.sh
#
# Optional:
#   SKIP_TRAIN=1          # only eval; require existing out/run_cascade_{norag|rag}
#   FORCE_TRAIN=1         # always retrain before eval
#   LIMIT=8               # debug subset
#   DATASETS="medqa_us_dev=dataset/RAG_MedQA_US_dev_1272.json,medqa_us_test=dataset/RAG_MedQA_US_test_1273.json"
#   PROTOCOLS="no_qf,qf_norag,qf_rag"
#   RESULTS_DIR=results/protocol_benchmark
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXTRA=()
[[ "${SKIP_TRAIN:-0}" == "1" ]] && EXTRA+=(--skip_train)
[[ "${FORCE_TRAIN:-0}" == "1" ]] && EXTRA+=(--force_train)
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${DATASETS:-}" ]] && EXTRA+=(--datasets "$DATASETS")
[[ -n "${PROTOCOLS:-}" ]] && EXTRA+=(--protocols "$PROTOCOLS")
[[ -n "${RESULTS_DIR:-}" ]] && EXTRA+=(--results_dir "$RESULTS_DIR")
[[ -n "${DEVICE:-}" ]] && EXTRA+=(--device "$DEVICE")
[[ -n "${SIDECAR_BACKEND:-}" ]] && EXTRA+=(--sidecar_backend "$SIDECAR_BACKEND")
[[ -n "${NPROC:-}" ]] && EXTRA+=(--nproc "$NPROC")

python3 "$ROOT/scripts/run_protocol_benchmark.py" "${EXTRA[@]}"

#!/usr/bin/env bash
# MedQA / USMLE 评测入口（一体流程：系统权重 + PMT），结果写入 result_summary*。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DATASET_NAME="${DATASET_NAME:-USMLE}"
export DATA_PATH="${DATA_PATH:-$ROOT/dataset/RAG_MedQA_US_test_1273.json}"
export CHECKPOINT="${CHECKPOINT:-$ROOT/out/run_pmt/cascade_PMT_en_1024_best.pth}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "缺少评测数据: $DATA_PATH（包 04 检索增强 JSON）" >&2
  exit 1
fi

bash "$ROOT/scripts/run_cascade_infer.sh"

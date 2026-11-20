#!/usr/bin/env bash
# Download ModelScope asset packs and extract into this repository.
#
# Usage (project root):
#   bash scripts/download_modelscope_assets.sh
#   PACKS="01 04" bash scripts/download_modelscope_assets.sh   # Phase-modulation-transformer init + MedQA retrieval JSON
#   PACKS="01 04" bash scripts/download_modelscope_assets.sh
#   PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh   # rebuild retrieval remotely
#
# Env:
#   MODELSCOPE_REPO   default zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
#   DOWNLOAD_DIR      default $ROOT/.modelscope_packs
#   PACKS             space-separated: 01 03b 03d 04 05 06 07 08
#   SKIP_EXTRACT=1
#   FORCE=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELSCOPE_REPO="${MODELSCOPE_REPO:-zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT/.modelscope_packs}"
PACKS="${PACKS:-01 04}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
FORCE="${FORCE:-0}"

mkdir -p "$DOWNLOAD_DIR"

declare -A PACK_FILE=(
  [01]="01_parameters.tar"
  [03b]="03b_corpus_openhermes2_5.tar"
  [03d]="03d_corpus_sft_mini_512_zh.tar"
  [04]="04_medqa_rag_json.tar"
  [05]="05_embedding_stella.tar"
  [06]="06_textbooks.tar"
  [07]="07_vector_stores.tar"
  [08]="08_questions.tar"
)

if ! command -v modelscope >/dev/null 2>&1; then
  echo "modelscope CLI missing; installing: pip install -U modelscope" >&2
  pip install -U "modelscope>=1.11.0"
fi

files=()
for key in $PACKS; do
  fname="${PACK_FILE[$key]:-}"
  if [[ -z "$fname" ]]; then
    echo "unknown pack id: $key (allowed: ${!PACK_FILE[*]})" >&2
    exit 1
  fi
  files+=("$fname")
done

echo "repo: $MODELSCOPE_REPO"
echo "cache: $DOWNLOAD_DIR"
echo "download: ${files[*]}"

dl_args=(download --repo-type model "$MODELSCOPE_REPO" --local-dir "$DOWNLOAD_DIR")
if [[ "$FORCE" == "1" ]]; then
  dl_args+=(--force)
fi

modelscope "${dl_args[@]}" "${files[@]}" SHA256SUMS.txt MANIFEST.md 2>/dev/null \
  || modelscope "${dl_args[@]}" "${files[@]}" \
  || true

for key in $PACKS; do
  fname="${PACK_FILE[$key]}"
  src="$(find "$DOWNLOAD_DIR" -name "$fname" -type f | head -n 1 || true)"
  if [[ -z "$src" ]]; then
    echo "missing pack file for $key ($fname)" >&2
    exit 1
  fi
  if [[ "$SKIP_EXTRACT" == "1" ]]; then
    continue
  fi
  echo "extract $src → $ROOT"
  tar -xf "$src" -C "$ROOT"
done

if [[ "$SKIP_EXTRACT" != "1" ]]; then
  python3 "$ROOT/scripts/strip_to_retrieval_json.py" --dataset_dir "$ROOT/dataset" || true
  # Normalize pack-01 filenames to project branding (accept either on-disk name)
  python3 "$ROOT/scripts/normalize_pmt_weight_names.py" || true
fi

echo
echo "done. check:"
echo "  ls -lh parameters/sft_PMT_en_1024.pth"
echo "  ls -lh dataset/RAG_MedQA_US_test_1273.json"
echo "  ls -d dataset/pipeline/models/infgrad/stella-base-en-v2 2>/dev/null || true"
echo "  ls -d dataset/pipeline_data/textbooks/en_json 2>/dev/null || true"
echo "  ls -d dataset/pipeline_data/vector_stores/en/stella-base-en-v2 2>/dev/null || true"
echo "  ls -lh dataset/pipeline_data/questions/US/test.jsonl 2>/dev/null || true"
echo "benchmark:"
echo "  bash scripts/run_cascade_infer.sh"

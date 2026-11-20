# Phase-modulation-transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

中文：[README_CN.md](README_CN.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

This repository releases a **Phase-Modulation Retrieval-Augmented Transformer** for medical multiple-choice QA: implementation, data construction, and reference evaluation results. By default the full pipeline runs under a **retrieval-augmented** setting: load the jointly released system weights and the published PMT checkpoint, predict option letters, and report accuracy. Large assets are hosted on ModelScope (see [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)).

Weights you can download and use directly (**no** local pretraining or general instruction SFT required):

| Asset | Path | Role |
|------|------|------|
| System weights | `parameters/ckpt_0/` | ModelScope directory download |
| Released PMT | `parameters/sft_PMT_en_1024.pth` | Pack `01`; default init for retrieval-augmented train / eval |
| Trained PMT (this repo) | `out/run_pmt/cascade_PMT_en_1024_best.pth` | After retrieval-augmented training on the released weights |
| **Finished reference eval / ablation** | **`results/result_summary/`** | **Completed USMLE / MedMCQA results tracked in git** (accuracy summary + per-sample CSVs); see [results/result_summary/README.md](results/result_summary/README.md) |
| New local eval output | `results/result_summaryN/` (N=1,2,…) | Auto-increments if `result_summary` already exists so reference results are **not** overwritten; detail CSVs **omit** intermediate input-text columns |

Default eval sets: USMLE, MedMCQA. Re-run temperature × NTK ablation:

```bash
bash scripts/run_ablation_temp_ntk.sh
# Input: pack-04 retrieval-augmented JSON; output → result_summary or result_summaryN
```

## Environment

```bash
cd Phase-modulation-transformer
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA. Ascend NPUs also need `torch_npu` (see `requirements.txt`).

| Item | Recommendation |
|------|----------------|
| GPU | Multi-GPU, large VRAM (full pipeline, e.g. 8×80 GB) |
| Disk | ≥400 GB (including `ckpt_0` and archive packs) |
| RAM | ≥256 GB |

## ModelScope assets

The GitHub repo is primarily source code; model weights, corpora, and embeddings are published on ModelScope.

**ModelScope repo**: [zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering](https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
Includes packs `01`/`04`/`05` and the integrated-pipeline directory `parameters/ckpt_0/`. Details: [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md).

### System weights (required for the integrated pipeline)

The default pipeline always loads `parameters/ckpt_0/` (used together with PMT). From the **project root**:

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

You can also download the directory manually from the ModelScope page into local `parameters/ckpt_0/`. Routing: `model/sidecar_map.json`.

You also need the released PMT weights (`parameters/sft_PMT_en_1024.pth` from pack `01`) and sample JSON with `question_with_options` / `answer_idx` (optionally `retrieve_info`).

### Archive packs (`.tar`) batch download

From the **project root** (the script downloads and extracts into `parameters/`, `dataset/`, etc.):

```bash
pip install -U modelscope

# Released PMT weights + MedQA retrieval samples
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# Rebuild retrieval: Embedding + textbooks + FAISS + raw questions (incl. MedMCQA)
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

Or download `.tar` files from the ModelScope page and run `tar -xf <pack>.tar` at the **project root**.

### Asset list and local paths

| ModelScope file / dir | Local path | Role |
|------------------------|----------|------|
| `parameters/ckpt_0` | `parameters/ckpt_0/` | System weights (retrieval-augmented integrated pipeline) |
| `01_parameters.tar` | `parameters/sft_PMT_en_1024.pth` | Released PMT weights (train init / optional direct eval) |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA / USMLE retrieval-augmented samples |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | Stella Embedding for retrieval (en/zh) |
| `06_textbooks.tar` | `dataset/pipeline_data/textbooks/` | MedQA textbooks (en / zh_paragraph / zh_sentence) |
| `07_vector_stores.tar` | `dataset/pipeline_data/vector_stores/` | Language-matched FAISS indexes (en / zh_paragraph / zh_sentence) |
| `08_questions.tar` | `dataset/pipeline_data/questions/` | MedQA US/Mainland and MedMCQA raw questions (**no ready-made MedMCQA RAG JSON pack**) |

By scenario:

| Goal | Suggestion |
|------|------|
| Retrieval-augmented train / eval (USMLE) | `ckpt_0` + `PACKS="01 04"` |
| MedMCQA eval | Also build `dataset/RAG_MedMCQA_*.json` per the data section, or supply equivalent files |
| Rebuild retrieval from raw questions | `PACKS="05 06 07 08"` |

## Quick start

```bash
# Train (default: MedQA retrieval-augmented train set)
bash scripts/run_cascade_train.sh

# Eval USMLE + MedMCQA → results/result_summary or result_summaryN
bash scripts/run_cascade_infer.sh
```

> MedMCQA: ModelScope currently ships raw questions only (pack `08`). Build retrieval-augmented JSON per the data section below, or place files such as `dataset/RAG_MedMCQA_dev.json` before evaluating.

## Repository layout

```
Phase-modulation-transformer/
├── model/                 # Model + system-weight routing / prompt config
├── trainer/               # Retrieval-augmented training
├── inference_cascade_pmt.py
├── scripts/               # Train / eval launchers, ModelScope asset scripts
├── dataset/
│   ├── pipeline/          # Retrieval build pipeline + Embedding
│   ├── pipeline_data/     # Textbooks / FAISS (PACKS 06+07) + runtime retrieval artifacts
│   └── *.json             # Eval / train samples
├── parameters/
│   ├── ckpt_0/            # System weights (ModelScope directory download)
│   └── sft_PMT_en_1024.pth # Released PMT weights (pack 01)
├── out/run_pmt/           # PMT checkpoints from retrieval-augmented training
├── results/result_summary/ # Finished reference eval + ablation (git-tracked)
├── requirements.txt
└── LICENSE
```

## Data

External train / eval / ablation inputs are retrieval-pipeline JSON products (fields include `question_with_options`, `answer_idx`, and `retrieve_info` when retrieval-augmented). **No** other intermediate sample formats are written to disk. The integrated pipeline uses system weights and PMT online.

### Sample inventory (after unpacking `PACKS`)

| Use | Path | Pack |
|------|------|-----|
| MedQA US train | `dataset/RAG_MedQA_US_train_10178.json` (10178) | `04` |
| MedQA US dev | `dataset/RAG_MedQA_US_dev_1272.json` (1272) | `04` |
| MedQA US test | `dataset/RAG_MedQA_US_test_1273.json` (1273) | `04` |

Obtain with `PACKS="04"`, or together with checkpoints via `PACKS="01 04"`. You may also rebuild from raw corpora below; details: [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md), [dataset/pipeline/README.md](dataset/pipeline/README.md).

### MedQA dataset

- Recommended (ModelScope): `PACKS="06 08"` → `dataset/pipeline_data/textbooks/` and `questions/`
- Upstream: https://github.com/jind11/MedQA
- Download: https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing
- Chinese overview: https://zhuanlan.zhihu.com/p/679590312

Directory layout:

```
dataset/pipeline_data/
├── textbooks/     # en / zh_paragraph / zh_sentence (txt)
└── questions/     # train.jsonl / dev.jsonl / test.jsonl under US|Mainland
```

### MedMCQA dataset

- Recommended (ModelScope): `PACKS="08"` includes `questions/MedMCQA/{train,dev,test}.json`
- Project: https://medmcqa.github.io/
- Upstream: https://github.com/medmcqa/medmcqa
- Download: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky
- Hugging Face: https://huggingface.co/datasets/openlifescienceai/medmcqa
- Paper: https://arxiv.org/abs/2203.14371
- **No ready-made MedMCQA retrieval-augmented JSON pack on ModelScope**; build as below, then arrange as `dataset/RAG_MedMCQA_{train,dev}.json` for eval discovery.

Place `train.json` / `dev.json` under `dataset/pipeline_data/questions/MedMCQA/`; before retrieval, convert to MedQA-aligned jsonl fields with `convert_medmcqa_to_medqa_jsonl.py`.

Retrieval embeddings live under `dataset/pipeline/models/` (`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`). Paths are set in `dataset/pipeline/config.py`; if you move them, update `model_path`. Before rebuilding, fetch:

```bash
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

(If you already have pack `07` FAISS indexes, you can skip the matching language index builds in steps 1–2 below.)

### Dataset construction (from raw corpora)

Run the following under `dataset/pipeline/`. For a single GPU use `--device_ids="0"` and `--num_process=1`.  
The retrieval query key is always `--query_key_name="question_with_options"` (stem + options); do not write results from different query keys into the same output path.

Outputs are JSON arrays with fields such as `id`, `question_with_options`, `answer_idx`, `retrieve_info` (same schema as pack `04`).

#### 1. Chunk textbook text into JSON

```bash
cd dataset/pipeline

python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

#### 2. Build vector indexes

```bash
cd dataset/pipeline

python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'
```

#### 3. Retrieval-augment train / dev / test

Query key: `--query_key_name="question_with_options"`. Output path:

`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

Single GPU: `--device_ids="0"` and `--num_process=1`. Inputs must be **jsonl** (one sample per line).

```bash
cd dataset/pipeline

# ---------- MedQA English US (default eval subset; or use pack 04 directly)----------
python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/US/test.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/US/test" \
    --output_json_file="test.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

# ---------- MedQA Chinese Mainland (optional)----------
python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/train.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/dev.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/Mainland/test.jsonl" \
    --embedding_model_name="stella-base-zh-v2" \
    --store_path="../pipeline_data/vector_stores/zh_paragraph" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/Mainland/test" \
    --output_json_file="test.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

# ---------- MedMCQA: align schema then retrieve (train / dev)----------
# Raw fields: opa/opb/opc/opd + cop; convert to MedQA-aligned jsonl first
# Place train.json / dev.json under ../pipeline_data/questions/MedMCQA/
python convert_medmcqa_to_medqa_jsonl.py \
  --in_jsonl ../pipeline_data/questions/MedMCQA/train.json \
  --out_jsonl ../pipeline_data/questions/MedMCQA/train.jsonl

python convert_medmcqa_to_medqa_jsonl.py \
  --in_jsonl ../pipeline_data/questions/MedMCQA/dev.json \
  --out_jsonl ../pipeline_data/questions/MedMCQA/dev.jsonl

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/MedMCQA/train.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/MedMCQA/train" \
    --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"

python generate_question_with_knowledges.py \
    --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
    --input_jsonl_path="../pipeline_data/questions/MedMCQA/dev.jsonl" \
    --embedding_model_name="stella-base-en-v2" \
    --store_path="../pipeline_data/vector_stores/en" \
    --output_json_dir="../pipeline_data/questions_with_knowledge/MedMCQA/dev" \
    --output_json_file="dev.json" --device_ids="0,1,2,3,4,5,6,7" \
    --query_key_name="question_with_options"
```

#### 4. Arrange under `dataset/` (for train / eval discovery)

Copy or symlink step-3 JSON outputs to the expected filenames, e.g.:

```bash
# MedQA US (if not using pack 04)
cp dataset/pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
   dataset/RAG_MedQA_US_train_10178.json
cp dataset/pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
   dataset/RAG_MedQA_US_dev_1272.json
cp dataset/pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test.json \
   dataset/RAG_MedQA_US_test_1273.json

# MedMCQA (default eval lookup paths)
cp dataset/pipeline_data/questions_with_knowledge/MedMCQA/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
   dataset/RAG_MedMCQA_dev.json
cp dataset/pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
   dataset/RAG_MedMCQA_train.json
```

You can then train and evaluate. For MedQA you may use pack `04` directly and skip the US parts of steps 1–4.

### Checkpoint paths (after ModelScope extract)

After `PACKS="01"` (or `PACKS="01 04"`) and downloading `ckpt_0`, train / eval paths are:

| Path | Role | Source |
|------|------|------|
| `parameters/ckpt_0/` | System weights; loaded by default in the integrated train/infer pipeline | ModelScope directory download |
| `parameters/sft_PMT_en_1024.pth` | Released PMT; default init for retrieval-augmented training | Pack `01` |
| `out/run_pmt/cascade_PMT_en_1024_best.pth` | Best PMT from retrieval-augmented training; default eval load | Local training |

Suggested order:

1. Download system weights `parameters/ckpt_0/` (see snapshot_download above)
2. `PACKS="01 04" bash scripts/download_modelscope_assets.sh` → `sft_PMT_en_1024.pth` + MedQA retrieval JSON
3. (Optional) `PACKS="05 06 07 08"` to rebuild retrieval or prepare MedMCQA

## Train and eval

```bash
# Train (default init: parameters/sft_PMT_en_1024.pth)
DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
```

**Default training output locations** (override with `OUT` / `INIT`):

| Path | Role |
|------|------|
| `out/run_pmt/` | Training output directory (default) |
| `out/run_pmt/cascade_PMT_en_1024_best.pth` | Best checkpoint on the validation set (default eval load) |
| `out/run_pmt/cascade_PMT_en_1024_epochN.pth` | Per-epoch snapshots (if written by the script) |

```bash
# Custom output dir / init weights
INIT=parameters/sft_PMT_en_1024.pth OUT=out/run_pmt_exp1 \
  DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
```

### Eval (loads trained checkpoint by default)

```bash
# Default: load out/run_pmt/cascade_PMT_en_1024_best.pth; eval USMLE + MedMCQA
DATASETS=USMLE,MEDMCQA bash scripts/run_cascade_infer.sh
```

Results go to `results/result_summary`; if that directory already exists, `result_summary1`, `result_summary2`, … are used (no overwrite). See [results/README.md](results/README.md) and [results/result_summary/README.md](results/result_summary/README.md).

### Eval a specific PMT checkpoint

The integrated pipeline still loads `parameters/ckpt_0/`; set `CHECKPOINT` (or `--checkpoint`) to choose which PMT weights to evaluate:

```bash
# Best weights trained in this repo
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth \
  DATASETS=USMLE bash scripts/run_cascade_infer.sh

# Released init weights (eval without this repo's retrieval-augmented training)
CHECKPOINT=parameters/sft_PMT_en_1024.pth \
  DATASETS=USMLE bash scripts/run_cascade_infer.sh

# Single JSON + explicit result directory
CHECKPOINT=out/run_pmt_exp1/cascade_PMT_en_1024_best.pth \
  DATA=dataset/RAG_MedQA_US_test_1273.json \
  DATASETS=USMLE \
  RESULT_DIR=results/result_summary_manual \
  bash scripts/run_cascade_infer.sh
```

Equivalent Python entrypoint:

```bash
python inference_cascade_pmt.py \
  --datasets USMLE \
  --checkpoint out/run_pmt/cascade_PMT_en_1024_best.pth
```

## License

See `LICENSE` at the repository root.

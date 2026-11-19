# Phase-modulation-transforme

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

中文说明：[README.md](README.md) / [README_CN.md](README_CN.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

This repository releases the complete experimental protocol for a Phase-Modulation Retrieval-Augmented Transformer on MedQA multiple-choice medical question answering, including implementation code, data-construction scripts, and reference evaluation results. Released Phase-modulation-transforme checkpoints are already pretrained; users start from these weights for supervised fine-tuning (SFT) and evaluation, and need not run pretraining. Large model checkpoints, corpora, and embedding resources are hosted on ModelScope (see below and [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)).

Three contrastive evaluation protocols are provided (each predicts an option letter, compares it with the gold label, and reports accuracy):

| Protocol | Description | Entry |
|----------|-------------|-------|
| `no_pmt` | Upstream only (no Phase-modulation-transforme) | `--protocol no_pmt` |
| `pmt_norag` | Non-retrieval upstream → Phase-modulation-transforme | `--protocol pmt_norag` |
| `pmt_rag` | Retrieval-augmented upstream → Phase-modulation-transforme | `--protocol pmt_rag` |

Training and inference automatically resolve distinct Phase-modulation-transforme checkpoints. Default paths:

- **Init weights (download / pretrain output)**: `parameters/sft_PMT_en_1024.pth`
- **`pmt_norag` train default save dir**: `out/run_pmt_norag/`, best checkpoint `cascade_PMT_en_1024_best.pth`
- **`pmt_rag` train default save dir**: `out/run_pmt_rag/`, best checkpoint `cascade_PMT_en_1024_best.pth`
- **Eval default load**: the matching `*_best.pth` under those dirs (override via env/args)

Evaluation artifacts record the final option letter and accuracy-related fields.

> The default training seed is `1337`; GPU nondeterminism may induce negligible numerical variation.
>
> Released MedQA samples follow the `generate_question_with_knowledges.py` schema only.

## Environment and dependencies

```bash
cd Phase-modulation-transforme
pip install -r requirements.txt
```

Install a PyTorch build matched to the local CUDA driver. Ascend platforms additionally require `torch_npu` (see comments in `requirements.txt`).

Optional downstream benchmarks:

```bash
pip install lm-eval==0.4.9
```

### Core dependencies

Full list: [requirements.txt](./requirements.txt). Primary components:

| Package | Notes |
|---------|-------|
| python | 3.8+ |
| torch | ≥2.0, matched to local CUDA |
| transformers | ≥4.37 |
| datasets / wandb / tqdm | Data loading, experiment logging, progress |
| modelscope | Optional; ModelScope asset retrieval |

### Compute requirements

Contrastive protocols load upstream modules sequentially by peak demand (all three need not be resident at once). Recommended configuration (bfloat16; peaks vary with context length, `max_new_tokens`, and backend):

| Item | Recommendation |
|------|----------------|
| GPUs | **8×80 GB** (e.g. A100 / A800 / H100) |
| Disk | ≥400 GB (all upstream weights plus packs `01`/`04`, etc.) |
| Host RAM | ≥256 GB (weight load / sharding) |

Additional notes: default training `NPROC=1`; prefer multi-GPU `device_map=auto` or vLLM tensor parallelism (`SIDECAR_TP*`). NVIDIA GPU with a CUDA-matched PyTorch build; Ascend requires `torch_npu` (see `requirements.txt`).

## ModelScope asset retrieval

The GitHub repository primarily contains source code; checkpoints, corpora, and embeddings are published on ModelScope.

**ModelScope repository:** [`zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`](https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
Includes archive packs `01`–`05` and the three contrastive-protocol directories `parameters/ckpt_a`, `parameters/ckpt_b`, `parameters/ckpt_c`. Details: [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md).

### Upstream generation checkpoints (required for contrastive protocols)

`pmt_norag` uses `parameters/ckpt_a`, `pmt_rag` uses `parameters/ckpt_b`, and `no_pmt` uses `parameters/ckpt_c`. From the project root:

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_a/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_b/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_c/**'])"
```

Directories may also be obtained manually from the ModelScope page. Branch routing is defined in `model/sidecar_map.json` and selected via `--protocol`.

Also required: Phase-modulation-transforme initialization weights (`parameters/sft_PMT_en_1024.pth` in pack `01`) and sample JSON with `question_with_options` and `answer_idx` (retrieval fields optional).

### Batch retrieval of archive packs (`.tar`)

From the **project root** (extracts into the conventional `parameters/`, `out/`, `dataset/` layout):

```bash
pip install -U modelscope

# Phase-modulation-transforme init weights + MedQA retrieval samples
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# Retrieval rebuild: embeddings + textbooks + FAISS + raw questions (includes English)
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

Alternatively, download `.tar` files from ModelScope and run `tar -xf <pack>.tar` in the project root.

### Asset inventory and local paths

| ModelScope file / directory | Local path | Role |
|-----------------------------|------------|------|
| `parameters/ckpt_a` | `parameters/ckpt_a/` | Cascade upstream module, slot 0 |
| `parameters/ckpt_b` | `parameters/ckpt_b/` | Cascade upstream module, slot 1 |
| `parameters/ckpt_c` | `parameters/ckpt_c/` | Upstream module for protocol `no_pmt` |
| `01_parameters.tar` | `parameters/PMT_*`, `parameters/MiniMind_*`, `parameters/sft_PMT_en_1024.pth` | Pretrained Phase-modulation-transforme (Hugging Face and SFT initialization checkpoints) |
| `03b_corpus_openhermes2_5.tar` | `dataset/openhermes2_5.jsonl` | English general instruction corpus |
| `03d_corpus_sft_mini_512_zh.tar` | `dataset/sft_mini_512_zh.jsonl` | Chinese instruction corpus (optional) |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA retrieval samples (`generate_question_with_knowledges.py` schema) |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | Stella embeddings for retrieval (en/zh) |
| `06_textbooks.tar` | `dataset/pipeline_data/textbooks/` | MedQA textbooks (en / zh_paragraph / zh_sentence and matching json) |
| `07_vector_stores.tar` | `dataset/pipeline_data/vector_stores/` | Language-matched FAISS indices (en / zh_paragraph / zh_sentence) |
| `08_questions.tar` | `dataset/pipeline_data/questions/` | MedQA US/Mainland and MedMCQA raw questions |

### Minimal usage after retrieval

```bash
# One-click contrastive benchmark → comparison.csv
bash scripts/run_protocol_benchmark.sh
```

## Repository layout

```
Phase-modulation-transforme/
├── model/                 # Phase-modulation-transforme and upstream routing / prompt config
├── trainer/               # General SFT, cascade training
├── inference_cascade_pmt.py
├── tools/                 # Checkpoint format conversion and length extrapolation
├── scripts/               # Cascade launchers, baseline eval, ModelScope asset scripts
├── dataset/
│   ├── pipeline/          # MedQA retrieval construction pipeline and embeddings
│   ├── pipeline_data/     # Textbooks / FAISS (PACKS 06+07) and runtime retrieval artifacts
│   └── *.json(l)          # SFT corpora and evaluation samples
├── parameters/
│   ├── ckpt_a/            # Cascade upstream module, slot 0
│   ├── ckpt_b/            # Cascade upstream module, slot 1
│   ├── ckpt_c/            # no_pmt upstream module
│   └── sft_PMT_en_1024.pth # Pretrained Phase-modulation-transforme (train init)
├── out/run_pmt_norag/ # pmt_norag checkpoints
├── out/run_pmt_rag/   # pmt_rag checkpoints
├── results/
│   └── protocol_benchmark/               # Contrastive accuracy tables
├── requirements.txt
└── LICENSE
```

## Data

### Corpus and sample inventory (after corresponding `PACKS` extraction)

| Stage | Path | Pack |
|-------|------|------|
| English general instruction tuning | `dataset/openhermes2_5.jsonl` | `03b` |
| Chinese instruction tuning (optional) | `dataset/sft_mini_512_zh.jsonl` | `03d` |
| MedQA US train | `dataset/RAG_MedQA_US_train_10178.json` (10178) | `04` |
| MedQA US validation | `dataset/RAG_MedQA_US_dev_1272.json` (1272) | `04` |
| MedQA US test | `dataset/RAG_MedQA_US_test_1273.json` (1273) | `04` |

Retrieval: `PACKS="04"`, or jointly with checkpoints via `PACKS="01 04"`. Samples follow the retrieval-pipeline schema (with or without `retrieve_info`). Public corpus mirrors are listed in [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md) when available.

To rebuild retrieval-augmented samples from raw MedQA, see **MedQA dataset**, **Dataset construction**, [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md), and [dataset/pipeline/README.md](dataset/pipeline/README.md).

### MedQA dataset

- Preferred (ModelScope): `PACKS="06 08"` → `dataset/pipeline_data/textbooks/` and `questions/`  
- Upstream release: https://github.com/jind11/MedQA  
- Download: https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing  
- Chinese overview: https://zhuanlan.zhihu.com/p/679590312  

Layout:

```
dataset/pipeline_data/
├── textbooks/     # en / zh_paragraph / zh_sentence (txt)
└── questions/     # US|Mainland with train.jsonl / dev.jsonl / test.jsonl
```

### MedMCQA dataset

- Preferred (ModelScope): `PACKS="08"` includes `questions/MedMCQA/{train,dev,test}.json`  
- Project page: https://medmcqa.github.io/  
- Upstream release: https://github.com/medmcqa/medmcqa  
- Download: https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky  
- Hugging Face: https://huggingface.co/datasets/openlifescienceai/medmcqa  
- Paper: https://arxiv.org/abs/2203.14371  

Place `train.json` / `dev.json` under `dataset/pipeline_data/questions/MedMCQA/`; convert to MedQA-aligned jsonl fields with `convert_medmcqa_to_medqa_jsonl.py` before retrieval.

Retrieval embeddings reside in `dataset/pipeline/models/` (`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`). Paths are set in `dataset/pipeline/config.py`; update `model_path` if relocated.

### Dataset construction (from raw corpora)

Run all commands under `dataset/pipeline/`. Single-GPU: `--device_ids="0"` and `--num_process=1`.  
The retrieval query key is uniformly `--query_key_name="question_with_options"` (stem concatenated with options). Do not write results for different query keys into the same output path.

#### 1. Chunk textbook text into JSON

```bash
cd dataset/pipeline

python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

#### 2. Build vector indices

```bash
cd dataset/pipeline

python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'
```

#### 3. Retrieval augmentation for train / dev / test

Query key: `--query_key_name="question_with_options"`. Output path:

`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

Single-GPU: `--device_ids="0"` and `--num_process=1`. Input must be **jsonl** (one sample per line).

```bash
cd dataset/pipeline

# ---------- MedQA English US (default evaluation subset) ----------
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

# ---------- MedQA Chinese Mainland (optional) ----------
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

# ---------- MedMCQA: schema alignment then retrieval (train / val) ----------
# Raw fields: opa/opb/opc/opd + cop; convert to MedQA-aligned jsonl schema first
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

After retrieval augmentation, proceed to **Online cascade training and inference**. Sample JSON from ModelScope pack `04` may be used directly. See [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md).

### Checkpoint paths (after ModelScope extraction)

After `PACKS="01"` / `PACKS="02"` or a full download:

| Path | Description | Pack |
|------|-------------|------|
| `parameters/PMT_Base_300M/pytorch_model.bin` | QF Base | `01` |
| `parameters/PMT_Struct_300M/pytorch_model.bin` | QF Struct | `01` |
| `parameters/MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base | `01` |
| `parameters/MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct | `01` |
| `parameters/sft_PMT_en_1024.pth` | Pretrained Phase-modulation-transforme (SFT / cascade initialization) | `01` |

Phase-modulation-transforme weights in pack `01` are the intended SFT starting point; SFT outputs may also be obtained under the protocols below.

## General instruction tuning (optional)

If `parameters/sft_PMT_en_1024.pth` is already available, skip this section and proceed to protocol training / evaluation.

```bash
cd trainer
python train_full_sft.py \
    --model_type QF \
    --language en \
    --hidden_size 1024 \
    --num_hidden_layers 24 \
    --num_attention_heads 8 \
    --num_key_value_heads 2 \
    --max_seq_len 1024 \
    --batch_size 100 \
    --learning_rate 5e-4 \
    --sft_lr 1e-4 \
    --epochs 2 \
    --data_path ../dataset/openhermes2_5.jsonl \
    --out_dir ../parameters
```

Produces `parameters/sft_PMT_en_1024.pth`. For Chinese, set `--language zh` with `../dataset/sft_mini_512_zh.jsonl`.

Export from a Hugging Face directory: `bash scripts/export_stage1_from_hf.sh`.

## Contrastive protocols: training and evaluation

```text
no_pmt      : standalone upstream → option letter → accuracy
pmt_norag   : non-retrieval branch + Phase-modulation-transforme → option letter → accuracy
pmt_rag     : retrieval-augmented branch + Phase-modulation-transforme → option letter → accuracy
```

- Training: `trainer/train_cascade_pmt.py --protocol pmt_norag|pmt_rag` (auto-loads pretrained init weights)
- Evaluation: `inference_cascade_pmt.py --protocol no_pmt|pmt_norag|pmt_rag|all` (auto-loads protocol `*_best.pth` when Phase-modulation-transforme is used)
- Routing: `model/sidecar_map.json`, `model/pmt_protocol.py`, `model/sidecar_runtime.py`

### Prerequisites

1. Obtain `parameters/ckpt_a`, `parameters/ckpt_b`, `parameters/ckpt_c` (ModelScope section)
2. Obtain Phase-modulation-transforme initialization weights: `PACKS="01" bash scripts/download_modelscope_assets.sh`
3. Prepare sample JSON (pack `04`, or retrieval-augmented item JSON)

### Training (per protocol)

By default training loads init weights from `parameters/sft_PMT_en_1024.pth` and writes checkpoints to the protocol directory:

- `PROTOCOL=pmt_norag` → save under `out/run_pmt_norag/`, best file `out/run_pmt_norag/cascade_PMT_en_1024_best.pth`
- `PROTOCOL=pmt_rag` → save under `out/run_pmt_rag/`, best file `out/run_pmt_rag/cascade_PMT_en_1024_best.pth`

Override with env vars: `INIT` (init checkpoint path), `OUT` (output directory).

```bash
DATA=dataset/RAG_MedQA_US_train_10178.json \
  PROTOCOL=pmt_norag bash scripts/run_cascade_train.sh

DATA=dataset/RAG_MedQA_US_train_10178.json \
  PROTOCOL=pmt_rag bash scripts/run_cascade_train.sh
```

### Evaluation (three protocols)

Eval loads trained weights by protocol: `pmt_norag` from `out/run_pmt_norag/cascade_PMT_en_1024_best.pth`, `pmt_rag` from `out/run_pmt_rag/cascade_PMT_en_1024_best.pth`; `no_pmt` does not load them. Results can be written under `results/` (e.g. `OUT_CSV=results/protocol_eval.csv`).

```bash
DATA=dataset/RAG_MedQA_US_test_1273.json \
PROTOCOL=all OUT_CSV=results/protocol_eval.csv \
  bash scripts/run_cascade_infer.sh
```

Result CSVs contain prediction fields such as `extracted_choice` and `matches_answer_idx`.

### Checkpoint metadata scrubbing and upload (maintenance)

```bash
python scripts/scrub_ckpt_metadata.py
# Inject credentials via environment variables only; do not commit tokens
export MODELSCOPE_API_TOKEN=your_token
python scripts/upload_ckpts_to_modelscope.py
```

## Reproduction summary

### A. Three contrastive protocols (one-click benchmark)

```bash
pip install -r requirements.txt
pip install -U modelscope

# Obtain ckpt_a / ckpt_b / ckpt_c (see snapshot_download above)
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# One-click: train missing pmt_norag/pmt_rag if needed, then eval on US dev + US test.
# Train default save: out/run_pmt_norag|rag/cascade_PMT_en_1024_best.pth
# Eval comparison table default: results/protocol_benchmark/latest/comparison.csv
bash scripts/run_protocol_benchmark.sh
# Equivalent: python scripts/run_protocol_benchmark.py
```

Eval only (best checkpoints already under `out/run_pmt_norag` / `out/run_pmt_rag`):

```bash
SKIP_TRAIN=1 bash scripts/run_protocol_benchmark.sh
```

Custom test sets (`name=path`, comma-separated):

```bash
DATASETS="medqa_us_test=dataset/RAG_MedQA_US_test_1273.json" \
  bash scripts/run_protocol_benchmark.sh
```

#### Where results are and how to read them

After evaluation, check the accuracy comparison table (each run also writes a timestamped directory and syncs `latest/`):

| Path | Contents |
|------|----------|
| `results/protocol_benchmark/latest/comparison.csv` | Quick entry: dataset × protocol accuracy (open in Excel / any editor) |
| `results/protocol_benchmark/latest/comparison.md` | Same table as Markdown for reading in-repo |
| `results/protocol_benchmark/<timestamp>/` | Full run artifacts: `comparison.csv` / `comparison.md` / `summary.json` |
| `results/protocol_benchmark/<timestamp>/details/` | Per-sample predictions, e.g. `medqa_us_test__pmt_rag.csv` |

```bash
# Summary table (recommended)
cat results/protocol_benchmark/latest/comparison.md
# or
head -n 20 results/protocol_benchmark/latest/comparison.csv

# Per-protocol detail CSVs
ls results/protocol_benchmark/*/details/*.csv
```

In `comparison.*`, focus on `dataset`, `protocol`, `accuracy` (plus `n_match` / `n_scored`). Detail CSVs contain final option letters and scoring fields.


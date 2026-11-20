# Phase-modulation-transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

中文：[README_CN.md](README_CN.md) · [README.md](README.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

This repository releases a **Phase-Modulation Retrieval-Augmented Transformer** for medical multiple-choice QA: code, data-construction scripts, and reference evaluation results. The default pipeline runs under a **retrieval-augmented** setting with jointly released system weights and PMT checkpoints, producing option-letter predictions and accuracy reports. Large assets are hosted on ModelScope ([MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)).

| Stage | Role | Typical artifact |
|------|------|------------------|
| Optional general instruction alignment | OpenHermes-style SFT | `parameters/sft_PMT_en_1024.pth` |
| Main: retrieval-augmented train / eval | USMLE (MedQA) and MedMCQA | `out/run_pmt/…_best.pth`, `results/result_summary*` |

- **System weights**: `parameters/ckpt_0/` (ModelScope)
- **PMT init**: `parameters/sft_PMT_en_1024.pth` (pack `01`)
- **Train output**: `out/run_pmt/`
- **Default eval sets**: USMLE, MedMCQA
- **Eval output**: `results/result_summary`, or `result_summary1`, `result_summary2`, … if the folder already exists. Detail CSVs omit intermediate prompt text columns.

Historical temperature × NTK ablations live under `results/result_summary/` (git-tracked).

## Environment

```bash
cd Phase-modulation-transformer
pip install -r requirements.txt
```

Install a CUDA-matched PyTorch build. Ascend needs `torch_npu` (see `requirements.txt`).

| Item | Suggestion |
|------|------------|
| GPU | Multi-GPU, large VRAM (full stack, e.g. 8×80 GB) |
| Disk | ≥400 GB (`ckpt_0` + packs) |
| RAM | ≥256 GB |

## ModelScope assets

GitHub holds source code; weights, corpora, and embeddings are on ModelScope:

**Repo**: [zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering](https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)

Download **from the project root** so archives extract into the paths below (full Chinese table: [README_CN.md](README_CN.md) / [README.md](README.md)).

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

# System weights → parameters/ckpt_0/
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"

# Pack 01 → parameters/PMT_Base_300M/, pretrain_*.pth, sft_PMT_en_1024.pth
# Pack 04 → dataset/RAG_MedQA_US_{train,dev,test}_*.json
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# Packs 05–08 → dataset/pipeline/models/, pipeline_data/{textbooks,vector_stores,questions}/
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

Or download `.tar` files manually and run `tar -xf <pack>.tar` at the project root.

| After download | Local path | Use |
|----------------|------------|-----|
| `parameters/ckpt_0` | `parameters/ckpt_0/` | System weights |
| `01_parameters.tar` | `parameters/PMT_Base_300M/`, `parameters/pretrain_PMT_en_1024_final.pth`, `parameters/sft_PMT_en_1024.pth` | PMT pretrain / SFT init |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_*.json` | MedQA / USMLE RAG samples |
| `05`–`08` | `dataset/pipeline/models/`, `dataset/pipeline_data/...` | Rebuild retrieval / MedMCQA raw questions |
| Train output | `out/run_pmt/cascade_PMT_en_1024_best.pth` | Default eval PMT checkpoint |

Routing: `model/sidecar_map.json` (`ckpt_0`, always retrieval-augmented).

## Quick start

```bash
bash scripts/run_cascade_train.sh
bash scripts/run_cascade_infer.sh
```

> MedMCQA: ModelScope currently ships raw questions only (pack `08`). Full rebuild steps (textbook chunking → FAISS → retrieval for MedQA US / Mainland / MedMCQA, then copy to `dataset/RAG_*.json`) are documented in the Chinese README **数据** section ([README_CN.md](README_CN.md) / [README.md](README.md)). Or place `dataset/RAG_MedMCQA_dev.json` before eval.

## Train / eval

```bash
DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
DATASETS=USMLE,MEDMCQA bash scripts/run_cascade_infer.sh
```

Outputs go to `results/result_summary` or auto-incremented `result_summaryN/`. See [results/README.md](results/README.md) and [results/result_summary/README.md](results/result_summary/README.md).

## License

See `LICENSE` at the repository root.

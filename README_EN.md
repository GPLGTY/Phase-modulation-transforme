# Phase-modulation-transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

中文：[README_CN.md](README_CN.md) · [README.md](README.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

This repository releases a **Phase-Modulation Retrieval-Augmented Transformer** for medical multiple-choice QA: code, data-construction scripts, and reference evaluation results. The default pipeline runs under a **retrieval-augmented** setting with jointly released system weights and a ready-to-use PMT checkpoint. Large assets are hosted on ModelScope ([MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)).

Download and use directly (**no** local pretraining or general instruction SFT required):

| Asset | Path | Role |
|------|------|------|
| System weights | `parameters/ckpt_0/` | ModelScope directory download |
| Released PMT | `parameters/sft_PMT_en_1024.pth` | Pack `01`; default train init / optional eval |
| Trained PMT | `out/run_pmt/cascade_PMT_en_1024_best.pth` | After retrieval-augmented training in this repo |
| Eval output | `results/result_summary` or `result_summaryN/` | Auto-increments; detail CSVs omit intermediate prompt columns |

Default eval sets: USMLE, MedMCQA. Full paths, train save locations, and `CHECKPOINT=…` inference examples: Chinese README **训练与评测** section.

## Environment

```bash
cd Phase-modulation-transformer
pip install -r requirements.txt
```

## ModelScope assets

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
PACKS="01 04" bash scripts/download_modelscope_assets.sh   # → sft_PMT_en_1024.pth + MedQA RAG JSON
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

| After download | Local path |
|----------------|------------|
| `parameters/ckpt_0` | `parameters/ckpt_0/` |
| `01_parameters.tar` | `parameters/sft_PMT_en_1024.pth` |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_*.json` |

## Train / eval / pick a checkpoint

```bash
# Train → out/run_pmt/cascade_PMT_en_1024_best.pth
DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh

# Default eval (loads out/run_pmt/..._best.pth)
DATASETS=USMLE,MEDMCQA bash scripts/run_cascade_infer.sh

# Eval a specific PMT checkpoint (system weights still from ckpt_0)
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth DATASETS=USMLE bash scripts/run_cascade_infer.sh
CHECKPOINT=parameters/sft_PMT_en_1024.pth DATASETS=USMLE bash scripts/run_cascade_infer.sh
```

## License

See `LICENSE` at the repository root.

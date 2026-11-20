# Phase-modulation-transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

English: [README_EN.md](README_EN.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

本仓库发布 **Phase-Modulation Retrieval-Augmented Transformer** 在医学选择题问答上的实现、数据构建与参考评测结果。默认在**检索增强**设定下运行完整流程：加载一体发布的系统权重与 PMT 检查点，完成选项字母预测并报告准确率。大体量资源托管于 ModelScope（见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)）。

| 阶段 | 说明 | 典型产物 |
|------|------|----------|
| 通用指令对齐（可选） | OpenHermes 等通用对话 SFT | `parameters/sft_PMT_en_1024.pth` |
| 检索增强训练 / 评测（主流程） | USMLE（MedQA）与 MedMCQA | `out/run_pmt/cascade_PMT_en_1024_best.pth`、`results/result_summary*` |

- **系统权重**：`parameters/ckpt_0/`（ModelScope）
- **PMT 初始化**：`parameters/sft_PMT_en_1024.pth`（归档包 `01`）
- **训练输出**：`out/run_pmt/`
- **默认评测集**：USMLE、MedMCQA
- **评测输出**：`results/result_summary`；若目录已存在则写入 `result_summary1`、`result_summary2`…（不覆盖）。明细 CSV **不含**中间输入文本列。

历史温度 × NTK 消融结果见 `results/result_summary/`（git 追踪）。新消融：

```bash
bash scripts/run_ablation_temp_ntk.sh
# 输入为包 04 检索增强 JSON；输出写入 result_summary 或 result_summaryN
```

## 环境

```bash
cd Phase-modulation-transformer
pip install -r requirements.txt
```

请安装与本机 CUDA 匹配的 PyTorch。昇腾需额外安装 `torch_npu`（见 `requirements.txt`）。

| 项目 | 建议 |
|------|------|
| GPU | 多卡大显存（完整流程，如 8×80 GB） |
| 磁盘 | ≥400 GB（含 `ckpt_0` 与归档包） |
| 内存 | ≥256 GB |

## ModelScope 资源获取

GitHub 仓库以源代码为主；模型权重、语料与 Embedding 统一发布于 ModelScope。

**ModelScope 仓库**：[zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering](https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
包含归档包 `01`/`04`/`05` 等，以及一体流程所需目录 `parameters/ckpt_0/`。细节见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)。

### 系统权重（一体流程必需）

默认流程始终加载 `parameters/ckpt_0/`（与 PMT 一体使用）。于**项目根目录**执行：

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

亦可在 ModelScope 页面按目录手动获取到本地 `parameters/ckpt_0/`。路由见 `model/sidecar_map.json`。

另需 Phase-modulation-transformer 初始化权重（归档包 `01` 中的 `parameters/sft_PMT_en_1024.pth`），以及含 `question_with_options`、`answer_idx` 的样本 JSON（可含 `retrieve_info`）。

### 归档包（`.tar`）批量获取

于**项目根目录**执行（脚本下载后解压至 `parameters/`、`dataset/` 等约定路径）：

```bash
pip install -U modelscope

# Phase-modulation-transformer 初始化权重 + MedQA 检索样本
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 检索重建：Embedding + 教材 + FAISS + 原始题目（含 MedMCQA）
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

亦可自 ModelScope 页面下载 `.tar` 后，在**项目根目录**执行 `tar -xf <包名>.tar`。

### 资源清单与本地路径

| ModelScope 文件 / 目录 | 本地路径 | 用途 |
|------------------------|----------|------|
| `parameters/ckpt_0` | `parameters/ckpt_0/` | 系统权重（检索增强一体流程） |
| `01_parameters.tar` | `parameters/PMT_Base_300M/`、`parameters/pretrain_PMT_en_1024_final.pth`（或由 Base 转换）、`parameters/sft_PMT_en_1024.pth` | Phase-modulation-transformer 预训练权重与通用指令对齐 / 训练初始化检查点 |
| `03b_corpus_openhermes2_5.tar` | `dataset/openhermes2_5.jsonl` | 英文通用指令微调语料 |
| `03d_corpus_sft_mini_512_zh.tar` | `dataset/sft_mini_512_zh.jsonl` | 中文指令微调语料（可选） |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA / USMLE 检索增强样本 |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | 检索用 Stella Embedding（en/zh） |
| `06_textbooks.tar` | `dataset/pipeline_data/textbooks/` | MedQA 教材（en / zh_paragraph / zh_sentence） |
| `07_vector_stores.tar` | `dataset/pipeline_data/vector_stores/` | 语言匹配 FAISS 索引（en / zh_paragraph / zh_sentence） |
| `08_questions.tar` | `dataset/pipeline_data/questions/` | MedQA US/Mainland 与 MedMCQA 原始题目（**尚无现成 MedMCQA RAG JSON 包**） |

按场景选用：

| 目标 | 建议 |
|------|------|
| 检索增强训练 / 评测（USMLE） | `ckpt_0` + `PACKS="01 04"` |
| MedMCQA 评测 | 另按数据节生成 `dataset/RAG_MedMCQA_*.json`，或自备同口径文件 |
| 通用指令微调 | 追加 `03b`（中文追加 `03d`） |
| 自原始题库重建检索 | `PACKS="05 06 07 08"` |

## 快速开始

```bash
# 训练（默认 MedQA 检索增强训练集）
bash scripts/run_cascade_train.sh

# 评测 USMLE + MedMCQA → results/result_summary 或 result_summaryN
bash scripts/run_cascade_infer.sh
```

> MedMCQA：ModelScope 目前仅提供原始题目（包 `08`）。请先按下方数据节生成检索增强 JSON，或自备 `dataset/RAG_MedMCQA_dev.json` 等路径后再评测。

## 仓库结构

```
Phase-modulation-transformer/
├── model/                 # Phase-modulation-transformer 与系统权重路由 / 提示配置
├── trainer/               # 通用指令对齐、检索增强训练
├── inference_cascade_pmt.py
├── tools/                 # 权重格式转换与长度外推
├── scripts/               # 训练 / 评测启动、ModelScope 资源脚本
├── dataset/
│   ├── pipeline/          # MedQA 检索构建流水线与 Embedding
│   ├── pipeline_data/     # 教材 / FAISS（PACKS 06+07）与运行时检索产物
│   └── *.json(l)          # SFT 语料与评测样本
├── parameters/
│   ├── ckpt_0/            # 系统权重（ModelScope 目录直传）
│   ├── PMT_Base_300M/     # 预训练 HF 权重（包 01，可选）
│   ├── pretrain_PMT_en_1024_final.pth  # 预训练 Torch 权重（包 01 或本地转换）
│   └── sft_PMT_en_1024.pth # 通用指令对齐后的训练初始化权重（包 01）
├── out/run_pmt/           # 检索增强训练产出的 PMT 检查点
├── results/result_summary/ # 参考评测与消融结果
├── requirements.txt
└── LICENSE
```

## 数据

对外训练 / 评测 / 消融输入均为检索流水线产物 JSON（字段含 `question_with_options`、`answer_idx`，检索增强时含 `retrieve_info`）。**不**落盘其它中间样本格式。一体流程在线使用系统权重与 PMT。

### 语料与样本清单（对应 `PACKS` 解压后）

| 阶段 | 路径 | 包 |
|------|------|-----|
| 英文通用指令微调 | `dataset/openhermes2_5.jsonl` | `03b` |
| 中文指令微调（可选） | `dataset/sft_mini_512_zh.jsonl` | `03d` |
| MedQA US 训练集 | `dataset/RAG_MedQA_US_train_10178.json`（10178） | `04` |
| MedQA US 验证集 | `dataset/RAG_MedQA_US_dev_1272.json`（1272） | `04` |
| MedQA US 测试集 | `dataset/RAG_MedQA_US_test_1273.json`（1273） | `04` |

获取方式：`PACKS="04"`，或与检查点一并 `PACKS="01 04"`。亦可自原始语料按下文重建；细节见 [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md)、[dataset/pipeline/README.md](dataset/pipeline/README.md)。

### MedQA 数据集

- 推荐（ModelScope）：`PACKS="06 08"` 解压至 `dataset/pipeline_data/textbooks/` 与 `questions/`
- 原始发布：https://github.com/jind11/MedQA
- 数据下载：https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing
- 中文简介：https://zhuanlan.zhihu.com/p/679590312

目录约定：

```
dataset/pipeline_data/
├── textbooks/     # en / zh_paragraph / zh_sentence（txt）
└── questions/     # US|Mainland 下的 train.jsonl / dev.jsonl / test.jsonl
```

### MedMCQA 数据集

- 推荐（ModelScope）：`PACKS="08"` 含 `questions/MedMCQA/{train,dev,test}.json`
- 项目主页：https://medmcqa.github.io/
- 原始发布：https://github.com/medmcqa/medmcqa
- 数据下载：https://drive.google.com/uc?export=download&id=15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky
- Hugging Face：https://huggingface.co/datasets/openlifescienceai/medmcqa
- 论文：https://arxiv.org/abs/2203.14371
- **尚无 ModelScope 上的现成 MedMCQA 检索增强 JSON 归档包**；需按下述步骤自建，再整理为 `dataset/RAG_MedMCQA_{train,dev}.json` 供评测发现。

将 `train.json` / `dev.json` 置于 `dataset/pipeline_data/questions/MedMCQA/`；检索前以 `convert_medmcqa_to_medqa_jsonl.py` 转换为与 MedQA 一致的 jsonl 字段。

检索用 Embedding 位于 `dataset/pipeline/models/`（`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`）。路径由 `dataset/pipeline/config.py` 指定；若迁移存放位置，请同步更新 `model_path`。重建前请先获取：

```bash
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

（若已有包 `07` 的 FAISS，可跳过下方步骤 1–2 中对应语言的索引构建。）

### 数据集构建（自原始语料）

下列命令均在 `dataset/pipeline/` 下执行。单卡设置 `--device_ids="0"` 且 `--num_process=1`。  
检索查询键统一为 `--query_key_name="question_with_options"`（题干与选项拼接）；请勿将不同查询键的结果写入同一输出路径以免覆盖。

产物为 JSON 数组，主要字段：`id`、`question_with_options`、`answer_idx`、`retrieve_info` 等（与包 `04` 口径一致）。

#### 1. 教材文本分块为 JSON

```bash
cd dataset/pipeline

python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/zh_sentence' --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk
python -u txt2json.py --input_txt_dir '../pipeline_data/textbooks/en' --output_json_dir '../pipeline_data/textbooks/en_json' --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

#### 2. 构建向量索引

```bash
cd dataset/pipeline

python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' --store_top_path '../pipeline_data/vector_stores/zh_sentence'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' --store_top_path '../pipeline_data/vector_stores/zh_paragraph'
python -u vector_store.py --input_json_dir '../pipeline_data/textbooks/en_json' --store_top_path '../pipeline_data/vector_stores/en'
```

#### 3. 对 train / dev / test 执行检索增强

查询键统一为 `--query_key_name="question_with_options"`。输出路径：

`{output_json_dir}/retrive_use_question_with_options/{embedding_model_name}/{output_json_file}`

单卡：`--device_ids="0"` 且 `--num_process=1`。输入须为 **jsonl**（每行一条样本）。

```bash
cd dataset/pipeline

# ---------- MedQA 英文 US（默认评测子集；亦可直接使用包 04）----------
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

# ---------- MedQA 中文 Mainland（可选）----------
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

# ---------- MedMCQA：格式对齐后检索（训练 / 验证）----------
# 原始字段为 opa/opb/opc/opd + cop；须先转换为与 MedQA 一致的 jsonl schema
# 将 train.json / dev.json 置于 ../pipeline_data/questions/MedMCQA/
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

#### 4. 整理到 `dataset/`（供训练 / 评测发现）

将步骤 3 的输出 JSON 复制或软链到约定文件名，例如：

```bash
# MedQA US（若未使用包 04）
cp dataset/pipeline_data/questions_with_knowledge/US/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
   dataset/RAG_MedQA_US_train_10178.json
cp dataset/pipeline_data/questions_with_knowledge/US/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
   dataset/RAG_MedQA_US_dev_1272.json
cp dataset/pipeline_data/questions_with_knowledge/US/test/retrive_use_question_with_options/stella-base-en-v2/test.json \
   dataset/RAG_MedQA_US_test_1273.json

# MedMCQA（评测默认查找下列路径）
cp dataset/pipeline_data/questions_with_knowledge/MedMCQA/dev/retrive_use_question_with_options/stella-base-en-v2/dev.json \
   dataset/RAG_MedMCQA_dev.json
cp dataset/pipeline_data/questions_with_knowledge/MedMCQA/train/retrive_use_question_with_options/stella-base-en-v2/train.json \
   dataset/RAG_MedMCQA_train.json
```

完成后即可进入训练与评测。MedQA 亦可直接使用包 `04`，跳过步骤 1–4 中的 US 部分。

### 检查点路径（ModelScope 解压后）

执行 `PACKS="01"`（或 `PACKS="01 04"`）及 `ckpt_0` 下载后，与训练 / 评测相关的路径如下：

| 路径 | 说明 | 来源 |
|------|------|------|
| `parameters/ckpt_0/` | 系统权重；推理与训练时一体流程默认加载 | ModelScope 目录直传 |
| `parameters/PMT_Base_300M/` | Phase-modulation-transformer **预训练**权重（Hugging Face）；自行跑 `train_full_sft.py` 时的前置模型 | 包 `01` |
| `parameters/pretrain_PMT_en_1024_final.pth` | 同上，Torch 格式；`train_full_sft.py` **实际加载**此文件（可由 `PMT_Base_300M` 经 `tools/convert_model.py` 转换得到） | 包 `01` 或本地转换 |
| `parameters/sft_PMT_en_1024.pth` | 通用指令微调产物；检索增强训练的默认初始化权重 | 包 `01` 或本地 SFT |
| `out/run_pmt/cascade_PMT_en_1024_best.pth` | 检索增强训练产出的最佳 PMT 检查点；评测默认加载 | 本地训练 |

若包中已有 `sft_PMT_en_1024.pth`，可跳过通用指令对齐，直接训练 / 评测。获取顺序建议：

1. 下载系统权重 `parameters/ckpt_0/`（见上文 snapshot_download）
2. `PACKS="01 04" bash scripts/download_modelscope_assets.sh` → 得到 `sft_PMT_en_1024.pth` 与 MedQA 检索 JSON
3. （可选）`PACKS="05 06 07 08"` 用于重建检索或准备 MedMCQA

## 训练与评测

```bash
# 训练
DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
# → out/run_pmt/cascade_PMT_en_1024_best.pth

# 评测
DATASETS=USMLE,MEDMCQA bash scripts/run_cascade_infer.sh
```

结果目录：`results/result_summary` 或自动递增的 `result_summaryN/`。说明见 [results/README.md](results/README.md) 与 [results/result_summary/README.md](results/result_summary/README.md)。

## 可选：通用指令对齐（得到 `sft_PMT_en_1024.pth`）

本节**不是**主流程必需。仅在需要从 `PMT_Base` / `pretrain_*.pth` **自行**得到训练初始化权重时使用；若已下载 `parameters/sft_PMT_en_1024.pth`，请跳过。

**加载**：`parameters/pretrain_PMT_en_1024_final.pth`（对应 `PMT_Base_300M`）。若只有 HF 目录，先转换：

```bash
python tools/convert_model.py --mode transformers2torch \
  --transformers_path parameters/PMT_Base_300M \
  --torch_path parameters/pretrain_PMT_en_1024_final.pth \
  --model_type PMT --language en \
  --hidden_size 1024 --num_hidden_layers 24 \
  --num_attention_heads 8 --num_key_value_heads 2 --max_seq_len 1024

cd trainer
python train_full_sft.py --epochs 2 \
  --data_path ../dataset/openhermes2_5.jsonl \
  --out_dir ../parameters
```

## 许可

见仓库根目录 `LICENSE`。

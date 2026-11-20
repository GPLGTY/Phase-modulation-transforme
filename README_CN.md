# Phase-modulation-transformer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

English: [README_EN.md](README_EN.md)

GitHub: https://github.com/ZYXL-QB/Phase-modulation-transformer

本仓库发布 **Phase-Modulation Retrieval-Augmented Transformer** 在医学选择题问答上的实现、数据构建与参考评测结果。默认在**检索增强**设定下运行完整流程：加载一体发布的系统权重与已发布的 PMT 检查点，完成选项字母预测并报告准确率。大体量资源托管于 ModelScope（见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)）。

使用者直接下载即可使用的权重（**无需**自行预训练或通用指令微调）：

| 资源 | 路径 | 说明 |
|------|------|------|
| 系统权重 | `parameters/ckpt_0/` | ModelScope 目录直传 |
| PMT 发布权重 | `parameters/sft_PMT_en_1024.pth` | 归档包 `01`；检索增强训练 / 评测的默认初始化 |
| **最佳评测权重（推荐）** | `out/run_pmt/cascade_PMT_en_1024_best.pth` | ModelScope 归档包 `09`；检索增强训练后的最佳 PMT；推荐 `temperature=0.7`、`ntk_alpha=0.4` |
| **已完成的参考评测 / 消融结果** | **`results/result_summary/`** | **仓库内已跑通并 git 追踪的 USMLE / MedMCQA 结果**（准确率总表与逐条 CSV）；说明见 [results/result_summary/README.md](results/result_summary/README.md) |
| 新跑评测输出 | `results/result_summaryN/`（N=1,2,…） | 若 `result_summary` 已存在则自动递增目录，**不覆盖**上述参考结果；明细 CSV **不含**中间输入文本列 |

默认评测集：USMLE、MedMCQA。重新跑温度 × NTK 消融：

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
包含归档包 `01`/`04`/`05`/`09` 等，以及一体流程所需目录 `parameters/ckpt_0/`。细节见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)。

### 系统权重（一体流程必需）

默认流程始终加载 `parameters/ckpt_0/`（与 PMT 一体使用）。于**项目根目录**执行：

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

亦可在 ModelScope 页面按目录手动获取到本地 `parameters/ckpt_0/`。路由见 `model/sidecar_map.json`。

### 最佳评测权重（推荐，包 `09`）

检索增强训练后的最佳 PMT，本地路径固定为 `out/run_pmt/cascade_PMT_en_1024_best.pth`（与评测脚本默认 `CHECKPOINT` 一致）。

```bash
pip install -U modelscope
# 解压后得到 out/run_pmt/cascade_PMT_en_1024_best.pth
PACKS="09" bash scripts/download_modelscope_assets.sh
```

**推荐推理设置**：`temperature=0.7`，`ntk_alpha=0.4`（训练期 NTK 与推理期 NTK 均开启，即 `TRAIN_NTK=TRUE`、`INFER_NTK=TRUE`）。

最小评测示例（需已有 `parameters/ckpt_0/` 与评测 JSON）：

```bash
# 推荐设置：temperature=0.7，ntk_alpha=0.4
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth \
TEMPERATURES=0.7 \
NTK_ALPHAS=0.4 \
TRAIN_NTK=TRUE \
INFER_NTK=TRUE \
DATA=dataset/RAG_MedQA_US_test_1273.json \
DATASET_NAME=USMLE \
  bash scripts/run_ablation_temp_ntk.sh

# 或使用默认评测入口（CHECKPOINT 默认同上路径；sidecar 温度默认 0.7）
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth \
DATASETS=USMLE \
  bash scripts/run_cascade_infer.sh
```

若尚未下载初始化权重 / MedQA 样本，可一并获取：

```bash
PACKS="01 04 09" bash scripts/download_modelscope_assets.sh
```

### 归档包（`.tar`）批量获取

于**项目根目录**执行（脚本下载后解压至 `parameters/`、`dataset/`、`out/run_pmt/` 等约定路径）：

```bash
pip install -U modelscope

# 已发布的 PMT 权重 + MedQA 检索样本
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 本仓库最佳评测权重
PACKS="09" bash scripts/download_modelscope_assets.sh

# 检索重建：Embedding + 教材 + FAISS + 原始题目（含 MedMCQA）
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

亦可自 ModelScope 页面下载 `.tar` 后，在**项目根目录**执行 `tar -xf <包名>.tar`。

### 资源清单与本地路径

| ModelScope 文件 / 目录 | 本地路径 | 用途 |
|------------------------|----------|------|
| `parameters/ckpt_0` | `parameters/ckpt_0/` | 系统权重（检索增强一体流程） |
| `01_parameters.tar` | `parameters/sft_PMT_en_1024.pth` | 已发布的 PMT 权重（训练初始化 / 可直接评测） |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA / USMLE 检索增强样本 |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | 检索用 Stella Embedding（en/zh） |
| `06_textbooks.tar` | `dataset/pipeline_data/textbooks/` | MedQA 教材（en / zh_paragraph / zh_sentence） |
| `07_vector_stores.tar` | `dataset/pipeline_data/vector_stores/` | 语言匹配 FAISS 索引（en / zh_paragraph / zh_sentence） |
| `08_questions.tar` | `dataset/pipeline_data/questions/` | MedQA US/Mainland 与 MedMCQA 原始题目（**尚无现成 MedMCQA RAG JSON 包**） |
| `09_cascade_pmt_best.tar` | `out/run_pmt/cascade_PMT_en_1024_best.pth` | **最佳评测权重**（推荐 `temperature=0.7`、`ntk_alpha=0.4`） |

按场景选用：

| 目标 | 建议 |
|------|------|
| **直接复现本仓库最佳权重评测** | `ckpt_0` + `PACKS="09"`（+ `04` 或自备 MedMCQA RAG JSON） |
| 检索增强训练 / 从初始化权重评测（USMLE） | `ckpt_0` + `PACKS="01 04"` |
| MedMCQA 评测 | 另按数据节生成 `dataset/RAG_MedMCQA_*.json`，或自备同口径文件 |
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
├── model/                 # 模型与系统权重路由 / 提示配置
├── trainer/               # 检索增强训练
├── inference_cascade_pmt.py
├── scripts/               # 训练 / 评测启动、ModelScope 资源脚本
├── dataset/
│   ├── pipeline/          # 检索构建流水线与 Embedding
│   ├── pipeline_data/     # 教材 / FAISS（PACKS 06+07）与运行时检索产物
│   └── *.json             # 评测 / 训练样本
├── parameters/
│   ├── ckpt_0/            # 系统权重（ModelScope 目录直传）
│   └── sft_PMT_en_1024.pth # 已发布的 PMT 权重（包 01）
├── out/run_pmt/           # 检索增强训练产出的 PMT 检查点
├── results/result_summary/ # 已完成的参考评测与消融结果（git 追踪）
├── requirements.txt
└── LICENSE
```

## 数据

对外训练 / 评测 / 消融输入均为检索流水线产物 JSON（字段含 `question_with_options`、`answer_idx`，检索增强时含 `retrieve_info`）。**不**落盘其它中间样本格式。一体流程在线使用系统权重与 PMT。

### 样本清单（对应 `PACKS` 解压后）

| 用途 | 路径 | 包 |
|------|------|-----|
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
| `parameters/sft_PMT_en_1024.pth` | 已发布的 PMT 权重；检索增强训练的默认初始化 | 包 `01` |
| `out/run_pmt/cascade_PMT_en_1024_best.pth` | 最佳评测 PMT；评测默认加载；推荐 `temperature=0.7`、`ntk_alpha=0.4` | 包 `09`（亦可本地训练生成） |

获取顺序建议：

1. 下载系统权重 `parameters/ckpt_0/`（见上文 snapshot_download）
2. 若只需复现最佳评测：`PACKS="04 09" bash scripts/download_modelscope_assets.sh`
3. 若需自行训练：`PACKS="01 04" bash scripts/download_modelscope_assets.sh` → `sft_PMT_en_1024.pth` 与 MedQA 检索 JSON
4. （可选）`PACKS="05 06 07 08"` 用于重建检索或准备 MedMCQA

## 训练与评测

```bash
# 训练（默认初始化：parameters/sft_PMT_en_1024.pth）
DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
```

**训练产物默认保存位置**（可用环境变量 `OUT` / `INIT` 覆盖）：

| 路径 | 说明 |
|------|------|
| `out/run_pmt/` | 训练输出目录（默认） |
| `out/run_pmt/cascade_PMT_en_1024_best.pth` | 验证集最优检查点（评测默认加载） |
| `out/run_pmt/cascade_PMT_en_1024_epochN.pth` | 各 epoch 快照（若脚本写出） |

```bash
# 自定义输出目录 / 初始化权重
INIT=parameters/sft_PMT_en_1024.pth OUT=out/run_pmt_exp1 \
  DATA=dataset/RAG_MedQA_US_train_10178.json bash scripts/run_cascade_train.sh
```

### 评测（默认加载训练产物）

```bash
# 默认：加载 out/run_pmt/cascade_PMT_en_1024_best.pth，评测 USMLE + MedMCQA
DATASETS=USMLE,MEDMCQA bash scripts/run_cascade_infer.sh
```

结果写入 `results/result_summary`；若该目录已存在则自动使用 `result_summary1`、`result_summary2`…（不覆盖）。说明见 [results/README.md](results/README.md) 与 [results/result_summary/README.md](results/result_summary/README.md)。

### 指定某个 PMT 检查点做推理

一体流程仍会加载 `parameters/ckpt_0/`；通过 `CHECKPOINT`（或 `--checkpoint`）指定要评测的 PMT 权重。

**示例：加载 `out/run_pmt/cascade_PMT_en_1024_best.pth`（推荐 `temperature=0.7`，`ntk_alpha=0.4`）**

```bash
# 单点设置推理 / 消融（TRAINING=TRUE, Inference=TRUE）
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth \
TEMPERATURES=0.7 \
NTK_ALPHAS=0.4 \
TRAIN_NTK=TRUE \
INFER_NTK=TRUE \
DATA=dataset/RAG_MedQA_US_test_1273.json \
DATASET_NAME=USMLE \
  bash scripts/run_ablation_temp_ntk.sh
```

等价 Python 入口：

```bash
python scripts/run_ablation_temp_ntk.py \
  --checkpoint out/run_pmt/cascade_PMT_en_1024_best.pth \
  --temperatures 0.7 \
  --ntk_alphas 0.4 \
  --train_ntk TRUE \
  --infer_ntk TRUE \
  --data_path dataset/RAG_MedQA_US_test_1273.json \
  --dataset_name USMLE
```

其他常用写法：

```bash
# 默认评测入口（加载 out/run_pmt/cascade_PMT_en_1024_best.pth）
CHECKPOINT=out/run_pmt/cascade_PMT_en_1024_best.pth \
  DATASETS=USMLE bash scripts/run_cascade_infer.sh

# 使用发布的初始化权重（未做本仓库检索增强训练时也可评测）
CHECKPOINT=parameters/sft_PMT_en_1024.pth \
  DATASETS=USMLE bash scripts/run_cascade_infer.sh
```

## 许可

见仓库根目录 `LICENSE`。

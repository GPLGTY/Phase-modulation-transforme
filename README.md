# Phase-modulation-transforme

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

English: [README_EN.md](README_EN.md)

GitHub: https://github.com/GPLGTY/Phase-modulation-transforme

本仓库发布相位调制检索增强 Transformer（Phase-Modulation Retrieval-Augmented Transformer）于 MedQA 医学选择题问答任务上的完整实验协议，包括实现代码、数据构建脚本与参考评测结果。所附 QubitFormer 检查点业已经过预训练；使用者基于该初始化权重开展指令微调（SFT）与评测，无需自行预训练。体量较大的模型检查点、语料与 Embedding 资源托管于 ModelScope（见下文及 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)）。

本仓库提供三条对照评测协议（均输出选项字母并与标准答案比对、报告准确率）：

| 协议 | 含义 | 入口 |
|------|------|------|
| `no_qf` | 仅上游模块完成选项预测（不经 QubitFormer） | `--protocol no_qf` |
| `qf_norag` | 非检索增强上游 → QubitFormer | `--protocol qf_norag` |
| `qf_rag` | 检索增强上游 → QubitFormer | `--protocol qf_rag` |

训练与推理自动区分 QubitFormer 检查点：训练自 `parameters/sft_QF_en_1024.pth` 初始化并写入 `out/run_cascade_norag/` 或 `out/run_cascade_rag/`；推理默认加载对应目录下的 `*_best.pth`。评测产物记录最终选项字母与准确率相关字段。

> 训练默认随机种子为 `1337`；受 GPU 非确定性算子影响，数值结果可能存在可忽略的波动。
>
> 对外样本仅为 `generate_question_with_knowledges.py` 口径的检索/非检索 JSON。

## 环境与依赖

```bash
cd Phase-modulation-transforme
pip install -r requirements.txt
```

请安装与本机 CUDA 驱动匹配的 PyTorch。华为昇腾平台需额外安装 `torch_npu`（见 `requirements.txt` 注释）。

可选下游基准评测：

```bash
pip install lm-eval==0.4.9
```

### 主要依赖

完整列表见 [requirements.txt](./requirements.txt)。核心组件：

| 软件包 | 说明 |
|--------|------|
| python | 3.8+ |
| torch | ≥2.0，需与本机 CUDA 匹配 |
| transformers | ≥4.37 |
| datasets / wandb / tqdm | 数据加载、实验记录与进度显示 |
| modelscope | 可选；用于获取 ModelScope 托管资源 |

### 计算资源要求

对照协议按峰值顺序加载上游模块（不必三套权重同时驻留）。建议配置（bfloat16；实际上限随上下文长度、`max_new_tokens` 与后端实现而变）：

| 项目 | 建议 |
|------|------|
| GPU | **8×80 GB**（如 A100 / A800 / H100） |
| 磁盘 | ≥400 GB（含全部上游权重与归档包 `01`/`04` 等） |
| 主机内存 | ≥256 GB（权重加载 / 分片） |

补充约定：训练默认 `NPROC=1`；多卡优先用于 `device_map=auto` 或 vLLM 张量并行（`SIDECAR_TP*`）。需 NVIDIA GPU 与匹配的 CUDA / PyTorch；昇腾需 `torch_npu`（见 `requirements.txt`）。

## ModelScope 资源获取

GitHub 仓库以源代码为主；模型权重、语料与 Embedding 统一发布于 ModelScope。

**ModelScope 仓库**：[`GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`](https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering)  
包含归档包 `01`/`04`/`05` 等，以及三条对照协议所需目录 `parameters/ckpt_a`、`parameters/ckpt_b`、`parameters/ckpt_c`。细节见 [MODELSCOPE_ASSETS.md](MODELSCOPE_ASSETS.md)。

### 上游生成模块检查点（对照协议必需）

`qf_norag` 使用 `parameters/ckpt_a`，`qf_rag` 使用 `parameters/ckpt_b`，`no_qf` 使用 `parameters/ckpt_c`。于项目根目录执行：

```bash
pip install -U modelscope
REPO=GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_a/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_b/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_c/**'])"
```

亦可在 ModelScope 页面按目录手动获取。分支路由见 `model/sidecar_map.json`（由 `--protocol` 选择加载）。

另需预训练后的 QubitFormer 初始化权重（归档包 `01` 中的 `parameters/sft_QF_en_1024.pth`），以及含 `question_with_options`、`answer_idx` 的样本 JSON（可含检索相关字段）。

### 归档包（`.tar`）批量获取

于**项目根目录**执行（解压至 `parameters/`、`out/`、`dataset/` 等约定路径）：

```bash
pip install -U modelscope

# QubitFormer 初始化权重 + MedQA 检索样本
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 检索重建：Embedding + 教材 + FAISS + 原始题目（含英文）
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

亦可自 ModelScope 页面下载 `.tar` 后，在项目根目录执行 `tar -xf <包名>.tar`。

### 资源清单与本地路径

| ModelScope 文件 / 目录 | 本地路径 | 用途 |
|------------------------|----------|------|
| `parameters/ckpt_a` | `parameters/ckpt_a/` | 级联上游模块 slot 0 |
| `parameters/ckpt_b` | `parameters/ckpt_b/` | 级联上游模块 slot 1 |
| `parameters/ckpt_c` | `parameters/ckpt_c/` | `no_qf` 协议上游模块 |
| `01_parameters.tar` | `parameters/QF_*`、`parameters/MiniMind_*`、`parameters/sft_QF_en_1024.pth` | 预训练后的 QubitFormer（Hugging Face 与 SFT 初始化检查点） |
| `03b_corpus_openhermes2_5.tar` | `dataset/openhermes2_5.jsonl` | 英文通用指令微调语料 |
| `03d_corpus_sft_mini_512_zh.tar` | `dataset/sft_mini_512_zh.jsonl` | 中文指令微调语料（可选） |
| `04_medqa_rag_json.tar` | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA 检索样本（`generate_question_with_knowledges.py` 口径） |
| `05_embedding_stella.tar` | `dataset/pipeline/models/` | 检索用 Stella Embedding（en/zh） |
| `06_textbooks.tar` | `dataset/pipeline_data/textbooks/` | MedQA 教材（en / zh_paragraph / zh_sentence 及对应 json） |
| `07_vector_stores.tar` | `dataset/pipeline_data/vector_stores/` | 语言匹配 FAISS 索引（en / zh_paragraph / zh_sentence） |
| `08_questions.tar` | `dataset/pipeline_data/questions/` | MedQA US/Mainland 与 MedMCQA 原始题目 |

### 获取后的基本用法

```bash
# 一键对照基准（训练缺失检查点 + 多测试集 × 三协议 → comparison.csv）
bash scripts/run_protocol_benchmark.sh
```

## 仓库结构

```
Phase-modulation-transforme/
├── model/                 # QubitFormer 与上游模块路由 / 提示配置
├── trainer/               # 通用 SFT、级联训练
├── inference_cascade_qf.py
├── tools/                 # 权重格式转换与长度外推
├── scripts/               # 级联启动、基线评测、ModelScope 资源脚本
├── dataset/
│   ├── pipeline/          # MedQA 检索构建流水线与 Embedding
│   ├── pipeline_data/     # 教材 / FAISS（PACKS 06+07）与运行时检索产物
│   └── *.json(l)          # SFT 语料与评测样本
├── parameters/
│   ├── ckpt_a/            # 级联上游模块 slot 0
│   ├── ckpt_b/            # 级联上游模块 slot 1
│   ├── ckpt_c/            # no_qf 协议上游模块
│   └── sft_QF_en_1024.pth # 预训练后的 QubitFormer（训练初始化）
├── out/run_cascade_norag/ # qf_norag 训练检查点
├── out/run_cascade_rag/   # qf_rag 训练检查点
├── results/
│   └── protocol_benchmark/               # 对照准确率对比表
├── requirements.txt
└── LICENSE
```

## 数据

### 语料与样本清单（对应 `PACKS` 解压后）

| 阶段 | 路径 | 包 |
|------|------|-----|
| 英文通用指令微调 | `dataset/openhermes2_5.jsonl` | `03b` |
| 中文指令微调（可选） | `dataset/sft_mini_512_zh.jsonl` | `03d` |
| MedQA US 训练集 | `dataset/RAG_MedQA_US_train_10178.json`（10178） | `04` |
| MedQA US 验证集 | `dataset/RAG_MedQA_US_dev_1272.json`（1272） | `04` |
| MedQA US 测试集 | `dataset/RAG_MedQA_US_test_1273.json`（1273） | `04` |

获取方式：`PACKS="04"`，或与检查点一并 `PACKS="01 04"`。样本为检索流水线产物（可含或不含 `retrieve_info`）。公开语料备份：https://www.modelscope.cn/datasets/longview/QubitFormerDataset

若需自原始 MedQA 语料重建检索增强样本，参见下文「MedQA 数据集」「数据集构建」及 [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md)、[dataset/pipeline/README.md](dataset/pipeline/README.md)。

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

将 `train.json` / `dev.json` 置于 `dataset/pipeline_data/questions/MedMCQA/`；检索前以 `convert_medmcqa_to_medqa_jsonl.py` 转换为与 MedQA 一致的 jsonl 字段。

检索用 Embedding 位于 `dataset/pipeline/models/`（`stella-base-en-v2` / `stella-base-zh-v2` / `stella-large-zh-v2`）。路径由 `dataset/pipeline/config.py` 指定；若迁移存放位置，请同步更新 `model_path`。

### 数据集构建（自原始语料）

下列命令均在 `dataset/pipeline/` 下执行。单卡设置 `--device_ids="0"` 且 `--num_process=1`。  
检索查询键统一为 `--query_key_name="question_with_options"`（题干与选项拼接）；请勿将不同查询键的结果写入同一输出路径以免覆盖。

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

# ---------- MedQA 英文 US（默认评测子集）----------
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

完成检索增强后，进入下文「在线级联训练与推理」。亦可直接使用 ModelScope 归档包 `04` 中的样本 JSON。详见 [dataset/DATA_PIPELINE_CN.md](dataset/DATA_PIPELINE_CN.md)。

### 检查点路径（ModelScope 解压后）

执行 `PACKS="01"` / `PACKS="02"` 或全量下载后，主要路径如下：

| 路径 | 说明 | 包 |
|------|------|-----|
| `parameters/QF_Base_300M/pytorch_model.bin` | QF Base | `01` |
| `parameters/QF_Struct_300M/pytorch_model.bin` | QF Struct | `01` |
| `parameters/MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base | `01` |
| `parameters/MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct | `01` |
| `parameters/sft_QF_en_1024.pth` | 预训练后的 QubitFormer（SFT / 级联初始化） | `01` |

包 `01` 中的 QubitFormer 权重可直接作为 SFT 起点；SFT 产物亦可按下文协议自行得到。

## 通用指令微调（可选）

若已获取 `parameters/sft_QF_en_1024.pth`，可跳过本节，直接进入对照协议训练 / 评测。

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

得到 `parameters/sft_QF_en_1024.pth`。中文设置 `--language zh`，语料为 `../dataset/sft_mini_512_zh.jsonl`。

亦可自 Hugging Face 目录导出：`bash scripts/export_stage1_from_hf.sh`。

## 对照协议：训练与评测

```text
no_qf      ：独立上游 → 选项字母 → 准确率
qf_norag   ：非检索分支 + QubitFormer → 选项字母 → 准确率
qf_rag     ：检索增强分支 + QubitFormer → 选项字母 → 准确率
```

- 训练：`trainer/train_cascade_qf.py --protocol qf_norag|qf_rag`（自动加载预训练初始化权重）
- 评测：`inference_cascade_qf.py --protocol no_qf|qf_norag|qf_rag|all`（含 QubitFormer 时自动加载对应 `*_best.pth`）
- 路由：`model/sidecar_map.json`、`model/qf_protocol.py`、`model/sidecar_runtime.py`

### 前置条件

1. 获取 `parameters/ckpt_a`、`parameters/ckpt_b`、`parameters/ckpt_c`（见 ModelScope 一节）
2. 获取 QubitFormer 初始化权重：`PACKS="01" bash scripts/download_modelscope_assets.sh`
3. 准备样本 JSON（归档包 `04`，或检索增强后的题目 JSON）

### 训练（按协议分别进行）

```bash
# 默认 NPROC=1。多进程：NPROC=8 PROTOCOL=qf_norag bash scripts/run_cascade_train.sh
DATA=dataset/RAG_MedQA_US_train_10178.json \
  PROTOCOL=qf_norag bash scripts/run_cascade_train.sh

DATA=dataset/RAG_MedQA_US_train_10178.json \
  PROTOCOL=qf_rag bash scripts/run_cascade_train.sh
```

输出：

- `out/run_cascade_norag/cascade_QF_en_1024_best.pth`
- `out/run_cascade_rag/cascade_QF_en_1024_best.pth`

等价 Python：

```bash
python trainer/train_cascade_qf.py \
  --protocol qf_norag \
  --data_path dataset/RAG_MedQA_US_train_10178.json \
  --sidecar_backend transformers \
  --hidden_size 1024 --num_hidden_layers 24 \
  --num_attention_heads 16 --num_key_value_heads 4 \
  --max_seq_len 4096 --accumulation_steps 8 --epochs 1
```

> 说明：每个 DDP rank 均会加载对应上游模块；大规模部署优先单进程配合 vLLM 张量并行或 `device_map=auto`。

### 评测（三条对照）

```bash
# 一次跑完三条协议并分别报告准确率（自动选择 QubitFormer 检查点）
DATA=dataset/RAG_MedQA_US_test_1273.json \
PROTOCOL=all OUT_CSV=results/protocol_eval.csv \
  bash scripts/run_cascade_infer.sh

# 单条协议
PROTOCOL=no_qf bash scripts/run_cascade_infer.sh
PROTOCOL=qf_norag bash scripts/run_cascade_infer.sh
PROTOCOL=qf_rag bash scripts/run_cascade_infer.sh
```

结果 CSV 含 `extracted_choice`、`matches_answer_idx` 等预测字段；标准输出给出各协议准确率摘要。

### 检查点元数据清理与上传（维护用途）

```bash
python scripts/scrub_ckpt_metadata.py
# 鉴权令牌仅通过环境变量注入，勿写入版本库
set MODELSCOPE_API_TOKEN=你的token   # PowerShell: $env:MODELSCOPE_API_TOKEN=...
python scripts/upload_ckpts_to_modelscope.py
```

## 复现流程摘要

### A. 三条对照协议（一键基准）

```bash
pip install -r requirements.txt
pip install -U modelscope

# 获取 ckpt_a / ckpt_b / ckpt_c（见上文 snapshot_download）
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 一键：缺权重则先训练 qf_norag/qf_rag，再在默认测试集（US dev + US test）上评测三条协议，
# 输出准确率对比表 results/protocol_benchmark/latest/comparison.csv
bash scripts/run_protocol_benchmark.sh
# 等价：python scripts/run_protocol_benchmark.py
```

仅评测（已有 `out/run_cascade_norag` / `out/run_cascade_rag`）：

```bash
SKIP_TRAIN=1 bash scripts/run_protocol_benchmark.sh
```

自定义测试集（`name=path` 逗号分隔）：

```bash
DATASETS="medqa_us_test=dataset/RAG_MedQA_US_test_1273.json" \
  bash scripts/run_protocol_benchmark.sh
```

#### 结果在哪里、怎么看

评测结束后，主要看准确率对比表（每次运行还会写入带时间戳的目录，并同步到 `latest/`）：

| 路径 | 内容 |
|------|------|
| `results/protocol_benchmark/latest/comparison.csv` | 快捷入口：数据集 × 协议 的准确率表（Excel / 文本编辑器均可打开） |
| `results/protocol_benchmark/latest/comparison.md` | 同上，Markdown 透视表，便于直接阅读 |
| `results/protocol_benchmark/<时间戳>/` | 该次完整产物：`comparison.csv` / `comparison.md` / `summary.json` |
| `results/protocol_benchmark/<时间戳>/details/` | 逐条预测明细，如 `medqa_us_test__qf_rag.csv`（含预测字母与是否答对） |

查看方式示例：

```bash
# 总表（推荐）
cat results/protocol_benchmark/latest/comparison.md
# 或
head -n 20 results/protocol_benchmark/latest/comparison.csv

# 某协议明细
ls results/protocol_benchmark/*/details/*.csv
```

`comparison.*` 中关注列：`dataset`、`protocol`、`accuracy`（以及 `n_match` / `n_scored`）。明细 CSV 含最终选项字母等评测字段。


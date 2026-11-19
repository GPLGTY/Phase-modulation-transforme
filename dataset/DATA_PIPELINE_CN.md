# MedQA 数据构建流水线

> **工作目录**：下文命令均在 `dataset/pipeline/` 下执行（脚本与 Embedding 检查点位于该目录）。  
> **对外样本**：仅发布 / 使用 `generate_question_with_knowledges.py` 产物（带或不带检索知识）。

---

## 0. 流水线概览与成品对应关系

```
MedQA 原始题库 + 教材
        │
        ▼
 ① 教材 txt → JSON 分块
        │
        ▼
 ② JSON → 向量索引
        │
        ▼
 ③ generate_question_with_knowledges.py（题干+选项检索增强）
        │
        ▼
  dataset/RAG_MedQA_US_{train,dev,test}_*.json
        │
        ▼
  对照协议：no_qf / qf_norag / qf_rag（见根目录 README）
```

| 本仓库 / ModelScope 归档包 `04` | 用途 |
|--------------------------------|------|
| `RAG_MedQA_US_train_10178.json` | 训练（10178） |
| `RAG_MedQA_US_dev_1272.json` | 验证（1272） |
| `RAG_MedQA_US_test_1273.json` | 测试（1273） |

第 ③ 步产物可直接作为级联输入；亦可获取上表 JSON，跳过 ①–③。  
无检索对照：使用不含 `retrieve_info.knowledges`（或空列表）的同结构样本即可。

建议的中间产物目录（运行时创建，可不入库）：

```
dataset/pipeline_data/
├── textbooks/
├── questions/
├── vector_stores/
└── questions_with_knowledge/
```

---

## 1. 原始数据

- MedQA 上游发布：https://github.com/jind11/MedQA  
- 数据下载：https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing  
- 中文简介：https://zhuanlan.zhihu.com/p/679590312  

解压后将题目与教材分别置于 `dataset/pipeline_data/questions/`、`dataset/pipeline_data/textbooks/`。

---

## 2. 预处理步骤

### 2.1 教材文本分块为 JSON

```bash
cd dataset/pipeline

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/zh_paragraph' \
  --output_json_dir '../pipeline_data/textbooks/zh_paragraph_json' \
  --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/zh_sentence' \
  --output_json_dir '../pipeline_data/textbooks/zh_sentence_json' \
  --max_knowledge_len 1800 --language_type chinese --min_knowledge_len 5 --do_chunk

python -u txt2json.py \
  --input_txt_dir '../pipeline_data/textbooks/en' \
  --output_json_dir '../pipeline_data/textbooks/en_json' \
  --max_knowledge_len 1800 --language_type english --min_knowledge_len 5 --do_chunk
```

### 2.2 构建向量索引

```bash
cd dataset/pipeline

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/zh_sentence_json' \
  --store_top_path '../pipeline_data/vector_stores/zh_sentence'

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/zh_paragraph_json' \
  --store_top_path '../pipeline_data/vector_stores/zh_paragraph'

python -u vector_store.py \
  --input_json_dir '../pipeline_data/textbooks/en_json' \
  --store_top_path '../pipeline_data/vector_stores/en'
```

运行前确认 `dataset/pipeline/config.py` 中 Embedding 的 `model_path`。

### 2.3 题目检索增强（对外样本的唯一生成入口）

统一使用 `--query_key_name="question_with_options"`。

```bash
cd dataset/pipeline

python generate_question_with_knowledges.py \
  --num_process=8 --index_key_name="id" --topk_knowledge=10 --knowledge_threshold=0.65 \
  --input_jsonl_path="../pipeline_data/questions/US/train.jsonl" \
  --embedding_model_name="stella-base-en-v2" \
  --store_path="../pipeline_data/vector_stores/en" \
  --output_json_dir="../pipeline_data/questions_with_knowledge/US/train" \
  --output_json_file="train.json" --device_ids="0,1,2,3,4,5,6,7" \
  --query_key_name="question_with_options"
```

将输出整理 / 复制为仓库根下约定文件名，例如：

`dataset/RAG_MedQA_US_train_10178.json`

字段应与 `generate_question_with_knowledges.py` 一致。

### 2.4 进入训练与推理

采用根目录三条对照协议（`no_qf` / `qf_norag` / `qf_rag`）。推荐一键基准：

```bash
bash scripts/run_protocol_benchmark.sh
```

---

## 3. 环境与模型（数据处理相关）

| 用途 | 说明 |
|------|------|
| Python / PyTorch | 与仓库根目录 `requirements.txt` 对齐 |
| Embedding | `stella-base-en-v2` / `stella-base-zh-v2`（路径在 config 中配置） |
| 级联上游模块 | `parameters/ckpt_a`、`parameters/ckpt_b`、`parameters/ckpt_c`（ModelScope） |

## 4. 已纳入内容

| 类别 | 路径 |
|------|------|
| 流水线脚本 | `dataset/pipeline/*.py`、`build_vectorstore.sh` |
| MedMCQA 转换 | `dataset/pipeline/convert_medmcqa_to_medqa_jsonl.py` |
| Embedding 检查点 | `dataset/pipeline/models/` |
| 路径配置 | `dataset/pipeline/config.py` |

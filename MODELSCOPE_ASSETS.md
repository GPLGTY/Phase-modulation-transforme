# ModelScope 检查点与语料

大体积资源托管于：

**https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering**

仓库标识：`GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`

GitHub 仓库以源代码为主；检查点、语料、Embedding 与级联上游模块按下列方式获取后，本地路径与训练 / 评测脚本的默认配置一致。

## 级联上游模块（目录直传）

| 远程路径 | 本地路径 | 约大小 | 用途 |
|----------|----------|--------|------|
| `parameters/ckpt_a/` | `parameters/ckpt_a/` | ~131G | `qf_norag` 上游模块 |
| `parameters/ckpt_b/` | `parameters/ckpt_b/` | ~149G | `qf_rag` 上游模块 |
| `parameters/ckpt_c/` | `parameters/ckpt_c/` | ~65G | `no_qf` 上游模块 |

```bash
pip install -U modelscope
REPO=GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_a/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_b/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_c/**'])"
```

路由配置见 `model/sidecar_map.json`。维护上传脚本：`scripts/upload_ckpts_to_modelscope.py`（鉴权令牌经环境变量 `MODELSCOPE_API_TOKEN` 注入）。

## 归档包批量获取

于**项目根目录**执行：

```bash
pip install -U modelscope

# 常用：QubitFormer 初始化权重 + MedQA 检索样本
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 检索重建：Embedding + 教材 + FAISS + 原始题目
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

脚本将下载指定 `.tar` 并解压至约定路径。若远程仍为旧包名 `04_medqa_intent_json.tar`，下载脚本会回退拉取并自动执行 `scripts/strip_to_retrieval_json.py`，将样本规范为检索流水线字段（去除 `messages` 等中间字段，并重命名为无 `_intent` 的文件名）。

## 远程归档包一览

| 文件 | 约大小 | 解压后位置 | 用途 |
|------|--------|------------|------|
| `01_parameters.tar` | 6.4G | `parameters/` | 预训练后的 QubitFormer（HF 与 `sft_QF_en_1024.pth`） |
| `04_medqa_rag_json.tar` | — | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA 检索增强样本（`generate_question_with_knowledges.py` 口径） |
| `03b_corpus_openhermes2_5.tar` | 1.6G | `dataset/openhermes2_5.jsonl` | 英文通用指令微调（可选） |
| `03d_corpus_sft_mini_512_zh.tar` | 1.2G | `dataset/sft_mini_512_zh.jsonl` | 中文指令微调（可选） |
| `05_embedding_stella.tar` | ~450M | `dataset/pipeline/models/` | Stella Embedding（en/zh） |
| `06_textbooks.tar` | ~0.5G | `dataset/pipeline_data/textbooks/` | MedQA 教材（en / zh_paragraph / zh_sentence 及 json） |
| `07_vector_stores.tar` | ~2G | `dataset/pipeline_data/vector_stores/` | 语言匹配 FAISS（en / zh_paragraph / zh_sentence） |
| `08_questions.tar` | ~0.2G | `dataset/pipeline_data/questions/` | MedQA US/Mainland 与 MedMCQA 原始题目 |
| `SHA256SUMS.txt` / `MANIFEST.md` | 小 | — | 校验与清单 |

> 对外实验协议不包含预训练语料包与 Intent SFT 相关归档。样本 JSON **仅**保留检索脚本产物字段（如 `question_with_options`、`answer_idx`、`retrieve_info`），不含离线 Intent / `messages` 中间结果。教材与向量库按语言匹配索引发布（英文 `en/stella-base-en-v2`）。自原始题库重建检索需同时具备 `05`–`08`。

## 按实验场景选用

| 目标 | 建议 |
|------|------|
| 对照协议训练 / 一键基准 | `ckpt_a` + `ckpt_b` + `ckpt_c` + `PACKS="01 04"` |
| 通用指令微调 | 追加 `03b`（中文追加 `03d`） |
| 自原始题库重建检索流水线 | `PACKS="05 06 07 08"`（或 `05`+`06`+`08` 后本地跑 `vector_store.py`） |

## 手动获取

1. 打开上述 ModelScope 页面，下载所需目录或 `.tar`
2. 于**项目根目录**执行：`tar -xf /path/to/xxx.tar`
3. 若解压得到 `*_intent.json`，执行：`python scripts/strip_to_retrieval_json.py`

校验（可选）：

```bash
cd .modelscope_packs
sha256sum -c SHA256SUMS.txt
```

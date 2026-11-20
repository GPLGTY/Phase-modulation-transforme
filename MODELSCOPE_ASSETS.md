# ModelScope 检查点与语料

大体积资源托管于：

**https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering**

仓库标识：`zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering`

GitHub 仓库以源代码为主；检查点、语料、Embedding 与系统权重按下列方式获取后，本地路径与训练 / 评测脚本的默认配置一致。

## 系统权重（目录直传）

| 远程路径 | 本地路径 | 说明 |
|----------|----------|------|
| `parameters/ckpt_0/` | `parameters/ckpt_0/` | Phase-modulation-transformer 系统权重（检索增强流程，与 PMT 一体发布） |

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

路由配置见 `model/sidecar_map.json`。维护上传脚本：`scripts/upload_ckpts_to_modelscope.py`（`SLOTS=ckpt_0`，鉴权令牌经 `MODELSCOPE_API_TOKEN`）。

> 历史目录 `ckpt_a` / `ckpt_b` / `ckpt_c` 已从本仓库流程移除；请勿再依赖三路对照协议。

## 归档包批量获取

于**项目根目录**执行：

```bash
pip install -U modelscope

# 常用：Phase-modulation-transformer 初始化权重 + MedQA 检索样本
PACKS="01 04" bash scripts/download_modelscope_assets.sh

# 检索重建：Embedding + 教材 + FAISS + 原始题目（含 MedMCQA）
PACKS="05 06 07 08" bash scripts/download_modelscope_assets.sh
```

脚本将下载指定 `.tar` 并解压至约定路径。

## 远程归档包一览

| 文件 | 约大小 | 解压后位置 | 用途 |
|------|--------|------------|------|
| `01_parameters.tar` | 6.4G | `parameters/sft_PMT_en_1024.pth` | 已发布的 PMT 权重（训练初始化 / 可直接评测） |
| `04_medqa_rag_json.tar` | — | `dataset/RAG_MedQA_US_{train,dev,test}_*.json` | MedQA / USMLE 检索增强样本 |
| `05_embedding_stella.tar` | ~450M | `dataset/pipeline/models/` | Stella Embedding（en/zh） |
| `06_textbooks.tar` | ~0.5G | `dataset/pipeline_data/textbooks/` | MedQA 教材 |
| `07_vector_stores.tar` | ~2G | `dataset/pipeline_data/vector_stores/` | FAISS |
| `08_questions.tar` | ~0.2G | `dataset/pipeline_data/questions/` | MedQA US/Mainland 与 MedMCQA 原始题目 |
| `SHA256SUMS.txt` / `MANIFEST.md` | 小 | — | 校验与清单 |

> 样本 JSON **仅**保留检索脚本产物字段（如 `question_with_options`、`answer_idx`、`retrieve_info`）。自原始题库重建检索需同时具备 `05`–`08`。

## 按实验场景选用

| 目标 | 建议 |
|------|------|
| 检索增强训练 / 评测（USMLE） | `ckpt_0` + `PACKS="01 04"` |
| MedMCQA | 包 `08` 仅含**原始题目**；**尚无现成 MedMCQA RAG JSON 归档包**，需本地流水线生成或自备后放入 `dataset/RAG_MedMCQA_*.json` |
| 自原始题库重建检索流水线 | `PACKS="05 06 07 08"` |

## 手动获取

1. 打开上述 ModelScope 页面，下载所需目录或 `.tar`
2. 于**项目根目录**执行：`tar -xf /path/to/xxx.tar`

# 数据

大体积文件不纳入 Git。请自 ModelScope 获取并解压至本目录（或项目根下对应路径）。

仓库：https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering  

```bash
# 于项目根目录
PACKS="04" bash scripts/download_modelscope_assets.sh          # MedQA 检索样本 JSON
PACKS="03b 03d" bash scripts/download_modelscope_assets.sh     # 指令微调语料（可选）
```

详见根目录 [MODELSCOPE_ASSETS.md](../MODELSCOPE_ASSETS.md)、[README_CN.md](../README_CN.md)。

## 解压后文件清单

对外可见的 MedQA 样本即为 `generate_question_with_knowledges.py` 口径（可含或不含 `retrieve_info`）：

| 路径 | 用途 | 条数 | 包 |
|------|------|------|-----|
| `RAG_MedQA_US_train_10178.json` | MedQA US 训练 | 10178 | `04` |
| `RAG_MedQA_US_dev_1272.json` | MedQA US 验证 | 1272 | `04` |
| `RAG_MedQA_US_test_1273.json` | MedQA US 测试 | 1273 | `04` |
| `openhermes2_5.jsonl` | 英文通用指令微调 | — | `03b` |
| `sft_mini_512_zh.jsonl` | 中文指令微调（可选） | — | `03d` |
| `pipeline/models/` | Stella Embedding | — | `05` |

主要字段：`id`、`question` / `question_with_options`、`options`、`answer_idx`、`retrieve_info`（检索增强时）等。

公开语料备份：https://www.modelscope.cn/datasets/longview/QubitFormerDataset

训练数据读取类：`lm_en_dataset.py` / `lm_zh_dataset.py`（请保留）。

## 数据处理与构建

| 内容 | 路径 |
|------|------|
| 流程说明 | [DATA_PIPELINE_CN.md](./DATA_PIPELINE_CN.md) |
| 脚本与 Embedding | [pipeline/](./pipeline/)（见 [pipeline/README.md](./pipeline/README.md)） |
| 教材 / FAISS / 运行时检索产物 | `pipeline_data/`（`PACKS="06 07"` 可解压教材与向量库） |

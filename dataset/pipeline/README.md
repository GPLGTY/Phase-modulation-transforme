# 数据处理脚本与 Embedding 检查点

本目录提供 MedQA 检索增强数据构建流水线代码及检索用 Embedding 模型。

## 目录结构

```
pipeline/
├── txt2json.py / vector_store.py / custom_faiss.py / embedding.py / retriever.py
├── generate_question_with_knowledges.py
├── convert_medmcqa_to_medqa_jsonl.py   # MedMCQA → MedQA 对齐 jsonl
├── generate_ragmedqa_dataset.py
├── format_json_dataset_for_training_llm.py / *_en.py
├── format_json_dataset_for_eval_from_jsonl.py / *_v2.py
├── add_abstract.py / build_vectorstore.sh / test_code.py / config.py
└── models/
    ├── infgrad/stella-base-en-v2/
    ├── infgrad/stella-base-zh-v2/
    └── stella-large-zh-v2/
```

中间产物写入同级 `../pipeline_data/`（默认不入库）。检索流水线说明见 [../DATA_PIPELINE_CN.md](../DATA_PIPELINE_CN.md)。  

对外可见的样本**仅**来自 `generate_question_with_knowledges.py`（带或不带检索知识）。

**默认实验协议**见根目录 README 对照协议与一键基准：`bash scripts/run_protocol_benchmark.sh`。

## 配置

`config.py` 已将 Embedding 路径指向本目录下 `models/`；通常无需修改。运行脚本前请 `cd` 至本目录。

## Embedding 检查点

`models/` 下 Stella 权重请自 ModelScope 获取归档包 `05_embedding_stella.tar`（项目根目录执行 `PACKS="05" bash scripts/download_modelscope_assets.sh`）。详见 [../../MODELSCOPE_ASSETS.md](../../MODELSCOPE_ASSETS.md)。

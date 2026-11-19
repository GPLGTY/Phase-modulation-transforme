# 检查点目录

大体积文件不纳入 Git。请自 ModelScope 获取：

https://www.modelscope.cn/models/GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

## 级联上游模块

| 路径 | 说明 |
|------|------|
| `ckpt_a/` | `pmt_norag` 上游模块（`model/sidecar_map.json`） |
| `ckpt_b/` | `pmt_rag` 上游模块 |
| `ckpt_c/` | `no_pmt` 协议上游模块（不经 Phase-modulation-transforme） |

获取示例（项目根目录）：

```bash
REPO=GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_a/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_b/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_c/**'])"
```

获取后可执行 `python scripts/scrub_ckpt_metadata.py` 清理 README / 路径类元数据（维护用途）。

## Hugging Face / Phase-modulation-transforme 初始化权重（归档包 `01`）

```bash
PACKS="01" bash scripts/download_modelscope_assets.sh   # → parameters/
```

| 路径 | 说明 | 包 |
|------|------|-----|
| `sft_PMT_en_1024.pth` | 预训练后的 Phase-modulation-transforme（SFT / 级联初始化） | `01` |
| `PMT_Base_300M/pytorch_model.bin` | QF Base（Hugging Face） | `01` |
| `PMT_Struct_300M/pytorch_model.bin` | QF Struct（Hugging Face） | `01` |
| `MiniMind_Base_300M/pytorch_model.bin` | MiniMind Base（Hugging Face） | `01` |
| `MiniMind_Struct_300M/pytorch_model.bin` | MiniMind Struct（Hugging Face） | `01` |

亦可按根目录 README 中的实验协议自行 SFT。Hugging Face 目录含 config、tokenizer 与模型代码；可用 `tools/convert_model.py` 在 `.pth` 与 Hugging Face 格式间转换。

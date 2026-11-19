# 检查点目录

大体积文件不纳入 Git。请自 ModelScope 获取：

https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

## 级联上游模块

| 路径 | 说明 |
|------|------|
| `ckpt_a/` | `pmt_norag` 上游模块（`model/sidecar_map.json`） |
| `ckpt_b/` | `pmt_rag` 上游模块 |
| `ckpt_c/` | `no_pmt` 协议上游模块（不经 Phase-modulation-transformer） |

获取示例（项目根目录）：

```bash
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_a/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_b/**'])"
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_c/**'])"
```

获取后可执行 `python scripts/scrub_ckpt_metadata.py` 清理 README / 路径类元数据（维护用途）。

## Phase-modulation-transformer 初始化权重（归档包 `01`）

```bash
PACKS="01" bash scripts/download_modelscope_assets.sh   # → parameters/
```

| 路径 | 说明 | 包 |
|------|------|-----|
| `PMT_Base_300M/` | 预训练权重（HF）；自行 `train_full_sft.py` 时的前置模型 | `01` |
| `pretrain_PMT_en_1024_final.pth` | 同上 Torch 格式；`train_full_sft.py` 实际加载（可由 Base 转换） | `01` 或本地转换 |
| `sft_PMT_en_1024.pth` | 通用 SFT 产物；级联协议默认初始化 | `01` 或本地 SFT |

可用 `tools/convert_model.py` 在 `.pth` 与 Hugging Face 格式间转换。详情见根目录 README「通用指令微调」。

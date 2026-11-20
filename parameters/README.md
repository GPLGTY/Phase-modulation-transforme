# parameters/

大文件托管于 ModelScope（勿提交 Git）：

https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

## 系统权重

| 目录 | 说明 |
|------|------|
| `ckpt_0/` | Phase-modulation-transformer 系统权重（检索增强流程，与 PMT 一体使用） |

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

路由见 `model/sidecar_map.json`（`parameters/ckpt_0`，始终检索增强）。

## Phase-modulation-transformer 初始化权重（归档包 `01`）

- `sft_PMT_en_1024.pth`：检索增强训练 / 评测初始化
- `PMT_Base_300M/`、`pretrain_PMT_en_1024_final.pth`：预训练相关

训练后的 PMT 检查点默认写在 `out/run_pmt/`（不在本目录）。多组消融结果见 `results/result_summary/`。

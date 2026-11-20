# parameters/

大文件托管于 ModelScope（勿提交 Git）：

https://www.modelscope.cn/models/zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering

## 系统权重

| 目录 | 说明 |
|------|------|
| `ckpt_0/` | 系统权重（检索增强一体流程，与 PMT 一体使用） |

```bash
pip install -U modelscope
REPO=zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering
python -c "from modelscope import snapshot_download; snapshot_download('$REPO', local_dir='.', allow_patterns=['parameters/ckpt_0/**'])"
```

路由见 `model/sidecar_map.json`（`parameters/ckpt_0`，始终检索增强）。

## 已发布的 PMT 权重（归档包 `01`）

| 文件 | 说明 |
|------|------|
| `sft_PMT_en_1024.pth` | 直接下载使用的 PMT 发布权重；检索增强训练初始化，或指定 `CHECKPOINT` 评测 |

训练后的 PMT 检查点默认写在 `out/run_pmt/`（不在本目录）。多组消融结果见 `results/result_summary/`。

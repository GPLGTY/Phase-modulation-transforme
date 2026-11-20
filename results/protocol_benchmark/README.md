# 已废弃

本目录对应的三路对照协议评测（`no_pmt` / `pmt_norag` / `pmt_rag`）已从本仓库移除。

请改用：

```bash
bash scripts/run_cascade_infer.sh
# 或
python inference_cascade_pmt.py --datasets USMLE,MEDMCQA
```

输出写入 `results/result_summary` 或 `result_summaryN`。

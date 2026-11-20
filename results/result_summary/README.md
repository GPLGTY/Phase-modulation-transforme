# result_summary

本目录存放 **USMLE（MedQA）** 与 **MedMCQA** 上的参考评测与温度 × NTK 消融结果（git 追踪）。

## 对外数据口径（与发布包一致）

训练 / 评测 / 消融的输入均为检索流水线产物 JSON（ModelScope 包 `04` 等），例如：

- `dataset/RAG_MedQA_US_train_10178.json`
- `dataset/RAG_MedQA_US_dev_1272.json`
- `dataset/RAG_MedQA_US_test_1273.json`

主要字段：`id`、`question_with_options`、`answer_idx`，检索增强时含 `retrieve_info`。  
由 `generate_question_with_knowledges.py` 生成；**一体流程在线使用系统权重与 PMT**，不对外提供或落盘其它中间样本格式。

## 目录约定

| 路径 | 含义 |
|------|------|
| `aggregated_accuracy_summary.csv` | 各设定准确率总表 |
| `USMLE__TRAINING-*__Inference-*/` | USMLE 消融子表 |
| `MEDMCQA__TRAINING-*__Inference-*/` | MedMCQA 消融子表 |
| `summary.csv` / `manifest.json` / `README_对应说明.txt` | 子目录元数据 |
| `eval_temp*_ntk*.csv` | 逐条结果 |

`TRAINING` / `Inference` 标记表示所用 PMT 训练设定与本次推理设定标签（见各子目录说明），与温度、`ntk_alpha` 网格一并报告。

## 明细 CSV 列

保留：`id`、`question_with_options`、`answer_idx`、`model_raw_output`、`extracted_choice`、`matches_answer_idx` 等。  
**不包含**中间输入文本列。

## 新评测 / 消融写入规则

- 默认评测：`bash scripts/run_cascade_infer.sh` → `result_summary` 或 `result_summaryN`
- 消融：`bash scripts/run_ablation_temp_ntk.sh` → 同上根目录下的 `{DATASET}__TRAINING-*__Inference-*/`

若 `results/result_summary` 已存在，则自动使用 `result_summary1`、`result_summary2`…，**不覆盖**本目录历史结果。

## MedMCQA

ModelScope 当前仅有原题包 `08`；检索增强 JSON 需自流水线生成后再评测 / 消融。

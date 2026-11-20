# result_summary

This directory holds **USMLE (MedQA)** and **MedMCQA** reference evaluation and temperature × NTK ablation results (git-tracked).

## External data schema (aligned with release packs)

Train / eval / ablation inputs are retrieval-pipeline JSON products (ModelScope pack `04`, etc.), for example:

- `dataset/RAG_MedQA_US_train_10178.json`
- `dataset/RAG_MedQA_US_dev_1272.json`
- `dataset/RAG_MedQA_US_test_1273.json`

Main fields: `id`, `question_with_options`, `answer_idx`, plus `retrieve_info` when retrieval-augmented.  
Produced by `generate_question_with_knowledges.py`. The **integrated pipeline uses system weights and PMT online**; other intermediate sample formats are not released or written to disk.

## Layout

| Path | Meaning |
|------|---------|
| `aggregated_accuracy_summary.csv` | Accuracy summary across settings (columns: `dataset`, train/inference ntk flags, `source_dir`, `copied_dir`, `temperature`, `ntk_alpha`, `n_total`, `n_scored`, `n_match`, `accuracy`, `elapsed_sec`, `output_csv`) |
| `USMLE__TRAINING-*__Inference-*/` | USMLE ablation tables |
| `MEDMCQA__TRAINING-*__Inference-*/` | MedMCQA ablation tables |
| `summary.csv` / `manifest.json` / `README_对应说明.txt` | Per-subdirectory metadata |
| `eval_temp*_ntk*.csv` | Per-sample results |

The `TRAINING` / `Inference` labels mark the PMT training setting and this run’s inference setting (see each subdirectory), reported together with the temperature × `ntk_alpha` grid.

## Detail CSV columns

Kept: `id`, `question_with_options`, `answer_idx`, `model_raw_output`, `extracted_choice`, `matches_answer_idx`, etc.  
Intermediate input-text columns are **not** included.

## Writing new eval / ablation runs

- Default eval: `bash scripts/run_cascade_infer.sh` → `result_summary` or `result_summaryN`
- Ablation: `bash scripts/run_ablation_temp_ntk.sh` → `{DATASET}__TRAINING-*__Inference-*/` under the same root

If `results/result_summary` already exists, new runs use `result_summary1`, `result_summary2`, … and **do not** overwrite this reference directory.

## MedMCQA

ModelScope currently provides raw questions only (pack `08`). Build retrieval-augmented JSON via the pipeline before eval / ablation.

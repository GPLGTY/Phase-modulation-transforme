# results/

## `result_summary/` (reference results, finished evals, git-tracked)

This directory holds **completed** USMLE (MedQA) and MedMCQA reference evaluations and temperature × NTK ablation results. You can inspect the accuracy summary and per-sample CSVs directly. See [result_summary/README.md](result_summary/README.md).

For new local eval / ablation runs: if `result_summary` already exists, outputs go to `result_summary1`, `result_summary2`, … so this directory is not overwritten.

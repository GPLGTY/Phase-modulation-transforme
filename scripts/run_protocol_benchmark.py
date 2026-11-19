#!/usr/bin/env python3
"""One-click protocol × dataset accuracy benchmark.

Runs contrastive protocols (no_pmt / pmt_norag / pmt_rag) on one or more test sets,
optionally trains missing PhaseModulationTransformer checkpoints first, then writes a comparison
table (CSV + Markdown).

Example:

  python scripts/run_protocol_benchmark.py
  python scripts/run_protocol_benchmark.py --skip_train --limit 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pmt_protocol import (  # noqa: E402
    ALL_PROTOCOLS,
    PROTOCOL_PMT_NORAG,
    PROTOCOL_PMT_RAG,
    default_out_dir,
    protocol_needs_pmt,
    resolve_pmt_infer_ckpt,
)
from inference_cascade_pmt import run_protocol  # noqa: E402

DEFAULT_DATASETS = {
    "medqa_us_dev": ROOT / "dataset" / "RAG_MedQA_US_dev_1272.json",
    "medqa_us_test": ROOT / "dataset" / "RAG_MedQA_US_test_1273.json",
}


def _parse_datasets(spec: str) -> list[tuple[str, Path]]:
    """name=path,name=path  or bare paths (stem used as name)."""
    items: list[tuple[str, Path]] = []
    if not spec.strip():
        for name, path in DEFAULT_DATASETS.items():
            items.append((name, path))
        return items
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, raw = part.split("=", 1)
            items.append((name.strip(), Path(raw.strip())))
        else:
            p = Path(part)
            items.append((p.stem, p))
    return items


def _ckpt_ready(protocol: str, hidden_size: int) -> bool:
    try:
        resolve_pmt_infer_ckpt(protocol, hidden_size=hidden_size)
        return True
    except FileNotFoundError:
        return False


def _train_protocol(protocol: str, train_data: Path, args) -> None:
    out_dir = default_out_dir(protocol)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        str(ROOT / "trainer" / "train_cascade_pmt.py"),
        "--protocol",
        protocol,
        "--data_path",
        str(train_data),
        "--out_dir",
        str(out_dir),
        "--sidecar_backend",
        args.sidecar_backend,
        "--sidecar_tp0",
        str(args.sidecar_tp0),
        "--sidecar_tp1",
        str(args.sidecar_tp1),
        "--sidecar_mem0",
        str(args.sidecar_mem0),
        "--sidecar_mem1",
        str(args.sidecar_mem1),
        "--batch_size",
        "1",
        "--accumulation_steps",
        "8",
        "--epochs",
        "1",
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_key_value_heads",
        str(args.num_key_value_heads),
        "--max_seq_len",
        str(args.max_seq_len),
        "--device",
        args.device,
    ]
    print(f"[benchmark] train protocol={protocol} → {out_dir}", flush=True)
    if args.nproc > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc}",
            *common,
            "--ddp",
        ]
    else:
        cmd = [sys.executable, *common]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _write_comparison(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    md_path = out_dir / "comparison.md"
    fields = [
        "dataset",
        "protocol",
        "n_scored",
        "n_match",
        "n_skip",
        "accuracy",
        "qf_checkpoint",
        "detail_csv",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "dataset": r.get("dataset", ""),
                    "protocol": r.get("protocol", ""),
                    "n_scored": r.get("n_scored", 0),
                    "n_match": r.get("n_match", 0),
                    "n_skip": r.get("n_skip", 0),
                    "accuracy": r.get("accuracy", 0),
                    "qf_checkpoint": r.get("qf_checkpoint", ""),
                    "detail_csv": r.get("output_csv", ""),
                }
            )

    # Pivot: rows=dataset, cols=protocol
    datasets = []
    for r in rows:
        d = r.get("dataset", "")
        if d not in datasets:
            datasets.append(d)
    protocols = [p for p in ALL_PROTOCOLS if any(r.get("protocol") == p for r in rows)]
    cell = {(r["dataset"], r["protocol"]): r for r in rows}

    lines = [
        "# Protocol × Dataset Accuracy",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| dataset | " + " | ".join(protocols) + " |",
        "|---|" + "|".join(["---"] * len(protocols)) + "|",
    ]
    for d in datasets:
        cells = []
        for p in protocols:
            r = cell.get((d, p))
            if not r:
                cells.append("—")
            else:
                cells.append(f"{r['accuracy']:.4f} ({r['n_match']}/{r['n_scored']})")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Long form:")
    lines.append("")
    lines.append("| dataset | protocol | n_scored | n_match | accuracy |")
    lines.append("|---|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r.get('dataset','')} | {r.get('protocol','')} | "
            f"{r.get('n_scored',0)} | {r.get('n_match',0)} | {r.get('accuracy',0):.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def main():
    p = argparse.ArgumentParser(description="One-click protocol × dataset benchmark")
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma list: name=path,...  Default: MedQA US dev + test",
    )
    p.add_argument(
        "--protocols",
        type=str,
        default="no_pmt,pmt_norag,pmt_rag",
        help="Comma list of protocols",
    )
    p.add_argument(
        "--train_data",
        type=str,
        default=str(ROOT / "dataset" / "RAG_MedQA_US_train_10178.json"),
    )
    p.add_argument("--skip_train", action="store_true", help="Do not train; require existing QF ckpts")
    p.add_argument(
        "--force_train",
        action="store_true",
        help="Always retrain pmt_norag / pmt_rag before eval",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "results" / "protocol_benchmark"),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--nproc", type=int, default=1, help="train NPROC for run_cascade_train.sh")
    p.add_argument("--limit", type=int, default=0, help="debug: first N samples per dataset")
    p.add_argument("--sidecar_backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--sidecar_map", type=str, default=str(ROOT / "model" / "sidecar_map.json"))
    p.add_argument("--sidecar_dtype", default="bfloat16")
    p.add_argument("--sidecar_max_tokens", type=int, default=4096)
    p.add_argument("--sidecar_temperature", type=float, default=0.7)
    p.add_argument("--sidecar_max_model_len", type=int, default=8192)
    p.add_argument("--sidecar_tp0", type=int, default=1)
    p.add_argument("--sidecar_tp1", type=int, default=1)
    p.add_argument("--sidecar_tp2", type=int, default=1)
    p.add_argument("--sidecar_mem0", type=float, default=0.45)
    p.add_argument("--sidecar_mem1", type=float, default=0.45)
    p.add_argument("--sidecar_mem2", type=float, default=0.85)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_hidden_layers", type=int, default=24)
    p.add_argument("--num_attention_heads", type=int, default=16)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="", help="optional QF override (discouraged)")
    p.add_argument("--out_dir", type=str, default="")
    args = p.parse_args()

    datasets = _parse_datasets(args.datasets)
    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    for proto in protocols:
        if proto not in ALL_PROTOCOLS:
            raise SystemExit(f"unknown protocol: {proto}")

    missing_data = [f"{n}: {p}" for n, p in datasets if not p.is_file()]
    if missing_data:
        raise SystemExit(
            "missing dataset files:\n  "
            + "\n  ".join(missing_data)
            + '\nDownload with: PACKS="04" bash scripts/download_modelscope_assets.sh'
        )

    # Train PhaseModulationTransformer protocols if needed
    train_data = Path(args.train_data)
    for proto in protocols:
        if not protocol_needs_pmt(proto):
            continue
        need = args.force_train or not _ckpt_ready(proto, args.hidden_size)
        if need and args.skip_train:
            raise SystemExit(
                f"missing trained PhaseModulationTransformer for {proto} under {default_out_dir(proto)} "
                "(run without --skip_train, or train first)"
            )
        if need:
            if not train_data.is_file():
                raise SystemExit(f"train data missing: {train_data}")
            _train_protocol(proto, train_data, args)

    results_dir = Path(args.results_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = run_dir / "details"
    detail_dir.mkdir(exist_ok=True)

    # Namespace compatible with inference_cascade_pmt.run_protocol
    class _Args:
        pass

    eval_args = _Args()
    for k, v in vars(args).items():
        setattr(eval_args, k, v)
    eval_args.protocol = "all"  # unused when output_csv override is set
    eval_args.output_csv = str(detail_dir / "unused.csv")

    rows: list[dict] = []
    for ds_name, ds_path in datasets:
        data = json.loads(ds_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"dataset must be JSON list: {ds_path}")
        if args.limit > 0:
            data = data[: args.limit]
        print(f"[benchmark] dataset={ds_name} n={len(data)} path={ds_path}", flush=True)
        for proto in protocols:
            out_csv = detail_dir / f"{ds_name}__{proto}.csv"
            summary = run_protocol(
                proto,
                data,
                eval_args,
                output_csv=out_csv,
                dataset_name=ds_name,
            )
            rows.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    csv_path, md_path = _write_comparison(rows, run_dir)
    # Also refresh a stable "latest" pointer for convenience
    latest_dir = results_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for src, name in ((csv_path, "comparison.csv"), (md_path, "comparison.md")):
        dest = latest_dir / name
        dest.write_bytes(src.read_bytes())

    meta = {
        "run_dir": str(run_dir),
        "comparison_csv": str(csv_path),
        "comparison_md": str(md_path),
        "latest_csv": str(latest_dir / "comparison.csv"),
        "latest_md": str(latest_dir / "comparison.md"),
        "datasets": [{ "name": n, "path": str(p)} for n, p in datasets],
        "protocols": protocols,
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"comparison_csv": str(csv_path), "comparison_md": str(md_path), "latest": str(latest_dir)}, indent=2), flush=True)
    print(f"\n对比表: {csv_path}\nMarkdown: {md_path}\n快捷入口: {latest_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()

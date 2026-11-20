#!/usr/bin/env python3
"""Normalize MedQA JSON to the retrieval-pipeline schema.

Keeps fields produced by ``generate_question_with_knowledges.py``; drops
``messages`` and other offline intermediates. Operates in-place on canonical
``RAG_MedQA_US_*.json`` files under ``dataset/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP_KEYS = (
    "id",
    "question",
    "options",
    "answer",
    "answer_idx",
    "meta_info",
    "question_with_options",
    "answer_with_idx",
    "retrieve_info",
)

CANONICAL = (
    "RAG_MedQA_US_train_10178.json",
    "RAG_MedQA_US_dev_1272.json",
    "RAG_MedQA_US_test_1273.json",
)


def strip_sample(sample: dict) -> dict:
    out = {k: sample[k] for k in KEEP_KEYS if k in sample}
    if "answer_idx" not in out and sample.get("gold_letter"):
        out["answer_idx"] = sample["gold_letter"]
    if "question_with_options" not in out and sample.get("question"):
        q = str(sample.get("question") or "")
        opts = sample.get("options") or {}
        if isinstance(opts, dict) and opts:
            lines = [q] + [f"{k}. {v}" for k, v in opts.items()]
            out["question_with_options"] = "\n".join(lines)
    return out


def convert_file(src: Path, dst: Path) -> int:
    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"expect JSON list: {src}")
    cleaned = [strip_sample(x) for x in raw if isinstance(x, dict)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(cleaned)


def main():
    p = argparse.ArgumentParser(description="Strip offline intermediates → retrieval JSON")
    p.add_argument(
        "--dataset_dir",
        type=str,
        default=str(ROOT / "dataset"),
        help="directory containing MedQA JSON samples",
    )
    p.add_argument("--input", type=str, default="", help="optional single input JSON")
    p.add_argument("--output", type=str, default="", help="optional single output JSON")
    args = p.parse_args()

    if args.input:
        src = Path(args.input)
        dst = Path(args.output) if args.output else src
        n = convert_file(src, dst)
        print(f"wrote {n} samples → {dst}", flush=True)
        return

    data_dir = Path(args.dataset_dir)
    for name in CANONICAL:
        path = data_dir / name
        if not path.is_file():
            print(f"skip missing: {name}", flush=True)
            continue
        n = convert_file(path, path)
        print(f"normalized {name} ({n})", flush=True)


if __name__ == "__main__":
    main()

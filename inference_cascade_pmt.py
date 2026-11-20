#!/usr/bin/env python3
"""RAG evaluation for the Phase-modulation-transformer stack (ckpt_0 + PMT).

Does not persist cascade intermediates (no assistant_content / model_input_prompt).
Default multi-dataset runs write under results/result_summary or result_summaryN.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import LlamaTokenizerFast
from tqdm import tqdm

from inference import format_inst_prompt_sft, merge_system_and_user_body
from model.PMTConfig import PMTConfig
from model.pmtmodel import PhaseModulationTransformer
from model.pmt_protocol import (
    PROTOCOL_PMT_RAG,
    allocate_result_summary_dir,
    prepare_sample_for_protocol,
    protocol_slot,
    resolve_pmt_infer_ckpt,
)
from model.sidecar_runtime import SidecarPool, gold_letter, load_map

ROOT = Path(__file__).resolve().parent

# Default eval sets: USMLE (MedQA) + MedMCQA (first existing path wins)
DEFAULT_DATASETS = {
    "USMLE": [
        ROOT / "dataset" / "RAG_MedQA_US_test_1273.json",
        ROOT / "dataset" / "RAG_MedQA_US_dev_1272.json",
    ],
    "MEDMCQA": [
        ROOT / "dataset" / "RAG_MedMCQA_dev.json",
        ROOT / "dataset" / "RAG_MedMCQA_test.json",
        Path(r"G:\mycodes\data_clean_gp\questions\MedMCQA\dev.json"),
    ],
}


def extract_letter(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    ch = t[0].upper()
    return ch if ch in "ABCDE" else ""


def _load_pmt(checkpoint: Path, args, device: torch.device):
    tok = LlamaTokenizerFast.from_pretrained(str(ROOT / "model" / "tokenizer_en"))
    cfg = PMTConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
    )
    model = PhaseModulationTransformer(cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if args.dtype != "float32":
        model = model.to(dtype=dtype)
    return model, tok


def _build_sidecar(args, route_cfg: dict) -> SidecarPool:
    slot = protocol_slot(PROTOCOL_PMT_RAG, route_cfg)
    return SidecarPool(
        map_path=args.sidecar_map,
        backend=args.sidecar_backend,
        dtype=args.sidecar_dtype,
        max_model_len=args.sidecar_max_model_len,
        tp_by_slot={slot: args.sidecar_tp},
        mem_util_by_slot={slot: args.sidecar_mem},
        load_slots=[slot],
    )


def run_rag_eval(
    data: list,
    args,
    *,
    output_csv: Path | str,
    dataset_name: str = "",
) -> dict:
    route_cfg = load_map(args.sidecar_map)
    sidecar = _build_sidecar(args, route_cfg)
    force_slot = protocol_slot(PROTOCOL_PMT_RAG, route_cfg)

    device = torch.device(args.device)
    ckpt_used = resolve_pmt_infer_ckpt(
        PROTOCOL_PMT_RAG,
        explicit=args.checkpoint or None,
        out_dir=args.out_dir or None,
        hidden_size=args.hidden_size,
    )
    model, tok = _load_pmt(ckpt_used, args, device)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id",
        "dataset",
        "question_with_options",
        "answer_idx",
        "extracted_choice",
        "matches_answer_idx",
    ]
    n_ok = 0
    n_scored = 0
    n_skip = 0

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        bar = dataset_name or "rag"
        for sample in tqdm(data, desc=bar):
            view = prepare_sample_for_protocol(sample, PROTOCOL_PMT_RAG, route_cfg)
            if view is None:
                n_skip += 1
                continue
            gold = gold_letter(sample)
            _slot, ctx, system = sidecar.generate(
                view,
                temperature=args.sidecar_temperature,
                max_tokens=args.sidecar_max_tokens,
                force_slot=force_slot,
            )
            merged = merge_system_and_user_body(system, ctx)
            del ctx
            prompt = format_inst_prompt_sft(merged)
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) > args.max_seq_len - args.max_new_tokens:
                ids = ids[-(args.max_seq_len - args.max_new_tokens) :]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                gen = model.generate(
                    x,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            new_ids = gen[0, x.shape[-1] :]
            raw = tok.decode(new_ids, skip_special_tokens=True)
            pred = extract_letter(raw)

            match = bool(pred and gold and pred == gold)
            if gold in "ABCDE":
                n_scored += 1
                n_ok += int(match)
            writer.writerow(
                {
                    "id": sample.get("id", ""),
                    "dataset": dataset_name,
                    "question_with_options": sample.get("question_with_options", ""),
                    "answer_idx": gold,
                    "extracted_choice": pred,
                    "matches_answer_idx": match,
                }
            )

    del sidecar
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = n_ok / max(n_scored, 1)
    return {
        "dataset": dataset_name,
        "output_csv": str(out_path),
        "n_scored": n_scored,
        "n_match": n_ok,
        "n_skip": n_skip,
        "accuracy": round(acc, 4),
        "pmt_checkpoint": str(ckpt_used),
    }


# Backward-compatible alias
def run_protocol(protocol: str, data: list, args, **kwargs) -> dict:
    if protocol != PROTOCOL_PMT_RAG:
        raise ValueError(f"only {PROTOCOL_PMT_RAG} is supported")
    return run_rag_eval(data, args, **kwargs)


def main():
    p = argparse.ArgumentParser(description="RAG cascade accuracy evaluation")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="optional override; default: out/run_pmt/cascade_PMT_en_*_best.pth",
    )
    p.add_argument("--out_dir", type=str, default="", help="PMT checkpoint directory override")
    p.add_argument(
        "--data_path",
        type=str,
        default="",
        help="single JSON list; if empty and --datasets used, ignore",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="USMLE,MEDMCQA",
        help="comma names from built-in map when --data_path empty",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="single-run CSV; default allocates under results/result_summary[N]/",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="explicit result_summary root; default auto-allocate",
    )
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_hidden_layers", type=int, default=24)
    p.add_argument("--num_attention_heads", type=int, default=16)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sidecar_map", type=str, default=str(ROOT / "model" / "sidecar_map.json"))
    p.add_argument("--sidecar_backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--sidecar_dtype", default="bfloat16")
    p.add_argument("--sidecar_max_tokens", type=int, default=4096)
    p.add_argument("--sidecar_temperature", type=float, default=0.7)
    p.add_argument("--sidecar_max_model_len", type=int, default=8192)
    p.add_argument("--sidecar_tp", type=int, default=1)
    p.add_argument("--sidecar_mem", type=float, default=0.85)
    # deprecated aliases kept so old shells do not crash
    p.add_argument("--protocol", type=str, default=PROTOCOL_PMT_RAG, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_tp0", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_tp1", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_tp2", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_mem0", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_mem1", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--sidecar_mem2", type=float, default=0.0, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.sidecar_tp1:
        args.sidecar_tp = args.sidecar_tp1
    elif args.sidecar_tp0:
        args.sidecar_tp = args.sidecar_tp0
    if args.sidecar_mem1:
        args.sidecar_mem = args.sidecar_mem1
    elif args.sidecar_mem0:
        args.sidecar_mem = args.sidecar_mem0
    elif args.sidecar_mem2:
        args.sidecar_mem = args.sidecar_mem2

    result_root = Path(args.result_dir) if args.result_dir else allocate_result_summary_dir()
    result_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    if args.data_path:
        jobs.append(("custom", Path(args.data_path)))
    else:
        for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            key = name.upper()
            cands = DEFAULT_DATASETS.get(key)
            if cands is None:
                raise SystemExit(f"unknown dataset {name}; known={list(DEFAULT_DATASETS)}")
            path = next((p for p in cands if p.is_file()), None)
            if path is None:
                print(f"[warn] missing dataset files for {key}, skip: {cands}", flush=True)
                continue
            jobs.append((key, path))

    if not jobs:
        raise SystemExit("no datasets to evaluate")

    summaries = []
    for ds_name, data_path in jobs:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"data must be JSON list: {data_path}")
        if args.limit > 0:
            data = data[: args.limit]
        if args.output_csv and len(jobs) == 1:
            out_csv = Path(args.output_csv)
        else:
            out_csv = result_root / f"{ds_name}_rag_eval.csv"
        summaries.append(run_rag_eval(data, args, output_csv=out_csv, dataset_name=ds_name))

    summary_path = result_root / "summary.json"
    summary_path.write_text(
        json.dumps({"results": summaries, "result_dir": str(result_root)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"results": summaries, "result_dir": str(result_root)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

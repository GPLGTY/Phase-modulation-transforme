#!/usr/bin/env python3
"""Protocol evaluation: option-letter accuracy only (no upstream intermediates).

Protocols:
  no_pmt     — standalone upstream (no PhaseModulationTransforme)
  pmt_norag  — PhaseModulationTransforme + non-retrieval cascade branch
  pmt_rag    — PhaseModulationTransforme + retrieval-augmented cascade branch
  all       — run the three protocols and report each accuracy

PhaseModulationTransforme checkpoints are resolved automatically:
  train writes out/run_cascade_{norag|rag}/; infer loads *_best.pth from there.
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
from model.pmtmodel import PhaseModulationTransforme
from model.pmt_protocol import (
    ALL_PROTOCOLS,
    PROTOCOL_NO_PMT,
    PROTOCOL_PMT_NORAG,
    PROTOCOL_PMT_RAG,
    prepare_sample_for_protocol,
    protocol_needs_pmt,
    protocol_slot,
    resolve_pmt_infer_ckpt,
)
from model.sidecar_runtime import SidecarPool, gold_letter, load_map

ROOT = Path(__file__).resolve().parent


def extract_letter(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    ch = t[0].upper()
    return ch if ch in "ABCDE" else ""


def _load_pmt(checkpoint: Path, args, device: torch.device) -> tuple[PhaseModulationTransforme, LlamaTokenizerFast]:
    tok = LlamaTokenizerFast.from_pretrained(str(ROOT / "model" / "tokenizer_en"))
    cfg = PMTConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
    )
    model = PhaseModulationTransforme(cfg)
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


def _build_sidecar(protocol: str, args, route_cfg: dict) -> SidecarPool:
    slot = protocol_slot(protocol, route_cfg)
    if protocol == PROTOCOL_NO_PMT:
        tp = {slot: args.sidecar_tp2}
        mem = {slot: args.sidecar_mem2}
    elif protocol == PROTOCOL_PMT_NORAG:
        tp = {slot: args.sidecar_tp0}
        mem = {slot: args.sidecar_mem0}
    else:
        tp = {slot: args.sidecar_tp1}
        mem = {slot: args.sidecar_mem1}
    return SidecarPool(
        map_path=args.sidecar_map,
        backend=args.sidecar_backend,
        dtype=args.sidecar_dtype,
        max_model_len=args.sidecar_max_model_len,
        tp_by_slot=tp,
        mem_util_by_slot=mem,
        load_slots=[slot],
    )


def run_protocol(
    protocol: str,
    data: list,
    args,
    *,
    output_csv: Path | str | None = None,
    dataset_name: str = "",
) -> dict:
    route_cfg = load_map(args.sidecar_map)
    sidecar = _build_sidecar(protocol, args, route_cfg)
    force_slot = protocol_slot(protocol, route_cfg)

    model = None
    tok = None
    device = torch.device(args.device)
    ckpt_used = None
    if protocol_needs_pmt(protocol):
        ckpt_used = resolve_pmt_infer_ckpt(
            protocol,
            explicit=args.checkpoint or None,
            out_dir=args.out_dir or None,
            hidden_size=args.hidden_size,
        )
        model, tok = _load_pmt(ckpt_used, args, device)

    if output_csv is not None:
        out_path = Path(output_csv)
    else:
        out_path = Path(args.output_csv)
        if getattr(args, "protocol", "") == "all":
            out_path = out_path.with_name(f"{out_path.stem}_{protocol}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluation artifacts: final letter only — no upstream intermediates
    fields = [
        "id",
        "dataset",
        "protocol",
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
        bar = f"{dataset_name}:{protocol}" if dataset_name else protocol
        for sample in tqdm(data, desc=bar):
            view = prepare_sample_for_protocol(sample, protocol, route_cfg)
            if view is None:
                n_skip += 1
                continue
            gold = gold_letter(sample)
            if protocol == PROTOCOL_NO_PMT:
                _slot, _raw, pred = sidecar.generate_letter(
                    view,
                    temperature=min(args.sidecar_temperature, 0.2),
                    max_tokens=min(args.sidecar_max_tokens, 16),
                )
                del _raw  # do not persist upstream text
            else:
                _slot, ctx, system = sidecar.generate(
                    view,
                    temperature=args.sidecar_temperature,
                    max_tokens=args.sidecar_max_tokens,
                    force_slot=force_slot,
                )
                merged = merge_system_and_user_body(system, ctx)
                del ctx  # drop intermediate ASAP
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
                    "protocol": protocol,
                    "question_with_options": sample.get("question_with_options", ""),
                    "answer_idx": gold,
                    "extracted_choice": pred,
                    "matches_answer_idx": match,
                }
            )

    # Free GPU memory before next protocol in --protocol all
    del sidecar
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    acc = n_ok / max(n_scored, 1)
    summary = {
        "dataset": dataset_name,
        "protocol": protocol,
        "output_csv": str(out_path),
        "n_scored": n_scored,
        "n_match": n_ok,
        "n_skip": n_skip,
        "accuracy": round(acc, 4),
    }
    if ckpt_used is not None:
        summary["qf_checkpoint"] = str(ckpt_used)
    return summary


def main():
    p = argparse.ArgumentParser(description="Protocol accuracy evaluation")
    p.add_argument(
        "--protocol",
        choices=(*ALL_PROTOCOLS, "all"),
        default="all",
        help="no_pmt | pmt_norag | pmt_rag | all",
    )
    p.add_argument(
        "--mode",
        choices=("cascade", "sidecar_only"),
        default="",
        help="deprecated alias: cascade→pmt_norag+pmt_rag via all; sidecar_only→no_pmt",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="optional override; by default infer auto-loads protocol-trained PhaseModulationTransforme",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="optional override of protocol PhaseModulationTransforme directory",
    )
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_csv", type=str, default=str(ROOT / "results" / "protocol_eval.csv"))
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
    p.add_argument("--sidecar_tp0", type=int, default=1)
    p.add_argument("--sidecar_tp1", type=int, default=1)
    p.add_argument("--sidecar_tp2", type=int, default=1)
    p.add_argument("--sidecar_mem0", type=float, default=0.45)
    p.add_argument("--sidecar_mem1", type=float, default=0.45)
    p.add_argument("--sidecar_mem2", type=float, default=0.85)
    args = p.parse_args()

    # Backward-compatible --mode mapping
    if args.mode and args.protocol == "all":
        if args.mode == "sidecar_only":
            args.protocol = PROTOCOL_NO_PMT
        elif args.mode == "cascade":
            # Prefer explicit single protocol via --protocol; cascade alone is ambiguous
            args.protocol = "all"

    data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("data_path must be a JSON list")
    if args.limit > 0:
        data = data[: args.limit]

    protocols = list(ALL_PROTOCOLS) if args.protocol == "all" else [args.protocol]
    summaries = []
    for proto in protocols:
        summaries.append(run_protocol(proto, data, args))

    print(json.dumps({"results": summaries}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

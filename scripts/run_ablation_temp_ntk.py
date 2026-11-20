#!/usr/bin/env python3
"""temperature × ntk_alpha 推理消融（对外数据口径与 v5 一致）。

输入：检索增强样本 JSON（包 04 等），字段含 question_with_options、answer_idx，
可含 retrieve_info。运行时由一体流程在线生成上下文并完成选项预测；
不读写中间样本格式，也不落盘中间文本。

输出：写入 results/result_summary 或 result_summaryN 下的
  {DATASET}__TRAINING-{T|F}__Inference-{T|F}/
结构对齐现有 result_summary（eval_temp*_ntk*.csv + summary.csv + manifest.json）。

用法（项目根目录）：

  python scripts/run_ablation_temp_ntk.py --limit 8
  DATASET_NAME=USMLE python scripts/run_ablation_temp_ntk.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import LlamaTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import format_inst_prompt_sft, merge_system_and_user_body
from model.PMTConfig import PMTConfig
from model.pmt_protocol import (
    PROTOCOL_PMT_RAG,
    allocate_result_summary_dir,
    prepare_sample_for_protocol,
    protocol_slot,
    resolve_pmt_infer_ckpt,
)
from model.pmtmodel import PhaseModulationTransformer
from model.sidecar_runtime import SidecarPool, gold_letter, load_map

CSV_FIELDS = [
    "id",
    "meta_info",
    "question_with_options",
    "answer_with_idx",
    "answer_idx",
    "model_raw_output",
    "extracted_choice",
    "matches_answer_idx",
    "error",
]

DEFAULT_TEMPERATURES = [0.7, 0.9, 1.1, 1.3, 1.5]
DEFAULT_NTK_ALPHAS = [0.4, 0.6, 0.8, 1.0, 1.2]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _param_tag(name: str, value: float) -> str:
    return f"{name}{str(value).replace('.', 'p')}"


def _bool_tag(v: bool) -> str:
    return "TRUE" if v else "FALSE"


def extract_letter(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    ch = t[0].upper()
    return ch if ch in "ABCDE" else ""


def apply_ablation_settings(
    model: PhaseModulationTransformer,
    temperature: float,
    ntk_alpha: float,
    ntk_max_train_len: int,
) -> None:
    model.tok_embeddings.temperature = float(temperature)
    model.params.ntk_alpha = float(ntk_alpha)
    model.dynamic_ntk.alpha = float(ntk_alpha)
    model.dynamic_ntk.max_train_len = int(ntk_max_train_len)
    for layer in model.layers:
        layer.attn.dynamic_ntk.alpha = float(ntk_alpha)
        layer.attn.dynamic_ntk.max_train_len = int(ntk_max_train_len)


def load_pmt(args, device: torch.device):
    tok = LlamaTokenizerFast.from_pretrained(str(ROOT / "model" / "tokenizer_en"))
    tok.pad_token = tok.eos_token
    cfg = PMTConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
    )
    model = PhaseModulationTransformer(cfg)
    ckpt = resolve_pmt_infer_ckpt(
        PROTOCOL_PMT_RAG,
        explicit=args.checkpoint or None,
        out_dir=args.out_dir or None,
        hidden_size=args.hidden_size,
    )
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    if args.dtype != "float32":
        model = model.to(dtype=dtype)
    return model, tok, ckpt


def build_sidecar(args, route_cfg: dict) -> SidecarPool:
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


def run_one_combo(
    *,
    samples: list,
    model,
    tok,
    sidecar: SidecarPool,
    route_cfg: dict,
    force_slot: str,
    args,
    device: torch.device,
    temperature: float,
    ntk_alpha: float,
    output_csv: Path,
) -> dict:
    apply_ablation_settings(model, temperature, ntk_alpha, args.ntk_max_train_len)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_match = n_err = n_skip = 0
    t0 = time.time()
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sample in tqdm(samples, desc=f"T={temperature} ntk={ntk_alpha}", leave=False):
            row = {
                "id": sample.get("id", ""),
                "meta_info": sample.get("meta_info", ""),
                "question_with_options": sample.get("question_with_options", ""),
                "answer_with_idx": sample.get("answer_with_idx", ""),
                "answer_idx": gold_letter(sample),
                "model_raw_output": "",
                "extracted_choice": "",
                "matches_answer_idx": False,
                "error": "",
            }
            view = prepare_sample_for_protocol(sample, PROTOCOL_PMT_RAG, route_cfg)
            if view is None:
                n_skip += 1
                row["error"] = "no_retrieval"
                writer.writerow(row)
                continue
            try:
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
                budget = args.max_seq_len - args.max_new_tokens
                if len(ids) > budget:
                    ids = ids[-budget:]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.inference_mode():
                    gen = model.generate(
                        x,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                raw = tok.decode(gen[0, x.shape[-1] :], skip_special_tokens=True)
                pred = extract_letter(raw)
                gold = row["answer_idx"]
                row["model_raw_output"] = raw
                row["extracted_choice"] = pred
                row["matches_answer_idx"] = bool(pred and gold and pred == gold)
                if gold in "ABCDE":
                    n_ok += 1
                    n_match += int(row["matches_answer_idx"])
            except Exception as e:
                row["error"] = str(e)
                n_err += 1
            writer.writerow(row)

    elapsed = time.time() - t0
    acc = (n_match / n_ok) if n_ok else 0.0
    return {
        "temperature": temperature,
        "ntk_alpha": ntk_alpha,
        "ntk_max_train_len": args.ntk_max_train_len,
        "n_total": len(samples),
        "n_scored": n_ok,
        "n_match": n_match,
        "n_error": n_err,
        "n_skip": n_skip,
        "accuracy": acc,
        "elapsed_sec": round(elapsed, 1),
        "output_csv": str(output_csv),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="RAG JSON 上的 temperature × ntk_alpha 消融")
    p.add_argument("--checkpoint", type=str, default="", help="PMT 权重；默认 out/run_pmt/*_best.pth")
    p.add_argument("--out_dir", type=str, default="", help="PMT 目录覆盖")
    p.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "dataset" / "RAG_MedQA_US_test_1273.json"),
        help="检索增强样本 JSON（包 04 口径，非中间产物）",
    )
    p.add_argument("--dataset_name", type=str, default="USMLE", help="子目录前缀：USMLE 或 MEDMCQA")
    p.add_argument(
        "--train_ntk",
        choices=("TRUE", "FALSE"),
        default="TRUE",
        help="目录标签 TRAINING-*（对应所用 PMT 训练设定）",
    )
    p.add_argument(
        "--infer_ntk",
        choices=("TRUE", "FALSE"),
        default="TRUE",
        help="目录标签 Inference-*",
    )
    p.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="显式结果根；默认 allocate result_summary / result_summaryN",
    )
    p.add_argument("--temperatures", type=str, default=",".join(map(str, DEFAULT_TEMPERATURES)))
    p.add_argument("--ntk_alphas", type=str, default=",".join(map(str, DEFAULT_NTK_ALPHAS)))
    p.add_argument("--ntk_max_train_len", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_hidden_layers", type=int, default=24)
    p.add_argument("--num_attention_heads", type=int, default=16)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--sidecar_map", type=str, default=str(ROOT / "model" / "sidecar_map.json"))
    p.add_argument("--sidecar_backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--sidecar_dtype", default="bfloat16")
    p.add_argument("--sidecar_max_tokens", type=int, default=4096)
    p.add_argument("--sidecar_temperature", type=float, default=0.7)
    p.add_argument("--sidecar_max_model_len", type=int, default=8192)
    p.add_argument("--sidecar_tp", type=int, default=1)
    p.add_argument("--sidecar_mem", type=float, default=0.85)
    args = p.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise SystemExit(f"data not found: {data_path} (expect pack-04 RAG JSON)")
    if "messages" in data_path.name.lower() or data_path.name.lower().endswith("_msgs.json"):
        raise SystemExit(
            f"refusing intermediate filename {data_path.name}; "
            "use retrieval JSON e.g. dataset/RAG_MedQA_US_test_1273.json"
        )

    samples = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit("data_path must be a JSON list")
    if args.limit > 0:
        samples = samples[: args.limit]

    root = Path(args.result_dir) if args.result_dir else allocate_result_summary_dir()
    root.mkdir(parents=True, exist_ok=True)
    table_dir = (
        root
        / f"{args.dataset_name.upper()}__TRAINING-{args.train_ntk}__Inference-{args.infer_ntk}"
    )
    table_dir.mkdir(parents=True, exist_ok=True)

    temperatures = _parse_float_list(args.temperatures)
    ntk_alphas = _parse_float_list(args.ntk_alphas)
    device = torch.device(args.device)
    route_cfg = load_map(args.sidecar_map)
    force_slot = protocol_slot(PROTOCOL_PMT_RAG, route_cfg)
    sidecar = build_sidecar(args, route_cfg)
    model, tok, ckpt_used = load_pmt(args, device)

    print(f"result_root={root}", flush=True)
    print(f"table_dir={table_dir}", flush=True)
    print(f"data={data_path} n={len(samples)}", flush=True)
    print(f"pmt={ckpt_used}", flush=True)

    summary_rows = []
    summary_fields = [
        "run_id",
        "temperature",
        "ntk_alpha",
        "ntk_max_train_len",
        "n_total",
        "n_scored",
        "n_match",
        "n_error",
        "n_skip",
        "accuracy",
        "elapsed_sec",
        "output_csv",
    ]
    run_id = 0
    for temperature in temperatures:
        for ntk_alpha in ntk_alphas:
            run_id += 1
            fname = (
                f"eval_{_param_tag('temp', temperature)}_{_param_tag('ntk', ntk_alpha)}.csv"
            )
            out_csv = table_dir / fname
            if args.skip_existing and out_csv.is_file():
                print(f"[{run_id}] skip {out_csv.name}", flush=True)
                continue
            print(
                f"[{run_id}/{len(temperatures)*len(ntk_alphas)}] "
                f"T={temperature} ntk={ntk_alpha} -> {fname}",
                flush=True,
            )
            row = run_one_combo(
                samples=samples,
                model=model,
                tok=tok,
                sidecar=sidecar,
                route_cfg=route_cfg,
                force_slot=force_slot,
                args=args,
                device=device,
                temperature=temperature,
                ntk_alpha=ntk_alpha,
                output_csv=out_csv,
            )
            row["run_id"] = run_id
            summary_rows.append(row)
            print(
                f"    acc={row['accuracy']:.4f} ({row['n_match']}/{row['n_scored']})",
                flush=True,
            )
            with open(table_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as sf:
                w = csv.DictWriter(sf, fieldnames=summary_fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(summary_rows)

    manifest = {
        "dataset_name": args.dataset_name.upper(),
        "train_ntk": args.train_ntk,
        "infer_ntk": args.infer_ntk,
        "data_path": str(data_path),
        "pmt_checkpoint": str(ckpt_used),
        "n_samples": len(samples),
        "temperatures": temperatures,
        "ntk_alphas": ntk_alphas,
        "runs": summary_rows,
    }
    (table_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (table_dir / "README_对应说明.txt").write_text(
        f"对应 {args.dataset_name.upper()} | TRAINING {args.train_ntk} | Inference {args.infer_ntk}\n"
        f"数据：检索增强 JSON（包 04 口径）\n"
        f"PMT：{ckpt_used}\n",
        encoding="utf-8",
    )
    print(f"done -> {table_dir}", flush=True)


if __name__ == "__main__":
    main()

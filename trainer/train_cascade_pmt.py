#!/usr/bin/env python3
"""Cascade train: protocol-specific PhaseModulationTransforme SFT (online upstream context).

Protocols:
  pmt_norag — non-retrieval branch + PhaseModulationTransforme → out/run_pmt_norag/
  pmt_rag   — retrieval-augmented branch + PhaseModulationTransforme → out/run_pmt_rag/

Training always initializes from the released pretrained PhaseModulationTransforme weights.
Inference must load the protocol-trained *_best.pth (see inference_cascade_pmt.py).
Upstream intermediate text is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from transformers import LlamaTokenizerFast

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.PMTConfig import PMTConfig
from model.pmtmodel import PhaseModulationTransforme
from model.pmt_protocol import (
    PROTOCOL_PMT_NORAG,
    PROTOCOL_PMT_RAG,
    default_out_dir,
    prepare_sample_for_protocol,
    protocol_slot,
    resolve_pmt_init_ckpt,
)
from model.sidecar_runtime import SidecarPool, gold_letter, load_map
from inference import format_inst_prompt_sft, merge_system_and_user_body

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def Logger(msg: str) -> None:
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        print(msg, flush=True)


def load_samples(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"expect JSON list: {path}")
    return raw


def encode_qf_example(
    tokenizer,
    *,
    system: str,
    user_long: str,
    letter: str,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    merged = merge_system_and_user_body(system, user_long)
    prompt = format_inst_prompt_sft(merged)
    # SFT-style: prompt + " " + letter + " </s>"
    full = f"{prompt} {letter} </s>"
    ids = tokenizer.encode(full, add_special_tokens=False)
    if len(ids) > max_seq_len:
        half = max_seq_len // 2
        ids = ids[:half] + ids[-(max_seq_len - half) :]
    pad = tokenizer.pad_token_id or 0
    if len(ids) < max_seq_len:
        ids = ids + [pad] * (max_seq_len - len(ids))

    inst_end = tokenizer.encode("[/INST]", add_special_tokens=False)
    mask = [0] * len(ids)
    for i in range(len(ids) - len(inst_end) + 1):
        if ids[i : i + len(inst_end)] == inst_end:
            for j in range(i + len(inst_end), len(ids)):
                mask[j] = 0 if ids[j] == pad else 1
            break

    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    m = torch.tensor(mask[1:], dtype=torch.long)
    return x, y, m


def masked_ce(logits, y, mask) -> torch.Tensor:
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    lp = loss_fct(logits.view(-1, logits.size(-1)), y.view(-1)).view(y.size())
    den = mask.sum().clamp_min(1)
    return (lp * mask).sum() / den


def parse_args():
    p = argparse.ArgumentParser(description="Protocol-specific cascade PhaseModulationTransforme SFT")
    p.add_argument(
        "--protocol",
        choices=(PROTOCOL_PMT_NORAG, PROTOCOL_PMT_RAG),
        required=True,
        help="pmt_norag or pmt_rag (separate PhaseModulationTransforme checkpoints)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="default: out/run_pmt_norag or out/run_pmt_rag by protocol",
    )
    p.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="default: parameters/sft_PMT_en_1024.pth (released pretrained PhaseModulationTransforme)",
    )
    p.add_argument("--data_path", type=str, required=True, help="JSON list with question_with_options + answer_idx")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="micro-batch after upstream (usually 1)")
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--sft_lr", type=float, default=5e-5)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--num_hidden_layers", type=int, default=24)
    p.add_argument("--num_attention_heads", type=int, default=16)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--sidecar_map", type=str, default=str(PROJECT_ROOT / "model" / "sidecar_map.json"))
    p.add_argument("--sidecar_backend", choices=("transformers", "vllm"), default="transformers")
    p.add_argument("--sidecar_dtype", default="bfloat16")
    p.add_argument("--sidecar_max_tokens", type=int, default=4096)
    p.add_argument("--sidecar_temperature", type=float, default=0.7)
    p.add_argument("--sidecar_max_model_len", type=int, default=8192)
    p.add_argument("--sidecar_tp0", type=int, default=1, help="vLLM TP for norag branch")
    p.add_argument("--sidecar_tp1", type=int, default=1, help="vLLM TP for rag branch")
    p.add_argument("--sidecar_mem0", type=float, default=0.45)
    p.add_argument("--sidecar_mem1", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N samples")
    return p.parse_args()


def main():
    args = parse_args()
    ddp = args.ddp or int(os.environ.get("RANK", -1)) != -1
    local_rank = 0
    device = torch.device(args.device)

    if ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.protocol)
    os.makedirs(out_dir, exist_ok=True)
    init_ckpt = resolve_pmt_init_ckpt(args.init_ckpt or None)

    route_cfg = load_map(args.sidecar_map)
    force_slot = protocol_slot(args.protocol, route_cfg)

    samples_raw = load_samples(args.data_path)
    if args.limit > 0:
        samples_raw = samples_raw[: args.limit]
    samples = []
    for s in samples_raw:
        view = prepare_sample_for_protocol(s, args.protocol, route_cfg)
        if view is not None:
            samples.append(view)
    Logger(f"protocol={args.protocol} samples={len(samples)}/{len(samples_raw)} out_dir={out_dir}")

    # shard samples across ranks
    if ddp:
        world = dist.get_world_size()
        rank = dist.get_rank()
        samples = [s for i, s in enumerate(samples) if i % world == rank]

    tp = {force_slot: args.sidecar_tp0 if args.protocol == PROTOCOL_PMT_NORAG else args.sidecar_tp1}
    mem = {force_slot: args.sidecar_mem0 if args.protocol == PROTOCOL_PMT_NORAG else args.sidecar_mem1}
    sidecar = SidecarPool(
        map_path=args.sidecar_map,
        backend=args.sidecar_backend,
        dtype=args.sidecar_dtype,
        max_model_len=args.sidecar_max_model_len,
        tp_by_slot=tp,
        mem_util_by_slot=mem,
        load_slots=[force_slot],
    )

    tok_dir = PROJECT_ROOT / "model" / "tokenizer_en"
    tokenizer = LlamaTokenizerFast.from_pretrained(str(tok_dir))
    lm_config = PMTConfig(
        dim=args.hidden_size,
        n_layers=args.num_hidden_layers,
        n_heads=args.num_attention_heads,
        n_kv_heads=args.num_key_value_heads,
        max_seq_len=args.max_seq_len,
        flash_attn=True,
    )
    model = PhaseModulationTransforme(lm_config)
    Logger(f"load PhaseModulationTransforme init (train) ← {init_ckpt}")
    state = torch.load(str(init_ckpt), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=args.sft_lr)
    use_amp = args.dtype != "float32" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    stem = f"cascade_PMT_en_{args.hidden_size}"
    latest = out_dir / f"{stem}.pth"
    best = out_dir / f"{stem}_best.pth"
    best_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, sample in enumerate(samples):
            letter = gold_letter(sample)
            if letter not in "ABCDE":
                continue
            _slot, user_long, system = sidecar.generate(
                sample,
                temperature=args.sidecar_temperature,
                max_tokens=args.sidecar_max_tokens,
                force_slot=force_slot,
            )
            if len(user_long) < 200:
                # skip without logging intermediate content/length details
                continue

            x, y, m = encode_qf_example(
                tokenizer,
                system=system,
                user_long=user_long,
                letter=letter,
                max_seq_len=args.max_seq_len,
            )
            del user_long
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=getattr(torch, args.dtype) if args.dtype != "float32" else torch.float32):
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out[0]
                loss = masked_ce(logits.float(), y, m) / args.accumulation_steps

            if use_amp and args.dtype == "float16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1
            if global_step % args.accumulation_steps == 0:
                if use_amp and args.dtype == "float16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * args.accumulation_steps
            if global_step % args.log_interval == 0:
                Logger(
                    f"protocol={args.protocol} epoch={epoch+1}/{args.epochs} "
                    f"i={i}/{len(samples)} loss={step_loss:.4f} elapsed={time.time()-t0:.0f}s"
                )

            if (not ddp or dist.get_rank() == 0) and step_loss < best_loss:
                best_loss = step_loss
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), best)
                Logger(f"best → {best} loss={best_loss:.4f}")

            if (not ddp or dist.get_rank() == 0) and global_step % args.save_interval == 0:
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), latest)
                Logger(f"latest → {latest}")

        if not ddp or dist.get_rank() == 0:
            raw = model.module if hasattr(model, "module") else model
            torch.save(raw.state_dict(), latest)
            Logger(f"epoch done → {latest}")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    Logger(f"cascade train finished protocol={args.protocol}")


if __name__ == "__main__":
    main()

"""In-process dual sidecar loaders + opaque sample routing.

Main train/infer scripts should only call ``select_slot`` / ``SidecarPool.generate``.
Routing rules live in ``sidecar_map.json`` (not in call sites).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from model.sidecar_prompts import PROMPT_AUGMENTED, PROMPT_LETTER, PROMPT_PLAIN, SYSTEM

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = Path(__file__).resolve().parent / "sidecar_map.json"

BARE_THEREFORE = "Therefore, the answer to the above question is: "
LEAD = "In response to your question, my analysis process and answer are as follows:"
THEREFORE_LINE_RE = re.compile(
    r"(?:Therefore,\s*)?(?:the )?answer to (?:the )?(?:above )?question is\s*[:：].*$",
    re.IGNORECASE | re.DOTALL,
)


def load_map(map_path: str | Path | None = None) -> dict:
    path = Path(map_path) if map_path else DEFAULT_MAP
    return json.loads(path.read_text(encoding="utf-8"))


def _dig(obj: Any, path: list[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def select_slot(sample: dict, cfg: dict | None = None) -> str:
    """Return slot id ('0' / '1') using map probe rules only."""
    cfg = cfg or load_map()
    probe = cfg.get("probe") or {}
    path = list(probe.get("path") or [])
    rule = str(probe.get("rule") or "list_nonempty")
    hit = False
    if path:
        val = _dig(sample, path)
        if rule == "list_nonempty":
            hit = isinstance(val, list) and len(val) > 0
        elif rule == "truthy":
            hit = bool(val)
        else:
            hit = val is not None
    return str(cfg["hit_slot"] if hit else cfg["default_slot"])


def resolve_slot_dir(slot: str, cfg: dict | None = None, root: Path | None = None) -> Path:
    cfg = cfg or load_map()
    root = root or ROOT
    rel = cfg["slots"][str(slot)]
    p = Path(rel)
    return p if p.is_absolute() else (root / p)


def _knowledges_text(sample: dict) -> str:
    if sample.get("knowledges_for_llm"):
        return str(sample["knowledges_for_llm"]).strip()
    cfg = load_map()
    path = list((cfg.get("probe") or {}).get("path") or [])
    val = _dig(sample, path) if path else None
    if isinstance(val, list):
        return "\n\n".join(str(x) for x in val if x)
    return ""


def build_sidecar_prompt(sample: dict, slot: str, cfg: dict | None = None) -> str:
    cfg = cfg or load_map()
    kind = (cfg.get("prompt_slot") or {}).get(str(slot), "plain")
    q = sample.get("question_with_options") or ""
    if kind == "letter":
        return PROMPT_LETTER.format(question_with_options=q)
    if kind == "augmented":
        kb = _knowledges_text(sample)
        if not kb:
            return PROMPT_PLAIN.format(question_with_options=q)
        return PROMPT_AUGMENTED.format(knowledges_for_llm=kb, question_with_options=q)
    return PROMPT_PLAIN.format(question_with_options=q)


def extract_choice_letter(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"(?is)<think>.*?</think>\s*", "", raw).strip()
    if not raw:
        return ""
    m = re.search(r"\b([A-E])\b", raw.upper())
    if m:
        return m.group(1)
    ch = raw[0].upper()
    return ch if ch in "ABCDE" else ""


def gold_letter(sample: dict) -> str:
    g = str(sample.get("gold_letter") or sample.get("answer_idx") or "").strip().upper()
    if g in "ABCDE":
        return g
    return g[0] if g and g[0] in "ABCDE" else ""


def clean_sidecar_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"</?think>\s*", "", text, flags=re.I)
    if not text.startswith(LEAD):
        body = text.lstrip()
        if body.startswith("**Analysis:**") or body.startswith("Analysis"):
            text = LEAD + "\n\n" + body
    text = re.sub(r"(?is)<knowledge>.*?</knowledge>\s*", "", text)
    matches = list(THEREFORE_LINE_RE.finditer(text))
    if matches:
        m = matches[-1]
        tail = text[m.start() :]
        if len(tail) < 500 or m.start() > len(text) * 0.45:
            text = text[: m.start()].rstrip() + "\n\n" + BARE_THEREFORE
    if not text.endswith(BARE_THEREFORE) and re.search(r"Conclusion:", text, re.I):
        text = text.rstrip() + "\n\n" + BARE_THEREFORE
    return text.lstrip() if text.endswith(BARE_THEREFORE) else text.strip()


class _HFEngine:
    def __init__(self, model_dir: Path, *, device_map: str, dtype: str, max_model_len: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_model_len = max_model_len
        self.device_map = device_map

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max(64, self.max_model_len - max_tokens),
        )
        # device_map=auto: move batch to first param device
        first = next(self.model.parameters()).device
        inputs = {k: v.to(first) for k, v in inputs.items()}
        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = max(temperature, 1e-5)
            kwargs["top_p"] = 0.95
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        new_ids = out[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


class _VLLMEngine:
    def __init__(
        self,
        model_dir: Path,
        *,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
    ):
        from vllm import LLM

        self.llm = LLM(
            model=str(model_dir),
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        from vllm import SamplingParams

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        params = SamplingParams(
            temperature=max(temperature, 0.0),
            max_tokens=max_tokens,
            top_p=0.95 if temperature > 0 else 1.0,
        )
        out = self.llm.generate([text], params)
        return out[0].outputs[0].text if out and out[0].outputs else ""


class SidecarPool:
    """Load selected slot engines; cascade routes 0/1, solo uses solo_slot."""

    def __init__(
        self,
        *,
        map_path: str | Path | None = None,
        backend: str = "transformers",
        dtype: str = "bfloat16",
        max_model_len: int = 8192,
        device_map_by_slot: dict[str, str] | None = None,
        tp_by_slot: dict[str, int] | None = None,
        mem_util_by_slot: dict[str, float] | None = None,
        load_slots: list[str] | None = None,
        root: Path | None = None,
    ) -> None:
        self.cfg = load_map(map_path)
        self.root = root or ROOT
        self.backend = backend
        self.engines: dict[str, Any] = {}
        device_map_by_slot = device_map_by_slot or {}
        tp_by_slot = tp_by_slot or {}
        mem_util_by_slot = mem_util_by_slot or {}

        if load_slots is None:
            load_slots = list(self.cfg.get("cascade_slots") or self.cfg["slots"].keys())
        load_slots = [str(s) for s in load_slots]

        for slot in load_slots:
            if slot not in self.cfg["slots"]:
                raise KeyError(f"unknown slot {slot} in sidecar_map")
            model_dir = resolve_slot_dir(slot, self.cfg, self.root)
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"missing slot weight dir: {model_dir} (slot={slot})"
                )
            # Opaque log: do not print weight directory names
            print(f"[upstream] load branch={slot} backend={backend}", flush=True)
            if backend == "vllm":
                self.engines[slot] = _VLLMEngine(
                    model_dir,
                    dtype=dtype,
                    tensor_parallel_size=int(tp_by_slot.get(slot, 1)),
                    gpu_memory_utilization=float(mem_util_by_slot.get(slot, 0.45)),
                    max_model_len=max_model_len,
                )
            else:
                self.engines[slot] = _HFEngine(
                    model_dir,
                    device_map=str(device_map_by_slot.get(slot, "auto")),
                    dtype=dtype,
                    max_model_len=max_model_len,
                )

    def generate(
        self,
        sample: dict,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        force_slot: str | None = None,
    ) -> tuple[str, str, str]:
        """Cascade path: return (slot, cleaned_context, system_prompt).

        Intermediate upstream text is for in-memory PhaseModulationTransformer input only;
        callers must not persist or print it.
        """
        slot = str(force_slot) if force_slot is not None else select_slot(sample, self.cfg)
        if slot not in self.engines:
            raise RuntimeError(f"branch {slot} not loaded; loaded={list(self.engines)}")
        prompt = build_sidecar_prompt(sample, slot, self.cfg)
        raw = self.engines[slot].generate(
            prompt, temperature=temperature, max_tokens=max_tokens
        )
        return slot, clean_sidecar_text(raw), SYSTEM

    def generate_letter(
        self,
        sample: dict,
        *,
        temperature: float = 0.0,
        max_tokens: int = 8,
    ) -> tuple[str, str, str]:
        """Solo path (no QF): return (slot, raw_text, extracted_letter).

        Only the extracted letter should be written to evaluation artifacts.
        """
        slot = str(self.cfg.get("solo_slot") or "2")
        if slot not in self.engines:
            raise RuntimeError(f"solo branch {slot} not loaded; loaded={list(self.engines)}")
        prompt = build_sidecar_prompt(sample, slot, self.cfg)
        raw = self.engines[slot].generate(
            prompt, temperature=temperature, max_tokens=max_tokens
        )
        letter = extract_choice_letter(raw)
        return slot, raw, letter

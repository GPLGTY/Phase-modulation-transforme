#!/usr/bin/env python3
"""[DEPRECATED] 离线 Intent 生成已停用（D1）。

训练 / 测试 / 推理请改用在线级联：
  - trainer/train_cascade_qf.py
  - inference_cascade_qf.py
  - scripts/run_cascade_train.sh / run_cascade_infer.sh

本文件仅保留兼容；新流程不再依赖离线 Intent JSON 生成。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt / 解析逻辑（与原先 Intent 格式一致）
# ---------------------------------------------------------------------------
SYSTEM = (
    "You are an expert in intent recognition. Based on the provided information, "
    "you can determine which of the options A through E the intent of the given "
    "text corresponds to."
)
LEAD = "In response to your question, my analysis process and answer are as follows:"

PROMPT_NO_RAG = """You write the user message for an intent-recognition example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Analyze the question and options using your own medical knowledge, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.

Question:
{question_with_options}
"""

PROMPT_RAG = """You write the user message for an intent-recognition example. Output only the user message, in English. No JSON, no system line, no assistant line, no title.

You are NOT given the correct answer. Below are medical knowledge passages retrieved from reference textbooks; they may or may not be relevant — use them only when helpful. Analyze the question and options, then choose the single best option yourself.

Match this layout exactly:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

<3 to 5 numbered points about the question's key facts. Bold important terms. Do not paste the whole question. Do not paste or quote long retrieved passages; paraphrase only what you use.>

Then one bullet per option, covering every option exactly once:
- **A. <option text>** – <why it fits or fails>
- **B. <option text>** – ...
- **C. <option text>** – ...
- **D. <option text>** – ...
- **E. <option text>** – ... (include E only if the question has option E)

Copy each option letter and option text exactly from the question. Do not invent, rename, merge, or add options. If the question has no option E, do not write an E bullet. Do not repeat the same sentence.

**Conclusion:**
<2 to 4 sentences. Name the chosen option in a full sentence, including its letter and its text, and the main reason.>

Therefore, the answer to the above question is: 

Length: about 2000 to 5000 characters.
The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon. Do not put a letter, option text, bold marks, or any other characters on that line. The letter belongs only in the conclusion sentences above it.
Do NOT include a "#Medical knowledge#" section or any raw retrieved text in your output.

Retrieved knowledge (for your reasoning only; do not copy into the output):
<knowledge>
{knowledges_for_llm}
</knowledge>

Question:
{question_with_options}
"""

INTENT_RE = re.compile(
    r"(?:Therefore[, ]+(?:the )?(?:correct )?answer(?: to the (?:above )?question(?: above)?)? is\s*[:：]?\s*"
    r"(?:\*\*)?([A-E])(?:\*\*)?"
    r"|the (?:correct )?answer is\s*[:：]?\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|correct choice is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|best (?:choice|option|answer) is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|chosen option is\s*(?:\*\*)?([A-E])(?:\*\*)?"
    r"|Answer\s*[:：]\s*(?:\*\*)?([A-E])(?:\*\*)?)",
    re.IGNORECASE,
)
LETTER_RE = re.compile(r"\b([A-E])\b")
OPTION_RE = re.compile(r"(?m)^\s*([A-E])[\.\)]\s*(.+?)\s*$")
BARE_THEREFORE = "Therefore, the answer to the above question is: "
THEREFORE_LINE_RE = re.compile(
    r"(?:Therefore,\s*)?(?:the )?answer to (?:the )?(?:above )?question is\s*[:：].*$",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# 本地推理后端
# ---------------------------------------------------------------------------
class LocalModel:
    """统一本地生成接口：一次可喂多条 prompt。"""

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        raise NotImplementedError


class TransformersLocalModel(LocalModel):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_model_len: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_model_len = max_model_len
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        print(f"[transformers] loading {model_path} dtype={dtype} device={device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            trust_remote_code=trust_remote_code,
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def _apply_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return prompt

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        import torch

        del presence_penalty, frequency_penalty  # transformers 无 repetition_penalty 近似
        outs: list[str] = []
        for prompt in prompts:
            text = self._apply_chat(prompt)
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max(64, self.max_model_len - max_tokens),
            )
            if self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            do_sample = temperature > 0
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": float(repetition_penalty) if repetition_penalty else 1.0,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(temperature, 1e-5)
                gen_kwargs["top_p"] = 0.95
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs["input_ids"].shape[-1]
            new_ids = output_ids[0, prompt_len:]
            outs.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return outs


class VLLMLocalModel(LocalModel):
    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ) -> None:
        from vllm import LLM

        print(
            f"[vllm] loading {model_path} dtype={dtype} "
            f"tp={tensor_parallel_size} max_model_len={max_model_len}",
            flush=True,
        )
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _apply_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return prompt

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> list[str]:
        from vllm import SamplingParams

        texts = [self._apply_chat(p) for p in prompts]
        params = SamplingParams(
            temperature=max(temperature, 0.0),
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty if repetition_penalty else 1.0,
            top_p=0.95 if temperature > 0 else 1.0,
        )
        results = self.llm.generate(texts, params)
        return [r.outputs[0].text if r.outputs else "" for r in results]


def build_local_model(args: argparse.Namespace) -> LocalModel:
    if args.backend == "vllm":
        return VLLMLocalModel(
            args.model_path,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=not args.no_trust_remote_code,
        )
    return TransformersLocalModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        max_model_len=args.max_model_len,
    )


# ---------------------------------------------------------------------------
# 样本字段 / 清洗 / 解析
# ---------------------------------------------------------------------------
def knowledges_for_llm(sample: dict) -> str:
    if sample.get("knowledges_for_llm"):
        return str(sample["knowledges_for_llm"]).strip()
    ri = sample.get("retrieve_info") or {}
    ki = ri.get("knowledges_info") or {}
    knowledges = ki.get("knowledges") if isinstance(ki, dict) else None
    if isinstance(knowledges, list) and knowledges:
        return "\n\n".join(str(x) for x in knowledges if x)
    return ""


def rag_k(sample: dict, with_rag: bool) -> int:
    if not with_rag:
        return 0
    ri = sample.get("retrieve_info") or {}
    declared = ri.get("topk_knowledge")
    if isinstance(declared, int) and declared > 0:
        return declared
    ki = (ri.get("knowledges_info") or {}).get("knowledges") or []
    return len(ki) if isinstance(ki, list) else 0


def gold_letter(sample: dict) -> str:
    g = str(sample.get("gold_letter") or sample.get("answer_idx") or "").strip().upper()
    if g in "ABCDE":
        return g
    return g[0] if g and g[0] in "ABCDE" else ""


def _norm_phrase(text: str) -> str:
    text = (text or "").lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_options(question: str) -> dict[str, str]:
    return {
        m.group(1).upper(): m.group(2).strip()
        for m in OPTION_RE.finditer(question or "")
    }


def _conclusion(text: str) -> str:
    m = re.search(r"\*\*Conclusion:\*\*([\s\S]*)$", text or "", re.I)
    if not m:
        m = re.search(r"(?m)^Conclusion:([\s\S]*)$", text or "", re.I)
    return m.group(1) if m else ""


def _significant_tokens(text: str) -> set[str]:
    return {tok for tok in _norm_phrase(text).split() if len(tok) > 3}


def _letter_conflicts(text: str, letter: str, options: dict[str, str]) -> bool:
    real = options.get(letter) or ""
    if not real:
        return False
    stated = re.findall(rf"\*\*\s*{letter}\.\s*([^*\n]+)", text or "", flags=re.I)
    if not stated:
        return False
    real_toks = _significant_tokens(real)
    if not real_toks:
        return False
    for chunk in stated:
        got = _significant_tokens(chunk)
        if got and got.isdisjoint(real_toks):
            return True
    return False


def _letter_from_option_names(text: str, question: str) -> str | None:
    options = parse_options(question)
    conc = _conclusion(text)
    if not options or not conc:
        return None
    conc = THEREFORE_LINE_RE.sub(" ", conc)
    sentences = re.split(r"(?<=[.!?])\s+", conc)
    choice = [
        s
        for s in sentences
        if re.search(r"best|correct|appropriate|answer|choice", s, re.I)
    ]
    regions = [_norm_phrase(s) for s in choice] or [_norm_phrase(conc)]
    hits: list[tuple[int, str]] = []
    for letter, name in options.items():
        norm_name = _norm_phrase(name)
        if len(norm_name) < 4:
            continue
        if any(norm_name in region for region in regions):
            hits.append((len(norm_name), letter))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


def extract_intent_letter(text: str, question: str = "") -> str | None:
    hits = []
    for m in INTENT_RE.finditer(text or ""):
        g = next((x for x in m.groups() if x), None)
        if g:
            hits.append(g.upper())
    letter = hits[-1] if hits else None
    if not letter:
        conc = _conclusion(text or "")
        tail = conc or (text or "")[-500:]
        m2 = re.findall(r"\*\*\s*([A-E])\.\s+", tail)
        if m2:
            letter = m2[-1].upper()
        else:
            letters = LETTER_RE.findall(tail)
            letter = letters[-1].upper() if letters else None
    options = parse_options(question)
    if letter and _letter_conflicts(text or "", letter, options):
        letter = None
    if letter:
        return letter
    return _letter_from_option_names(text or "", question)


def _is_repetition(text: str) -> bool:
    raw = (text or "").strip()
    if len(raw) < 400:
        return False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 8:
        keys = [re.sub(r"\s+", " ", ln.lower())[:120] for ln in lines]
        top = max(keys.count(k) for k in set(keys))
        if top >= 5 and top / len(keys) >= 0.35:
            return True
    parts = [p.strip() for p in re.split(r"[\n.]+", raw) if len(p.strip()) > 40]
    if len(parts) >= 8:
        top = max(parts.count(p) for p in set(parts))
        if top >= 5 and top / len(parts) >= 0.4:
            return True
    for size in (32, 64):
        if len(raw) < size * 10:
            continue
        counts: dict[str, int] = {}
        for i in range(0, min(len(raw) - size, 12000), size):
            piece = raw[i : i + size]
            counts[piece] = counts.get(piece, 0) + 1
        top = max(counts.values(), default=0)
        if top >= 8 and top * size / len(raw) >= 0.25:
            return True
    return False


def _force_bare_therefore(text: str) -> str:
    matches = list(THEREFORE_LINE_RE.finditer(text))
    if matches:
        m = matches[-1]
        tail = text[m.start() :]
        if len(tail) < 500 or m.start() > len(text) * 0.45:
            text = text[: m.start()].rstrip() + "\n\n" + BARE_THEREFORE
    if not text.endswith(BARE_THEREFORE) and re.search(r"Conclusion:", text, re.I):
        text = text.rstrip() + "\n\n" + BARE_THEREFORE
    if text.endswith(BARE_THEREFORE):
        return text.lstrip()
    return text.strip()


def clean_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"</sam>\s*", "", text, flags=re.I)
    text = re.sub(r"</?think>\s*", "", text, flags=re.I)
    # DeepSeek-R1 等思考模型可能夹带 reasoning 标记
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"^(?:User|USER|user message)\s*:\s*", "", text, flags=re.I)
    if not text.startswith(LEAD):
        body = text.lstrip()
        if body.startswith("**Analysis:**") or body.startswith("Analysis"):
            text = LEAD + "\n\n" + body
    text = re.sub(
        r"(?is)#\s*Medical knowledge\s*#.*?(?=\n\s*#\s*Medical question\s*#|\n\s*\*\*Analysis:\*\*|\n\s*Analysis:)",
        "",
        text,
    )
    text = re.sub(r"(?is)<knowledge>.*?</knowledge>\s*", "", text)
    return _force_bare_therefore(text)


def make_prompt(sample: dict, with_rag: bool) -> str:
    q = sample.get("question_with_options") or ""
    if with_rag:
        kb = knowledges_for_llm(sample)
        if not kb:
            raise RuntimeError("with_rag set but knowledges empty")
        return PROMPT_RAG.format(knowledges_for_llm=kb, question_with_options=q)
    return PROMPT_NO_RAG.format(question_with_options=q)


def repair_prompt(question: str, notes: str) -> str:
    options = parse_options(question)
    option_lines = "\n".join(f"{letter}. {name}" for letter, name in options.items())
    clipped = (notes or "").strip()[:4000]
    return f"""Rewrite the scratch notes into the required user message. Output only that message, in English. No scratch notes, no title, no JSON.

Start with this exact text:

In response to your question, my analysis process and answer are as follows:

**Analysis:**

Then write 3 to 5 numbered points and one bullet for every option below, copying the letter and option text exactly. Do not add or rename options.

**Conclusion:**
In 2 to 4 sentences, name the chosen option with its letter and its text.

Therefore, the answer to the above question is: 

The last line must be exactly "Therefore, the answer to the above question is: " with a space after the colon and nothing else on that line. Do not repeat a sentence.

Options:
{option_lines}

Question:
{question}

Scratch notes to rewrite, not to paste:
{clipped}
"""


def option_lock(question: str) -> str:
    options = parse_options(question)
    if not options:
        return ""
    lines = "\n".join(f"{letter}. {name}" for letter, name in options.items())
    return (
        "\n\nHard constraint: copy the option bullets verbatim from this list. "
        "Do not add, remove, rename, or replace any option. "
        "Start the output with **Analysis:**. Do not repeat any sentence or line.\n"
        + lines
    )


def call_model(
    model: LocalModel,
    prompt: str,
    temperature: float,
    max_tokens: int,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> str:
    texts = model.generate(
        [prompt],
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
    )
    return clean_output(texts[0] if texts else "")


def build_sample(
    src: dict,
    user_text: str,
    model_name: str,
    with_rag: bool,
    source: str,
) -> dict:
    intent = extract_intent_letter(user_text, src.get("question_with_options") or "")
    if not intent:
        raise RuntimeError("unresolved intent_letter")
    gold = gold_letter(src)
    return {
        "id": src.get("id"),
        "meta_info": src.get("meta_info") or src.get("subject_name"),
        "question_with_options": src.get("question_with_options"),
        "answer_with_idx": src.get("answer_with_idx"),
        "answer_idx": gold or intent,
        "gold_letter": gold or None,
        "intent_letter": intent,
        "intent_extract_source": "rewrite_conclusion",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": intent},
        ],
        "model_prompt_text": f"[INST] {SYSTEM}\n\n{user_text} [/INST]",
        "source": source,
        "user_rewrite_model": model_name,
        "user_rewrite_no_gold": True,
        "user_rewrite_with_rag": bool(with_rag),
    }


def load_done(partial_path: Path) -> dict[str, dict]:
    if not partial_path.exists():
        return {}
    raw = json.loads(partial_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "samples" in raw:
        items = raw["samples"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    done = {}
    for s in items:
        sid = str(s.get("id"))
        if sid and sid != "None":
            done[sid] = s
    return done


def save_partial(partial_path: Path, done: dict[str, dict], meta: dict) -> None:
    payload = {"meta": meta, "samples": [done[k] for k in sorted(done, key=str)]}
    partial_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def model_tag(model_path: str) -> str:
    return Path(model_path.rstrip("/\\")).name or model_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="检索 JSON → Intent JSON（本地部署大模型，无云端/HTTP 接口）。"
    )
    p.add_argument("--in_json", type=Path, required=True)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地模型目录或 HF 模型名，例如 /models/DeepSeek-R1-Distill-Qwen-32B",
    )
    p.add_argument(
        "--backend",
        choices=("transformers", "vllm"),
        default="transformers",
        help="本地推理后端；默认 transformers，吞吐更高可选 vllm（进程内，非 HTTP）",
    )
    p.add_argument("--with_rag", action="store_true", help="仅把检索段落放进 prompt，不写入输出")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=0, help="0 = offset 之后全部样本")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--device", default="cuda", help="transformers: cuda / cpu / auto")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--tensor_parallel_size", type=int, default=1, help="仅 vllm")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="仅 vllm")
    p.add_argument("--no_trust_remote_code", action="store_true")
    p.add_argument("--no_resume", action="store_true", help="忽略已有 partial，从头生成")
    p.add_argument(
        "--hard_retry",
        action="store_true",
        help="跳过无惩罚首轮，直接用更强的反重复采样",
    )
    return p.parse_args()


def process_one(
    model: LocalModel,
    sample: dict,
    *,
    with_rag: bool,
    model_name: str,
    source_placeholder: str,
    temperature: float,
    max_tokens: int,
    hard_retry: bool,
) -> tuple[str, dict | None, str | None]:
    sid = str(sample.get("id"))
    base_prompt = make_prompt(sample, with_rag)
    question = sample.get("question_with_options") or ""
    schedules = (
        (temperature, max_tokens, 0.0, 0.0, 1.0, False),
        (min(temperature, 0.4), min(max_tokens, 3072), 0.8, 0.4, 1.12, True),
        (0.3, min(max_tokens, 3072), 1.2, 0.8, 1.25, True),
        (0.2, min(max_tokens, 3072), 1.6, 1.2, 1.4, True),
        (0.15, min(max_tokens, 3072), 1.8, 1.5, 1.55, True),
    )
    if hard_retry:
        schedules = schedules[1:]
    last_err = "unknown"
    for attempt, (temp, tok, presence, frequency, repetition, lock) in enumerate(schedules):
        prompt = base_prompt + (option_lock(question) if lock else "")
        try:
            text = call_model(
                model, prompt, temp, tok, presence, frequency, repetition
            )
            has_analysis = "**Analysis:**" in text or "Analysis:" in text
            has_conclusion = "**Conclusion:**" in text or "Conclusion:" in text
            if (
                (not has_analysis or not has_conclusion)
                and len(text) >= 400
                and not _is_repetition(text)
            ):
                repaired = call_model(
                    model,
                    repair_prompt(question, text),
                    0.2,
                    min(tok, 3072),
                    0.6,
                    0.4,
                    1.15,
                )
                if "**Analysis:**" in repaired or "Analysis:" in repaired:
                    text = repaired
            if _is_repetition(text):
                raise RuntimeError("repetition_loop")
            if len(text) < 400:
                raise RuntimeError(f"too_short len={len(text)}")
            if "**Analysis:**" not in text and "Analysis:" not in text:
                raise RuntimeError("missing Analysis")
            if "**Conclusion:**" not in text and "Conclusion:" not in text:
                raise RuntimeError("missing Conclusion")
            if not re.search(
                r"Therefore, the answer to the above question is: $",
                text,
            ):
                raise RuntimeError("missing bare therefore line")
            built = build_sample(
                sample, text, model_name, with_rag, source_placeholder
            )
            return sid, built, None
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 * (attempt + 1), 8))
    return sid, None, last_err


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if model_path.exists() and not (model_path.is_dir() or model_path.is_file()):
        raise SystemExit(f"invalid --model_path: {args.model_path}")

    all_data = json.loads(args.in_json.read_text(encoding="utf-8"))
    if not isinstance(all_data, list):
        raise SystemExit("input JSON must be a list")
    end = None if args.n <= 0 else args.offset + args.n
    slice_data = all_data[args.offset : end]
    if not slice_data:
        raise SystemExit("empty slice")

    tag = model_tag(args.model_path)
    k = rag_k(slice_data[0], args.with_rag)
    planned_n = len(slice_data)
    source_placeholder = f"{tag}_local_{args.backend}_rag{k}_n{planned_n}"

    partial_path = args.out_json.with_suffix(args.out_json.suffix + ".partial.json")
    if args.out_json.suffix == ".json":
        partial_path = args.out_json.with_name(args.out_json.stem + ".partial.json")

    done: dict[str, dict] = {} if args.no_resume else load_done(partial_path)
    todo = [s for s in slice_data if str(s.get("id")) not in done]

    print(
        json.dumps(
            {
                "in_json": str(args.in_json),
                "out_json": str(args.out_json),
                "model_path": args.model_path,
                "backend": args.backend,
                "with_rag": args.with_rag,
                "rag_k": k,
                "slice_n": planned_n,
                "resume_done": len(done),
                "todo": len(todo),
            },
            indent=2,
        ),
        flush=True,
    )

    if not todo:
        out = [done[str(s.get("id"))] for s in slice_data if str(s.get("id")) in done]
        source = f"{tag}_local_{args.backend}_rag{k}_n{len(out)}"
        for s in out:
            s["source"] = source
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"nothing to run; wrote {args.out_json} n={len(out)} source={source}", flush=True)
        return

    model = build_local_model(args)
    errors: dict[str, str] = {}
    t0 = time.time()

    def run_batch(samples: list[dict], label: str) -> None:
        if not samples:
            return
        print(f"{label}: {len(samples)} items", flush=True)
        for i, sample in enumerate(samples, 1):
            sid, built, err = process_one(
                model,
                sample,
                with_rag=args.with_rag,
                model_name=tag,
                source_placeholder=source_placeholder,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                hard_retry=args.hard_retry,
            )
            if err:
                errors[sid] = err
                print(f"[{i}/{len(samples)}] FAIL id={sid} {err[:160]}", flush=True)
            else:
                errors.pop(sid, None)
                done[sid] = built
                print(
                    f"[{i}/{len(samples)}] OK id={sid} "
                    f"intent={built['intent_letter']} gold={built.get('gold_letter')} "
                    f"done={len(done)}/{planned_n}",
                    flush=True,
                )
            if len(done) % 10 == 0:
                save_partial(
                    partial_path,
                    done,
                    {"source": source_placeholder, "ok": len(done), "todo": planned_n},
                )
                print(f"checkpoint {partial_path} n={len(done)}", flush=True)

    run_batch(todo, "pass1")
    retry_ids = set(errors)
    if retry_ids:
        retry_samples = [s for s in todo if str(s.get("id")) in retry_ids]
        print(f"retrying {len(retry_samples)} failures...", flush=True)
        run_batch(retry_samples, "pass2_retry")

    save_partial(
        partial_path,
        done,
        {"source": source_placeholder, "ok": len(done), "todo": planned_n},
    )

    missing = [str(s.get("id")) for s in slice_data if str(s.get("id")) not in done]
    if missing:
        print(
            f"incomplete: ok={len(done)} missing={len(missing)} partial={partial_path}",
            flush=True,
        )
        for sid in missing[:20]:
            print(sid, errors.get(sid, "not_done"), flush=True)
        sys.exit(1)

    out = [done[str(s.get("id"))] for s in slice_data]
    source = f"{tag}_local_{args.backend}_rag{k}_n{len(out)}"
    for s in out:
        s["source"] = source

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    both = [
        s
        for s in out
        if s.get("gold_letter") in list("ABCDE") and s.get("intent_letter") in list("ABCDE")
    ]
    agree = sum(1 for s in both if s["gold_letter"] == s["intent_letter"])
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "source": source,
                "backend": args.backend,
                "model_path": args.model_path,
                "n": len(out),
                "with_rag": args.with_rag,
                "rag_k": k,
                "elapsed_sec": round(time.time() - t0, 1),
                "intent_vs_gold_agree": agree,
                "intent_vs_gold_n": len(both),
                "intent_vs_gold_rate": round(agree / max(len(both), 1), 4),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

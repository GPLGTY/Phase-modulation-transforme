"""Opaque experimental protocols for cascade / standalone evaluation.

Protocols (public names only; slot wiring stays in sidecar_map.json):

  no_pmt     — no PhaseModulationTransformer; standalone upstream letter prediction
  pmt_norag  — PhaseModulationTransformer + non-retrieval cascade branch
  pmt_rag    — PhaseModulationTransformer + retrieval-augmented cascade branch

Train vs infer automatically resolve distinct PhaseModulationTransformer checkpoints.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from model.sidecar_runtime import _dig, load_map

ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_NO_PMT = "no_pmt"
PROTOCOL_PMT_NORAG = "pmt_norag"
PROTOCOL_PMT_RAG = "pmt_rag"
ALL_PROTOCOLS = (PROTOCOL_NO_PMT, PROTOCOL_PMT_NORAG, PROTOCOL_PMT_RAG)

# Default layout: train writes here; infer reads *_best.pth from here.
DEFAULT_OUT = {
    PROTOCOL_PMT_NORAG: ROOT / "out" / "run_pmt_norag",
    PROTOCOL_PMT_RAG: ROOT / "out" / "run_pmt_rag",
}
DEFAULT_INIT_CKPT = ROOT / "parameters" / "sft_PMT_en_1024.pth"


def probe_has_retrieval(sample: dict, cfg: dict | None = None) -> bool:
    cfg = cfg or load_map()
    probe = cfg.get("probe") or {}
    path = list(probe.get("path") or [])
    rule = str(probe.get("rule") or "list_nonempty")
    if not path:
        return False
    val = _dig(sample, path)
    if rule == "list_nonempty":
        return isinstance(val, list) and len(val) > 0
    if rule == "truthy":
        return bool(val)
    return val is not None


def strip_retrieval_fields(sample: dict, cfg: dict | None = None) -> dict:
    """Shallow copy with retrieval fields cleared so the norag branch is used."""
    cfg = cfg or load_map()
    out = dict(sample)
    out.pop("knowledges_for_llm", None)
    path = list((cfg.get("probe") or {}).get("path") or [])
    if not path:
        return out
    # Mutate nested copy along probe path → empty list at leaf
    root_key = path[0]
    if root_key not in out or not isinstance(out.get(root_key), dict):
        return out
    nested = copy.deepcopy(out[root_key])
    cur: Any = nested
    for key in path[1:-1]:
        if not isinstance(cur, dict) or key not in cur:
            out[root_key] = nested
            return out
        cur = cur[key]
    if isinstance(cur, dict) and path[-1] in cur:
        cur[path[-1]] = []
    out[root_key] = nested
    return out


def protocol_slot(protocol: str, cfg: dict | None = None) -> str:
    cfg = cfg or load_map()
    if protocol == PROTOCOL_NO_PMT:
        return str(cfg.get("solo_slot") or "2")
    if protocol == PROTOCOL_PMT_NORAG:
        return str(cfg.get("default_slot") or "0")
    if protocol == PROTOCOL_PMT_RAG:
        return str(cfg.get("hit_slot") or "1")
    raise ValueError(f"unknown protocol: {protocol}")


def protocol_needs_pmt(protocol: str) -> bool:
    return protocol in (PROTOCOL_PMT_NORAG, PROTOCOL_PMT_RAG)


def default_out_dir(protocol: str) -> Path:
    if protocol not in DEFAULT_OUT:
        raise ValueError(f"protocol {protocol} has no PhaseModulationTransformer out_dir")
    return DEFAULT_OUT[protocol]


def resolve_pmt_init_ckpt(explicit: str | None = None) -> Path:
    """Training always starts from the released pretrained PhaseModulationTransformer weights."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {p}")
        return p
    if DEFAULT_INIT_CKPT.is_file():
        return DEFAULT_INIT_CKPT
    # Accept pre-rename pack-01 filename if download normalize has not run yet
    alt = ROOT / "parameters" / ("sft_" + "QF" + "_en_1024.pth")
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        f"default init checkpoint missing: {DEFAULT_INIT_CKPT} "
        "(download pack 01 / parameters/sft_PMT_en_1024.pth)"
    )


def resolve_pmt_infer_ckpt(
    protocol: str,
    *,
    explicit: str | None = None,
    out_dir: str | Path | None = None,
    hidden_size: int = 1024,
) -> Path:
    """Inference loads the protocol-specific trained PhaseModulationTransformer (not the init weights)."""
    if not protocol_needs_pmt(protocol):
        raise ValueError(f"protocol {protocol} does not use PhaseModulationTransformer")
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"infer checkpoint not found: {p}")
        return p
    base = Path(out_dir) if out_dir else default_out_dir(protocol)
    stems = [f"cascade_PMT_en_{hidden_size}", f"cascade_{'QF'}_en_{hidden_size}"]
    for stem in stems:
        best = base / f"{stem}_best.pth"
        latest = base / f"{stem}.pth"
        if best.is_file():
            return best
        if latest.is_file():
            return latest
    raise FileNotFoundError(
        f"no trained PhaseModulationTransformer for protocol={protocol} under {base} "
        f"(expected cascade_PMT_en_{hidden_size}_best.pth; train with --protocol {protocol} first)"
    )


def prepare_sample_for_protocol(sample: dict, protocol: str, cfg: dict | None = None) -> dict | None:
    """Return a sample view for the protocol, or None to skip."""
    cfg = cfg or load_map()
    if protocol == PROTOCOL_NO_PMT:
        return sample
    if protocol == PROTOCOL_PMT_NORAG:
        return strip_retrieval_fields(sample, cfg)
    if protocol == PROTOCOL_PMT_RAG:
        if not probe_has_retrieval(sample, cfg):
            return None
        return sample
    raise ValueError(f"unknown protocol: {protocol}")

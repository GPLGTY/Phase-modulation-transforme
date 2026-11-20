"""Single RAG protocol for the Phase-modulation-transformer stack (ckpt_0 + PMT).

Always uses retrieval-augmented samples. Slot wiring lives in sidecar_map.json.
"""

from __future__ import annotations

from pathlib import Path

from model.sidecar_runtime import _dig, load_map

ROOT = Path(__file__).resolve().parents[1]

# Kept as the only public protocol name for CLI compatibility.
PROTOCOL_PMT_RAG = "pmt_rag"
ALL_PROTOCOLS = (PROTOCOL_PMT_RAG,)

DEFAULT_OUT = {
    PROTOCOL_PMT_RAG: ROOT / "out" / "run_pmt",
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


def protocol_slot(protocol: str = PROTOCOL_PMT_RAG, cfg: dict | None = None) -> str:
    cfg = cfg or load_map()
    if protocol != PROTOCOL_PMT_RAG:
        raise ValueError(f"unsupported protocol: {protocol} (only {PROTOCOL_PMT_RAG})")
    return str(cfg.get("hit_slot") or cfg.get("default_slot") or "0")


def protocol_needs_pmt(protocol: str = PROTOCOL_PMT_RAG) -> bool:
    return protocol == PROTOCOL_PMT_RAG


def default_out_dir(protocol: str = PROTOCOL_PMT_RAG) -> Path:
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
    alt = ROOT / "parameters" / ("sft_" + "QF" + "_en_1024.pth")
    if alt.is_file():
        return alt
    raise FileNotFoundError(
        f"default init checkpoint missing: {DEFAULT_INIT_CKPT} "
        "(download pack 01 / parameters/sft_PMT_en_1024.pth)"
    )


def resolve_pmt_infer_ckpt(
    protocol: str = PROTOCOL_PMT_RAG,
    *,
    explicit: str | None = None,
    out_dir: str | Path | None = None,
    hidden_size: int = 1024,
) -> Path:
    """Inference loads the trained PhaseModulationTransformer (not the init weights)."""
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
        f"no trained PhaseModulationTransformer under {base} "
        f"(expected cascade_PMT_en_{hidden_size}_best.pth; train first)"
    )


def prepare_sample_for_protocol(
    sample: dict, protocol: str = PROTOCOL_PMT_RAG, cfg: dict | None = None
) -> dict | None:
    """Return sample if it has retrieval context; otherwise skip."""
    if protocol != PROTOCOL_PMT_RAG:
        raise ValueError(f"unsupported protocol: {protocol}")
    cfg = cfg or load_map()
    if not probe_has_retrieval(sample, cfg):
        return None
    return sample


def allocate_result_summary_dir(results_root: Path | None = None) -> Path:
    """results/result_summary if free; else result_summary1, result_summary2, ..."""
    root = results_root or (ROOT / "results")
    base = root / "result_summary"
    if not base.exists():
        return base
    i = 1
    while (root / f"result_summary{i}").exists():
        i += 1
    return root / f"result_summary{i}"

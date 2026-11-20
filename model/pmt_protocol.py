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


# Columns for results/*/aggregated_accuracy_summary.csv (no checkpoint/data_path/note/table_dir).
AGGREGATED_SUMMARY_FIELDS: tuple[str, ...] = (
    "dataset",
    "train_with_position_ntk_alpha",
    "inference_with_position_ntk_alpha",
    "source_dir",
    "copied_dir",
    "temperature",
    "ntk_alpha",
    "n_total",
    "n_scored",
    "n_match",
    "accuracy",
    "elapsed_sec",
    "output_csv",
)


def upsert_aggregated_accuracy_summary(
    result_root: Path,
    *,
    dataset: str,
    train_with_position_ntk_alpha: bool | str,
    inference_with_position_ntk_alpha: bool | str,
    copied_dir: str,
    summary_rows: list[dict],
    source_dir: str | None = None,
) -> Path:
    """Replace rows for one ablation table in aggregated_accuracy_summary.csv."""
    import csv

    def _flag(v: bool | str) -> str:
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes"):
            return "True"
        if s in ("0", "false", "f", "no"):
            return "False"
        return str(v)

    agg_path = Path(result_root) / "aggregated_accuracy_summary.csv"
    train_f = _flag(train_with_position_ntk_alpha)
    infer_f = _flag(inference_with_position_ntk_alpha)
    src = source_dir or f"results/{copied_dir}"

    existing: list[dict] = []
    if agg_path.is_file() and agg_path.stat().st_size > 0:
        with agg_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("copied_dir") or "") == copied_dir:
                    continue
                if (row.get("dataset") or "") == dataset and _flag(
                    row.get("train_with_position_ntk_alpha") or ""
                ) == train_f and _flag(
                    row.get("inference_with_position_ntk_alpha") or ""
                ) == infer_f:
                    continue
                existing.append({k: row.get(k, "") for k in AGGREGATED_SUMMARY_FIELDS})

    new_rows: list[dict] = []
    for row in summary_rows:
        out_csv = str(row.get("output_csv") or "")
        # Prefer path under results/result_summary...
        if out_csv and not out_csv.replace("\\", "/").startswith("results/"):
            out_csv = f"results/{Path(result_root).name}/{copied_dir}/{Path(out_csv).name}"
        new_rows.append(
            {
                "dataset": dataset,
                "train_with_position_ntk_alpha": train_f,
                "inference_with_position_ntk_alpha": infer_f,
                "source_dir": src,
                "copied_dir": copied_dir,
                "temperature": row.get("temperature", ""),
                "ntk_alpha": row.get("ntk_alpha", ""),
                "n_total": row.get("n_total", ""),
                "n_scored": row.get("n_scored", ""),
                "n_match": row.get("n_match", ""),
                "accuracy": row.get("accuracy", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "output_csv": out_csv,
            }
        )

    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with agg_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(AGGREGATED_SUMMARY_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(existing + new_rows)
    return agg_path

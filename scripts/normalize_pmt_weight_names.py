#!/usr/bin/env python3
"""Rename pack-01 weight paths to Phase-modulation-transforme branding."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pairs = [
        (ROOT / "parameters" / "sft_QF_en_1024.pth", ROOT / "parameters" / "sft_PMT_en_1024.pth"),
        (ROOT / "parameters" / "QF_Base_300M", ROOT / "parameters" / "PMT_Base_300M"),
        (ROOT / "parameters" / "QF_Struct_300M", ROOT / "parameters" / "PMT_Struct_300M"),
    ]
    for src, dst in pairs:
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"renamed {src.name} -> {dst.name}", flush=True)


if __name__ == "__main__":
    main()

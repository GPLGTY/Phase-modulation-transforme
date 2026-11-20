#!/usr/bin/env python3
"""Redirect to project scripts/run_ablation_temp_ntk.py (RAG JSON + integrated stack)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts" / "run_ablation_temp_ntk.py"), run_name="__main__")

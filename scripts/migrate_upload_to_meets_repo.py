#!/usr/bin/env python3
"""Upload local packs + ckpts to the renamed ModelScope repo.

Repo: zyxl2026/Phase-Modulation-Meets-Retrieval-Augmented-Transformers
Requires MODELSCOPE_API_TOKEN.
  UPLOAD_PACKS=1 (default)  UPLOAD_CKPTS=1 (default)
  SLOTS=ckpt_a,ckpt_b,ckpt_c
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "zyxl2026/Phase-Modulation-Meets-Retrieval-Augmented-Transformers"

PACKS = [
    ROOT / ".modelscope_publish_assets" / "packs" / "01_parameters.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "03b_corpus_openhermes2_5.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "03d_corpus_sft_mini_512_zh.tar",
    ROOT / ".modelscope_publish" / "packs" / "04_medqa_rag_json.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "05_embedding_stella.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "06_textbooks.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "07_vector_stores.tar",
    ROOT / ".modelscope_publish_assets" / "packs" / "08_questions.tar",
]

CKPT_SLOTS = [
    (ROOT / "parameters" / "ckpt_a", "parameters/ckpt_a"),
    (ROOT / "parameters" / "ckpt_b", "parameters/ckpt_b"),
    (ROOT / "parameters" / "ckpt_c", "parameters/ckpt_c"),
]


def token() -> str:
    t = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not t:
        raise SystemExit("MODELSCOPE_API_TOKEN not set")
    return t.strip()


def main() -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    do_packs = os.environ.get("UPLOAD_PACKS", "1") == "1"
    do_ckpts = os.environ.get("UPLOAD_CKPTS", "1") == "1"

    print(f"target repo: {REPO_ID}", flush=True)

    if do_packs:
        for p in PACKS:
            if not p.is_file() or p.stat().st_size < 1000:
                print(f"[skip pack] {p}", flush=True)
                continue
            print(f"[upload pack] {p.name} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
                repo_id=REPO_ID,
                commit_message=f"add/update {p.name}",
                token=token(),
            )
            print(f"[done pack] {p.name}", flush=True)

    if do_ckpts:
        want = {
            s.strip()
            for s in (os.environ.get("SLOTS") or "ckpt_a,ckpt_b,ckpt_c").split(",")
            if s.strip()
        }
        for local_dir, path_in_repo in CKPT_SLOTS:
            if local_dir.name not in want:
                continue
            if not local_dir.is_dir():
                print(f"[skip ckpt] missing {local_dir}", flush=True)
                continue
            print(f"[upload ckpt] {local_dir} -> {path_in_repo}", flush=True)
            api.upload_folder(
                repo_id=REPO_ID,
                folder_path=str(local_dir),
                path_in_repo=path_in_repo,
                commit_message=f"add {path_in_repo}",
                token=token(),
            )
            print(f"[done ckpt] {path_in_repo}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

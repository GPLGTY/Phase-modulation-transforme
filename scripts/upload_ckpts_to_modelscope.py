#!/usr/bin/env python3
"""Upload parameters/ckpt_a|ckpt_b|ckpt_c to the project ModelScope repo.

Auth: set env MODELSCOPE_API_TOKEN (do NOT commit the token).

  set MODELSCOPE_API_TOKEN=...
  python scripts/upload_ckpts_to_modelscope.py
  # only one slot:
  SLOTS=ckpt_c python scripts/upload_ckpts_to_modelscope.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
ALL_SLOTS = [
    (ROOT / "parameters" / "ckpt_a", "parameters/ckpt_a"),
    (ROOT / "parameters" / "ckpt_b", "parameters/ckpt_b"),
    (ROOT / "parameters" / "ckpt_c", "parameters/ckpt_c"),
]


def main() -> None:
    token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not token:
        print("Set MODELSCOPE_API_TOKEN before upload.", file=sys.stderr)
        sys.exit(1)

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    want = {
        s.strip()
        for s in (os.environ.get("SLOTS") or "ckpt_a,ckpt_b,ckpt_c").split(",")
        if s.strip()
    }
    local_slots = [(d, p) for d, p in ALL_SLOTS if d.name in want]

    for local_dir, path_in_repo in local_slots:
        if not local_dir.is_dir():
            print(f"missing {local_dir}, skip", flush=True)
            continue
        print(f"upload {local_dir} -> {REPO_ID}:{path_in_repo}", flush=True)
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            commit_message=f"add {path_in_repo}",
            token=token,
        )
        print(f"done {path_in_repo}", flush=True)

    print("upload finished", flush=True)


if __name__ == "__main__":
    main()

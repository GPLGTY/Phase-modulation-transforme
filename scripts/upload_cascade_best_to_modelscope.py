#!/usr/bin/env python3
"""Upload best cascade PMT checkpoint to ModelScope as pack 09.

Auth: set MODELSCOPE_API_TOKEN (do NOT commit the token).

  set MODELSCOPE_API_TOKEN=...
  python scripts/upload_cascade_best_to_modelscope.py

Uploads ``09_cascade_pmt_best.tar`` which extracts to:
  out/run_pmt/cascade_PMT_en_1024_best.pth
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
SRC = ROOT / "out" / "run_pmt" / "cascade_PMT_en_1024_best.pth"
PACK_NAME = "09_cascade_pmt_best.tar"
ARCNAME = "out/run_pmt/cascade_PMT_en_1024_best.pth"


def token() -> str:
    t = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not t:
        raise SystemExit("MODELSCOPE_API_TOKEN not set")
    return t.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"missing checkpoint: {SRC}")

    size_mb = SRC.stat().st_size / (1024 * 1024)
    digest = sha256_file(SRC)
    print(f"src={SRC}", flush=True)
    print(f"size_mb={size_mb:.1f}", flush=True)
    print(f"sha256={digest}", flush=True)

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())

    work = Path(tempfile.mkdtemp(prefix="ms_pack09_"))
    tar_path = work / PACK_NAME
    print(f"[1/2] build {PACK_NAME}", flush=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(SRC, arcname=ARCNAME)
    print(f"  tar_mb={tar_path.stat().st_size / (1024 * 1024):.1f}", flush=True)

    print(f"[2/2] upload {PACK_NAME} → {REPO_ID}", flush=True)
    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo=PACK_NAME,
        repo_id=REPO_ID,
        commit_message=(
            "add 09_cascade_pmt_best.tar "
            "(best retrieval-augmented PMT; recommended temperature=0.7, ntk_alpha=0.4)"
        ),
        token=token(),
    )
    print(f"  uploaded {PACK_NAME}", flush=True)

    note = ROOT / ".modelscope_publish" / "09_cascade_pmt_best.sha256"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(f"{digest}  {ARCNAME}\n", encoding="utf-8")
    print(f"wrote {note}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publish MedQA/MedMCQA question json(l) as pack 08 for remote retrieval rebuild.

Layout:
  dataset/pipeline_data/questions/{US,Mainland,MedMCQA}/

Requires MODELSCOPE_API_TOKEN.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(r"g:\mycodes\data_clean_gp\questions")
REPO_ID = "GPLYMX/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
WORK = ROOT / ".modelscope_publish_assets"
PACK_DIR = WORK / "packs"
CLONE_DIR = WORK / "repo_manifest_08"
DST = ROOT / "dataset" / "pipeline_data" / "questions"

# Pipeline-facing files only (no metamap / checkpoints / Taiwan extras)
KEEP = {
    "US": ["train.jsonl", "dev.jsonl", "test.jsonl"],
    "Mainland": ["train.jsonl", "dev.jsonl", "test.jsonl"],
    "MedMCQA": ["train.json", "dev.json", "test.json"],
}


def token() -> str:
    t = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not t:
        raise SystemExit("MODELSCOPE_API_TOKEN not set")
    return t.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage() -> Path:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for split, names in KEEP.items():
        src_dir = SRC / split
        dst_dir = DST / split
        dst_dir.mkdir(parents=True)
        for name in names:
            src = src_dir / name
            if not src.is_file():
                raise SystemExit(f"missing {src}")
            print(f"[cp] {split}/{name}", flush=True)
            shutil.copy2(src, dst_dir / name)
    return DST


def make_tar(questions_dir: Path) -> Path:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    out = PACK_DIR / "08_questions.tar"
    if out.exists():
        out.unlink()
    print(f"[tar] {out}", flush=True)
    n = 0
    with tarfile.open(out, "w") as tf:
        for p in questions_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(questions_dir)
                tf.add(p, arcname=f"dataset/pipeline_data/questions/{rel.as_posix()}")
                n += 1
    print(f"[tar] files={n} sizeMB={out.stat().st_size / 1e6:.1f}", flush=True)
    return out


def upload(path: Path) -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    print(f"[upload] {path.name}", flush=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=REPO_ID,
        commit_message=f"add/update {path.name}",
        token=token(),
    )
    print(f"[upload] done {path.name}", flush=True)


def update_manifest(pack: Path) -> None:
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
    CLONE_DIR.mkdir(parents=True)
    url = f"https://oauth2:{token()}@www.modelscope.cn/{REPO_ID}.git"
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", url, str(CLONE_DIR)], check=True, env=env)

    manifest = [
        "# ModelScope asset manifest (public protocol)",
        "",
        "- `01_parameters.tar`: pretrained PhaseModulationTransforme init weights",
        "- `04_medqa_rag_json.tar`: MedQA retrieval samples",
        "- `05_embedding_stella.tar`: Stella embeddings under `dataset/pipeline/models/`",
        "- `06_textbooks.tar`: MedQA textbooks under `dataset/pipeline_data/textbooks/`",
        "- `07_vector_stores.tar`: FAISS indices under `dataset/pipeline_data/vector_stores/`",
        "- `08_questions.tar`: MedQA/MedMCQA questions under `dataset/pipeline_data/questions/`",
        "- `03b` / `03d`: optional SFT corpora",
        "",
        "Retrieval rebuild needs packs `05`+`06`+`07`+`08`",
        "(or `05`+`06`+`08` then run `vector_store.py` locally).",
        "Language-matched stores include English `en/stella-base-en-v2`.",
        "",
    ]
    (CLONE_DIR / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

    existing: dict[str, str] = {}
    sha_path = CLONE_DIR / "SHA256SUMS.txt"
    if sha_path.is_file():
        for line in sha_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                existing[parts[-1]] = parts[0]
    existing[pack.name] = sha256_file(pack)

    lines = ["# SHA256 of root archive packs", ""]
    for name in [
        "01_parameters.tar",
        "03b_corpus_openhermes2_5.tar",
        "03d_corpus_sft_mini_512_zh.tar",
        "04_medqa_rag_json.tar",
        "05_embedding_stella.tar",
        "06_textbooks.tar",
        "07_vector_stores.tar",
        "08_questions.tar",
    ]:
        if name in existing:
            lines.append(f"{existing[name]}  {name}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "add", "MANIFEST.md", "SHA256SUMS.txt"],
        check=True,
    )
    st = subprocess.run(
        ["git", "-C", str(CLONE_DIR), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not st:
        print("[git] nothing to commit", flush=True)
        return
    print(st, flush=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(CLONE_DIR),
            "commit",
            "-m",
            "docs: add pack 08 questions for remote retrieval rebuild",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(CLONE_DIR), "push", "origin", "HEAD"], check=True, env=env)
    print("[git] push ok", flush=True)


def verify() -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    files = api.get_model_files(REPO_ID, recursive=False)
    names = {(f.get("Path") or f.get("path") or "") for f in files if isinstance(f, dict)}
    if "08_questions.tar" not in names:
        raise SystemExit("missing remote 08_questions.tar")
    print("remote 08_questions.tar: OK", flush=True)


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"missing source: {SRC}")
    qs = stage()
    pack = make_tar(qs)
    upload(pack)
    update_manifest(pack)
    verify()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

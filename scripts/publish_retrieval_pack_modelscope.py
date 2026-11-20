#!/usr/bin/env python3
"""Build/upload MedQA retrieval pack (package 04) to ModelScope.

Requires MODELSCOPE_API_TOKEN. Does not print the token.
Builds from local ``dataset/RAG_MedQA_US_*.json`` (retrieval schema).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
WORK = ROOT / ".modelscope_publish"
PACK_DIR = WORK / "packs"
STRIP_DIR = WORK / "stripped"
CLONE_DIR = WORK / "repo_sparse"

KEEP_KEYS = (
    "id",
    "question",
    "options",
    "answer",
    "answer_idx",
    "meta_info",
    "question_with_options",
    "answer_with_idx",
    "retrieve_info",
)

CANONICAL = (
    "RAG_MedQA_US_train_10178.json",
    "RAG_MedQA_US_dev_1272.json",
    "RAG_MedQA_US_test_1273.json",
)

NEW_PACK = "04_medqa_rag_json.tar"


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


def strip_sample(sample: dict) -> dict:
    out = {k: sample[k] for k in KEEP_KEYS if k in sample}
    if "answer_idx" not in out and sample.get("gold_letter"):
        out["answer_idx"] = sample["gold_letter"]
    if "question_with_options" not in out and sample.get("question"):
        q = str(sample.get("question") or "")
        opts = sample.get("options") or {}
        if isinstance(opts, dict) and opts:
            out["question_with_options"] = "\n".join(
                [q] + [f"{k}. {v}" for k, v in opts.items()]
            )
    return out


def build_new_pack() -> Path:
    print("[1/4] build retrieval JSON tar from local dataset/", flush=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    if STRIP_DIR.exists():
        shutil.rmtree(STRIP_DIR)
    out_root = STRIP_DIR / "dataset"
    out_root.mkdir(parents=True)

    for name in CANONICAL:
        src = ROOT / "dataset" / name
        if not src.is_file():
            raise SystemExit(f"missing local sample file: {src}")
        raw = json.loads(src.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit(f"expect list in {src}")
        cleaned = [strip_sample(x) for x in raw if isinstance(x, dict)]
        if any("messages" in x for x in cleaned):
            raise SystemExit("messages still present after strip")
        dst = out_root / name
        dst.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  dataset/{name} n={len(cleaned)}", flush=True)

    new_tar = PACK_DIR / NEW_PACK
    if new_tar.exists():
        new_tar.unlink()
    with tarfile.open(new_tar, "w") as tf:
        for p in sorted(out_root.glob("*.json")):
            tf.add(p, arcname=f"dataset/{p.name}")
    print(f"  wrote {new_tar} ({new_tar.stat().st_size/1e6:.1f} MB)", flush=True)
    return new_tar


def upload_new_pack(new_tar: Path) -> None:
    from modelscope.hub.api import HubApi

    print(f"[2/4] upload {NEW_PACK}", flush=True)
    api = HubApi()
    api.login(token())
    api.upload_file(
        path_or_fileobj=str(new_tar),
        path_in_repo=NEW_PACK,
        repo_id=REPO_ID,
        commit_message=f"add {NEW_PACK} (MedQA retrieval samples)",
        token=token(),
    )
    print("  upload ok", flush=True)


def git_update_manifest(new_tar: Path) -> None:
    print("[3/4] git clone (skip LFS) and refresh checksums", flush=True)
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
    CLONE_DIR.mkdir(parents=True)

    url = f"https://oauth2:{token()}@www.modelscope.cn/{REPO_ID}.git"
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(CLONE_DIR)],
        check=True,
        env=env,
    )

    shutil.copy2(new_tar, CLONE_DIR / NEW_PACK)

    packs = sorted(
        [p for p in CLONE_DIR.glob("*.tar") if p.is_file() and p.stat().st_size > 0]
    )
    lines = ["# SHA256 of root archive packs", ""]
    manifest = [
        "# ModelScope asset manifest",
        "",
        f"- `{NEW_PACK}`: MedQA retrieval samples (`generate_question_with_knowledges.py` schema)",
        "- `01_parameters.tar`: pretrained PhaseModulationTransformer init weights",
        "- `03b_corpus_openhermes2_5.tar` / `03d_corpus_sft_mini_512_zh.tar`: optional SFT corpora",
        "- `05_embedding_stella.tar`: retrieval embeddings",
        "",
    ]
    (CLONE_DIR / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

    for p in packs:
        if p.stat().st_size < 1_000_000:
            continue
        digest = sha256_file(p)
        lines.append(f"{digest}  {p.name}")
    digest = sha256_file(CLONE_DIR / NEW_PACK)
    if f"{digest}  {NEW_PACK}" not in lines:
        lines.append(f"{digest}  {NEW_PACK}")
    (CLONE_DIR / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    subprocess.run(["git", "-C", str(CLONE_DIR), "add", "-A"], check=True)
    status = subprocess.run(
        ["git", "-C", str(CLONE_DIR), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not status:
        print("  nothing to commit", flush=True)
        return
    print(status, flush=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(CLONE_DIR),
            "commit",
            "-m",
            "refresh MedQA retrieval pack checksums",
        ],
        check=True,
    )
    print("[4/4] git push", flush=True)
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "push", "origin", "HEAD"],
        check=True,
        env=env,
    )
    print("  push ok", flush=True)


def verify_remote() -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    files = api.get_model_files(REPO_ID, recursive=False)
    names = sorted(
        {(f.get("Path") or f.get("path") or "") for f in files if isinstance(f, dict)}
    )
    print("remote root files:", flush=True)
    for n in names:
        if n and "/" not in n:
            print(" ", n, flush=True)
    if NEW_PACK not in names:
        raise SystemExit(f"{NEW_PACK} missing on remote")
    print("verify ok", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    new_tar = build_new_pack()
    upload_new_pack(new_tar)
    git_update_manifest(new_tar)
    verify_remote()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

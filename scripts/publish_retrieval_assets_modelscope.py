#!/usr/bin/env python3
"""Publish textbooks + embeddings + vector stores to ModelScope (option B).

Layout (project-relative):
  dataset/pipeline/models/                 ← Stella embeddings (pack 05)
  dataset/pipeline_data/textbooks/         ← textbooks txt/json (pack 06)
  dataset/pipeline_data/vector_stores/     ← FAISS indices (pack 07)

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
SRC = Path(r"g:\mycodes\data_clean_gp")
REPO_ID = "zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
WORK = ROOT / ".modelscope_publish_assets"
PACK_DIR = WORK / "packs"
STAGE = WORK / "stage"
CLONE_DIR = WORK / "repo_sparse"

# Official pipeline language-matched indices only
VECTOR_KEEP = [
    ("en", "stella-base-en-v2"),
    ("zh_paragraph", "stella-base-zh-v2"),
    ("zh_sentence", "stella-base-zh-v2"),
]

TEXTBOOK_SUBDIRS = [
    "en",
    "en_json",
    "zh_paragraph",
    "zh_paragraph_json",
    "zh_sentence",
    "zh_sentence_json",
]


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


def ensure_embeddings() -> Path:
    """Ensure Stella weights under dataset/pipeline/models; prefer ModelScope pack 05."""
    models = ROOT / "dataset" / "pipeline" / "models"
    en = models / "infgrad" / "stella-base-en-v2"
    zh = models / "infgrad" / "stella-base-zh-v2"
    if en.is_dir() and zh.is_dir() and any(en.iterdir()) and any(zh.iterdir()):
        print(f"[emb] already present: {models}", flush=True)
        return models

    from modelscope import snapshot_download

    PACK_DIR.mkdir(parents=True, exist_ok=True)
    print("[emb] download 05_embedding_stella.tar from project ModelScope", flush=True)
    try:
        snapshot_download(
            REPO_ID,
            local_dir=str(PACK_DIR / "dl05"),
            allow_patterns=["05_embedding_stella.tar"],
        )
        tar_path = next((PACK_DIR / "dl05").rglob("05_embedding_stella.tar"))
        print(f"[emb] extract {tar_path}", flush=True)
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(ROOT)
    except Exception as e:
        print(f"[emb] pack 05 failed ({e}); fallback HuggingFace/ModelScope model ids", flush=True)
        models.mkdir(parents=True, exist_ok=True)
        for mid, dest in [
            ("AI-ModelScope/stella-base-en-v2", en),
            ("AI-ModelScope/stella-base-zh-v2", zh),
            ("infgrad/stella-base-en-v2", en),
            ("infgrad/stella-base-zh-v2", zh),
        ]:
            if dest.is_dir() and any(dest.iterdir()):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                print(f"[emb] snapshot_download {mid} → {dest}", flush=True)
                snapshot_download(mid, local_dir=str(dest))
            except Exception as e2:
                print(f"[emb] skip {mid}: {e2}", flush=True)

    if not (en.is_dir() and any(en.iterdir())):
        raise SystemExit(f"missing embedding: {en}")
    if not (zh.is_dir() and any(zh.iterdir())):
        raise SystemExit(f"missing embedding: {zh}")
    return models


def stage_textbooks() -> Path:
    dst_root = ROOT / "dataset" / "pipeline_data" / "textbooks"
    print(f"[tb] sync textbooks → {dst_root}", flush=True)
    dst_root.mkdir(parents=True, exist_ok=True)
    for sub in TEXTBOOK_SUBDIRS:
        src = SRC / "textbooks" / sub
        dst = dst_root / sub
        if not src.is_dir():
            print(f"[tb] missing {src}, skip", flush=True)
            continue
        if dst.exists():
            shutil.rmtree(dst)
        print(f"[tb] copy {sub} ...", flush=True)
        shutil.copytree(src, dst)
    return dst_root


def stage_vector_stores() -> Path:
    dst_root = ROOT / "dataset" / "pipeline_data" / "vector_stores"
    print(f"[vs] sync language-matched stores → {dst_root}", flush=True)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    for store_name, emb_name in VECTOR_KEEP:
        src = SRC / "vector_stores" / store_name / emb_name
        dst = dst_root / store_name / emb_name
        if not src.is_dir():
            raise SystemExit(f"missing vector store: {src}")
        print(f"[vs] copy {store_name}/{emb_name} ...", flush=True)
        shutil.copytree(src, dst)
    return dst_root


def make_tar(out_path: Path, base: Path, arc_prefix: str) -> Path:
    """Tar all files under base with arcnames arc_prefix/..."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    print(f"[tar] {out_path.name} from {base}", flush=True)
    n = 0
    with tarfile.open(out_path, "w") as tf:
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(base)
                tf.add(p, arcname=f"{arc_prefix}/{rel.as_posix()}")
                n += 1
    print(f"[tar] {out_path.name}: files={n} sizeMB={out_path.stat().st_size/1e6:.1f}", flush=True)
    return out_path


def pack_embeddings(models_dir: Path) -> Path:
    # Only ship en/zh base models used by US/Mainland pipelines
    stage = STAGE / "emb_stage" / "dataset" / "pipeline" / "models"
    if stage.parent.parent.exists():
        shutil.rmtree(STAGE / "emb_stage")
    for rel in [
        Path("infgrad") / "stella-base-en-v2",
        Path("infgrad") / "stella-base-zh-v2",
    ]:
        src = models_dir / rel
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return make_tar(
        PACK_DIR / "05_embedding_stella.tar",
        STAGE / "emb_stage" / "dataset" / "pipeline" / "models",
        "dataset/pipeline/models",
    )


def pack_textbooks(tb_dir: Path) -> Path:
    return make_tar(
        PACK_DIR / "06_textbooks.tar",
        tb_dir,
        "dataset/pipeline_data/textbooks",
    )


def pack_vector_stores(vs_dir: Path) -> Path:
    return make_tar(
        PACK_DIR / "07_vector_stores.tar",
        vs_dir,
        "dataset/pipeline_data/vector_stores",
    )


def upload_packs(paths: list[Path]) -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    for p in paths:
        print(f"[upload] {p.name}", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=REPO_ID,
            commit_message=f"add/update {p.name}",
            token=token(),
        )
        print(f"[upload] done {p.name}", flush=True)


def git_update_manifest(pack_paths: list[Path]) -> None:
    print("[git] clone sparse (skip LFS) + update MANIFEST/SHA256", flush=True)
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

    # Packs are already on Hub via upload_file; only refresh docs here.
    manifest = [
        "# ModelScope asset manifest (public protocol)",
        "",
        "- `01_parameters.tar`: pretrained PhaseModulationTransformer init weights",
        "- `04_medqa_rag_json.tar`: MedQA retrieval samples",
        "- `05_embedding_stella.tar`: Stella embeddings under `dataset/pipeline/models/`",
        "- `06_textbooks.tar`: MedQA textbooks under `dataset/pipeline_data/textbooks/`",
        "- `07_vector_stores.tar`: FAISS indices under `dataset/pipeline_data/vector_stores/`",
        "- `03b` / `03d`: optional SFT corpora",
        "",
        "Retrieval rebuild needs packs `05`+`06`+`07` (or run vector_store.py after `05`+`06`).",
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
    for p in pack_paths:
        existing[p.name] = sha256_file(p)
    lines = ["# SHA256 of root archive packs", ""]
    for name in [
        "01_parameters.tar",
        "03b_corpus_openhermes2_5.tar",
        "03d_corpus_sft_mini_512_zh.tar",
        "04_medqa_rag_json.tar",
        "05_embedding_stella.tar",
        "06_textbooks.tar",
        "07_vector_stores.tar",
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
            "add textbooks, embeddings, and FAISS vector stores for retrieval rebuild",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(CLONE_DIR), "push", "origin", "HEAD"],
        check=True,
        env=env,
    )
    print("[git] push ok", flush=True)


def verify() -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token())
    files = api.get_model_files(REPO_ID, recursive=False)
    names = {
        (f.get("Path") or f.get("path") or "")
        for f in files
        if isinstance(f, dict)
    }
    for need in ("05_embedding_stella.tar", "06_textbooks.tar", "07_vector_stores.tar"):
        print(f"remote {need}: {need in names}", flush=True)
        if need not in names:
            raise SystemExit(f"missing remote {need}")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"missing source: {SRC}")
    WORK.mkdir(parents=True, exist_ok=True)
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    models = ensure_embeddings()
    tb = stage_textbooks()
    vs = stage_vector_stores()

    p05 = pack_embeddings(models)
    p06 = pack_textbooks(tb)
    p07 = pack_vector_stores(vs)

    upload_packs([p05, p06, p07])
    git_update_manifest([p05, p06, p07])
    verify()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

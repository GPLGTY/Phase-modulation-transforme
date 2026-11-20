#!/usr/bin/env python3
"""After new-repo migrate finishes: delete from OLD only what already exists on NEW.

Keeps anything present only on the old repo (user's newer uploads).
Requires MODELSCOPE_API_TOKEN.
  DRY_RUN=1 (default)  — list actions only
  DRY_RUN=0            — actually delete
"""

from __future__ import annotations

import os
import sys

OLD_REPO = "zyxl2026/Phase-Modulation-Retrieval-Augmented-Transformer-for-Medical-Question-Answering"
NEW_REPO = "zyxl2026/Phase-Modulation-Meets-Retrieval-Augmented-Transformers"

# Never delete these even if also on new.
KEEP_ALWAYS = {
    ".gitattributes",
    "README.md",
    "configuration.json",
}


def token() -> str:
    t = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
    if not t:
        raise SystemExit("MODELSCOPE_API_TOKEN not set")
    return t.strip()


def file_map(api, repo: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in api.get_model_files(model_id=repo, recursive=True) or []:
        if not isinstance(f, dict):
            continue
        path = f.get("Path") or f.get("Name") or ""
        size = int(f.get("Size") or 0)
        if not path or path.endswith("/") or size == 0 and path.count("/") == 0:
            # skip empty directory placeholders when Size==0 and looks like a dir marker
            if size == 0 and not path.rsplit("/", 1)[-1].count("."):
                continue
        if path:
            out[path] = size
    return out


def main() -> None:
    from modelscope.hub.api import HubApi

    dry = os.environ.get("DRY_RUN", "1") != "0"
    api = HubApi()
    api.login(token())

    old = file_map(api, OLD_REPO)
    new = file_map(api, NEW_REPO)
    print(f"old files: {len(old)}  new files: {len(new)}", flush=True)

    to_delete = []
    keep_only_old = []
    keep_policy = []
    for path, size in sorted(old.items()):
        if path in KEEP_ALWAYS:
            keep_policy.append(path)
            continue
        if path in new and new[path] == size:
            to_delete.append(path)
        elif path in new and new[path] != size:
            print(f"[skip size mismatch] {path}: old={size} new={new[path]}", flush=True)
            keep_only_old.append(path)
        else:
            keep_only_old.append(path)

    print(f"\nwill keep (only on old / mismatch): {len(keep_only_old)}", flush=True)
    for p in keep_only_old:
        print(f"  KEEP {p}", flush=True)
    print(f"\nwill keep (policy): {len(keep_policy)}", flush=True)
    for p in keep_policy:
        print(f"  KEEP {p}", flush=True)
    print(f"\nwill delete (mirrored on new): {len(to_delete)}", flush=True)
    for p in to_delete:
        print(f"  DEL  {p}", flush=True)

    if dry:
        print("\nDRY_RUN=1 — no deletes. Re-run with DRY_RUN=0 to apply.", flush=True)
        return

    if not to_delete:
        print("nothing to delete", flush=True)
        return

    # Prefer folder-level deletes when entire ckpt tree is mirrored.
    folders = []
    for folder in ("parameters/ckpt_a", "parameters/ckpt_b", "parameters/ckpt_c"):
        members = [p for p in to_delete if p == folder or p.startswith(folder + "/")]
        all_old = [p for p in old if p == folder or p.startswith(folder + "/")]
        if members and len(members) == len(all_old):
            folders.append(folder)
            to_delete = [p for p in to_delete if p != folder and not p.startswith(folder + "/")]

    for folder in folders:
        print(f"[delete folder] {OLD_REPO}:{folder}", flush=True)
        api.delete_folder(
            repo_id=OLD_REPO,
            path_in_repo=folder,
            commit_message=f"remove migrated {folder} (now on {NEW_REPO})",
            revision="master",
            token=token(),
        )

    for path in to_delete:
        print(f"[delete file] {OLD_REPO}:{path}", flush=True)
        api.delete_file(
            repo_id=OLD_REPO,
            path_in_repo=path,
            commit_message=f"remove migrated {path} (now on {NEW_REPO})",
            revision="master",
            token=token(),
        )

    print("DONE cleanup", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strip obvious source strings from ckpt_a / ckpt_b / ckpt_c metadata after download.

Keeps architecture fields required for loading; rewrites _name_or_path and deletes READMEs.
Does not print or embed original upstream ids in new files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLOTS = {
    "ckpt_a": ROOT / "parameters" / "ckpt_a",
    "ckpt_b": ROOT / "parameters" / "ckpt_b",
    "ckpt_c": ROOT / "parameters" / "ckpt_c",
}

DROP_NAME_RE = re.compile(
    r"(deepseek|llama|qwen|flash|distill|r1|v4)",
    re.IGNORECASE,
)


def scrub_json(path: Path, slot_name: str) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    changed = False
    if isinstance(data, dict):
        for key in list(data.keys()):
            if key in ("_name_or_path", "name_or_path", "model_name", "model_type_name"):
                data[key] = slot_name
                changed = True
            elif key in ("model_name_or_path",):
                data[key] = slot_name
                changed = True
            elif isinstance(data[key], str) and DROP_NAME_RE.search(data[key]):
                # keep architectures / auto_map values that are class paths — only scrub free-form names
                if key in ("architectures", "auto_map"):
                    continue
                if key.endswith("_name") or "name" in key.lower() or key in ("tokenizer_class",):
                    continue
                if key in ("_name_or_path",):
                    data[key] = slot_name
                    changed = True
        if "_name_or_path" in data:
            data["_name_or_path"] = slot_name
            changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"scrubbed {path}")


def main() -> None:
    for slot, directory in SLOTS.items():
        if not directory.is_dir():
            print(f"skip missing {directory}")
            continue
        for readme in directory.rglob("README*"):
            try:
                readme.unlink()
                print(f"removed {readme}")
            except Exception as e:
                print(f"warn remove {readme}: {e}")
        for cfg in directory.rglob("*.json"):
            if cfg.name.endswith(".index.json"):
                continue
            scrub_json(cfg, slot)
    print("scrub done")


if __name__ == "__main__":
    main()

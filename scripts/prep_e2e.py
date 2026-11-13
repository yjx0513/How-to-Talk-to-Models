from __future__ import annotations

import sys
import os
# 把项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List

from src.data.linearize import linearize_e2e
from src.utils import ensure_dir, read_jsonl, write_jsonl

'''mr = {
    "name": "The Eagle",
    "eatType": "coffee shop",
    "food": "French",
    "area": "riverside"
}'''
def parse_mr_e2e(mr_str: str) -> Dict[str, str]:
    """
    Parse standard E2E MR format like:
      name[The Eagle], eatType[coffee shop], food[French], area[riverside]
    """
    mr: Dict[str, str] = {}
    for m in re.finditer(r"([^,\[]+)\[([^\]]*)\]", mr_str):
        k = m.group(1).strip()
        v = m.group(2).strip()
        mr[k] = v
    return mr

'''mr_raw = "name[The Eagle], eatType[coffee shop], food[French], area[riverside]"
ref = "The Eagle is a coffee shop by the riverside that serves French food."'''
def load_e2e_csv(path: str, allow_missing_ref: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # common columns: 'mr' and 'ref'
        for r in reader:
            mr_raw = r.get("mr") or r.get("MR") or ""
            ref = r.get("ref") or r.get("reference") or r.get("text") or r.get("Ref") or ""
            if not mr_raw:
                continue
            if not ref and not allow_missing_ref:
                continue
            rows.append({"mr": parse_mr_e2e(mr_raw), "text": ref})
    return rows


def load_jsonl_generic(path: str) -> List[Dict[str, Any]]:
    # expects {"mr": {...}, "text": "..."}
    return read_jsonl(path)


def convert_split(
    raw_rows: List[Dict[str, Any]],
    out_path: str,
    strategy: str,
    canonical_missing: str,
    allow_empty_target: bool = False,
) -> None:
    out_rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        mr = r["mr"]
        text = r.get("text", "")
        # Skip only if mr is empty, or if text is empty and we don't allow empty targets
        if not mr or (not text and not allow_empty_target):
            continue
        source = linearize_e2e(mr, strategy=strategy, canonical_missing=canonical_missing)
        out_rows.append(
            {
                "source": source,
                "target": text,
                "meta": {"mr": mr},
            }
        )
    write_jsonl(out_path, out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="folder containing raw train/dev/test (csv or jsonl)")
    ap.add_argument("--out_dir", type=str, required=True, help="output folder (processed)")
    ap.add_argument("--strategy", type=str, required=True, choices=["plain", "canonical", "prompt", "tagged"])
    ap.add_argument("--canonical_missing", type=str, default="none", choices=["none", "skip"])
    ap.add_argument("--train", type=str, default="train.csv", help="train filename (csv or jsonl)")
    ap.add_argument("--dev", type=str, default="dev.csv", help="dev filename (csv or jsonl)")
    ap.add_argument("--test", type=str, default="test.csv", help="test filename (csv or jsonl)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    def load_split(fname: str, allow_missing_ref: bool = False) -> List[Dict[str, Any]]:
        p = os.path.join(args.raw_dir, fname)
        if fname.lower().endswith(".csv"):
            return load_e2e_csv(p, allow_missing_ref=allow_missing_ref)
        if fname.lower().endswith(".jsonl"):
            return load_jsonl_generic(p)
        raise ValueError(f"Unsupported file type: {fname}")

    train_rows = load_split(args.train)
    dev_rows = load_split(args.dev)
    test_rows = load_split(args.test, allow_missing_ref=True)

    convert_split(train_rows, os.path.join(args.out_dir, "train.jsonl"), args.strategy, args.canonical_missing)
    convert_split(dev_rows, os.path.join(args.out_dir, "dev.jsonl"), args.strategy, args.canonical_missing)
    convert_split(test_rows, os.path.join(args.out_dir, "test.jsonl"), args.strategy, args.canonical_missing, allow_empty_target=True)

    print(json.dumps({"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}, indent=2))


if __name__ == "__main__":
    main()


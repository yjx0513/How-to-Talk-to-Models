from __future__ import annotations

import sys
import os
# 把项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

from src.data.linearize import Triple, linearize_webnlg
from src.utils import ensure_dir, read_jsonl, write_jsonl


def _as_triples(obj: Any) -> List[Triple]:
    """
    Accepts triples in one of these formats:
      - [["S","R","O"], ...]
      - [{"subj":..,"rel":..,"obj":..}, ...]
      - [["S","R","O","..."]] (extra fields ignored)
    """
    triples: List[Triple] = []
    for t in obj:
        if isinstance(t, dict):
            triples.append((str(t["subj"]), str(t["rel"]), str(t["obj"])))
        else:
            triples.append((str(t[0]), str(t[1]), str(t[2])))
    return triples


def convert_split(
    raw_rows: List[Dict[str, Any]],
    out_path: str,
    strategy: str,
    triple_order: str,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    out_rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        triples = _as_triples(r["triples"])
        text = r["text"]
        source = linearize_webnlg(triples, strategy=strategy, triple_order=triple_order, rng=rng)
        out_rows.append({"source": source, "target": text, "meta": {"triples": triples}})
    write_jsonl(out_path, out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True, help="expects train/dev/test jsonl files with {triples, text}")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--strategy", type=str, required=True, choices=["plain", "canonical", "prompt", "tagged"])
    ap.add_argument("--triple_order", type=str, default="canonical", choices=["canonical", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=str, default="train.jsonl")
    ap.add_argument("--dev", type=str, default="dev.jsonl")
    ap.add_argument("--test", type=str, default="test.jsonl")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    train_rows = read_jsonl(os.path.join(args.raw_dir, args.train))
    dev_rows = read_jsonl(os.path.join(args.raw_dir, args.dev))
    test_rows = read_jsonl(os.path.join(args.raw_dir, args.test))

    convert_split(train_rows, os.path.join(args.out_dir, "train.jsonl"), args.strategy, args.triple_order, seed=args.seed)
    convert_split(dev_rows, os.path.join(args.out_dir, "dev.jsonl"), args.strategy, args.triple_order, seed=args.seed)
    convert_split(test_rows, os.path.join(args.out_dir, "test.jsonl"), args.strategy, args.triple_order, seed=args.seed)

    print(json.dumps({"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)}, indent=2))


if __name__ == "__main__":
    main()


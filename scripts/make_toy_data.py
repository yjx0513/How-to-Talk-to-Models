from __future__ import annotations

import sys
import os
# 把项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from typing import Any, Dict, List

from src.data.linearize import linearize_e2e, linearize_webnlg
from src.utils import ensure_dir, write_jsonl


def make_e2e(out_dir: str, strategy: str) -> None:
    ensure_dir(out_dir)
    examples = [
        ({"name": "The Eagle", "food": "French", "area": "riverside", "priceRange": "cheap"}, "The Eagle is a cheap French restaurant near the riverside."),
        ({"name": "Blue Spice", "food": "Indian", "area": "city centre", "customer rating": "high"}, "Blue Spice is a highly rated Indian place in the city centre."),
        ({"name": "Green Garden", "eatType": "coffee shop", "near": "Burger King"}, "Green Garden is a coffee shop near Burger King."),
    ]
    rows: List[Dict[str, Any]] = []
    for mr, text in examples:
        rows.append({"source": linearize_e2e(mr, strategy=strategy), "target": text, "meta": {"mr": mr}})
    # tiny split
    write_jsonl(os.path.join(out_dir, "train.jsonl"), rows)
    write_jsonl(os.path.join(out_dir, "dev.jsonl"), rows[:2])
    write_jsonl(os.path.join(out_dir, "test.jsonl"), rows[1:])


def make_webnlg(out_dir: str, strategy: str) -> None:
    ensure_dir(out_dir)
    examples = [
        ([("Paris", "country", "France"), ("Paris", "river", "Seine")], "Paris is a city in France and the Seine river runs through it."),
        ([("Albert Einstein", "field", "Physics"), ("Albert Einstein", "bornIn", "Ulm")], "Albert Einstein, born in Ulm, worked in physics."),
        ([("Python", "paradigm", "Object-oriented programming")], "Python supports object-oriented programming."),
    ]
    rows: List[Dict[str, Any]] = []
    for triples, text in examples:
        rows.append({"source": linearize_webnlg(triples, strategy=strategy), "target": text, "meta": {"triples": triples}})
    write_jsonl(os.path.join(out_dir, "train.jsonl"), rows)
    write_jsonl(os.path.join(out_dir, "dev.jsonl"), rows[:2])
    write_jsonl(os.path.join(out_dir, "test.jsonl"), rows[1:])


def main() -> None:
    # Generates small processed datasets so you can sanity-check training without downloading anything.
    make_e2e("data/e2e/processed", strategy="plain")
    make_webnlg("data/webnlg/processed", strategy="plain")
    print("Toy data written to data/e2e/processed and data/webnlg/processed (strategy=plain).")


if __name__ == "__main__":
    main()


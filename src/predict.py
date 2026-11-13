from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from src.eval.factuality_e2e import slot_error_rate
from src.eval.factuality_webnlg import triple_coverage
from src.eval.metrics import compute_text_metrics
from src.utils import read_jsonl, write_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    p.add_argument("--task", type=str, default=None, choices=["e2e", "webnlg"], help="infer if omitted")
    p.add_argument("--data_file", type=str, default=None, help="override JSONL with {source,target,meta...}")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt))
    # Try to infer config from run folder name
    task = args.task
    if task is None:
        task = "e2e" if "runs\\e2e" in args.ckpt or "/runs/e2e" in args.ckpt else "webnlg"

    if args.data_file is None:
        data_root = "data/e2e/processed" if task == "e2e" else "data/webnlg/processed"
        fname = "train.jsonl" if args.split == "train" else "dev.jsonl" if args.split == "dev" else "test.jsonl"
        data_file = os.path.join(data_root, fname)
    else:
        data_file = args.data_file

    rows = read_jsonl(data_file)
    if not rows:
        raise RuntimeError(f"No examples found in {data_file}")

    # Load model via Lightning checkpoint weights
    from src.models.t5_lora import T5LoRALitModule

    lit = T5LoRALitModule.load_from_checkpoint(args.ckpt, decode_cfg=None)  # decode_cfg unused here
    lit.eval()
    lit.freeze()
    model = lit.model
    tokenizer = AutoTokenizer.from_pretrained(lit.model_name, use_fast=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds: List[str] = []
    refs: List[str] = []
    out_rows: List[Dict[str, Any]] = []

    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        sources = [b["source"] for b in batch]
        targets = [b.get("target", "") for b in batch]
        enc = tokenizer(sources, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=3,
        )
        text_preds = [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]
        preds.extend(text_preds)
        refs.extend([t.strip() for t in targets])

        for b, ptxt in zip(batch, text_preds):
            r = dict(b)
            r["pred"] = ptxt
            out_rows.append(r)

    metrics = compute_text_metrics(preds, refs, enable_bertscore=True) if all(refs) else {}

    # factuality summary
    fact_scores: List[Dict[str, float]] = []
    for r in out_rows:
        meta = r.get("meta") or {}
        if task == "e2e" and "mr" in meta:
            fact_scores.append(slot_error_rate(meta["mr"], r["pred"]))
        if task == "webnlg" and "triples" in meta:
            fact_scores.append(triple_coverage(meta["triples"], r["pred"]))

    # Save predictions
    pred_path = os.path.join(ckpt_dir, f"pred_{task}_{args.split}.jsonl")
    write_jsonl(pred_path, out_rows)

    if metrics:
        print(json.dumps({"metrics": metrics}, indent=2))
    if fact_scores:
        # mean over keys
        keys = sorted({k for fs in fact_scores for k in fs.keys()})
        mean = {k: sum(fs.get(k, 0.0) for fs in fact_scores) / len(fact_scores) for k in keys}
        print(json.dumps({"factuality_mean": mean}, indent=2))

    print(f"saved_predictions={pred_path}")


if __name__ == "__main__":
    main()


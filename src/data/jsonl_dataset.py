from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.utils import read_jsonl


@dataclass
class JsonlExample:
    source: str
    target: str
    meta: Dict[str, Any]


class JsonlSeq2SeqDataset(Dataset):
    def __init__(self, path: str, source_field: str = "source", target_field: str = "target"):
        self.rows = read_jsonl(path)
        self.source_field = source_field
        self.target_field = target_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        src = r[self.source_field]
        tgt = r[self.target_field]
        meta = {k: v for k, v in r.items() if k not in {self.source_field, self.target_field}}
        return {"source": src, "target": tgt, "meta": meta}


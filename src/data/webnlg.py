from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.jsonl_dataset import JsonlSeq2SeqDataset


@dataclass
class WebNLGDataConfig:
    data_dir: str
    train_file: str
    val_file: str
    test_file: str
    max_source_length: int
    max_target_length: int


class WebNLGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: WebNLGDataConfig,
        tokenizer_name: str,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self._train: JsonlSeq2SeqDataset | None = None
        self._val: JsonlSeq2SeqDataset | None = None
        self._test: JsonlSeq2SeqDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        base = self.cfg.data_dir
        self._train = JsonlSeq2SeqDataset(os.path.join(base, self.cfg.train_file))
        self._val = JsonlSeq2SeqDataset(os.path.join(base, self.cfg.val_file))
        self._test = JsonlSeq2SeqDataset(os.path.join(base, self.cfg.test_file))

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sources = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]
        model_inputs = self.tokenizer(
            sources,
            max_length=self.cfg.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            targets,
            max_length=self.cfg.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return DataLoader(
            self._test,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.eval.metrics import compute_text_metrics
from src.utils import DecodeConfig


@dataclass
class LoraArgs:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05


class T5LoRALitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        weight_decay: float,
        decode_cfg: DecodeConfig | None,
        use_lora: bool = True,
        lora_args: LoraArgs | None = None,
        enable_bertscore: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["decode_cfg", "lora_args"])
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.decode_cfg = decode_cfg or DecodeConfig()
        self.enable_bertscore = enable_bertscore

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Gradient checkpointing is disabled due to pickling issues with local functions
        # in multiprocessing environments. If memory becomes an issue, consider:
        # 1. Reducing batch size or gradient accumulation
        # 2. Using a smaller model variant
        # try:
        #     model.gradient_checkpointing_enable()
        # except Exception as e:
        #     import warnings
        #     warnings.warn(f"Gradient checkpointing failed: {e}. Continuing without it.")

        if use_lora:
            la = lora_args or LoraArgs()
            peft_cfg = LoraConfig(
                r=la.r,
                lora_alpha=la.alpha,
                lora_dropout=la.dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
                # This covers T5 attention projections in PEFT for most versions.
                target_modules=["q", "k", "v", "o"],
            )
            model = get_peft_model(model, peft_cfg)
            model.print_trainable_parameters()
            # Note: enable_in   put_require_grads() is omitted due to pickling issues
            # with local functions in multiprocessing environments

        self.model = model

        self._val_preds: List[str] = []
        self._val_refs: List[str] = []

    def forward(self, **batch: torch.Tensor) -> Any:
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.model(**batch)
        loss = out.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.model(**batch)
        self.log("val/loss", out.loss, prog_bar=True, on_step=False, on_epoch=True)

        gen = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=self.decode_cfg.num_beams,
            length_penalty=self.decode_cfg.length_penalty,
            no_repeat_ngram_size=self.decode_cfg.no_repeat_ngram_size,
            max_new_tokens=self.decode_cfg.max_new_tokens,
        )
        preds = self.tokenizer.batch_decode(gen, skip_special_tokens=True)

        labels = batch["labels"].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self._val_preds.extend([p.strip() for p in preds])
        self._val_refs.extend([r.strip() for r in refs])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        metrics = compute_text_metrics(self._val_preds, self._val_refs, enable_bertscore=self.enable_bertscore)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=(k in {"rougeL", "bleu"}), on_epoch=True)
        self._val_preds.clear()
        self._val_refs.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.AdamW(params, lr=self.lr)
        return opt


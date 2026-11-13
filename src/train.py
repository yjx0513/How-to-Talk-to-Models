from __future__ import annotations

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import argparse
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.data.e2e import E2EDataConfig, E2EDataModule
from src.data.webnlg import WebNLGDataConfig, WebNLGDataModule
from src.models.t5_lora import LoraArgs, T5LoRALitModule
from src.utils import DecodeConfig, ensure_dir, load_yaml, set_global_seed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--strategy", type=str, default=None, help="plain|canonical|prompt|tagged (overrides config)")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    return p.parse_args()


def main() -> None:
    # Set multiprocessing start method for compatibility with Python 3.14+ and Windows
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    args = _parse_args()
    cfg: Dict[str, Any] = load_yaml(args.config)

    if args.strategy is not None:
        cfg["linearization"]["strategy"] = args.strategy
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)

    seed = int(cfg["train"]["seed"])
    set_global_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    run_dir = cfg["run_dir"]
    task = cfg["task"]
    strategy = cfg["linearization"]["strategy"]
    exp_name = f"strategy={strategy}_seed={seed}"
    out_dir = os.path.join(run_dir, exp_name)
    ensure_dir(out_dir)

    decode_cfg = DecodeConfig(**cfg["decode"])
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    train_cfg["num_workers"] = 0  # Force 0 for Windows compatibility

    lit = T5LoRALitModule(
        model_name=model_cfg["model_name"],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        decode_cfg=decode_cfg,
        use_lora=bool(model_cfg["use_lora"]),
        lora_args=LoraArgs(
            r=int(model_cfg["lora_r"]),
            alpha=int(model_cfg["lora_alpha"]),
            dropout=float(model_cfg["lora_dropout"]),
        ),
        enable_bertscore=bool(cfg.get("metrics", {}).get("enable_bertscore", True)),
    )

    data_cfg = cfg["data"]
    if task == "e2e":
        dm = E2EDataModule(
            cfg=E2EDataConfig(**data_cfg),
            tokenizer_name=model_cfg["model_name"],
            train_batch_size=int(train_cfg["train_batch_size"]),
            eval_batch_size=int(train_cfg["eval_batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
        )
    elif task == "webnlg":
        dm = WebNLGDataModule(
            cfg=WebNLGDataConfig(**data_cfg),
            tokenizer_name=model_cfg["model_name"],
            train_batch_size=int(train_cfg["train_batch_size"]),
            eval_batch_size=int(train_cfg["eval_batch_size"]),
            num_workers=int(train_cfg["num_workers"]),
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    ckpt = ModelCheckpoint(
        dirpath=os.path.join(out_dir, "checkpoints"),
        save_last=bool(train_cfg.get("save_last", True)),
        monitor="val/rougeL",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_rougeL:.4f}",
    )
    logger = CSVLogger(save_dir=out_dir, name="logs")

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        max_epochs=int(train_cfg["epochs"]),
        accumulate_grad_batches=int(train_cfg["grad_accum"]),
        precision=str(train_cfg["precision"]),
        log_every_n_steps=int(train_cfg["log_every_n_steps"]),
        val_check_interval=float(train_cfg["val_check_interval"]),
        callbacks=[ckpt],
        logger=logger,
        enable_checkpointing=True,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()


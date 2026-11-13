# Data-to-Text Linearization (T5 + LoRA + PyTorch Lightning)

This project runs controlled experiments for **data-to-text generation** where the **only variable** is the *input linearization strategy*.

## What you get
- Datasets: **E2E NLG** and **WebNLG** (English)
- Linearization strategies: `plain`, `canonical`, `prompt`, `tagged`
- Model: `t5-base` fine-tuned with **LoRA** (fits typical 8GB GPUs)
- Training/eval: PyTorch Lightning with generation-based validation and test evaluation
- Metrics: BLEU, METEOR, ROUGE-L, BERTScore (+ factuality scripts)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Create a local data directory:

```bash
mkdir data
```

## Quick sanity check (no downloads)

Generate tiny toy datasets and run a short training to verify your environment:

```bash
python scripts/make_toy_data.py
python -m src.train --config configs/e2e.yaml --strategy plain --seed 42
```

### E2E
- Put the E2E CSVs under `data/e2e/raw/` (train/dev/test).
- Run preprocessing **once per strategy** to produce `data/e2e/processed/*.jsonl`.

Example:

```bash
python scripts/prep_e2e.py --raw_dir data/e2e/raw --out_dir data/e2e/processed --strategy plain --train train.csv --dev dev.csv --test test.csv
```

### WebNLG
- Put WebNLG JSONL files under `data/webnlg/raw/` then run the prep script.

Expected JSONL format per line:
- `{"triples": [["S","R","O"], ...], "text": "..."}`

Example:

```bash
python scripts/prep_webnlg.py --raw_dir data/webnlg/raw --out_dir data/webnlg/processed --strategy plain --triple_order canonical
```

> Note: The prep scripts support a simple, reproducible JSONL format:
> `{"mr": {...}, "text": "..."}` for E2E and `{"triples": [...], "text": "..."}` for WebNLG.

## Train

```bash
python -m src.train --config configs/e2e.yaml --strategy plain --seed 42
```

Run all 4 strategies (and multiple seeds):

```bash
for %s in (plain canonical prompt tagged) do python -m src.train --config configs/e2e.yaml --strategy %s --seed 42
```

## Test / Predict

```bash
python -m src.predict --ckpt "runs/webnlg/strategy=tagged_seed=42/checkpoints/last.ckpt" --split test

```

## Outputs
- Checkpoints and logs are stored under `runs/`.
- Predictions are saved as JSONL with source/target/pred text.


● 这个项目是 数据到文本生成（Data-to-Text Generation）
  的实验框架，主要研究输入线性化策略对生成质量的影响。它使用 T5-base 模型配合 LoRA
  微调，在 E2E NLG（餐馆描述）和
  WebNLG（三元组生成文本）两个数据集上比较四种不同的线性化策略。

  项目核心目标

  -
  唯一变量：输入结构化数据（E2E的槽位对、WebNLG的三元组）转换为文本的方式（线性化策略）
  - 四种策略：
    a. plain：槽位=值 用分号连接
    b. canonical：按预定义顺序排列槽位/三元组
    c. prompt：用自然语言提示包裹
    d. tagged：XML风格的标签包裹值
  - 评估指标：BLEU、ROUGE-L、METEOR、BERTScore
  以及事实性指标（槽位错误率/三元组覆盖率）

  文结构说明件

  配置与文档

  - CLAUDE.md：给 Claude Code
  的详细操作指南，包含环境设置、数据准备、训练、预测等完整流程
  - README.md：用户文档，简要说明项目目的和基本使用方法
  - requirements.txt：Python 依赖包列表
  - configs/：YAML 配置文件
    - e2e.yaml：E2E 数据集的训练配置
    - webnlg.yaml：WebNLG 数据集的训练配置

  脚本工具 (scripts/)

  - prep_e2e.py：预处理 E2E 数据集（CSV → JSONL），支持四种线性化策略
  - prep_webnlg.py：预处理 WebNLG 数据集（JSONL → JSONL），支持策略和三元组排序
  - make_toy_data.py：创建玩具数据集，用于环境验证
  - rename_e2e.py：E2E 数据文件重命名辅助工具

  核心源代码 (src/)

  数据模块 (src/data/)

  - linearize.py：核心文件，实现四种线性化策略
    - linearize_e2e()：处理 E2E 槽位对
    - linearize_webnlg()：处理 WebNLG 三元组
  - e2e.py：E2EDataModule，E2E 数据加载与处理
  - webnlg.py：WebNLGDataModule，WebNLG 数据加载与处理
  - jsonl_dataset.py：JsonlSeq2SeqDataset，读取处理后的 JSONL 文件

  模型模块 (src/models/)

  - t5_lora.py：核心模型文件
    - T5LoRALitModule：PyTorch Lightning 模块，封装 T5-base + LoRA
    - 支持梯度检查点、可训练参数统计、训练/验证步骤

  训练与预测

  - train.py：训练入口，加载配置、设置数据模块、模型、训练器
  - predict.py：预测入口，加载检查点、运行推理、计算指标

  评估模块 (src/eval/)

  - metrics.py：计算文本生成指标（BLEU、ROUGE-L、METEOR、BERTScore）
  - factuality_e2e.py：E2E 事实性评估（槽位错误率）
  - factuality_webnlg.py：WebNLG 事实性评估（三元组覆盖率）

  工具函数 (src/utils.py)

  - 配置文件加载、JSONL 读写、目录创建、随机种子设置等辅助函数

  数据目录 (data/)

  - e2e/raw/：原始 E2E CSV 文件（需手动放置）
  - e2e/processed/：预处理后的 JSONL 文件
  - webnlg/raw/：原始 WebNLG JSONL 文件
  - webnlg/processed/：预处理后的 JSONL 文件

  运行输出 (runs/)

  - 训练检查点、日志、预测结果的保存位置
  - 目录结构：runs/<任务>/strategy=<策略>_seed=<种子>/

  工作流程

  1. 数据准备：运行对应脚本将原始数据转换为 JSONL 格式
  2. 训练：python -m src.train --config configs/e2e.yaml --strategy plain
  3. 预测评估：python -m src.predict --ckpt <检查点路径>

  这个项目的设计重点是控制实验：保持模型架构、训练超参数不变，仅改变输入线性化策略，从
  而研究不同策略对生成质量和事实性的影响。
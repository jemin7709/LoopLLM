#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


DEFAULT_TRAIN_DIR = Path("res/dpo/qwen3-0.6b_math_train_multi_prompt_gcg_b200_s23")
ResultRecord = dict[str, Any]
ResultData = dict[str, ResultRecord]
Message = dict[str, str]
PreferenceRow = dict[str, list[Message]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3 with LoRA DPO on LoopLLM reject pairs."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_result_dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/qwen3-0.6b-dpo-lora")
    )
    parser.add_argument("--save_dataset_dir", type=Path)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default="qwen3-0.6b-dpo-lora")
    parser.add_argument("--wandb_project", default="loopllm-dpo")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser.parse_args()


def result_files(result_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        result_dir.glob("res_*.json"),
        key=lambda path: int(path.stem.removeprefix("res_")),
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No res_*.json files found in {result_dir}")
    return files


def load_result(path: Path) -> ResultData:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def selected_rejected_record(data: ResultData) -> tuple[ResultRecord, bool]:
    last_step = max(int(key) for key in data if int(key) >= 0)
    first_success_step = data[str(last_step)].get("first_success_step")
    success = first_success_step is not None
    step = int(first_success_step) if success else last_step
    return data[str(step)], success


def make_preference_row(data: ResultData) -> tuple[PreferenceRow, bool]:
    rejected_record, success = selected_rejected_record(data)
    return (
        {
            "prompt": [
                {"role": "user", "content": str(rejected_record["adv_prompt"]).strip()}
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": str(data["-1"]["baseline_answer"]).strip(),
                }
            ],
            "rejected": [
                {"role": "assistant", "content": str(rejected_record["answer"]).strip()}
            ],
        },
        success,
    )


def load_preference_rows(
    result_dir: Path,
    limit: int | None,
    success_only: bool,
) -> tuple[list[PreferenceRow], dict[str, int]]:
    rows = []
    stats = {"files": 0, "pairs": 0, "success": 0, "fallback_final": 0}

    for path in result_files(result_dir, limit):
        stats["files"] += 1
        row, success = make_preference_row(load_result(path))
        stats["success" if success else "fallback_final"] += 1
        if success_only and not success:
            continue
        rows.append(row)

    if not rows:
        raise SystemExit(f"No DPO pairs left after filtering {result_dir}")

    stats["pairs"] = len(rows)
    return rows, stats


def save_jsonl(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_metadata(
    args: argparse.Namespace,
    source_stats: dict[str, int],
    train_stats: dict[str, int],
    validation_stats: dict[str, int],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "data": {
            "train_result_dir": str(args.train_result_dir),
            "save_dataset_dir": str(args.save_dataset_dir)
            if args.save_dataset_dir
            else None,
            "success_only": args.success_only,
            "max_train_samples": args.max_train_samples,
            "validation_ratio": args.validation_ratio,
            "source_stats": source_stats,
            "train_stats": train_stats,
            "validation_stats": validation_stats,
        },
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "seed": args.seed,
            "bf16": not args.fp16,
            "fp16": args.fp16,
        },
        "lora": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": "all-linear",
        },
        "wandb": {
            "report_to": args.report_to,
            "run_name": args.run_name,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def configure_wandb(args: argparse.Namespace) -> None:
    if args.report_to != "wandb":
        return
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_NAME", args.run_name)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)


def make_dpo_config(args: argparse.Namespace) -> DPOConfig:
    dtype = torch.float16 if args.fp16 else torch.bfloat16

    return DPOConfig(
        output_dir=str(args.output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        report_to=args.report_to,
        run_name=args.run_name,
        model_init_kwargs={
            "dtype": dtype,
            "trust_remote_code": True,
        },
    )


def main() -> None:
    args = parse_args()

    rows, source_stats = load_preference_rows(
        args.train_result_dir,
        args.max_train_samples,
        args.success_only,
    )
    split = Dataset.from_list(rows).train_test_split(
        test_size=args.validation_ratio,
        seed=args.seed,
    )
    train_dataset = split["train"]
    validation_dataset = split["test"]
    train_stats = {"pairs": len(train_dataset)}
    validation_stats = {"pairs": len(validation_dataset)}

    if args.save_dataset_dir is not None:
        save_jsonl(train_dataset, args.save_dataset_dir / "train.jsonl")
        save_jsonl(validation_dataset, args.save_dataset_dir / "validation.jsonl")

    print(f"source: {source_stats}")
    print(f"train: {train_stats}")
    print(f"validation: {validation_stats}")
    if args.dry_run:
        return

    dpo_config = make_dpo_config(args)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    save_metadata(args, source_stats, train_stats, validation_stats)
    configure_wandb(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = DPOTrainer(
        model=args.model_name_or_path,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "checkpoint-best"))


if __name__ == "__main__":
    main()

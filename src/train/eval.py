#!/usr/bin/env python3
"""Evaluate clean and attack MATH prompts with a simple-evals-style checker."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER = Path(
    "outputs/qwen3-0.6b-dpo-train95-val5-lr5e-6-b0.1-l4096-20260519-172427/checkpoint-best"
)
DEFAULT_CHECKER_MODEL = "google/gemma-4-31B-it"

ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
JsonDict = dict[str, Any]

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent.
Only perform trivial simplifications

Examples:
Expression 1: $2x+3$
Expression 2: $3+2x$
Yes

Expression 1: 3/2
Expression 2: 1.5
Yes

Expression 1: $x^2+2x+1$
Expression 2: $y^2+2y+1$
No

Expression 1: $x^2+2x+1$
Expression 2: $(x+1)^2$
Yes

Expression 1: 3245/5
Expression 2: 649
No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

Expression 1: 2/(-3)
Expression 2: -2/3
Yes
(trivial simplifications are allowed)

Expression 1: 72 degrees
Expression 2: 72
Yes
(give benefit of the doubt to units)

Expression 1: 64
Expression 2: 64 square feet
Yes
(give benefit of the doubt to units)

---
YOUR TASK
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Expression 1: %(expression1)s
Expression 2: %(expression2)s
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clean and attack MATH prompts with vLLM."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--split", choices=["math_test", "math_train"], default="math_test")
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--attack_result_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n_repeats", "--n", dest="n_repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--checker_model",
        "--equality_checker_model",
        dest="checker_model",
        default=DEFAULT_CHECKER_MODEL,
    )
    return parser.parse_args()


def select_attack_prompt(result: JsonDict) -> tuple[str, int, str]:
    steps = [int(key) for key in result if key.lstrip("-").isdigit() and int(key) >= 0]
    last_step = max(steps)
    first_success_step = result[str(last_step)].get("first_success_step")
    step = int(first_success_step) if first_success_step is not None else last_step
    source = "first_success_step" if first_success_step is not None else "last_step"
    record = result[str(step)]
    return str(record.get("adv_prompt") or record["prompt"]).strip(), step, source


def extract_answer(response: str) -> str | None:
    match = ANSWER_PATTERN.search(response)
    return match.group(1) if match else None


def build_examples(data_path: Path, attack_dir: Path, limit: int | None) -> list[JsonDict]:
    math_rows = {}
    with data_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                math_rows[index] = json.loads(line)
    if not math_rows:
        raise SystemExit(f"No MATH rows found in {data_path}")

    result_paths = sorted(
        attack_dir.glob("res_*.json"),
        key=lambda path: int(path.stem.removeprefix("res_")),
    )
    if limit is not None:
        result_paths = result_paths[:limit]
    if not result_paths:
        raise SystemExit(f"No res_*.json files found in {attack_dir}")

    examples = []
    for path in result_paths:
        index = int(path.stem.removeprefix("res_"))
        math_row = math_rows[index]
        with path.open(encoding="utf-8") as handle:
            attack_text, attack_step, attack_source = select_attack_prompt(
                json.load(handle)
            )

        base = {
            "index": index,
            "unique_id": math_row.get("unique_id"),
            "subject": math_row.get("subject"),
            "level": math_row.get("level"),
            "problem": math_row["problem"],
            "expected_answer": math_row["answer"],
            "result_path": str(path),
        }
        examples.append(
            base
            | {
                "prompt_version": "clean",
                "prompt": QUERY_TEMPLATE.format(Question=math_row["problem"]),
                "attack_step": None,
                "attack_source": None,
            }
        )
        examples.append(
            base
            | {
                "prompt_version": "attack",
                "prompt": attack_text,
                "attack_step": attack_step,
                "attack_source": attack_source,
            }
        )
    return examples


def generate_samples(args: argparse.Namespace, examples: list[JsonDict]) -> list[JsonDict]:
    model_kwargs: JsonDict = {
        "model": args.model_name_or_path,
        "dtype": "auto",
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
    }
    lora_request = None
    if not args.no_adapter:
        if not args.adapter_path.exists():
            raise SystemExit(f"Adapter path does not exist: {args.adapter_path}")
        lora_request = LoRARequest("dpo_adapter", 1, str(args.adapter_path))
        model_kwargs.update(enable_lora=True, max_lora_rank=16)

    model = LLM(**model_kwargs)
    params = SamplingParams(
        n=args.n_repeats,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    samples = []

    outputs = model.chat(
        [[{"role": "user", "content": row["prompt"]}] for row in examples],
        params,
        lora_request=lora_request,
        use_tqdm=True,
    )
    for row, output in zip(examples, outputs):
        for repeat_index, completion in enumerate(output.outputs):
            response = completion.text.strip()
            samples.append(
                row
                | {
                    "repeat_index": repeat_index,
                    "response": response,
                    "extracted_answer": extract_answer(response),
                    "output_tokens": len(completion.token_ids),
                    "equality_response": None,
                    "score": 0.0,
                }
            )

    del model
    free_cuda_cache()
    return samples


def grade_samples(args: argparse.Namespace, samples: list[JsonDict]) -> None:
    checker = LLM(
        model=args.checker_model,
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    params = SamplingParams(max_tokens=8, temperature=0, seed=args.seed)

    prompts = [
        EQUALITY_TEMPLATE
        % {
            "expression1": str(row["expected_answer"]),
            "expression2": row["extracted_answer"],
        }
        for row in samples
    ]
    outputs = checker.chat(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        params,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    for row, output in zip(samples, outputs):
        answer = output.outputs[0].text.strip()
        row["equality_response"] = answer
        row["score"] = float(answer.lower() == "yes")

    del checker
    free_cuda_cache()


def summarize(
    args: argparse.Namespace,
    data_path: Path,
    attack_dir: Path,
    samples: list[JsonDict],
) -> JsonDict:
    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_prompt_version = {}
    for version in ["clean", "attack"]:
        rows = [row for row in samples if row["prompt_version"] == version]
        lengths = [int(row["output_tokens"]) for row in rows]
        by_prompt_version[version] = {
            "num_examples": len({row["index"] for row in rows}),
            "num_samples": len(rows),
            "accuracy": average([float(row["score"]) for row in rows]),
            "answer_extraction_rate": average(
                [float(row["extracted_answer"] is not None) for row in rows]
            ),
            "avg_output_tokens": average([float(length) for length in lengths]),
            "max_token_hit_rate": average(
                [float(length >= args.max_new_tokens - 5) for length in lengths]
            ),
        }

    return {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": None if args.no_adapter else str(args.adapter_path),
        "split": args.split,
        "dataset_path": str(data_path),
        "attack_result_dir": str(attack_dir),
        "checker_model": args.checker_model,
        "checker": "simple-evals equality prompt",
        "num_examples": len({row["index"] for row in samples}),
        "n_repeats": args.n_repeats,
        "accuracy": average([float(row["score"]) for row in samples]),
        "by_prompt_version": by_prompt_version,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    }


def save_outputs(output_dir: Path, summary: JsonDict, samples: list[JsonDict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {samples_path}")


def free_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.n_repeats > 1 and args.temperature == 0:
        raise SystemExit("--n_repeats > 1 needs --temperature > 0.")

    split = "train" if args.split == "math_train" else "test"
    data_path = args.dataset_path or Path("dataset/math") / f"{split}.jsonl"
    attack_dir = args.attack_result_dir or Path(
        "res/dpo"
    ) / f"qwen3-0.6b_{args.split}_multi_prompt_gcg_b200_s23"

    examples = build_examples(data_path, attack_dir, args.limit)
    samples = generate_samples(args, examples)
    grade_samples(args, samples)
    summary = summarize(args, data_path, attack_dir, samples)

    if args.output_dir is None:
        adapter_label = "base"
        if not args.no_adapter:
            adapter_label = f"{args.adapter_path.parent.name}-{args.adapter_path.name}"
        model_label = args.model_name_or_path.rsplit("/", maxsplit=1)[-1]
        run_name = (
            f"{args.split}-{model_label}-{adapter_label}-{attack_dir.name}"
            f"-n{args.n_repeats}-s{args.seed}"
        )
        output_dir = Path("analysis/math_eval") / re.sub(
            r"[^A-Za-z0-9_.-]+", "-", run_name
        )
    else:
        output_dir = args.output_dir

    save_outputs(output_dir, summary, samples)


if __name__ == "__main__":
    main()

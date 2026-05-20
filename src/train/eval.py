#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_ADAPTER = Path(
    "outputs/qwen3-0.6b-dpo-train95-val5-lr5e-6-b0.1-l4096-20260519-172427/checkpoint-best"
)
ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")

MathRow = dict[str, Any]


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
        description="Evaluate a base model or LoRA adapter on the local MATH split with vLLM."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--split", choices=["math_test", "math_train"], default="math_test")
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--equality_checker_model")
    parser.add_argument("--no_adapter", action="store_true")
    return parser.parse_args()


def dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return args.dataset_path
    split = "train" if args.split == "math_train" else "test"
    return Path("dataset/math") / f"{split}.jsonl"


def load_math_rows(path: Path, limit: int | None) -> list[MathRow]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No examples found in {path}")
    return rows


def make_prompt(row: MathRow) -> str:
    return QUERY_TEMPLATE.format(Question=row["problem"])


def make_prompt_request(tokenizer: Any, row: MathRow) -> dict[str, list[int]]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt(row)}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    return {"prompt_token_ids": prompt_ids[0].tolist()}


def make_equality_prompt_request(
    tokenizer: Any,
    expected_answer: str,
    extracted_answer: str | None,
) -> dict[str, list[int]]:
    prompt = EQUALITY_TEMPLATE % {
        "expression1": expected_answer,
        "expression2": extracted_answer,
    }
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )
    return {"prompt_token_ids": prompt_ids[0].tolist()}


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def extract_answer(response_text: str) -> str | None:
    match = ANSWER_PATTERN.search(response_text)
    return match.group(1) if match else None


def make_completion(output: Any) -> dict[str, Any]:
    response_text = output.text.strip()
    extracted_answer = extract_answer(response_text)
    return {
        "response": response_text,
        "extracted_answer": extracted_answer,
        "output_tokens": len(output.token_ids),
        "equality_response": None,
        "score": 0.0,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir

    adapter_label = "base"
    if not args.no_adapter and args.adapter_path is not None:
        adapter_label = f"{args.adapter_path.parent.name}-{args.adapter_path.name}"
    model_label = args.model_name_or_path.rsplit("/", maxsplit=1)[-1]
    run_name = f"{args.split}-{model_label}-{adapter_label}-n{args.n}-s{args.seed}"
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name)
    return Path("analysis/math_eval") / run_name


def build_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model_name_or_path,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
    }
    if not args.no_adapter:
        kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
    return LLM(**kwargs)


def build_equality_llm(args: argparse.Namespace, llm: Any):
    if args.equality_checker_model is None:
        return llm

    from vllm import LLM

    return LLM(
        model=args.equality_checker_model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )


def build_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    if args.n > 1 and args.temperature == 0:
        raise SystemExit("--n > 1 needs --temperature > 0; deterministic repeats are identical.")

    kwargs: dict[str, Any] = {
        "n": args.n,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    return SamplingParams(**kwargs)


def build_equality_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    return SamplingParams(max_tokens=8, temperature=0, seed=args.seed)


def build_lora_request(args: argparse.Namespace):
    if args.no_adapter:
        return None
    if not args.adapter_path.exists():
        raise SystemExit(f"Adapter path does not exist: {args.adapter_path}")

    from vllm.lora.request import LoRARequest

    return LoRARequest("dpo_adapter", 1, str(args.adapter_path))


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_problem_result(row: MathRow, request_output: Any) -> dict[str, Any]:
    completions = [make_completion(output) for output in request_output.outputs]
    scores = [completion["score"] for completion in completions]
    return {
        "unique_id": row.get("unique_id"),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "problem": row["problem"],
        "expected_answer": row["answer"],
        "score_mean": mean(scores),
        "any_correct": any(score > 0 for score in scores),
        "completions": completions,
    }


def run_equality_checks(
    scored_rows: list[dict[str, Any]],
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    batch_size: int,
) -> None:
    pending = [
        (row["expected_answer"], completion)
        for row in scored_rows
        for completion in row["completions"]
        if completion["extracted_answer"] is not None
    ]

    for batch_index, batch in enumerate(batched(pending, batch_size), start=1):
        request_outputs = llm.generate(
            [
                make_equality_prompt_request(
                    tokenizer,
                    str(expected_answer),
                    completion["extracted_answer"],
                )
                for expected_answer, completion in batch
            ],
            sampling_params,
            use_tqdm=False,
        )
        for (_, completion), request_output in zip(batch, request_outputs):
            equality_response = request_output.outputs[0].text.strip()
            completion["equality_response"] = equality_response
            completion["score"] = float(equality_response.lower().strip() == "yes")

        print(f"checked {min(batch_index * batch_size, len(pending))}/{len(pending)}")

    for row in scored_rows:
        scores = [completion["score"] for completion in row["completions"]]
        row["score_mean"] = mean(scores)
        row["any_correct"] = any(score > 0 for score in scores)


def summarize_results(
    args: argparse.Namespace,
    data_path: Path,
    scored_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completions = [
        completion
        for row in scored_rows
        for completion in row["completions"]
    ]
    token_lengths = [completion["output_tokens"] for completion in completions]
    cap_hits = [length >= args.max_new_tokens - 5 for length in token_lengths]

    return {
        "model_name_or_path": args.model_name_or_path,
        "adapter_path": None if args.no_adapter else str(args.adapter_path),
        "split": args.split,
        "dataset_path": str(data_path),
        "num_examples": len(scored_rows),
        "n": args.n,
        "equality_checker_model": args.equality_checker_model or args.model_name_or_path,
        "checker": "simple-evals equality prompt",
        "accuracy": mean([completion["score"] for completion in completions]),
        "problem_score_mean": mean([row["score_mean"] for row in scored_rows]),
        "any_correct_rate": mean([float(row["any_correct"]) for row in scored_rows]),
        "answer_extraction_rate": mean(
            [
                float(completion["extracted_answer"] is not None)
                for completion in completions
            ]
        ),
        "avg_output_tokens": mean([float(length) for length in token_lengths]),
        "max_token_hit_rate": mean([float(hit) for hit in cap_hits]),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
    }


def save_results(
    out_dir: Path,
    summary: dict[str, Any],
    scored_rows: list[dict[str, Any]],
) -> None:
    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in scored_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {samples_path}")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_path = dataset_path(args)
    rows = load_math_rows(data_path, args.limit)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_name_or_path)
    sampling_params = build_sampling_params(args)
    equality_sampling_params = build_equality_sampling_params(args)
    lora_request = build_lora_request(args)
    llm = build_llm(args)
    equality_llm = build_equality_llm(args, llm)
    equality_tokenizer = tokenizer
    if args.equality_checker_model is not None:
        equality_tokenizer = load_tokenizer(args.equality_checker_model)

    scored_rows = []
    for batch_index, batch_rows in enumerate(batched(rows, args.batch_size), start=1):
        prompt_requests = [make_prompt_request(tokenizer, row) for row in batch_rows]
        request_outputs = llm.generate(
            prompt_requests,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        for row, request_output in zip(batch_rows, request_outputs):
            scored_rows.append(make_problem_result(row, request_output))

        print(f"generated {min(batch_index * args.batch_size, len(rows))}/{len(rows)}")

    run_equality_checks(
        scored_rows,
        equality_llm,
        equality_tokenizer,
        equality_sampling_params,
        args.batch_size,
    )
    summary = summarize_results(args, data_path, scored_rows)
    save_results(out_dir, summary, scored_rows)
    return summary


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

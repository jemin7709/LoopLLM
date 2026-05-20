#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
DEFAULT_JUDGE_MODEL = "google/gemma-4-31B-it"
JUDGE_MAX_MODEL_LEN = 4096
JsonDict = dict[str, Any]

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge
whether they are equivalent. Only perform trivial simplifications

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
(these are actually equal, don't mark them equivalent if you need to do
nontrivial simplifications)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--setting",
        choices=["original_math", "correct_output"],
        required=True,
    )
    parser.add_argument("--math_split", choices=["train", "test"], default="train")
    parser.add_argument("--math_jsonl", type=Path)
    parser.add_argument("--output_root", type=Path, default=Path("res/dpo/datasets"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.95)
    return parser.parse_args()


def load_json(path: Path) -> JsonDict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_math(path: Path) -> dict[int, JsonDict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                rows[index] = json.loads(line)
    return rows


def result_files(result_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        result_dir.glob("res_*.json"),
        key=lambda path: int(path.stem.removeprefix("res_")),
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No res_*.json files in {result_dir}")
    return files


def math_solution(math_row: JsonDict) -> str:
    solution = str(math_row.get("solution", "")).strip()
    answer = str(math_row["answer"]).strip()
    if not solution:
        return f"Answer: {answer}"
    if ANSWER_PATTERN.search(solution):
        return solution
    return f"{solution}\n\nAnswer: {answer}"


def parsed_answer(text: str) -> str | None:
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None


def output_candidates(data: JsonDict) -> list[JsonDict]:
    candidates: list[JsonDict] = []
    metadata = data["-1"]
    baseline_answer = str(metadata.get("baseline_answer", "")).strip()
    if baseline_answer:
        candidates.append(
            {
                "source": "baseline_answer",
                "step": "baseline",
                "prompt": str(metadata["baseline_prompt"]).strip(),
                "answer": baseline_answer,
                "success": bool(metadata["baseline_success"]),
            }
        )

    for key in sorted((key for key in data if key != "-1"), key=int):
        record = data[key]
        answer = str(record["answer"]).strip()
        if answer:
            candidates.append(
                {
                    "source": "attack_answer",
                    "step": int(key),
                    "prompt": str(record["adv_prompt"]).strip(),
                    "answer": answer,
                    "success": bool(record["success"]),
                }
            )
    return candidates


def first_success(candidates: list[JsonDict]) -> JsonDict | None:
    for candidate in candidates:
        if candidate["success"]:
            return candidate
    return None


def count_source(stats: JsonDict, key: str, source: str) -> None:
    counts = stats.setdefault(key, {})
    counts[source] = counts.get(source, 0) + 1


def dpo_row(prompt: str, chosen: str, rejected: str) -> JsonDict:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def judge_equal(
    pairs: list[tuple[str, str]],
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[bool]:
    if not pairs:
        return []

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=JUDGE_MAX_MODEL_LEN,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    prompts = [
        EQUALITY_TEMPLATE % {"expression1": gold, "expression2": candidate}
        for gold, candidate in pairs
    ]

    outputs = llm.generate(prompts, sampling_params)
    return [
        output.outputs[0].text.strip().lower().startswith("yes") for output in outputs
    ]


def base_stats(paths: list[Path]) -> JsonDict:
    return {
        "files": len(paths),
        "missing_math_row": 0,
        "no_success_output": 0,
        "no_answer_candidate": 0,
        "no_correct_chosen": 0,
        "rejected_sources": {},
        "chosen_sources": {},
    }


def build_original_math(
    paths: list[Path],
    math_rows: dict[int, JsonDict],
) -> tuple[list[JsonDict], JsonDict]:
    examples = []
    stats = base_stats(paths)

    for path in paths:
        index = int(path.stem.removeprefix("res_"))
        if index not in math_rows:
            stats["missing_math_row"] += 1
            continue

        rejected = first_success(output_candidates(load_json(path)))
        if rejected is None:
            stats["no_success_output"] += 1
            continue

        count_source(stats, "rejected_sources", rejected["source"])
        count_source(stats, "chosen_sources", "math_solution")
        examples.append(
            {
                "index": index,
                "path": str(path),
                "prompt": rejected["prompt"],
                "chosen": math_solution(math_rows[index]),
                "chosen_source": "math_solution",
                "rejected": rejected["answer"],
                "rejected_source": rejected["source"],
                "reject_step": rejected["step"],
            }
        )

    return examples, stats


def build_correct_output(
    paths: list[Path],
    math_rows: dict[int, JsonDict],
    args: argparse.Namespace,
) -> tuple[list[JsonDict], JsonDict]:
    examples = []
    pending: list[tuple[JsonDict, list[JsonDict], list[tuple[str, str]]]] = []
    stats = base_stats(paths)

    for path in paths:
        index = int(path.stem.removeprefix("res_"))
        if index not in math_rows:
            stats["missing_math_row"] += 1
            continue

        candidates = output_candidates(load_json(path))
        rejected = first_success(candidates)
        if rejected is None:
            stats["no_success_output"] += 1
            continue

        checks = []
        judged_candidates = []
        for candidate in candidates:
            answer = parsed_answer(candidate["answer"])
            if answer is None:
                continue
            checks.append((str(math_rows[index]["answer"]).strip(), answer))
            judged_candidates.append(candidate)

        if not checks:
            stats["no_answer_candidate"] += 1
            continue

        pending.append(
            (
                {
                    "index": index,
                    "path": str(path),
                    "prompt": rejected["prompt"],
                    "rejected": rejected["answer"],
                    "rejected_source": rejected["source"],
                    "reject_step": rejected["step"],
                },
                judged_candidates,
                checks,
            )
        )

    all_checks = [pair for _, _, checks in pending for pair in checks]
    stats["checked_candidates"] = len(all_checks)
    correct = judge_equal(
        all_checks,
        args.judge_model,
        args.judge_tensor_parallel_size,
        args.judge_gpu_memory_utilization,
    )

    offset = 0
    for base, candidates, checks in pending:
        flags = correct[offset : offset + len(checks)]
        offset += len(checks)

        chosen = None
        for candidate, is_correct in zip(candidates, flags):
            if not is_correct:
                continue
            if candidate["answer"] == base["rejected"]:
                continue
            chosen = candidate
            break

        if chosen is None:
            stats["no_correct_chosen"] += 1
            continue

        count_source(stats, "rejected_sources", base["rejected_source"])
        count_source(stats, "chosen_sources", chosen["source"])
        examples.append(
            {
                **base,
                "chosen": chosen["answer"],
                "chosen_source": chosen["source"],
                "chosen_step": chosen["step"],
            }
        )

    return examples, stats


def split_examples(
    examples: list[JsonDict],
    validation_ratio: float,
    seed: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_size = round(len(shuffled) * validation_ratio)
    if validation_ratio > 0 and len(shuffled) > 1:
        validation_size = max(1, validation_size)
    return shuffled[validation_size:], shuffled[:validation_size]


def write_jsonl(rows: list[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    examples: list[JsonDict],
    stats: JsonDict,
    args: argparse.Namespace,
) -> None:
    output_dir = args.output_root / args.result_dir.name / args.setting
    train, validation = split_examples(examples, args.validation_ratio, args.seed)

    write_jsonl(
        [dpo_row(row["prompt"], row["chosen"], row["rejected"]) for row in train],
        output_dir / "train.jsonl",
    )
    write_jsonl(
        [dpo_row(row["prompt"], row["chosen"], row["rejected"]) for row in validation],
        output_dir / "validation.jsonl",
    )
    write_jsonl(
        [{**row, "split": "train"} for row in train]
        + [{**row, "split": "validation"} for row in validation],
        output_dir / "audit.jsonl",
    )

    summary = {
        "setting": args.setting,
        "result_dir": str(args.result_dir),
        "math_jsonl": str(args.math_jsonl),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "counts": {
            "examples": len(examples),
            "train": len(train),
            "validation": len(validation),
        },
        "source_stats": stats,
    }
    if args.setting == "correct_output":
        summary["judge"] = {
            "model": args.judge_model,
            "tensor_parallel_size": args.judge_tensor_parallel_size,
            "gpu_memory_utilization": args.judge_gpu_memory_utilization,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"] | {"output_dir": str(output_dir)}))


def main() -> None:
    args = parse_args()
    args.math_jsonl = (
        args.math_jsonl or Path("dataset/math") / f"{args.math_split}.jsonl"
    )

    paths = result_files(args.result_dir, args.limit)
    math_rows = load_math(args.math_jsonl)
    if args.setting == "original_math":
        examples, stats = build_original_math(paths, math_rows)
    else:
        examples, stats = build_correct_output(paths, math_rows, args)
    write_outputs(examples, stats, args)


if __name__ == "__main__":
    main()

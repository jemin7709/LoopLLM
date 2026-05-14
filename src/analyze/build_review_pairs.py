#!/usr/bin/env python3
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ReadOnly, TypedDict, cast


ResultRecord = dict[str, Any]
ResultData = dict[str, ResultRecord]


class ReviewPair(TypedDict):
    index: ReadOnly[int]
    path: ReadOnly[str]
    step: ReadOnly[int]
    success: ReadOnly[bool]
    chosen: ReadOnly[str]
    rejected: ReadOnly[str]
    chosen_sentences: ReadOnly[int]
    rejected_sentences: ReadOnly[int]


@dataclass(frozen=True, slots=True)
class RejectedResult:
    step: int
    success: bool
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def default_output(result_dir: Path) -> Path:
    path = str(result_dir.resolve())
    if "/res/" in path:
        return Path(path.replace("/res/", "/analysis/", 1)) / "review_pairs.json"
    return result_dir / "review_pairs.json"


def res_files(result_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(
        result_dir.glob("res_*.json"),
        key=lambda path: int(path.stem.removeprefix("res_")),
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        sys.exit(f"No res_*.json files in {result_dir}")
    return files


def preprocess(text: str) -> str:
    text = (
        str(text).replace("<think>", "").replace("</think>", "").replace(":\n\n", ": ")
    )
    return re.sub(r"\s+", " ", text).strip()


def sentence_count(text: str) -> int:
    text = preprocess(text)
    if not text:
        return 0
    return len([part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()])


def last_step_record(data: ResultData) -> tuple[int, ResultRecord]:
    last_step = max(int(key) for key in data if int(key) >= 0)
    return last_step, data[str(last_step)]


def selected_rejected_result(data: ResultData) -> RejectedResult:
    last_step, last_record = last_step_record(data)
    first_success_step = last_record.get("first_success_step")
    success = first_success_step is not None
    step = int(first_success_step) if success else last_step
    return RejectedResult(
        step=step,
        success=success,
        answer=cast(str, data[str(step)]["answer"]),
    )


def load_result(path: Path) -> ResultData:
    with path.open(encoding="utf-8") as f:
        return cast(ResultData, json.load(f))


def review_pair(path: Path) -> ReviewPair:
    data = load_result(path)
    rejected = selected_rejected_result(data)
    chosen = cast(str, data["-1"]["baseline_answer"])

    return {
        "index": int(path.stem.removeprefix("res_")),
        "path": str(path),
        "step": rejected.step,
        "success": rejected.success,
        "chosen": chosen,
        "rejected": rejected.answer,
        "chosen_sentences": sentence_count(chosen),
        "rejected_sentences": sentence_count(rejected.answer),
    }


def main() -> None:
    args = parse_args()
    rows: list[ReviewPair] = [
        review_pair(path) for path in res_files(args.result_dir, args.limit)
    ]
    output = args.output or default_output(args.result_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

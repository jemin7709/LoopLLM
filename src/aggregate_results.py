#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

SUCCESS_THRESHOLD = 0.125


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate LoopLLMSolar result files.")
    parser.add_argument(
        "result_dir", type=Path, help="Directory containing res_*.json files"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where aggregated output directories are created",
    )
    return parser.parse_args()


def get_output_path(result_dir: Path, output_root: Optional[Path] = None) -> Path:
    if output_root is not None:
        output_root = output_root.resolve()
        out_dir = output_root / result_dir.relative_to(output_root.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "aggregated_results.json"

    out_dir_str = str(result_dir).replace("/res/", "/aggregate/")
    if out_dir_str == str(result_dir):
        out_dir = result_dir.parent / "aggregate" / result_dir.name
    else:
        out_dir = Path(out_dir_str)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "aggregated_results.json"


def latest_evaluated_row(data):
    for key in sorted(data.keys(), key=int, reverse=True):
        row = data[key]
        if row.get("evaluated", True):
            return row
    last_key = max(data.keys(), key=int)
    return data[last_key]


def prompt_success(row):
    if "success" in row:
        return bool(row["success"])
    if "success_rate" in row:
        return float(row["success_rate"]) >= SUCCESS_THRESHOLD
    raise KeyError("success or success_rate")


def prompt_avg_len(row):
    if "avg_len" in row:
        return float(row["avg_len"])
    raise KeyError("avg_len")


def process_result_files(files):
    successful_attacks = 0
    lengths = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        last_row = latest_evaluated_row(data)

        if prompt_success(last_row):
            successful_attacks += 1
        lengths.append(prompt_avg_len(last_row))

    total_files = len(files)
    mean_len = sum(lengths) / total_files
    std_len = float(np.std(lengths))
    p25_len, median_len, p75_len = [
        float(value) for value in np.percentile(lengths, [25, 50, 75])
    ]

    return {
        "files": total_files,
        "successful_attacks": successful_attacks,
        "average_asr": successful_attacks / total_files,
        "average_avg_len": mean_len,
        "std_avg_len": std_len,
        "median_len": median_len,
        "p25_len": p25_len,
        "p75_len": p75_len,
    }


def print_summary(result_dir, summary):
    print(f"Directory: {result_dir}")
    print(f"Files: {summary['files']}")
    print(f"Successful Attacks: {summary['successful_attacks']}")
    print(f"Average ASR: {summary['average_asr']:.4f}")
    print(f"Average Avg-len: {summary['average_avg_len']:.4f}")
    print(f"Std Avg-len: {summary['std_avg_len']:.4f}")
    print(f"Median Len: {summary['median_len']:.4f}")
    print(f"P25 Len: {summary['p25_len']:.4f}")
    print(f"P75 Len: {summary['p75_len']:.4f}")


def main():
    args = parse_args()
    result_dir = args.result_dir.resolve()

    files = list(result_dir.glob("res_*.json"))
    if not files:
        sys.exit(f"Error: No result files found in {result_dir}")

    summary = process_result_files(files)
    print_summary(result_dir, summary)

    output_file = get_output_path(result_dir, args.output_root)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved aggregated results to: {output_file}")


if __name__ == "__main__":
    main()

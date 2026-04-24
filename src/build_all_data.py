from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
ALPACA_PATH = DATA_DIR / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
SHAREGPT_PATH = DATA_DIR / "sharegpt_gpt4.jsonl"
OUTPUT_PATH = DATA_DIR / "all_data.json"

SAMPLE_SIZE_PER_SOURCE = 50
RANDOM_SEED = 42
MAX_INSTRUCTION_WORDS = 400


def load_alpaca_records() -> list[dict]:
    records = pd.read_parquet(ALPACA_PATH).to_dict(orient="records")
    filtered_records = []

    for record in records:
        instruction = record["instruction"]
        if record.get("input"):
            instruction += f"\n{record['input']}"

        if len(instruction.split()) <= MAX_INSTRUCTION_WORDS:
            filtered_records.append(
                {
                    "instruction": instruction,
                    "source": "alpaca",
                }
            )

    return filtered_records


def load_sharegpt_records() -> list[dict]:
    records = []

    with SHAREGPT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            prompt = row["conversations"][0]["value"]

            if len(prompt.split()) <= MAX_INSTRUCTION_WORDS:
                records.append(
                    {
                        "instruction": prompt,
                        "source": "sharegpt",
                    }
                )

    return records


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    alpaca_sample = rng.sample(load_alpaca_records(), SAMPLE_SIZE_PER_SOURCE)
    sharegpt_sample = rng.sample(load_sharegpt_records(), SAMPLE_SIZE_PER_SOURCE)

    all_records = alpaca_sample + sharegpt_sample

    for i, record in enumerate(all_records):
        record["index"] = i

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_records, file, ensure_ascii=False, indent=2)

    print(f"wrote {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

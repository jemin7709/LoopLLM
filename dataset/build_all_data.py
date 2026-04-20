from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ALPACA_PATH = BASE_DIR / "train-00000-of-00001-a09b74b3ef9c3b56.parquet"
SHAREGPT_PATH = BASE_DIR / "sharegpt_gpt4.jsonl"
OUTPUT_PATH = BASE_DIR / "all_data.json"

SAMPLE_SIZE_PER_SOURCE = 50
RANDOM_SEED = 42


def load_alpaca_records() -> list[dict[str, str]]:
    records = pd.read_parquet(ALPACA_PATH).to_dict(orient="records")
    if len(records) < SAMPLE_SIZE_PER_SOURCE:
        raise ValueError(f"alpaca source has only {len(records)} records")

    return [
        {
            "instruction": build_prompt(record["instruction"], record.get("input", "")),
            "source": "alpaca",
        }
        for record in records
    ]


def load_sharegpt_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    with SHAREGPT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            conversations = row.get("conversations", [])
            if not conversations:
                continue

            first_turn = conversations[0]
            if first_turn.get("from") != "human":
                continue

            prompt = first_turn.get("value", "").strip()
            if not prompt:
                continue

            records.append(
                {
                    "instruction": prompt,
                    "source": "sharegpt",
                }
            )

    if len(records) < SAMPLE_SIZE_PER_SOURCE:
        raise ValueError(f"sharegpt source has only {len(records)} usable records")

    return records


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()

    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    alpaca_sample = rng.sample(load_alpaca_records(), SAMPLE_SIZE_PER_SOURCE)
    sharegpt_sample = rng.sample(load_sharegpt_records(), SAMPLE_SIZE_PER_SOURCE)

    all_records = alpaca_sample + sharegpt_sample
    rng.shuffle(all_records)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(all_records, file, ensure_ascii=False, indent=2)

    print(f"wrote {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

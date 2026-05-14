#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


MODEL = "Qwen/Qwen3-Embedding-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--top-percent", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def default_output(result_dir: Path) -> Path:
    path = str(result_dir.resolve())
    if "/res/" in path:
        return Path(path.replace("/res/", "/analysis/", 1)) / "top_similarity.jsonl"
    return result_dir / "top_similarity.jsonl"


def all_output_for(output: Path) -> Path:
    if output.name == "top_similarity.jsonl":
        return output.with_name("all_similarity.jsonl")
    return output.with_name(f"{output.stem}_all{output.suffix}")


def res_files(result_dir: Path, limit: int | None):
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


def sentence_tokenize(text: str):
    from nltk.tokenize import sent_tokenize

    return [
        sentence.strip()
        for sentence in sent_tokenize(preprocess(text))
        if sentence.strip()
    ]


def selected_rejected_answer(data: dict):
    last_step = max(int(key) for key in data if int(key) >= 0)
    first_success_step = data[str(last_step)].get("first_success_step")
    step = first_success_step if first_success_step is not None else last_step
    return step, data[str(step)]["answer"]


def load_pair(path: Path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    step, rejected = selected_rejected_answer(data)
    return {
        "index": int(path.stem.removeprefix("res_")),
        "path": str(path),
        "step": step,
        "chosen": data["-1"]["baseline_answer"],
        "rejected": rejected,
    }


def load_model(device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL, device=device)


def sentence_similarity_score(text: str, model, threshold: float, batch_size: int):
    import numpy as np

    parts = sentence_tokenize(text)
    if len(parts) < 2:
        return {"sentences": len(parts), "high_ratio": 0.0, "max_sim": 0.0}

    embeddings = model.encode(
        parts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sims = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    rows, cols = np.triu_indices(len(parts), k=1)
    scores = sims[rows, cols]

    return {
        "sentences": len(parts),
        "high_ratio": float(np.mean(scores >= threshold)),
        "max_sim": float(np.max(scores)),
    }


def score_pair(pair: dict, model, threshold: float, batch_size: int):
    chosen = sentence_similarity_score(pair["chosen"], model, threshold, batch_size)
    rejected = sentence_similarity_score(pair["rejected"], model, threshold, batch_size)
    return {
        **pair,
        "gap": rejected["high_ratio"] - chosen["high_ratio"],
        "chosen_high_ratio": chosen["high_ratio"],
        "rejected_high_ratio": rejected["high_ratio"],
        "chosen_max_sim": chosen["max_sim"],
        "rejected_max_sim": rejected["max_sim"],
        "chosen_sentences": chosen["sentences"],
        "rejected_sentences": rejected["sentences"],
    }


def top_count(total: int, percent: float):
    return max(1, min(total, math.ceil(total * percent / 100)))


def main():
    args = parse_args()
    files = res_files(args.result_dir, args.limit)
    model = load_model(args.device)
    output = args.output or default_output(args.result_dir)
    all_output = all_output_for(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with all_output.open("w", encoding="utf-8") as f:
        for i, path in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] {path.name}")
            row = score_pair(load_pair(path), model, args.threshold, args.batch_size)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    rows.sort(
        key=lambda row: (
            row["gap"],
            row["rejected_high_ratio"],
            row["rejected_max_sim"],
        ),
        reverse=True,
    )

    with output.open("w", encoding="utf-8") as f:
        for row in rows[: top_count(len(rows), args.top_percent)]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved {all_output}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
REPETITION_NGRAMS = (2, 3, 4)
SUMMARY_STAT_PERCENTILES = {
    "median": 50,
    "p25": 25,
    "p75": 75,
}
SUMMARY_STATS = ("mean", *SUMMARY_STAT_PERCENTILES.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate degeneration metrics for transfer result JSON files."
    )
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def mean(values):
    return float(np.mean(values)) if values else None


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def percentile(values, percentile_value):
    return float(np.percentile(values, percentile_value)) if values else None


def distribution_summary(values):
    return {
        "mean": mean(values),
        **{
            name: percentile(values, percentile_value)
            for name, percentile_value in SUMMARY_STAT_PERCENTILES.items()
        },
    }


def ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def stat_deltas(attack_summary, clean_summary):
    return {
        stat: attack_summary[stat] - clean_summary[stat]
        for stat in SUMMARY_STATS
    }


def stat_ratios(attack_summary, clean_summary):
    return {
        stat: ratio(attack_summary[stat], clean_summary[stat])
        for stat in SUMMARY_STATS
    }


def add_summary_metrics(metrics, prefix, summary):
    for stat in SUMMARY_STATS:
        metrics[f"{prefix}.{stat}"] = summary.get(stat)


def load_metadata(result_file):
    metadata_path = result_file.parent / "metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_completion_cap(metadata, result, sample_block, cli_max_new_tokens=None):
    for val in (
        cli_max_new_tokens,
        sample_block.get("completion_token_cap"),
        result.get("completion_token_cap"),
        result.get("max_new_tokens"),
        metadata.get("generation_config", {}).get("max_new_tokens"),
    ):
        if val and int(val) > 0:
            return int(val)
    return None


def args_metadata(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def repetition_ratio(tokens, n):
    counts = ngram_counts(tokens, n)
    total = sum(counts.values())
    return 1.0 - (len(counts) / total) if total else 0.0


def repetition_summary(texts_tokens, n):
    return distribution_summary(
        [repetition_ratio(tokens, n) for tokens in texts_tokens]
    )


def evaluate_repetition(clean_tokens, attack_tokens):
    metrics = {}
    for n in REPETITION_NGRAMS:
        clean_summary = repetition_summary(clean_tokens, n)
        attack_summary = repetition_summary(attack_tokens, n)
        delta_summary = stat_deltas(attack_summary, clean_summary)

        add_summary_metrics(metrics, f"repetition.clean.rep_{n}", clean_summary)
        add_summary_metrics(metrics, f"repetition.adv.rep_{n}", attack_summary)
        add_summary_metrics(metrics, f"repetition.delta.rep_{n}", delta_summary)

    return metrics


def get_lengths(sample_block, tokens):
    return sample_block.get("length") or [len(row) for row in tokens]


def max_hit_metrics(lengths, cap):
    if cap is None:
        return None, None

    hit_count = sum(length >= cap - 5 for length in lengths)
    return hit_count, ratio(hit_count, len(lengths))


def evaluate_length(
    sample,
    clean_tokens,
    attack_tokens,
    result,
    metadata,
    max_new_tokens,
):
    clean_lengths = get_lengths(sample["baseline"], clean_tokens)
    attack_lengths = get_lengths(sample["adv"], attack_tokens)
    clean_summary = distribution_summary(clean_lengths)
    attack_summary = distribution_summary(attack_lengths)

    clean_cap = resolve_completion_cap(
        metadata,
        result,
        sample["baseline"],
        max_new_tokens,
    )
    attack_cap = resolve_completion_cap(
        metadata,
        result,
        sample["adv"],
        max_new_tokens,
    )
    clean_hit_count, clean_hit_rate = max_hit_metrics(clean_lengths, clean_cap)
    attack_hit_count, attack_hit_rate = max_hit_metrics(attack_lengths, attack_cap)

    metrics = {}
    add_summary_metrics(metrics, "length.clean", clean_summary)
    add_summary_metrics(metrics, "length.adv", attack_summary)
    add_summary_metrics(
        metrics,
        "length.adv_over_clean",
        stat_ratios(attack_summary, clean_summary),
    )
    metrics.update(
        {
            "length.clean.max_hit_count": clean_hit_count,
            "length.clean.max_hit_rate": clean_hit_rate,
            "length.adv.max_hit_count": attack_hit_count,
            "length.adv.max_hit_rate": attack_hit_rate,
        }
    )
    return metrics


class SemanticScorer:
    def __init__(self, device="auto"):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        transformers_logging.set_verbosity_error()
        self.device = device
        self.embedding_model_name = EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device,
        )

    @staticmethod
    def score_block(clean_intra, adv_intra, cross):
        return {
            "clean_intra": clean_intra,
            "adv_intra": adv_intra,
            "cross": cross,
            "cross_minus_clean_intra": stat_deltas(cross, clean_intra),
        }

    def encode(self, texts):
        return self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).float()

    @staticmethod
    def cosine_score(scores):
        values = scores.clamp(-1, 1).detach().cpu().tolist()
        return distribution_summary(values)

    @staticmethod
    def intra_cosine(embeddings):
        similarities = (embeddings @ embeddings.T).clamp(-1, 1).detach().cpu().numpy()
        rows, cols = np.triu_indices(len(similarities), k=1)
        return distribution_summary(similarities[rows, cols].tolist())

    def cross_cosine(self, source_embeddings, target_embeddings):
        return self.cosine_score((source_embeddings @ target_embeddings.T).flatten())

    def cosine(self, clean_embeddings, attack_embeddings):
        return self.score_block(
            self.intra_cosine(clean_embeddings),
            self.intra_cosine(attack_embeddings),
            self.cross_cosine(clean_embeddings, attack_embeddings),
        )

    def evaluate(self, clean_texts, attack_texts):
        clean_embeddings = self.encode(clean_texts)
        attack_embeddings = self.encode(attack_texts)

        metrics = {}
        for relation, summary in self.cosine(
            clean_embeddings,
            attack_embeddings,
        ).items():
            add_summary_metrics(metrics, f"semantic.cosine.{relation}", summary)
        return metrics


def evaluate_sample(sample, result, metadata, semantic_scorer, max_new_tokens=None):
    clean_texts = sample["baseline"]["answer"]
    attack_texts = sample["adv"]["answer"]
    clean_tokens = [tokenize(text) for text in clean_texts]
    attack_tokens = [tokenize(text) for text in attack_texts]

    metrics = {}
    metrics.update(
        evaluate_length(
            sample,
            clean_tokens,
            attack_tokens,
            result,
            metadata,
            max_new_tokens,
        )
    )
    metrics.update(evaluate_repetition(clean_tokens, attack_tokens))
    if semantic_scorer is not None:
        metrics.update(semantic_scorer.evaluate(clean_texts, attack_texts))

    return {
        "source": sample["source"],
        "index": sample["index"],
        "instruction": sample["instruction"],
        "metrics": metrics,
    }


def summarize_items(items):
    values = defaultdict(list)
    for item in items:
        for key, value in item["metrics"].items():
            if isinstance(value, (int, float)):
                values[key].append(value)

    return {
        "item_count": len(items),
        "sample_means": {
            key: mean(metric_values)
            for key, metric_values in sorted(values.items())
            if metric_values
        },
    }


def semantic_metadata(semantic_scorer):
    return {
        "enabled": semantic_scorer is not None,
        "embedding_model": semantic_scorer.embedding_model_name
        if semantic_scorer
        else None,
        "device": semantic_scorer.device if semantic_scorer else None,
    }


def evaluation_metadata(args, metadata):
    return {
        "args": args_metadata(args),
        "input_metadata_path": str(args.result_file.parent / "metadata.json"),
        "input_metadata": metadata,
    }


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    with args.result_file.open("r", encoding="utf-8") as f:
        result = json.load(f)
    metadata = load_metadata(args.result_file)

    samples = result["samples"]
    if args.limit is not None:
        samples = samples[: args.limit]

    semantic_scorer = (
        None
        if args.skip_semantic
        else SemanticScorer(
            args.device,
        )
    )
    items = [
        evaluate_sample(
            sample,
            result,
            metadata,
            semantic_scorer,
            args.max_new_tokens,
        )
        for sample in tqdm(samples, desc="Evaluating samples")
    ]

    payload = {
        "metric_version": "degeneration-v3-flat",
        "input_path": str(args.result_file),
        "schema": "transfer",
        "evaluation_metadata": evaluation_metadata(args, metadata),
        "semantic": semantic_metadata(semantic_scorer),
        "source_metadata": {
            key: value for key, value in result.items() if key != "samples"
        },
        "summary": summarize_items(items),
        "items": items,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved degeneration metrics to: {args.output}")


if __name__ == "__main__":
    main()

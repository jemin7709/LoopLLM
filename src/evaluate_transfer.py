#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

from tqdm import tqdm

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_BERTSCORE_MODEL = "jhu-clsp/mmBERT-base"
REPETITION_NGRAMS = (2, 3, 4)
COSINE_KEYS = ("embedding_cosine",)
BERTSCORE_KEYS = (
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)
SUMMARY_PERCENTILES = (25, 75, 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate degeneration metrics for transfer result JSON files."
    )
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def std(values):
    average = mean(values)
    return sqrt(sum((value - average) ** 2 for value in values) / len(values))


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def delta_metrics(attack, clean):
    return {key: attack[key] - clean[key] for key in clean}


def flatten_metrics(metrics, prefix=""):
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from flatten_metrics(value, f"{metric_name}.")
        else:
            yield metric_name, value


def percentile(values, percentile_value):
    sorted_values = sorted(values)
    rank = (percentile_value / 100) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class RepetitionScorer:
    def __init__(self, ngrams=REPETITION_NGRAMS):
        self.ngrams = ngrams

    @staticmethod
    def ngram_counts(tokens, n):
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @classmethod
    def rep_n(cls, tokens, n):
        counts = cls.ngram_counts(tokens, n)
        total = sum(counts.values())
        return 1.0 - (len(counts) / total) if total else 0.0

    def score(self, texts_tokens):
        return {
            f"rep_{n}": mean([self.rep_n(tokens, n) for tokens in texts_tokens])
            for n in self.ngrams
        }

    def evaluate(self, clean_tokens, attack_tokens):
        clean_repetition = self.score(clean_tokens)
        attack_repetition = self.score(attack_tokens)
        return {
            "clean": clean_repetition,
            "attack": attack_repetition,
            "delta": delta_metrics(attack_repetition, clean_repetition),
        }


class LengthScorer:
    def evaluate(self, clean_tokens, attack_tokens):
        ratios = [
            len(attack) / len(clean)
            for clean, attack in zip(clean_tokens, attack_tokens, strict=True)
        ]
        return {"length_ratio": mean(ratios)}


class SemanticScorer:
    def __init__(self, device="auto"):
        import torch
        from sentence_transformers import SentenceTransformer
        from torchmetrics.text import BERTScore
        from transformers.utils import logging as transformers_logging

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        transformers_logging.set_verbosity_error()
        self.device = device
        self.embedding_model_name = EMBEDDING_MODEL
        self.bertscore_model_name = DEFAULT_BERTSCORE_MODEL
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device,
        )
        self.bertscore_metric = BERTScore(
            model_name_or_path=DEFAULT_BERTSCORE_MODEL,
            device=self.device,
            truncation=True,
            max_length=8192,
        )

    @staticmethod
    def empty_scores(keys):
        return {key: 0.0 for key in keys}

    @classmethod
    def empty_score_block(cls, keys):
        clean_intra = cls.empty_scores(keys)
        adv_intra = cls.empty_scores(keys)
        clean_adv_cross = cls.empty_scores(keys)
        return {
            "clean_intra": clean_intra,
            "adv_intra": adv_intra,
            "clean_adv_cross": clean_adv_cross,
            "delta": delta_metrics(clean_adv_cross, clean_intra),
        }

    @classmethod
    def empty_scores_block(cls):
        return {
            "cosine": cls.empty_score_block(COSINE_KEYS),
            "bertscore": cls.empty_score_block(BERTSCORE_KEYS),
        }

    @staticmethod
    def intra_pairs(texts):
        sources = []
        targets = []
        for source, target in combinations(texts, 2):
            sources.append(source)
            targets.append(target)
        return sources, targets

    @staticmethod
    def cross_pairs(sources, targets):
        return (
            [source for source in sources for _ in targets],
            [target for _ in sources for target in targets],
        )

    @staticmethod
    def score_block(clean_intra, adv_intra, clean_adv_cross):
        return {
            "clean_intra": clean_intra,
            "adv_intra": adv_intra,
            "clean_adv_cross": clean_adv_cross,
            "delta": delta_metrics(clean_adv_cross, clean_intra),
        }

    def encode(self, texts):
        return self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @staticmethod
    def cosine_score(scores):
        return {"embedding_cosine": scores.clamp(-1, 1).mean().item()}

    def intra_cosine(self, embeddings):
        pair_count = len(embeddings) * (len(embeddings) - 1) / 2
        similarities = (embeddings @ embeddings.T).clamp(-1, 1)
        return {
            "embedding_cosine": (
                similarities.triu(diagonal=1).sum() / pair_count
            ).item()
        }

    def cross_cosine(self, source_embeddings, target_embeddings):
        return self.cosine_score((source_embeddings @ target_embeddings.T).flatten())

    def cosine(self, clean_embeddings, attack_embeddings):
        return self.score_block(
            self.intra_cosine(clean_embeddings),
            self.intra_cosine(attack_embeddings),
            self.cross_cosine(clean_embeddings, attack_embeddings),
        )

    @staticmethod
    def bertscore_scores(scores, start, end):
        return {
            "bertscore_precision": scores["precision"][start:end].mean().item(),
            "bertscore_recall": scores["recall"][start:end].mean().item(),
            "bertscore_f1": scores["f1"][start:end].mean().item(),
        }

    def bertscore(self, clean_texts, attack_texts):
        clean_sources, clean_targets = self.intra_pairs(clean_texts)
        attack_sources, attack_targets = self.intra_pairs(attack_texts)
        cross_sources, cross_targets = self.cross_pairs(clean_texts, attack_texts)

        sources = clean_sources + attack_sources + cross_sources
        targets = clean_targets + attack_targets + cross_targets
        scores = self.bertscore_metric(preds=targets, target=sources)

        clean_end = len(clean_sources)
        attack_end = clean_end + len(attack_sources)
        return self.score_block(
            self.bertscore_scores(scores, 0, clean_end),
            self.bertscore_scores(scores, clean_end, attack_end),
            self.bertscore_scores(scores, attack_end, len(sources)),
        )

    def evaluate(self, clean_texts, attack_texts):
        clean_embeddings = self.encode(clean_texts)
        attack_embeddings = self.encode(attack_texts)
        return {
            "cosine": self.cosine(clean_embeddings, attack_embeddings),
            "bertscore": self.bertscore(clean_texts, attack_texts),
        }


def evaluate_sample(sample, repetition_scorer, length_scorer, semantic_scorer):
    clean_texts = sample["baseline"]["answer"]
    attack_texts = sample["adv"]["answer"]
    clean_tokens = [tokenize(text) for text in clean_texts]
    attack_tokens = [tokenize(text) for text in attack_texts]

    return {
        "source": sample["source"],
        "index": sample["index"],
        "instruction": sample["instruction"],
        "repetition": repetition_scorer.evaluate(clean_tokens, attack_tokens),
        "semantic": (
            semantic_scorer.evaluate(clean_texts, attack_texts)
            if semantic_scorer
            else SemanticScorer.empty_scores_block()
        ),
        "length": length_scorer.evaluate(clean_tokens, attack_tokens),
    }


def summarize_items(items):
    values = defaultdict(list)
    for item in items:
        metrics = {key: item[key] for key in ("repetition", "semantic", "length")}
        for key, value in flatten_metrics(metrics):
            values[key].append(value)

    metric_values = {
        key: numbers for key, numbers in sorted(values.items()) if numbers
    }
    return {
        "item_count": len(items),
        "means": {
            key: mean(numbers)
            for key, numbers in metric_values.items()
        },
        "stds": {
            key: std(numbers)
            for key, numbers in metric_values.items()
        },
        "medians": {
            key: percentile(numbers, 50)
            for key, numbers in metric_values.items()
        },
        "percentiles": {
            f"p{percentile_value}": {
                key: percentile(numbers, percentile_value)
                for key, numbers in metric_values.items()
            }
            for percentile_value in SUMMARY_PERCENTILES
        },
        "mins": {
            key: min(numbers)
            for key, numbers in metric_values.items()
        },
        "maxs": {
            key: max(numbers)
            for key, numbers in metric_values.items()
        },
    }


def semantic_metadata(semantic_scorer):
    return {
        "enabled": semantic_scorer is not None,
        "embedding_model": semantic_scorer.embedding_model_name
        if semantic_scorer
        else None,
        "bertscore_model": semantic_scorer.bertscore_model_name
        if semantic_scorer
        else None,
        "device": semantic_scorer.device if semantic_scorer else None,
    }


def main():
    args = parse_args()

    with args.result_file.open("r", encoding="utf-8") as f:
        result = json.load(f)

    samples = result["samples"]
    if args.limit:
        samples = samples[: args.limit]

    repetition_scorer = RepetitionScorer()
    length_scorer = LengthScorer()
    semantic_scorer = None if args.skip_semantic else SemanticScorer(args.device)
    items = [
        evaluate_sample(sample, repetition_scorer, length_scorer, semantic_scorer)
        for sample in tqdm(samples, desc="Evaluating samples")
    ]

    payload = {
        "metric_version": "degeneration-v2-oop",
        "input_path": str(args.result_file),
        "schema": "transfer",
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

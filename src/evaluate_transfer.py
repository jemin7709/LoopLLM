import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
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
METRIC_PERCENTILES = {
    "median": 50,
    "p05": 5,
    "p10": 10,
    "p90": 90,
    "p95": 95,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate degeneration metrics for transfer result JSON files."
    )
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--semantic-top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def mean(values):
    return float(np.mean(values)) if values else None


def std(values):
    return float(np.std(values)) if values else None


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def metric_delta(attack_value, clean_value):
    if attack_value is None or clean_value is None:
        return None
    return attack_value - clean_value


def delta_metrics(attack, clean):
    return {k: metric_delta(attack.get(k), clean.get(k)) for k in clean}


def flatten_metrics(metrics, prefix=""):
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from flatten_metrics(value, f"{metric_name}.")
        else:
            yield metric_name, value


def percentile(values, percentile_value):
    return float(np.percentile(values, percentile_value)) if values else None


def distribution_summary(values):
    return {
        "mean": mean(values),
        **{k: percentile(values, p) for k, p in METRIC_PERCENTILES.items()},
    }


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def load_metadata(result_file):
    metadata_path = result_file.parent / "metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_max_token_cap(metadata, result, sample, cli_max_new_tokens=None):
    sources = [
        ("sample.adv.completion_token_cap", sample["adv"].get("completion_token_cap")),
        ("result.completion_token_cap", result.get("completion_token_cap")),
        ("result.max_new_tokens", result.get("max_new_tokens")),
        (
            "metadata.generation_config.max_new_tokens",
            metadata.get("generation_config", {}).get("max_new_tokens"),
        ),
        ("cli.max_new_tokens", cli_max_new_tokens),
    ]
    for source, val in sources:
        if val and int(val) > 0:
            return int(val), source
    return None, "none"


def args_metadata(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def effective_top_k_metadata(values):
    values = sorted(values)
    if not values:
        return None
    return values[0] if len(values) == 1 else values


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
        scores = {}
        for n in self.ngrams:
            summary = distribution_summary(
                [self.rep_n(tokens, n) for tokens in texts_tokens]
            )
            scores.update(
                {
                    f"rep_{n}": summary["mean"],
                    f"rep_{n}_median": summary["median"],
                    f"rep_{n}_p90": summary["p90"],
                    f"rep_{n}_p95": summary["p95"],
                }
            )
        return scores

    def evaluate(self, clean_tokens, attack_tokens):
        clean_repetition = self.score(clean_tokens)
        attack_repetition = self.score(attack_tokens)
        return {
            "clean": clean_repetition,
            "attack": attack_repetition,
            "delta": delta_metrics(attack_repetition, clean_repetition),
        }


class LengthScorer:
    @staticmethod
    def lengths(sample_block, tokens):
        if "length" in sample_block:
            return sample_block["length"]
        return [len(row) for row in tokens]

    def evaluate(self, sample, clean_tokens, attack_tokens, max_token_cap, cap_source):
        clean_lengths = self.lengths(sample["baseline"], clean_tokens)
        attack_lengths = self.lengths(sample["adv"], attack_tokens)
        ratios = [
            value
            for attack, clean in zip(attack_lengths, clean_lengths, strict=True)
            if (value := ratio(attack, clean)) is not None
        ]

        clean_summary = distribution_summary(clean_lengths)
        attack_summary = distribution_summary(attack_lengths)

        max_token_hit_count = (
            sum(length >= max_token_cap - 5 for length in attack_lengths)
            if max_token_cap
            else None
        )
        max_token_hit_rate = (
            ratio(max_token_hit_count, len(attack_lengths)) if max_token_cap else None
        )

        return {
            "length_ratio": mean(ratios),
            "clean_len_median": clean_summary["median"],
            "clean_len_p90": clean_summary["p90"],
            "clean_len_p95": clean_summary["p95"],
            "adv_len_median": attack_summary["median"],
            "adv_len_p90": attack_summary["p90"],
            "adv_len_p95": attack_summary["p95"],
            "median_len_ratio": ratio(
                attack_summary["median"], clean_summary["median"]
            ),
            "tail_len_ratio_p90": ratio(attack_summary["p90"], clean_summary["p90"]),
            "tail_len_ratio_p95": ratio(attack_summary["p95"], clean_summary["p95"]),
            "tail_vs_clean_median_ratio_p90": ratio(
                attack_summary["p90"], clean_summary["median"]
            ),
            "tail_vs_clean_median_ratio_p95": ratio(
                attack_summary["p95"], clean_summary["median"]
            ),
            "max_token_cap": max_token_cap,
            "max_token_cap_source": cap_source,
            "max_token_hit_count": max_token_hit_count,
            "max_token_hit_rate": max_token_hit_rate,
        }


class SemanticScorer:
    def __init__(self, device="auto", use_bertscore=True, semantic_top_k=5):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        transformers_logging.set_verbosity_error()
        self.device = device
        self.semantic_top_k = semantic_top_k
        self.effective_clean_to_clean_top_k_values = set()
        self.effective_adv_to_clean_top_k_values = set()
        self.embedding_model_name = EMBEDDING_MODEL
        self.bertscore_model_name = DEFAULT_BERTSCORE_MODEL if use_bertscore else None
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.device,
        )
        self.bertscore_metric = None
        if use_bertscore:
            from torchmetrics.text import BERTScore

            self.bertscore_metric = BERTScore(
                model_name_or_path=DEFAULT_BERTSCORE_MODEL,
                device=self.device,
                truncation=True,
                max_length=8192,
            )

    @classmethod
    def empty_score_block(cls, keys):
        return {
            "clean_intra": {k: 0.0 for k in keys},
            "adv_intra": {k: 0.0 for k in keys},
            "clean_adv_cross": {k: 0.0 for k in keys},
            "delta": {k: 0.0 for k in keys},
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

    @staticmethod
    def cosine_score_from_values(values):
        summary = distribution_summary(values)
        return {"embedding_cosine": summary["mean"], **summary}

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
        return SemanticScorer.cosine_score_from_values(values)

    def intra_cosine(self, embeddings):
        similarities = (embeddings @ embeddings.T).clamp(-1, 1).detach().cpu().numpy()
        rows, cols = np.triu_indices(len(similarities), k=1)
        values = similarities[rows, cols].tolist()
        return self.cosine_score_from_values(values)

    def cross_cosine(self, source_embeddings, target_embeddings):
        return self.cosine_score((source_embeddings @ target_embeddings.T).flatten())

    def cosine(self, clean_embeddings, attack_embeddings):
        block = self.score_block(
            self.intra_cosine(clean_embeddings),
            self.intra_cosine(attack_embeddings),
            self.cross_cosine(clean_embeddings, attack_embeddings),
        )
        block["delta"].update(
            {
                f"semantic_shift_{key}": block["delta"][key]
                for key in ("mean", *METRIC_PERCENTILES.keys())
            }
        )
        block.update(self.semantic_outliers(clean_embeddings, attack_embeddings))
        return block

    def semantic_outliers(self, clean_embeddings, attack_embeddings):
        clean_count = len(clean_embeddings)
        attack_count = len(attack_embeddings)
        effective_k = min(self.semantic_top_k, clean_count - 1)
        empty = {
            "clean_to_clean_topk_p05": None,
            **{
                f"adv_to_clean_topk_mean_{key}": None
                for key in ("mean", *METRIC_PERCENTILES.keys())
            },
            "semantic_outlier_count": None,
            "semantic_outlier_rate": None,
        }
        if effective_k <= 0:
            return empty

        self.effective_clean_to_clean_top_k_values.add(effective_k)
        clean_similarities = (clean_embeddings @ clean_embeddings.T).clamp(-1, 1)
        clean_topk_means = []
        for index in range(clean_count):
            candidates = [
                clean_similarities[index, other].item()
                for other in range(clean_count)
                if other != index
            ]
            topk_candidates = sorted(candidates, reverse=True)[:effective_k]
            clean_topk_means.append(mean(topk_candidates))

        threshold = percentile(clean_topk_means, 5)
        adv_effective_k = min(self.semantic_top_k, clean_count)
        self.effective_adv_to_clean_top_k_values.add(adv_effective_k)
        cross_similarities = (attack_embeddings @ clean_embeddings.T).clamp(-1, 1)
        adv_topk_means = []
        for index in range(attack_count):
            candidates = cross_similarities[index].detach().cpu().tolist()
            adv_topk_means.append(
                mean(sorted(candidates, reverse=True)[:adv_effective_k])
            )

        adv_summary = distribution_summary(adv_topk_means)
        outlier_count = sum(value < threshold for value in adv_topk_means)
        return {
            "clean_to_clean_topk_p05": threshold,
            **{
                f"adv_to_clean_topk_mean_{key}": value
                for key, value in adv_summary.items()
            },
            "semantic_outlier_count": outlier_count,
            "semantic_outlier_rate": ratio(outlier_count, attack_count),
        }

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
            "bertscore": (
                self.bertscore(clean_texts, attack_texts)
                if self.bertscore_metric is not None
                else self.empty_score_block(BERTSCORE_KEYS)
            ),
        }


def evaluate_sample(
    sample,
    result,
    metadata,
    repetition_scorer,
    length_scorer,
    semantic_scorer,
    cli_max_new_tokens=None,
):
    clean_texts = sample["baseline"]["answer"]
    attack_texts = sample["adv"]["answer"]
    clean_tokens = [tokenize(text) for text in clean_texts]
    attack_tokens = [tokenize(text) for text in attack_texts]
    max_token_cap, cap_source = resolve_max_token_cap(
        metadata,
        result,
        sample,
        cli_max_new_tokens,
    )

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
        "length": length_scorer.evaluate(
            sample,
            clean_tokens,
            attack_tokens,
            max_token_cap,
            cap_source,
        ),
    }


def summarize_items(items):
    values = defaultdict(list)
    for item in items:
        for key, value in flatten_metrics(
            {k: item[k] for k in ("repetition", "semantic", "length")}
        ):
            if isinstance(value, (int, float)):
                values[key].append(value)

    metrics = {k: v for k, v in sorted(values.items()) if v}
    return {
        "item_count": len(items),
        "means": {k: mean(v) for k, v in metrics.items()},
        "stds": {k: std(v) for k, v in metrics.items()},
        "medians": {k: percentile(v, 50) for k, v in metrics.items()},
        "percentiles": {
            f"p{p}": {k: percentile(v, p) for k, v in metrics.items()}
            for p in SUMMARY_PERCENTILES
        },
        "mins": {k: min(v) for k, v in metrics.items()},
        "maxs": {k: max(v) for k, v in metrics.items()},
    }


def semantic_metadata(semantic_scorer, semantic_top_k):
    return {
        "enabled": semantic_scorer is not None,
        "embedding_model": semantic_scorer.embedding_model_name
        if semantic_scorer
        else None,
        "bertscore_model": semantic_scorer.bertscore_model_name
        if semantic_scorer
        else None,
        "device": semantic_scorer.device if semantic_scorer else None,
        "semantic_top_k": semantic_top_k,
        "effective_clean_to_clean_top_k": effective_top_k_metadata(
            semantic_scorer.effective_clean_to_clean_top_k_values
        )
        if semantic_scorer
        else None,
        "effective_adv_to_clean_top_k": effective_top_k_metadata(
            semantic_scorer.effective_adv_to_clean_top_k_values
        )
        if semantic_scorer
        else None,
    }


def evaluation_metadata(args, metadata):
    return {
        "args": args_metadata(args),
        "input_metadata_path": str(args.result_file.parent / "metadata.json"),
        "input_metadata": metadata,
    }


def main():
    args = parse_args()
    if args.semantic_top_k <= 0:
        raise ValueError("--semantic-top-k must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    with args.result_file.open("r", encoding="utf-8") as f:
        result = json.load(f)
    metadata = load_metadata(args.result_file)

    samples = result["samples"]
    if args.limit:
        samples = samples[: args.limit]

    repetition_scorer = RepetitionScorer()
    length_scorer = LengthScorer()
    semantic_scorer = (
        None
        if args.skip_semantic
        else SemanticScorer(
            args.device,
            use_bertscore=not args.skip_bertscore,
            semantic_top_k=args.semantic_top_k,
        )
    )
    items = [
        evaluate_sample(
            sample,
            result,
            metadata,
            repetition_scorer,
            length_scorer,
            semantic_scorer,
            args.max_new_tokens,
        )
        for sample in tqdm(samples, desc="Evaluating samples")
    ]

    payload = {
        "metric_version": "degeneration-v2-oop",
        "input_path": str(args.result_file),
        "schema": "transfer",
        "evaluation_metadata": evaluation_metadata(args, metadata),
        "semantic": semantic_metadata(semantic_scorer, args.semantic_top_k),
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

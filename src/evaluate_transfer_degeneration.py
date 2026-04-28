#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_BERTSCORE_MODEL = "roberta-large"
REPETITION_NGRAMS = (2, 3, 4)
COSINE_KEYS = ("embedding_cosine",)
BERTSCORE_KEYS = (
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)


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


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def empty_scores(keys):
    return {key: 0.0 for key in keys}


def delta_metrics(attack, clean):
    return {key: attack[key] - clean[key] for key in clean}


def flatten_metrics(metrics, prefix=""):
    for key, value in metrics.items():
        metric_name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from flatten_metrics(value, f"{metric_name}.")
        else:
            yield metric_name, value


def intra_pairs(texts):
    sources = []
    targets = []
    for source, target in combinations(texts, 2):
        sources.append(source)
        targets.append(target)
    return sources, targets


def cross_pairs(sources, targets):
    return (
        [source for source in sources for _ in targets],
        [target for _ in sources for target in targets],
    )


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rep_n(tokens, n):
    counts = ngram_counts(tokens, n)
    total = sum(counts.values())
    return 1.0 - (len(counts) / total) if total else 0.0


def repetition_scores(texts_tokens):
    return {
        f"rep_{n}": mean([rep_n(tokens, n) for tokens in texts_tokens])
        for n in REPETITION_NGRAMS
    }


def length_scores(clean_tokens, attack_tokens):
    ratios = [
        len(attack) / len(clean)
        for clean, attack in zip(clean_tokens, attack_tokens, strict=True)
    ]
    return {"length_ratio": mean(ratios)}


def empty_score_block(keys):
    clean_intra = empty_scores(keys)
    adv_intra = empty_scores(keys)
    clean_adv_cross = empty_scores(keys)
    return {
        "clean_intra": clean_intra,
        "adv_intra": adv_intra,
        "clean_adv_cross": clean_adv_cross,
        "delta": delta_metrics(clean_adv_cross, clean_intra),
    }


def score_block(score_fn, clean_texts, attack_texts):
    clean_intra = score_fn(clean_texts)
    adv_intra = score_fn(attack_texts)
    clean_adv_cross = score_fn(clean_texts, attack_texts)
    return {
        "clean_intra": clean_intra,
        "adv_intra": adv_intra,
        "clean_adv_cross": clean_adv_cross,
        "delta": delta_metrics(clean_adv_cross, clean_intra),
    }


def empty_semantic_scores():
    return {
        "cosine": empty_score_block(COSINE_KEYS),
        "bertscore": empty_score_block(BERTSCORE_KEYS),
    }


class SemanticScorer:
    def __init__(self, device="auto"):
        import torch
        from sentence_transformers import SentenceTransformer
        from torchmetrics.text import BERTScore

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
        )

    def encode(self, texts):
        return self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def cosine(self, sources, targets=None):
        if targets is None:
            embeddings = self.encode(sources)
            scores = [
                (embeddings[i] @ embeddings[j]).clamp(-1, 1).item()
                for i, j in combinations(range(len(sources)), 2)
            ]
        else:
            source_embeddings = self.encode(sources)
            target_embeddings = self.encode(targets)
            scores = (
                (source_embeddings @ target_embeddings.T)
                .flatten()
                .clamp(-1, 1)
                .detach()
                .cpu()
                .tolist()
            )
        return {"embedding_cosine": mean(scores)}

    def bertscore(self, sources, targets=None):
        if targets is None:
            sources, targets = intra_pairs(sources)
        else:
            sources, targets = cross_pairs(sources, targets)

        scores = self.bertscore_metric(preds=targets, target=sources)
        return {
            "bertscore_precision": mean(scores["precision"].tolist()),
            "bertscore_recall": mean(scores["recall"].tolist()),
            "bertscore_f1": mean(scores["f1"].tolist()),
        }

    def evaluate(self, clean_texts, attack_texts):
        return {
            "cosine": score_block(self.cosine, clean_texts, attack_texts),
            "bertscore": score_block(self.bertscore, clean_texts, attack_texts),
        }


def evaluate_sample(sample, semantic_scorer):
    clean_texts = sample["baseline"]["answer"]
    attack_texts = sample["adv"]["answer"]
    clean_tokens = [tokenize(text) for text in clean_texts]
    attack_tokens = [tokenize(text) for text in attack_texts]

    clean_repetition = repetition_scores(clean_tokens)
    attack_repetition = repetition_scores(attack_tokens)

    return {
        "source": sample["source"],
        "index": sample["index"],
        "instruction": sample["instruction"],
        "repetition": {
            "clean": clean_repetition,
            "attack": attack_repetition,
            "delta": delta_metrics(attack_repetition, clean_repetition),
        },
        "semantic": (
            semantic_scorer.evaluate(clean_texts, attack_texts)
            if semantic_scorer
            else empty_semantic_scores()
        ),
        "length": length_scores(clean_tokens, attack_tokens),
    }


def summarize_items(items):
    values = defaultdict(list)
    for item in items:
        metrics = {key: item[key] for key in ("repetition", "semantic", "length")}
        for key, value in flatten_metrics(metrics):
            values[key].append(value)

    return {
        "item_count": len(items),
        "means": {
            key: sum(numbers) / len(numbers)
            for key, numbers in sorted(values.items())
            if numbers
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

    semantic_scorer = None if args.skip_semantic else SemanticScorer(args.device)
    items = [evaluate_sample(sample, semantic_scorer) for sample in samples]

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

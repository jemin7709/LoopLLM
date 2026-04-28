#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate degeneration metrics for raw_output result directories."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--semantic-model", default="bert-base-uncased")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-semantic-tokens", type=int, default=512)
    return parser.parse_args()


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def text_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def ngram_counts(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rep_n(text, n):
    tokens = tokenize(text)
    counts = ngram_counts(tokens, n)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - (len(counts) / total)


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def average_repetition(texts):
    return {
        f"rep_{n}": mean([rep_n(text, n) for text in texts])
        for n in (2, 3, 4)
    }


def delta_metrics(attack, clean):
    return {
        key: (
            attack[key] - clean[key]
            if attack.get(key) is not None and clean.get(key) is not None
            else None
        )
        for key in clean
    }


def closest_reference_length(candidate_len, references):
    lengths = [len(ref) for ref in references if ref]
    if not lengths:
        return 0
    return min(lengths, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len))


def modified_precision(candidate, references, n):
    candidate_counts = ngram_counts(candidate, n)
    total = sum(candidate_counts.values())
    if total == 0:
        return None

    max_reference_counts = Counter()
    for reference in references:
        max_reference_counts |= ngram_counts(reference, n)

    clipped = sum(
        min(count, max_reference_counts[gram])
        for gram, count in candidate_counts.items()
    )
    return (clipped + 1.0) / (total + 1.0)


def bleu_score(candidate_text, reference_texts, max_order=4):
    candidate = tokenize(candidate_text)
    references = [tokenize(text) for text in reference_texts if str(text).strip()]
    if not candidate or not references:
        return None

    order = min(max_order, len(candidate))
    precisions = []
    for n in range(1, order + 1):
        precision = modified_precision(candidate, references, n)
        if precision is not None:
            precisions.append(precision)
    if not precisions:
        return None

    candidate_len = len(candidate)
    reference_len = closest_reference_length(candidate_len, references)
    brevity_penalty = (
        1.0
        if candidate_len > reference_len
        else math.exp(1.0 - (reference_len / candidate_len))
    )
    log_precision = sum(math.log(value) for value in precisions) / len(precisions)
    return brevity_penalty * math.exp(log_precision)


def self_bleu(texts):
    texts = [text for text in texts if str(text).strip()]
    if len(texts) < 2:
        return None
    scores = []
    for index, text in enumerate(texts):
        references = texts[:index] + texts[index + 1 :]
        scores.append(bleu_score(text, references))
    return mean(scores)


def length_summary(clean_texts, attack_texts):
    clean_lengths = [len(tokenize(text)) for text in clean_texts]
    attack_lengths = [len(tokenize(text)) for text in attack_texts]
    pair_count = min(len(clean_lengths), len(attack_lengths))
    ratios = []
    for index in range(pair_count):
        if clean_lengths[index] > 0:
            ratios.append(attack_lengths[index] / clean_lengths[index])
    return {
        "clean_mean_tokens": mean(clean_lengths),
        "attack_mean_tokens": mean(attack_lengths),
        "length_ratio": mean(ratios),
        "length_pair_count": pair_count,
    }


class SemanticScorer:
    def __init__(self, model_name, device, max_tokens):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = self.resolve_device(device)
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.cache = {}

    def resolve_device(self, device):
        if device == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device

    def encode(self, text):
        text = str(text)
        if text in self.cache:
            return self.cache[text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            return_special_tokens_mask=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.torch.no_grad():
            output = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        hidden = output.last_hidden_state[0]
        attention_mask = encoded["attention_mask"][0].bool()
        special_mask = encoded.get("special_tokens_mask")
        if special_mask is not None:
            token_mask = attention_mask & ~special_mask[0].bool()
        else:
            token_mask = attention_mask
        token_embeddings = hidden[token_mask]
        if token_embeddings.numel() == 0:
            token_embeddings = hidden[attention_mask]

        normalized_tokens = self.torch.nn.functional.normalize(
            token_embeddings, p=2, dim=1
        ).cpu()
        pooled = self.torch.nn.functional.normalize(
            token_embeddings.mean(dim=0), p=2, dim=0
        ).cpu()
        self.cache[text] = (normalized_tokens, pooled)
        return self.cache[text]

    def score_pair(self, clean_text, attack_text):
        clean_tokens, clean_pooled = self.encode(clean_text)
        attack_tokens, attack_pooled = self.encode(attack_text)

        cosine = float(self.torch.dot(clean_pooled, attack_pooled).clamp(-1, 1))
        similarity = attack_tokens @ clean_tokens.T
        precision = float(similarity.max(dim=1).values.mean())
        recall = float(similarity.max(dim=0).values.mean())
        f1 = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
        return {
            "embedding_cosine": cosine,
            "bertscore_precision": precision,
            "bertscore_recall": recall,
            "bertscore_f1": f1,
        }


def average_pair_scores(scores):
    if not scores:
        return {
            "embedding_cosine": None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "pair_count": 0,
        }
    keys = ["embedding_cosine", "bertscore_precision", "bertscore_recall", "bertscore_f1"]
    result = {key: mean([score[key] for score in scores]) for key in keys}
    result["pair_count"] = len(scores)
    return result


def semantic_summary(clean_texts, attack_texts, scorer):
    pair_count = min(len(clean_texts), len(attack_texts))
    clean_attack_scores = [
        scorer.score_pair(clean_texts[index], attack_texts[index])
        for index in range(pair_count)
    ]
    clean_attack = average_pair_scores(clean_attack_scores)

    clean_clean_scores = [
        scorer.score_pair(first, second)
        for first, second in itertools.combinations(clean_texts, 2)
    ]
    clean_clean = average_pair_scores(clean_clean_scores)

    return {
        **clean_attack,
        "clean_clean_embedding_cosine": clean_clean["embedding_cosine"],
        "clean_clean_bertscore_f1": clean_clean["bertscore_f1"],
        "clean_clean_pair_count": clean_clean["pair_count"],
        "embedding_similarity_drop": (
            clean_clean["embedding_cosine"] - clean_attack["embedding_cosine"]
            if clean_clean["embedding_cosine"] is not None
            and clean_attack["embedding_cosine"] is not None
            else None
        ),
        "bertscore_f1_drop": (
            clean_clean["bertscore_f1"] - clean_attack["bertscore_f1"]
            if clean_clean["bertscore_f1"] is not None
            and clean_attack["bertscore_f1"] is not None
            else None
        ),
    }


def skipped_semantic_summary(clean_texts, attack_texts):
    return {
        "embedding_cosine": None,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
        "pair_count": min(len(clean_texts), len(attack_texts)),
        "clean_clean_embedding_cosine": None,
        "clean_clean_bertscore_f1": None,
        "clean_clean_pair_count": len(list(itertools.combinations(clean_texts, 2))),
        "embedding_similarity_drop": None,
        "bertscore_f1_drop": None,
    }


def final_step_record(data):
    step_keys = [int(key) for key in data if key.lstrip("-").isdigit() and int(key) >= 0]
    if not step_keys:
        raise ValueError("No non-negative optimization steps found.")
    step = max(step_keys)
    return step, data[str(step)]


def clean_outputs(record):
    for key in ("baseline_answer", "baseline_answers", "clean_answer", "clean_answers"):
        texts = text_list(record.get(key))
        if texts:
            return texts
    baseline = record.get("baseline")
    if isinstance(baseline, dict):
        texts = text_list(baseline.get("answer"))
        if texts:
            return texts
    return []


def attack_outputs(record):
    for key in ("answer", "answers", "adv_answer", "adv_answers", "attack_answer", "attack_answers"):
        texts = text_list(record.get(key))
        if texts:
            return texts
    adv = record.get("adv")
    if isinstance(adv, dict):
        texts = text_list(adv.get("answer"))
        if texts:
            return texts
    return []


def evaluate_file(path, scorer):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clean_record = data.get("-1")
    if not isinstance(clean_record, dict):
        raise ValueError(f"Missing clean '-1' record in {path}")
    final_step, attack_record = final_step_record(data)
    clean_texts = clean_outputs(clean_record)
    attack_texts = attack_outputs(attack_record)

    clean_repetition = average_repetition(clean_texts)
    attack_repetition = average_repetition(attack_texts)
    clean_self_bleu = self_bleu(clean_texts)
    attack_self_bleu = self_bleu(attack_texts)
    semantic = (
        semantic_summary(clean_texts, attack_texts, scorer)
        if scorer is not None
        else skipped_semantic_summary(clean_texts, attack_texts)
    )

    return {
        "path": str(path),
        "index": parse_result_index(path),
        "final_step": final_step,
        "prompt": clean_record.get("prompt") or clean_record.get("baseline_prompt"),
        "adv_prompt": attack_record.get("adv_prompt"),
        "clean_sample_count": len(clean_texts),
        "attack_sample_count": len(attack_texts),
        "repetition": {
            "clean": clean_repetition,
            "attack": attack_repetition,
            "delta": delta_metrics(attack_repetition, clean_repetition),
        },
        "self_bleu": {
            "clean": clean_self_bleu,
            "attack": attack_self_bleu,
            "delta": (
                attack_self_bleu - clean_self_bleu
                if attack_self_bleu is not None and clean_self_bleu is not None
                else None
            ),
        },
        "semantic": semantic,
        "length": length_summary(clean_texts, attack_texts),
    }


def parse_result_index(path):
    match = re.search(r"res_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def result_sort_key(path):
    index = parse_result_index(path)
    return (index is None, index if index is not None else path.name)


def flatten_numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from flatten_numbers(item, next_prefix)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield prefix, value
    elif value is None:
        yield prefix, None


def summarize_items(items):
    values = defaultdict(list)
    null_counts = defaultdict(int)
    for item in items:
        for key, value in flatten_numbers(item):
            if value is None:
                null_counts[key] += 1
            else:
                values[key].append(value)
    return {
        "item_count": len(items),
        "means": {
            key: sum(numbers) / len(numbers)
            for key, numbers in sorted(values.items())
            if numbers
        },
        "null_counts": dict(sorted(null_counts.items())),
    }


def default_output_path(result_dir):
    table = result_dir.parent.name
    model = result_dir.name
    return Path("res") / "aggregate" / "degeneration" / "raw" / table / model / "degeneration_metrics.json"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        sys.exit("--limit must be positive")
    if args.max_semantic_tokens <= 0:
        sys.exit("--max-semantic-tokens must be positive")
    if not args.result_dir.is_dir():
        sys.exit(f"Result directory does not exist: {args.result_dir}")

    files = sorted(args.result_dir.glob("res_*.json"), key=result_sort_key)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No res_*.json files found in {args.result_dir}")

    scorer = None
    if not args.skip_semantic:
        scorer = SemanticScorer(args.semantic_model, args.device, args.max_semantic_tokens)

    items = [evaluate_file(path, scorer) for path in files]
    output_path = args.output or default_output_path(args.result_dir)
    payload = {
        "metric_version": "degeneration-v1-raw",
        "input_path": str(args.result_dir),
        "schema": "raw_output",
        "semantic": {
            "enabled": not args.skip_semantic,
            "model": None if args.skip_semantic else args.semantic_model,
            "device": None if scorer is None else scorer.device,
            "max_tokens": args.max_semantic_tokens,
        },
        "summary": summarize_items(items),
        "items": items,
    }
    write_json(output_path, payload)
    print(f"Saved degeneration metrics to: {output_path}")


if __name__ == "__main__":
    main()

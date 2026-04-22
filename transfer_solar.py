import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

from utils import get_chat_prompt, load_model_and_tokenizer


DEFAULT_DATASET = Path("dataset/all_data.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model",
        type=str,
        default="nota-ai/Solar-Open-100B-Nota-FP8",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "adv", "both"],
    )
    parser.add_argument("--adv-result-dir", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--sample-times", type=int, default=16)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of repeated generations to run per forward pass. Reduce this if you hit OOM.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="The maximum number of output tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def load_selected_samples(dataset_path):
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    alpaca_samples = [row for row in raw_data if row["source"] == "alpaca"][:5]
    sharegpt_samples = [row for row in raw_data if row["source"] == "sharegpt"][:5]
    selected = alpaca_samples + sharegpt_samples

    if len(selected) != 10:
        raise ValueError(f"Expected 10 selected samples, found {len(selected)}")

    samples = []
    for row in selected:
        samples.append(
            {
                "source": row["source"],
                "index": int(row["index"]),
                "instruction": row["instruction"],
            }
        )

    return samples


def get_last_adv_prompt(result_path):
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    step_keys = [int(key) for key in result.keys() if int(key) >= 0]
    if not step_keys:
        raise ValueError(f"No optimization steps found in {result_path}")

    last_step = str(max(step_keys))
    last_record = result[last_step]
    adv_prompt = last_record.get("adv_prompt")
    if not adv_prompt:
        raise ValueError(f"Missing final adv_prompt in {result_path} step {last_step}")

    return adv_prompt, last_step


def load_adv_prompts(samples, adv_result_dir):
    adv_dir = Path(adv_result_dir)
    if not adv_dir.exists():
        raise FileNotFoundError(f"Adv result directory does not exist: {adv_dir}")

    prompts = {}
    for sample in samples:
        result_path = adv_dir / f"res_{sample['index']}.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing adv result file: {result_path}")

        adv_prompt, last_step = get_last_adv_prompt(result_path)
        prompts[sample["index"]] = {
            "prompt": adv_prompt,
            "result_file": str(result_path),
            "last_step": int(last_step),
        }

    return prompts


def get_completion_length(tokens, eos_token_ids, pad_token_id):
    eos_token_ids = set(eos_token_ids or [])

    for index, token_id in enumerate(tokens.tolist()):
        if token_id in eos_token_ids:
            return index + 1
        if pad_token_id is not None and token_id == pad_token_id:
            return index

    return tokens.shape[0]


@torch.inference_mode()
def run_prompt_generations(
    model, tokenizer, prompt, sample_times, batch_size, max_new_tokens
):
    prompt_ids = get_chat_prompt(
        tokenizer,
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    prompt_len = prompt_ids.shape[1]

    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_ids = [eos_token_id]
    else:
        eos_token_ids = eos_token_id

    pad_token_id = model.generation_config.pad_token_id
    lengths = []
    answers = []

    remaining = sample_times
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch_input_ids = prompt_ids.repeat(current_batch, 1).to(model.device)
        attention_mask = torch.ones_like(batch_input_ids)

        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
        )
        completions = outputs[:, prompt_len:]

        for completion in completions:
            completion_length = get_completion_length(
                completion, eos_token_ids, pad_token_id
            )
            lengths.append(completion_length)
            answers.append(
                tokenizer.decode(
                    completion[:completion_length],
                    skip_special_tokens=True,
                ).strip()
            )

        remaining -= current_batch

    completion_cap = max_new_tokens
    avg_len = sum(lengths) / len(lengths)
    is_success = any(length >= completion_cap for length in lengths)

    return {
        "prompt": prompt,
        "answer": answers,
        "length": lengths,
        "avg_len": avg_len,
        "count": len(lengths),
        "is_success": is_success,
        "prompt_token_length": prompt_len,
        "completion_token_cap": completion_cap,
    }


def evaluate_samples(
    model,
    tokenizer,
    samples,
    mode,
    sample_times,
    batch_size,
    max_new_tokens,
    adv_prompts,
):
    evaluated = []
    for sample in samples:
        sample_result = {
            "source": sample["source"],
            "index": sample["index"],
            "instruction": sample["instruction"],
        }

        if mode in {"baseline", "both"}:
            sample_result["baseline"] = run_prompt_generations(
                model=model,
                tokenizer=tokenizer,
                prompt=sample["instruction"],
                sample_times=sample_times,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )

        if mode in {"adv", "both"}:
            adv_entry = adv_prompts[sample["index"]]
            adv_result = run_prompt_generations(
                model=model,
                tokenizer=tokenizer,
                prompt=adv_entry["prompt"],
                sample_times=sample_times,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
            adv_result["result_file"] = adv_entry["result_file"]
            adv_result["last_step"] = adv_entry["last_step"]
            sample_result["adv"] = adv_result

        evaluated.append(sample_result)

    return evaluated


def write_output(output_path, payload):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.mode in {"adv", "both"} and not args.adv_result_dir:
        raise ValueError("--adv-result-dir is required for adv and both modes")
    if args.sample_times <= 0:
        raise ValueError("--sample-times must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    set_seed(args.seed)

    dataset_path = DEFAULT_DATASET
    samples = load_selected_samples(dataset_path)
    adv_prompts = None
    if args.mode in {"adv", "both"}:
        adv_prompts = load_adv_prompts(samples, args.adv_result_dir)

    device = "auto"
    model, tokenizer = load_model_and_tokenizer(
        args.target_model, device=device, dtype=torch.bfloat16, trust_remote_code=True
    )

    sample_results = evaluate_samples(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        mode=args.mode,
        sample_times=args.sample_times,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        adv_prompts=adv_prompts,
    )

    payload = {
        "target_model": args.target_model,
        "mode": args.mode,
        "sample_count": len(sample_results),
        "sample_times": args.sample_times,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dataset_path": str(dataset_path),
        "adv_result_dir": args.adv_result_dir,
        "samples": sample_results,
    }
    write_output(Path(args.output), payload)


if __name__ == "__main__":
    main()

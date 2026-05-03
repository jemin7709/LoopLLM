import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from transformers import GenerationConfig
from vllm import LLM

from utils.string_utils import _vllm_prompt_request, _vllm_sampling_params


DEFAULT_DATASET = Path("dataset/all_data.json")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
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
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sample-times", type=int, default=16)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="The maximum number of output tokens to generate",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=3,
        help="Number of pipeline parallel stages for vLLM.",
    )
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        args = OmegaConf.load(args.config)

    return args


def load_selected_samples(dataset_path):
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    alpaca_samples = [row for row in raw_data if row["source"] == "alpaca"]
    sharegpt_samples = [row for row in raw_data if row["source"] == "sharegpt"]
    selected = alpaca_samples + sharegpt_samples

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


def run_prompt_generations(
    llm, tokenizer, generation_config, prompt, sample_times, seed
):
    sampling_params = _vllm_sampling_params(
        generation_config,
        n=sample_times,
        seed=seed,
    )
    request_output = llm.generate(
        [_vllm_prompt_request(tokenizer, prompt)],
        sampling_params,
        use_tqdm=True,
    )[0]

    lengths = []
    answers = []
    for completion in request_output.outputs:
        lengths.append(len(completion.token_ids))
        answers.append(completion.text.strip())

    completion_cap = generation_config.max_new_tokens
    avg_len = sum(lengths) / len(lengths)
    is_success = any(length >= completion_cap for length in lengths)

    return {
        "prompt": prompt,
        "answer": answers,
        "length": lengths,
        "avg_len": avg_len,
        "count": len(lengths),
        "is_success": is_success,
        "prompt_token_length": len(request_output.prompt_token_ids),
        "completion_token_cap": completion_cap,
    }


def evaluate_samples(
    llm,
    tokenizer,
    samples,
    mode,
    sample_times,
    generation_config,
    seed,
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
                llm=llm,
                tokenizer=tokenizer,
                generation_config=generation_config,
                prompt=sample["instruction"],
                sample_times=sample_times,
                seed=seed,
            )

        if mode in {"adv", "both"}:
            adv_entry = adv_prompts[sample["index"]]
            adv_result = run_prompt_generations(
                llm=llm,
                tokenizer=tokenizer,
                generation_config=generation_config,
                prompt=adv_entry["prompt"],
                sample_times=sample_times,
                seed=seed,
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
    if not args.output:
        raise ValueError("--output is required")
    if args.mode in {"adv", "both"} and not args.adv_result_dir:
        raise ValueError("--adv-result-dir is required for adv and both modes")
    if args.sample_times <= 0:
        raise ValueError("--sample-times must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.pipeline_parallel_size <= 0:
        raise ValueError("--pipeline-parallel-size must be positive")

    dataset_path = DEFAULT_DATASET
    samples = load_selected_samples(dataset_path)
    adv_prompts = None
    if args.mode in {"adv", "both"}:
        adv_prompts = load_adv_prompts(samples, args.adv_result_dir)

    llm = LLM(
        model=args.target_model,
        trust_remote_code=True,
        generation_config="vllm",
        pipeline_parallel_size=args.pipeline_parallel_size,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    generation_config = GenerationConfig.from_pretrained(args.target_model)
    if not generation_config.do_sample:
        generation_config.do_sample = True
        generation_config.temperature = 0.6
        generation_config.top_p = 0.9
    generation_config.max_new_tokens = args.max_new_tokens

    sample_results = evaluate_samples(
        llm=llm,
        tokenizer=tokenizer,
        samples=samples,
        mode=args.mode,
        sample_times=args.sample_times,
        generation_config=generation_config,
        seed=args.seed,
        adv_prompts=adv_prompts,
    )

    payload = {
        "target_model": args.target_model,
        "mode": args.mode,
        "sample_count": len(sample_results),
        "sample_times": args.sample_times,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": {
            key: getattr(generation_config, key, None)
            for key in [
                "do_sample",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "eos_token_id",
                "pad_token_id",
            ]
        },
        "seed": args.seed,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dataset_path": str(dataset_path),
        "adv_result_dir": args.adv_result_dir,
        "samples": sample_results,
    }
    write_output(Path(args.output), payload)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# vLLM forks worker processes by default; this script touches CUDA before LLM init.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import set_seed

from utils import (
    MODEL_PATHS,
    SuffixManager,
    get_all_losses,
    get_filtered_cands,
    get_nonascii_toks,
    is_entropy_low,
    load_model_and_tokenizer,
    read_data,
    sample_control,
)
from utils.string_utils import _vllm_prompt_request, _vllm_sampling_params


logger = logging.getLogger(__name__)


@dataclass
class PromptState:
    dataset_index: int
    group_id: int
    group_size: int
    prompt: str
    suffix_manager: SuffixManager
    results: dict[int, dict[str, Any]]
    first_success_step: int | None = None

    @property
    def adv_prompt(self) -> str:
        return f"{self.prompt} {self.suffix_manager.adv_suffix.strip()}"


class ColocatedRuntime:
    def __init__(self, args: argparse.Namespace):
        if not torch.cuda.is_available():
            raise RuntimeError("multi_prompt_gcg.py requires CUDA")

        from vllm import LLM

        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model_path = MODEL_PATHS[args.model_name]
        self.active = "loading"

        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_path,
            device="cpu",
        )
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.generation_config.max_new_tokens = args.max_length

        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        warmup_bytes = min(1024**3, max(1, free_bytes - 512 * 1024**2))
        dummy_tensor = torch.ones(warmup_bytes, dtype=torch.uint8, device=self.device)
        torch.cuda.synchronize()
        time.sleep(1)
        del dummy_tensor
        torch.cuda.empty_cache()

        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            generation_config="vllm",
            enable_sleep_mode=True,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            seed=args.seed,
        )
        self.llm.sleep(level=args.vllm_sleep_level)
        self.active = "sleeping"

    def activate_hf(self) -> None:
        if self.active == "hf":
            return
        if self.active == "vllm":
            torch.cuda.synchronize()
            self.llm.sleep(level=self.args.vllm_sleep_level)
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.active = "hf"
        logger.info("vLLM sleep -> HF active")

    def activate_vllm(self) -> None:
        if self.active == "vllm":
            return
        if self.active == "hf":
            torch.cuda.synchronize()
            self.model.to("cpu")
        torch.cuda.empty_cache()
        self.llm.wake_up()
        self.active = "vllm"
        logger.info("HF offload -> vLLM wake")


def run_attack(args: argparse.Namespace, runtime: ColocatedRuntime) -> None:
    data = read_data(args.data_name, length=args.limit)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    initial_suffix, initial_token_id = build_initial_suffix(
        runtime.tokenizer,
        args.adv_len,
    )
    runtime.activate_hf()
    not_allowed_tokens = get_nonascii_toks(runtime.tokenizer, runtime.device)

    for group_id, dataset_indices in prompt_chunks(data, args.prompt_batch_size):
        logger.info("========== group %s / indices %s ==========", group_id, dataset_indices)
        states = initialize_group(
            args,
            runtime,
            data,
            group_id,
            dataset_indices,
            initial_suffix,
            initial_token_id,
            save_dir,
        )
        optimize_group(args, runtime, states, not_allowed_tokens, save_dir)
        torch.cuda.empty_cache()


def prompt_chunks(data: list[str], batch_size: int):
    for start in range(0, len(data), batch_size):
        dataset_indices = list(range(start, min(start + batch_size, len(data))))
        yield start // batch_size, dataset_indices


def initialize_group(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    data: list[str],
    group_id: int,
    dataset_indices: list[int],
    initial_suffix: str,
    initial_token_id: int,
    save_dir: Path,
) -> list[PromptState]:
    states = [
        make_prompt_state(
            args,
            runtime,
            data[dataset_index],
            dataset_index,
            group_id,
            len(dataset_indices),
            initial_suffix,
            initial_token_id,
        )
        for dataset_index in dataset_indices
    ]

    runtime.activate_vllm()
    baseline = generate_batch(runtime, [state.prompt for state in states], n=1)
    initial = generate_batch(runtime, [state.adv_prompt for state in states], n=1)

    for state, baseline_result, initial_result in zip(states, baseline, initial):
        state.suffix_manager.update(answer=initial_result["answer"])
        state.results[-1] = initial_record(state, baseline_result, initial_result)
        save_state(save_dir, state)

    runtime.activate_hf()
    return states


def make_prompt_state(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    prompt: str,
    dataset_index: int,
    group_id: int,
    group_size: int,
    initial_suffix: str,
    initial_token_id: int,
) -> PromptState:
    suffix_manager = SuffixManager(
        tokenizer=runtime.tokenizer,
        instruction=prompt,
        adv_len=args.adv_len,
        eos_token_id=runtime.model.generation_config.eos_token_id,
        pad_token_id=runtime.model.generation_config.pad_token_id,
    )
    suffix_manager.update(adv_suffix=initial_suffix)
    suffix_manager.adv_token_id = initial_token_id
    return PromptState(
        dataset_index=dataset_index,
        group_id=group_id,
        group_size=group_size,
        prompt=prompt,
        suffix_manager=suffix_manager,
        results={},
    )


def optimize_group(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    states: list[PromptState],
    not_allowed_tokens: torch.Tensor,
    save_dir: Path,
) -> None:
    for step in range(args.steps):
        start_time = time.time()
        logger.info("group %s step %s", states[0].group_id, step)

        best_suffixes, current_losses, group_current_loss = choose_suffixes(
            args,
            runtime,
            states,
            not_allowed_tokens,
        )
        for state, best_suffix in zip(states, best_suffixes):
            state.suffix_manager.update(adv_suffix=best_suffix)

        evaluations = None
        group_success_rate = 0.0
        if should_evaluate(args, step):
            runtime.activate_vllm()
            evaluations = evaluate_group(args, runtime, states)
            runtime.activate_hf()
            annotate_success(args, runtime, states, evaluations, step)
            group_success_rate = get_group_success_rate(evaluations)

        duration = time.time() - start_time
        for index, state in enumerate(states):
            evaluation = evaluations[index] if evaluations else None
            state.results[step] = step_record(
                state,
                current_losses[index],
                group_current_loss,
                group_success_rate,
                evaluation,
                duration,
            )
            save_state(save_dir, state)

        if group_success_rate >= args.success_rate_threshold:
            logger.info(
                "group %s reached success threshold %.3f",
                states[0].group_id,
                group_success_rate,
            )
            break


def choose_suffixes(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    states: list[PromptState],
    not_allowed_tokens: torch.Tensor,
) -> tuple[list[str], list[float], float]:
    best_suffixes = []
    current_losses = []

    for state in states:
        best_suffix, current_loss = choose_prompt_suffix(
            args,
            runtime,
            state,
            not_allowed_tokens,
        )
        best_suffixes.append(best_suffix)
        current_losses.append(current_loss)

    return best_suffixes, current_losses, sum(current_losses)


def choose_prompt_suffix(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    state: PromptState,
    not_allowed_tokens: torch.Tensor,
) -> tuple[str, float]:
    input_ids = state.suffix_manager.get_input_ids().to(runtime.device)
    grad = compute_prompt_gradient(runtime.model, input_ids, state.suffix_manager)
    control_tokens = input_ids[state.suffix_manager._control_slice].to(runtime.device)

    with torch.no_grad():
        candidate_tokens = sample_control(
            control_tokens,
            grad,
            args.num_candidate,
            args.topk,
            not_allowed_tokens=not_allowed_tokens,
        )
        candidate_suffixes, candidate_ids = get_filtered_cands(
            runtime.tokenizer,
            candidate_tokens,
            control_tokens,
            fill_cand=False,
            return_ids=True,
        )
        losses = get_all_losses(
            runtime.model,
            runtime.tokenizer,
            input_ids,
            candidate_ids,
            state.suffix_manager,
            batch_size=args.once_forward_batch,
        )
        best_index = int(losses.argmin().item())
        return candidate_suffixes[best_index], losses[best_index].item()


def sum_gradients(
    model,
    device: torch.device,
    states: list[PromptState],
) -> torch.Tensor:
    total_grad = None
    for state in states:
        input_ids = state.suffix_manager.get_input_ids().to(device)
        grad = compute_prompt_gradient(model, input_ids, state.suffix_manager)
        total_grad = grad if total_grad is None else total_grad + grad
    return total_grad


def get_prompt_losses(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    states: list[PromptState],
    candidate_ids: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        get_all_losses(
            runtime.model,
            runtime.tokenizer,
            state.suffix_manager.get_input_ids(),
            candidate_ids,
            state.suffix_manager,
            batch_size=args.once_forward_batch,
        )
        for state in states
    ]


def should_evaluate(args: argparse.Namespace, step: int) -> bool:
    return (step + 1) % args.eval_interval == 0 or step == args.steps - 1


def evaluate_group(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    states: list[PromptState],
) -> list[dict[str, Any]]:
    return generate_batch(
        runtime,
        [state.adv_prompt for state in states],
        n=args.sample_times,
    )


def generate_batch(
    runtime: ColocatedRuntime,
    prompts: list[str],
    n: int,
) -> list[dict[str, Any]]:
    sampling_params = _vllm_sampling_params(
        runtime.model.generation_config,
        n=n,
        seed=runtime.args.seed,
    )
    request_outputs = runtime.llm.generate(
        [_vllm_prompt_request(runtime.tokenizer, prompt) for prompt in prompts],
        sampling_params,
        use_tqdm=False,
    )
    return [
        summarize_request_output(runtime.model.generation_config.max_new_tokens, output)
        for output in request_outputs
    ]


def summarize_request_output(max_new_tokens: int, request_output) -> dict[str, Any]:
    answers = [output.text.strip() for output in request_output.outputs]
    lengths = [len(output.token_ids) for output in request_output.outputs]
    longest_index = max(range(len(lengths)), key=lengths.__getitem__)
    success_count = sum(length >= max_new_tokens - 5 for length in lengths)
    return {
        "answer": answers[longest_index],
        "answers": answers,
        "output_lengths": lengths,
        "avg_len": sum(lengths) / len(lengths),
        "success_rate": success_count / len(lengths),
    }


def annotate_success(
    args: argparse.Namespace,
    runtime: ColocatedRuntime,
    states: list[PromptState],
    evaluations: list[dict[str, Any]],
    step: int,
) -> None:
    for state, evaluation in zip(states, evaluations):
        state.suffix_manager.update(answer=evaluation["answer"])
        output_cap_hit = evaluation["success_rate"] >= args.prompt_success_rate_threshold
        entropy_low = False
        if output_cap_hit:
            input_ids = state.suffix_manager.get_input_ids().to(runtime.device).unsqueeze(0)
            entropy_low = is_entropy_low(runtime.model, input_ids)
        success = output_cap_hit and entropy_low
        evaluation["entropy_low"] = entropy_low
        evaluation["success"] = success
        if success and state.first_success_step is None:
            state.first_success_step = step


def get_group_success_rate(evaluations: list[dict[str, Any]]) -> float:
    successful_prompts = sum(evaluation["success"] for evaluation in evaluations)
    return successful_prompts / len(evaluations)


def initial_record(
    state: PromptState,
    baseline_result: dict[str, Any],
    initial_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "baseline_prompt": state.prompt,
        "baseline_answer": baseline_result["answer"],
        "baseline_output_len": baseline_result["avg_len"],
        "prompt": state.adv_prompt,
        "answer": initial_result["answer"],
        "output_len": initial_result["avg_len"],
        "adv_suffix": state.suffix_manager.adv_suffix,
        "adv_prompt": state.adv_prompt,
        "group_id": state.group_id,
        "group_size": state.group_size,
        "group_success_rate": 0.0,
        "group_current_loss": 0.0,
        "evaluated": True,
        "success": False,
        "entropy_low": False,
        "first_success_step": state.first_success_step,
        "time": 0.0,
    }


def step_record(
    state: PromptState,
    current_loss: float,
    group_current_loss: float,
    group_success_rate: float,
    evaluation: dict[str, Any] | None,
    duration: float,
) -> dict[str, Any]:
    evaluated = evaluation is not None
    record = {
        "prompt": state.prompt,
        "adv_suffix": state.suffix_manager.adv_suffix,
        "adv_prompt": state.adv_prompt,
        "current_losses": current_loss,
    }
    if evaluation is not None:
        record.update(
            {
                "answer": evaluation["answer"],
                "success_rate": evaluation["success_rate"],
                "avg_len": evaluation["avg_len"],
            }
        )
    record.update(
        {
            "time": duration,
            "group_id": state.group_id,
            "group_size": state.group_size,
            "group_success_rate": group_success_rate,
            "group_current_loss": group_current_loss,
            "evaluated": evaluated,
            "success": evaluation["success"] if evaluation is not None else False,
            "entropy_low": evaluation["entropy_low"] if evaluation is not None else False,
            "first_success_step": state.first_success_step,
        }
    )
    return record


def save_state(save_dir: Path, state: PromptState) -> None:
    with (save_dir / f"res_{state.dataset_index}.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(state.results, handle, indent=4, ensure_ascii=False)


def build_initial_suffix(tokenizer, adv_len: int) -> tuple[str, int]:
    adv_token_id = tokenizer.encode("* " * 20)[-5]
    return tokenizer.decode([adv_token_id] * adv_len), adv_token_id


def compute_prompt_gradient(
    model,
    input_ids: torch.Tensor,
    suffix_manager: SuffixManager,
) -> torch.Tensor:
    control_slice = suffix_manager._control_slice
    target_slice = suffix_manager._target_slice
    special_id = suffix_manager.adv_token_id

    embed_weights = model.get_input_embeddings().weight
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(1, input_ids[control_slice].unsqueeze(1), 1)
    one_hot.requires_grad_()

    input_embeds = one_hot @ embed_weights
    embeds = embed_weights[input_ids].detach()
    full_embeds = torch.cat(
        [
            embeds[: control_slice.start, :],
            input_embeds,
            embeds[control_slice.stop :, :],
        ],
        dim=0,
    ).unsqueeze(0)

    logits = model(inputs_embeds=full_embeds, use_cache=False).logits
    loss = target_loss(logits, target_slice, special_id)
    loss.backward(retain_graph=False)

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    one_hot.grad.zero_()
    model.zero_grad(set_to_none=True)
    return grad


def target_loss(logits: torch.Tensor, target_slice: slice, special_id) -> torch.Tensor:
    logits_t = logits[:, target_slice.start - 1 : -1, :]
    prob = torch.softmax(logits_t, dim=-1)
    special_p = prob[:, :, special_id]
    if isinstance(special_id, list):
        special_p = special_p.sum(dim=-1)
    return nn.BCELoss(reduction="none")(
        special_p,
        torch.ones_like(special_p),
    ).mean(dim=-1)


def generation_config_to_metadata(generation_config) -> dict[str, Any]:
    keys = [
        "do_sample",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "eos_token_id",
        "pad_token_id",
    ]
    return {key: getattr(generation_config, key, None) for key in keys}


def save_metadata(args: argparse.Namespace, runtime: ColocatedRuntime) -> None:
    metadata_path = Path(args.save_dir) / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "generation_config": generation_config_to_metadata(runtime.model.generation_config),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llama2-7b", choices=MODEL_PATHS.keys())
    parser.add_argument(
        "--data_name",
        type=str,
        default="alpaca",
        choices=["sharegpt", "alpaca", "all", "math", "math_test", "math_train"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adv_len", type=int, default=30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--num_candidate", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--once_forward_batch", type=int, default=16)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--root_dir", type=str, default="res/")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--prompt_batch_size", type=int, default=4)
    parser.add_argument("--sample_times", type=int, default=16)
    parser.add_argument("--prompt_success_rate_threshold", type=float, default=0.125)
    parser.add_argument("--success_rate_threshold", type=float, default=0.9)
    parser.add_argument("--vllm_sleep_level", type=int, default=1, choices=[1])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.35)
    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()
    args.save_dir = os.path.join(
        args.root_dir,
        f"{args.model_name}_{args.data_name}_multi_prompt_gcg_b{args.prompt_batch_size}_s{args.seed}",
    )
    return args


def configure_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)

    runtime = ColocatedRuntime(args)
    save_metadata(args, runtime)
    run_attack(args, runtime)


if __name__ == "__main__":
    main()

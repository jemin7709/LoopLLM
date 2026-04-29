#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
SAMPLE_TIMES="${SAMPLE_TIMES:-16}"
SEED="${SEED:-23}"

export CUDA_VISIBLE_DEVICES="$GPU"

run_transfer() {
    local split=$1
    local proxy=$2
    local target=$3
    local target_model=$4
    local max_new_tokens=$5

    echo "Transfer table1_${split}: ${proxy} -> ${target}"
    uv run python src/transfer_solar.py \
        --target_model "$target_model" \
        --mode "both" \
        --adv-result-dir "res/raw_output/table1_${split}/${proxy}_all" \
        --output "res/transfer/table1_${split}/${proxy}_${target}.json" \
        --sample-times "$SAMPLE_TIMES" \
        --max_new_tokens "$max_new_tokens" \
        --seed "$SEED" \
        --pipeline-parallel-size 1
}

for split in t p; do
    run_transfer "$split" "llama3-1b" "llama3-1b" "meta-llama/Llama-3.2-1B-Instruct" 1024
    run_transfer "$split" "llama3-1b" "llama3-3b" "meta-llama/Llama-3.2-3B-Instruct" 1024
    run_transfer "$split" "llama3-1b" "llama3-8b" "meta-llama/Llama-3.1-8B-Instruct" 4096
    run_transfer "$split" "llama3-3b" "llama3-8b" "meta-llama/Llama-3.1-8B-Instruct" 4096

    run_transfer "$split" "qwen3-0.6b" "qwen3-0.6b" "Qwen/Qwen3-0.6B" 1024
    run_transfer "$split" "qwen3-0.6b" "qwen3-1.7b" "Qwen/Qwen3-1.7B" 1024
    run_transfer "$split" "qwen3-1.7b" "qwen3-1.7b" "Qwen/Qwen3-1.7B" 1024
done

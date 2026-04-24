#!/bin/bash
set -uo pipefail
trap 'jobs -pr | xargs -r kill' INT TERM EXIT

export CUDA_VISIBLE_DEVICES="0,1,2"

for model_dir in res/raw_output/table1_t/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        uv run python src/transfer_solar.py \
            --mode "both" \
            --adv-result-dir "res/raw_output/table1_t/$model_name" \
            --output "res/transfer/table1_t/$model_name.json" \
            --sample-times 4 \
            --max_new_tokens 10000
    fi
done

for model_dir in res/raw_output/table1_p/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        uv run python src/transfer_solar.py \
            --mode "both" \
            --adv-result-dir "res/raw_output/table1_p/$model_name" \
            --output "res/transfer/table1_p/$model_name.json" \
            --sample-times 4 \
            --max_new_tokens 10000
    fi
done

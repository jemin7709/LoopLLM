#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
TRANSFER_DIR="${1:-res/transfer}"
EVALUATION_DIR="${TRANSFER_DIR}/evaluation"

export CUDA_VISIBLE_DEVICES="$GPU"

shopt -s nullglob

for result_file in "$TRANSFER_DIR"/table1_*/*.json; do
    output_file="${EVALUATION_DIR}/${result_file#"$TRANSFER_DIR"/}"

    echo "Evaluate transfer: ${result_file} -> ${output_file}"
    uv run python src/evaluate_transfer.py "$result_file" --output "$output_file"
done

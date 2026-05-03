#!/bin/bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-logs/evaluation_transfer.log}"
TRANSFER_DIR="${1:-res/transfer}"

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

shopt -s nullglob

log() {
  printf '%s\n' "$1" >> "$LOG_FILE"
}

run_gpu() {
  local gpu="$1"
  local input_root="$2"
  local evaluation_dir="$3"
  shift 3

  local result_file output_file
  for result_file in "$@"; do
    output_file="${evaluation_dir}/${result_file#"$input_root"/}"

    log "[INFO] GPU=${gpu} input=${result_file} output=${output_file}"
    if CUDA_VISIBLE_DEVICES="$gpu" uv run python src/evaluate_transfer.py "$result_file" --output "$output_file" --skip-bertscore; then
      log "[SUCCESS] GPU=${gpu} input=${result_file}"
    else
      local status=$?
      log "[FAILURE] GPU=${gpu} input=${result_file} exit_code=${status}"
      return "$status"
    fi
  done
}

# GPU랑 result file는 여기서 직접 지정하세요.
# 같은 GPU 안에서는 순차 실행되고, & 붙인 줄들은 동시에 실행됩니다.

run_gpu 0 \
    "$TRANSFER_DIR/sample16" \
    "$TRANSFER_DIR/sample16/evaluation" \
    "$TRANSFER_DIR"/sample16/table1_*/*.json &

run_gpu 1 \
    "$TRANSFER_DIR/sample64" \
    "$TRANSFER_DIR/sample64/evaluation" \
    "$TRANSFER_DIR"/sample64/table1_*/*.json &

wait

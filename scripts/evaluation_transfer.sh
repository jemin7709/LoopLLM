#!/bin/bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-logs/evaluation_transfer.log}"

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

log() {
  printf '%s\n' "$1" >> "$LOG_FILE"
}

run_gpu() {
  local gpu="$1"
  shift

  local transfer_dir result_file output_file
  for transfer_dir in "$@"; do
    transfer_dir="${transfer_dir%/}"

    for result_file in "$transfer_dir"/*/*.json; do
      output_file="$transfer_dir/evaluation/${result_file#"$transfer_dir"/}"

      log "[INFO] GPU=${gpu} input=${result_file} output=${output_file}"
      if CUDA_VISIBLE_DEVICES="$gpu" uv run python src/evaluate_transfer.py "$result_file" --output "$output_file" --skip-bertscore; then
        log "[SUCCESS] GPU=${gpu} input=${result_file}"
      else
        local status=$?
        log "[FAILURE] GPU=${gpu} input=${result_file} exit_code=${status}"
        return "$status"
      fi
    done
  done
}

# GPU랑 transfer 결과 폴더는 여기서 직접 지정하세요.
# 같은 GPU 안에서는 순차 실행되고, & 붙인 줄들은 동시에 실행됩니다.

run_gpu 2 \
    res/param_test/transfer_test1 &

wait

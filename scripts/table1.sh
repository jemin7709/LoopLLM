#!/bin/bash
set -uo pipefail
trap 'jobs -pr | xargs -r kill' INT TERM EXIT

LOG_FILE="execution.log"
> "$LOG_FILE"

run_and_log() {
    local gpus=$1
    local model=$2
    local c_val=$3
    shift 3

    if CUDA_VISIBLE_DEVICES=$gpus "$@"; then
        echo "[SUCCESS] gpus=$gpus model=$model c=$c_val" | tee -a "$LOG_FILE"
    else
        echo "[FAIL] gpus=$gpus model=$model c=$c_val" | tee -a "$LOG_FILE"
    fi
}

run_gpu0() {
    run_and_log "0,1" "llama3-1b" "1" uv run main.py --model_name "llama3-1b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "llama3-1b" "5" uv run main.py --model_name "llama3-1b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "gemma2-2b" "1" uv run main.py --model_name "gemma2-2b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "gemma2-2b" "5" uv run main.py --model_name "gemma2-2b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "qwen2.5-3b" "1" uv run main.py --model_name "qwen2.5-3b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "qwen2.5-3b" "5" uv run main.py --model_name "qwen2.5-3b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "llama3-3b" "1" uv run main.py --model_name "llama3-3b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "llama3-3b" "5" uv run main.py --model_name "llama3-3b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p


    run_and_log "0,1" "stablelm-3b" "1" uv run main.py --model_name "stablelm-3b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "stablelm-3b" "5" uv run main.py --model_name "stablelm-3b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "phi4-mini" "1" uv run main.py --model_name "phi4-mini" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "phi4-mini" "5" uv run main.py --model_name "phi4-mini" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 1024 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "mistral-7b" "1" uv run main.py --model_name "mistral-7b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "mistral-7b" "5" uv run main.py --model_name "mistral-7b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "llama2-7b" "1" uv run main.py --model_name "llama2-7b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "llama2-7b" "5" uv run main.py --model_name "llama2-7b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_p


    run_and_log "0,1" "vicuna-7b" "1" uv run main.py --model_name "vicuna-7b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "vicuna-7b" "5" uv run main.py --model_name "vicuna-7b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 2048 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "llama3-8b" "1" uv run main.py --model_name "llama3-8b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 4096 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "llama3-8b" "5" uv run main.py --model_name "llama3-8b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 4096 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "glm4-9b" "1" uv run main.py --model_name "glm4-9b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 4096 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "glm4-9b" "5" uv run main.py --model_name "glm4-9b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 4096 --seed 23 --root_dir res/table1_p

    run_and_log "0,1" "llama2-13b" "1" uv run main.py --model_name "llama2-13b" --data_name all --c 1 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 8192 --seed 23 --root_dir res/table1_t
    run_and_log "0,1" "llama2-13b" "5" uv run main.py --model_name "llama2-13b" --data_name all --c 5 --adv_len 30 --steps 20 --topk 64 --num_candidate 128 --max_length 8192 --seed 23 --root_dir res/table1_p
}

run_gpu0 

wait
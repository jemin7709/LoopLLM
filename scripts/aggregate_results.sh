#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 res/table1_p [res/table_t ...]" >&2
    exit 1
fi

for root_dir in "$@"; do
    [ -d "$root_dir" ] || { echo "Skip: $root_dir is not a directory" >&2; continue; }

    for result_dir in "$root_dir"/*/; do
        result_dir="${result_dir%/}"

        count=$(find "$result_dir" -maxdepth 1 -type f -name 'res_*.json' | wc -l)

        if [ "$count" -eq 100 ]; then
            echo "Aggregate: $result_dir"
            python src/aggregate_results.py "$result_dir"
        else
            echo "Skip: $result_dir ($count files)"
        fi
    done
done

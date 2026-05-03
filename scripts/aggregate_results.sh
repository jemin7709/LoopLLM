#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 res/test_output [res/another_test ...]" >&2
    exit 1
fi

for root_dir in "$@"; do
    root_dir="${root_dir%/}"
    [ -d "$root_dir" ] || { echo "Skip: $root_dir is not a directory" >&2; continue; }

    aggregate_root="$root_dir/aggregate"

    while IFS= read -r -d '' result_dir; do
        case "$result_dir" in
            "$aggregate_root"|"$aggregate_root"/*) continue ;;
        esac

        count=$(find "$result_dir" -maxdepth 1 -type f -name 'res_*.json' | wc -l)

        if [ "$count" -eq 100 ]; then
            echo "Aggregate: $result_dir"
            python src/aggregate_results.py "$result_dir" --output-root "$aggregate_root"
        else
            echo "Skip: $result_dir ($count files)"
        fi
    done < <(find "$root_dir" -mindepth 1 -type d -print0 | sort -z)
done

#!/usr/bin/env bash
# Reproduce the frozen compiler failure. Does not use the live source/target or NPU.
set -euo pipefail
candidate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$candidate_dir/../../../.." && pwd)
artifact_dir="$workspace_dir/moa-dev/experiments/artifacts/qkv-bf16-interleaved-kv-e55"
python3 "$candidate_dir/prepare_source.py"
docker exec moa2026-dev mkdir -p /tmp/moa-qkv-bf16-interleaved-kv-e55
docker cp "$artifact_dir/crate/." moa2026-dev:/tmp/moa-qkv-bf16-interleaved-kv-e55/
# Keep the original compile-attempt1.log immutable; a reproduction gets another log.
docker exec moa2026-dev bash -lc 'cd /tmp/moa-qkv-bf16-interleaved-kv-e55 && CARGO_HOME=/home/moa/.cargo CARGO_TARGET_DIR=/tmp/moa-qkv-bf16-interleaved-kv-e55/target FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-bf16-interleaved-kv-e55/target/furiosa-opt/kernel cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule candidate.json --dump-graph candidate-graph.json' > "$artifact_dir/reproduce.log" 2>&1

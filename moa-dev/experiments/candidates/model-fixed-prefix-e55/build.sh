#!/usr/bin/env bash
# Isolated host build only. Never launches a model or changes the live crate.
set -euo pipefail
candidate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$candidate_dir/../../../.." && pwd)
artifact_dir="$workspace_dir/moa-dev/experiments/artifacts/model-fixed-prefix-e55"
python3 "$candidate_dir/prepare_sources.py"
for variant in original e55; do
    docker exec moa2026-dev mkdir -p "/tmp/moa-model-fixed-prefix-e55/$variant"
    docker cp "$artifact_dir/$variant/crate/." "moa2026-dev:/tmp/moa-model-fixed-prefix-e55/$variant/"
    # Only the two constant variant names above enter the shell command.
    docker exec moa2026-dev bash -lc "cd /tmp/moa-model-fixed-prefix-e55/$variant && CARGO_HOME=/home/moa/.cargo CARGO_TARGET_DIR=/tmp/moa-model-fixed-prefix-e55/$variant/target FURIOSA_OPT_OUT_DIR=/tmp/moa-model-fixed-prefix-e55/$variant/target/furiosa-opt/kernel cargo furiosa-opt build --release --locked --bin probe_model_logits" > "$artifact_dir/$variant/rebuild.log" 2>&1
    docker cp "moa2026-dev:/tmp/moa-model-fixed-prefix-e55/$variant/target/release/probe_model_logits" "$artifact_dir/$variant/probe_model_logits"
done

#!/bin/sh
# Build-machine helper only. Arena needs only the resulting ELF.
set -eu
inspector_source=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
inspector_output=${1:?Usage: build.sh OUTPUT_DIRECTORY}
inspector_container=${MOA_INSPECTOR_CONTAINER:-moa2026-dev}
inspector_contract="$inspector_source/../loader-header-contract.json"
inspector_expected=8e2b9346ff1f4ee3d58581326c85bb89f2dca5c97cb6c5e2b1fd9c3260754055
inspector_actual=$(sha256sum "$inspector_contract")
inspector_actual=${inspector_actual%% *}
if [ "$inspector_actual" != "$inspector_expected" ]; then
    printf '%s\n' 'contract_sha256_mismatch' >&2
    exit 1
fi
inspector_scratch=$(docker exec "$inspector_container" mktemp -d /tmp/moa-asset-inspector-build.XXXXXX)
for inspector_file in Cargo.toml Cargo.lock audit_dependencies.py; do
    docker cp "$inspector_source/$inspector_file" "$inspector_container:$inspector_scratch/$inspector_file"
done
docker cp "$inspector_source/src" "$inspector_container:$inspector_scratch/src"
# include_bytes! needs the exact contract inside the staged crate, not a second
# tracked 349-KB copy in rust-inspector. The parent candidate owns that JSON.
docker cp "$inspector_contract" "$inspector_container:$inspector_scratch/loader-header-contract.json"
docker exec -w "$inspector_scratch" \
    -e CARGO_HOME=/home/moa/.cargo -e CARGO_TARGET_DIR="$inspector_scratch/target" \
    "$inspector_container" bash -c '
        cargo test --offline --locked --release > cpu-tests.log 2>&1 &&
        cargo build --offline --locked --release > build.log 2>&1 &&
        cargo metadata --offline --locked --format-version 1 > dependency-metadata.json &&
        python3 audit_dependencies.py . > dependency-audit.json
    '
mkdir -p "$inspector_output"
docker cp "$inspector_container:$inspector_scratch/target/release/moa-model-asset-inspector" "$inspector_output/moa-model-asset-inspector"
for inspector_file in cpu-tests.log build.log dependency-audit.json; do
    docker cp "$inspector_container:$inspector_scratch/$inspector_file" "$inspector_output/$inspector_file"
done
python3 "$inspector_source/cpu_cli_fixture_test.py" "$inspector_output/moa-model-asset-inspector" "$inspector_output" > "$inspector_output/cpu-cli-tests.json"
sha256sum "$inspector_output/moa-model-asset-inspector"

#!/bin/sh
# Same reference token history at every compared checkpoint; no accuracy verdict.
set -eu
cd "$(dirname "$0")"
finish() {
    result=$?
    printf '\nMODEL_CONTINUATION_PYTHON40_PAIR_EXIT=%s\n' "$result"
}
trap finish EXIT
prepare_binary() {
    binary=$1
    chmod +x "$binary" 2>/dev/null || true
    if [ ! -x "$binary" ]; then
        cp "$binary" "$binary.exec"
        chmod +x "$binary.exec"
        printf '%s\n' "$binary.exec"
    else
        printf '%s\n' "$binary"
    fi
}
reference_hashes() {
    sha256sum reference/manifest.json
    step=0
    while [ "$step" -lt 40 ]; do
        file="reference/case-0-step-$step.json"
        if [ -f "$file" ]; then
            sha256sum "$file"
        fi
        step=$((step + 1))
    done
}
original=$(prepare_binary ./probe_model_original)
candidate=$(prepare_binary ./probe_model_e55)
inspector=$(prepare_binary ./moa-model-asset-inspector)
printf 'MODEL_CONTINUATION_PYTHON40_PAIR_BEGIN\n'
sha256sum "$original" "$candidate" "$inspector"
printf 'MODEL_CONTINUATION_PYTHON40_ASSETS_BEFORE_BEGIN\n'
"$inspector"
printf 'MODEL_CONTINUATION_PYTHON40_ASSETS_BEFORE_END\n'
printf 'MODEL_CONTINUATION_PYTHON40_ORIGINAL_BEGIN\n'
FURIOSA_OPT_PROFILE=warn "$original" --emit reference
printf 'MODEL_CONTINUATION_PYTHON40_REFERENCE_MANIFEST_SHA256\n'
sha256sum reference/manifest.json
printf 'MODEL_CONTINUATION_PYTHON40_REFERENCE_FILES_SHA256_BEGIN\n'
reference_hashes
printf 'MODEL_CONTINUATION_PYTHON40_REFERENCE_FILES_SHA256_END\n'
printf 'MODEL_CONTINUATION_PYTHON40_E55_BEGIN\n'
FURIOSA_OPT_PROFILE=warn "$candidate" --compare reference
printf 'MODEL_CONTINUATION_PYTHON40_REFERENCE_FILES_AFTER_SHA256_BEGIN\n'
reference_hashes
printf 'MODEL_CONTINUATION_PYTHON40_REFERENCE_FILES_AFTER_SHA256_END\n'
printf 'MODEL_CONTINUATION_PYTHON40_ASSETS_AFTER_BEGIN\n'
"$inspector" --header-only
printf 'MODEL_CONTINUATION_PYTHON40_ASSETS_AFTER_END\n'
printf 'MODEL_CONTINUATION_PYTHON40_PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT\n'

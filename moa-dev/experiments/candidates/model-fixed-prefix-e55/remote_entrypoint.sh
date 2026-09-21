#!/bin/sh
# Sequential paired model diagnostic. Exit 0 means structurally valid comparison,
# never an accuracy PASS. The full reference logits stay in this job directory.
set -eu
cd "$(dirname "$0")"
finish() {
    result=$?
    printf '\nMODEL_FIXED_PREFIX_PAIR_EXIT=%s\n' "$result"
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
original=$(prepare_binary ./probe_model_original)
candidate=$(prepare_binary ./probe_model_e55)
printf 'MODEL_FIXED_PREFIX_PAIR_BEGIN\n'
sha256sum "$original" "$candidate"
printf 'MODEL_FIXED_PREFIX_ORIGINAL_BEGIN\n'
FURIOSA_OPT_PROFILE=warn "$original" --emit reference.json
printf 'MODEL_FIXED_PREFIX_REFERENCE_SHA256\n'
sha256sum reference.json
printf 'MODEL_FIXED_PREFIX_E55_BEGIN\n'
FURIOSA_OPT_PROFILE=warn "$candidate" --compare reference.json
printf 'MODEL_FIXED_PREFIX_PAIR_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT\n'

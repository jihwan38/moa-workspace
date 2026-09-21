#!/bin/sh
# Unmodified E55 CLI; merged direct stream, no accuracy verdict.
set -eu
cd "$(dirname "$0")"
finish() {
    result=$?
    printf '\nMODEL_CLI_PYTHON40_EXIT=%s\n' "$result"
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
model_cli=$(prepare_binary ./gemma4)
inspector=$(prepare_binary ./moa-model-asset-inspector)
printf 'MODEL_CLI_PYTHON40_BEGIN\n'
sha256sum "$model_cli" "$inspector"
printf 'MODEL_CLI_PYTHON40_ASSETS_BEFORE_BEGIN\n'
"$inspector"
printf 'MODEL_CLI_PYTHON40_ASSETS_BEFORE_END\n'
printf 'MODEL_CLI_PYTHON40_CLI_BEGIN\n'
if GEMMA4_MAX_NEW_TOKENS=40 FURIOSA_OPT_PROFILE=warn "$model_cli" 'Complete the Python expression: len([1, 2, 3])' 2>&1; then
    cli_result=0
else
    cli_result=$?
fi
printf '\nMODEL_CLI_PYTHON40_CLI_EXIT=%s\n' "$cli_result"
# Preserve post-run asset evidence even if the CLI returned an error. Its
# original exit code takes priority over any subsequent inspector failure.
printf 'MODEL_CLI_PYTHON40_ASSETS_AFTER_BEGIN\n'
if "$inspector" --header-only; then
    asset_result=0
else
    asset_result=$?
fi
printf 'MODEL_CLI_PYTHON40_ASSETS_AFTER_END\n'
if [ "$cli_result" -ne 0 ]; then
    exit "$cli_result"
fi
if [ "$asset_result" -ne 0 ]; then
    exit "$asset_result"
fi
printf 'MODEL_CLI_PYTHON40_STRUCTURAL_COMPLETION_NO_ACCURACY_VERDICT\n'

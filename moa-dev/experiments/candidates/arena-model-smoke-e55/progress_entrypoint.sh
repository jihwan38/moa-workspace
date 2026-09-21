#!/bin/sh
set -eu
cd "$(dirname "$0")"
chmod +x ./probe_model_logits
printf 'MOA_E55_MODEL_PROGRESS_BEGIN\n'
set +e
FURIOSA_OPT_PROFILE=warn ./probe_model_logits --emit reference.json
result=$?
set -e
printf '\nMOA_E55_MODEL_PROGRESS_EXIT=%s\n' "$result"
exit "$result"

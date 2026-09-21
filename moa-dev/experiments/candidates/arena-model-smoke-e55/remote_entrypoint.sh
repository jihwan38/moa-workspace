#!/bin/sh
# One generated token from the unchanged E55 model host; development smoke only.
set -eu
cd "$(dirname "$0")"
binary=./gemma4
chmod +x "$binary" 2>/dev/null || true
if [ ! -x "$binary" ]; then
    cp "$binary" ./gemma4.exec
    chmod +x ./gemma4.exec
    binary=./gemma4.exec
fi
printf 'MOA_E55_MODEL_SMOKE_BEGIN\n'
set +e
GEMMA4_MAX_NEW_TOKENS=1 FURIOSA_OPT_PROFILE=warn "$binary" Hi
result=$?
set -e
printf '\nMOA_E55_MODEL_SMOKE_EXIT=%s\n' "$result"
rm -f ./gemma4.exec
exit "$result"

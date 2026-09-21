#!/bin/sh
set -u
cd "$(dirname "$0")" || exit 1
binary=./test_runtime
if [ ! -x "$binary" ]; then
    cp "$binary" ./test_runtime.exec || exit 126
    chmod +x ./test_runtime.exec || exit 126
    binary=./test_runtime.exec
fi
"$binary" --fixture boundary.safetensors
status=$?
rm -f ./test_runtime.exec
exit "$status"

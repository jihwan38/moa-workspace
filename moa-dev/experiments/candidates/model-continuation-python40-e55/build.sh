#!/bin/sh
set -eu
continuation_source=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$continuation_source/build.py" "$@"

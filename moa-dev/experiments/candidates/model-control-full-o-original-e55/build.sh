#!/bin/sh
set -eu
control_source=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$control_source/build.py"

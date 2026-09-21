#!/usr/bin/env bash
# Build only in this control's isolated Docker scratch. No model/NPU execution.
set -euo pipefail
candidate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$candidate_dir/build.py"

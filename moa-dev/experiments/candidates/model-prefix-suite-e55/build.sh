#!/usr/bin/env bash
# Isolated host build only. Never launches a model or changes the live crate.
set -euo pipefail
candidate_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_dir=$(cd -- "$candidate_dir/../../../.." && pwd)
exec python3 "$candidate_dir/build.py"

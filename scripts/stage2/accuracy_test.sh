#!/bin/bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/arena_test.sh" accuracy "$@"

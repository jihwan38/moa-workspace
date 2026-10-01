#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Uploaded files may be owned by another user or lack executable permission.
# Arena mounts /tmp read-only; the uploaded job directory is writable.
work=$(mktemp -d "$PWD/stage2-work.XXXXXX")
trap 'rm -rf "$work"' EXIT
tar -xzf stage2-payload.tar.gz -C "$work"
mkdir "$work/py"
tar -xzf stage2-python.tar.gz -C "$work/py"
cp server "$work/server"
chmod +x "$work/server" "$work/py/bin/python3"
export PYTHONHOME="$work/py"
export LD_LIBRARY_PATH="$work/py/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$work"
# arena_test.sh writes the selected mode (accuracy or performance) into the payload.
mode=$(cat mode)
set +e
FURIOSA_OPT_PROFILE="${FURIOSA_OPT_PROFILE:-info}" "$work/py/bin/python3" -u "$mode.py" --remote
status=$?
set -e
if [ "$status" -eq 0 ]; then
    echo "remote_entrypoint.sh: all $mode tests passed"
else
    echo "remote_entrypoint.sh: $mode tests failed (exit $status)" >&2
fi
exit "$status"

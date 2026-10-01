#!/bin/bash
set -euo pipefail

CRATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CRATE"

MODE="${1:-}"
case "$MODE" in
    accuracy|performance) shift ;;
    *) echo "arena_test.sh: run accuracy_test.sh or performance_test.sh instead" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python3}"
HELPER="scripts/stage2/$MODE.py"
TEST=""
CONCURRENCY=1
POLL_SECONDS="${RNGD_POLL_SECONDS:-5}"
TIMEOUT="${RNGD_TIMEOUT:-1800}"
SUBMIT_TIMEOUT="${RNGD_SUBMIT_TIMEOUT:-120}"

build=1
wait_for_result=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --test)
            if [ "$#" -lt 2 ] || [ -n "$TEST" ]; then
                echo "${MODE}_test.sh: specify --test NAME exactly once" >&2
                exit 2
            fi
            TEST="$2"
            shift 2
            ;;
        --concurrency)
            if [ "$#" -lt 2 ]; then
                echo "${MODE}_test.sh: --concurrency needs a number" >&2
                exit 2
            fi
            CONCURRENCY="$2"
            shift 2
            ;;
        --no-build) build=0; shift ;;
        --no-wait) wait_for_result=0; shift ;;
        -h|--help)
            echo "Usage: scripts/stage2/${MODE}_test.sh --test NAME [--concurrency N] [--no-build] [--no-wait]"
            if [ "$MODE" = accuracy ]; then
                echo "Required: --test NAME (mmlu, gsm8k, ifeval, vqav2, fleurs_en, fleurs_ko)"
            else
                echo "Required: --test NAME (256, 1024, 2048, image, audio)"
            fi
            echo "Concurrency: 1-100 (default 1); send the same input 2N times, N at a time"
            exit 0
            ;;
        *) echo "${MODE}_test.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
"$PYTHON" "$HELPER" --prepare "$staging" --test "$TEST" --concurrency "$CONCURRENCY"

BINARY="target/release/server"

if [ "$build" -eq 1 ]; then
    echo "==> building server"
    cargo furiosa-opt build --release --bin server
fi

for required in "$BINARY" scripts/stage2/entrypoint.sh public-test/cases.jsonl; do
    if [ ! -f "$required" ]; then
        echo "${MODE}_test.sh: missing $required; build the server and check public-test/" >&2
        exit 1
    fi
done

cp scripts/stage2/entrypoint.sh "$staging/remote_entrypoint.sh"
cp "$BINARY" "$staging/server"
chmod +x "$staging/remote_entrypoint.sh" "$staging/server"
echo "$MODE" > "$staging/mode"
tar -czf "$staging/stage2-payload.tar.gz" \
    -C "$CRATE/scripts/stage2" accuracy.py scoring.py suites.json performance.py performance_text.json \
    -C "$staging" mode selection.json -C "$CRATE" public-test
"$PYTHON" "$HELPER" --bundle-python "$staging/stage2-python.tar.gz"

job_name="${RNGD_JOB_NAME:-stage2_${MODE}_$RANDOM}"

echo "==> submitting $job_name ($(du -ch "$staging"/* | tail -1 | cut -f1) total)"
set +e
submit_output=$(furiosa-arena submit \
    "$staging/remote_entrypoint.sh" \
    "$staging/server" \
    "$staging/stage2-payload.tar.gz" \
    "$staging/stage2-python.tar.gz" \
    --name "$job_name" \
    --entrypoint remote_entrypoint.sh \
    --timeout "$SUBMIT_TIMEOUT" 2>&1)
submit_status=$?
set -e
echo "$submit_output"
if [ "$submit_status" -ne 0 ]; then
    echo "${MODE}_test.sh: furiosa-arena submit failed (exit $submit_status, shown above)" >&2
    exit 1
fi

job=$(printf '%s\n' "$submit_output" | sed -n 's/.*submitted job \([0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "$job" ]; then
    echo "${MODE}_test.sh: could not find a job id in furiosa-arena's output (shown above)" >&2
    exit 1
fi

if [ "$wait_for_result" -eq 0 ]; then
    echo "==> submitted job $job; follow it with: furiosa-arena logs $job"
    exit 0
fi

json_field() {
    sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\1/p" <<<"$1" | head -1
}

is_terminal() {
    case "$1" in
        succeeded|failed|completed|cancelled|canceled) return 0 ;;
        *) return 1 ;;
    esac
}

echo "==> waiting on job $job (polling every ${POLL_SECONDS}s)"
deadline=$(( SECONDS + TIMEOUT ))
state=""
status_output=""
while [ "$SECONDS" -lt "$deadline" ]; do
    status_output=$(furiosa-arena status "$job" 2>&1 || true)
    state=$(json_field "$status_output" status | tr '[:upper:]' '[:lower:]')
    is_terminal "$state" && break
    case "$state" in
        queued|running) ;;
        *) echo "${MODE}_test.sh: unexpected status '${state:-unreadable}', retrying" >&2 ;;
    esac
    sleep "$POLL_SECONDS"
done

if ! is_terminal "$state"; then
    echo "${MODE}_test.sh: job $job still '${state:-unknown}' after ${TIMEOUT}s; cancel with: furiosa-arena cancel $job" >&2
    exit 1
fi

code=$(json_field "$status_output" exit_code)
echo "==> job $job $state (exit ${code:-?}); log follows"
furiosa-arena logs "$job" || true

[ "${code:-1}" = "0" ]

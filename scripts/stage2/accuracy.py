#!/usr/bin/env python3
"""Public accuracy test, and the helpers it shares with `performance.py`.

`arena_test.sh` calls this file twice on the submitting machine: `--prepare` validates the
selection and writes `selection.json`, and `--bundle-python` packs a Python runtime, since
the Arena worker has none. Inside the Arena job, `entrypoint.sh` runs it with `--remote`:
it starts the server, sends the selected case, scores every response and prints the result.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import sysconfig
import tarfile
import time
import threading
import urllib.request

import scoring

HERE = Path(__file__).resolve().parent
CRATE = HERE.parent.parent
SPECS = json.loads((HERE / "suites.json").read_text())
BASE_URL = "http://127.0.0.1:8000"

# Requests sent per run, as a multiple of the concurrency. With more requests than are in
# flight, later ones are admitted while others are still running.
ROUNDS = 2
STARTUP_TIMEOUT = 90
REQUEST_TIMEOUT = 120


def media_path(base: Path, value: str) -> Path:
    """Resolve a media reference, refusing anything outside `base` or missing."""
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"Missing or invalid media path: {value}")
    return path


def expand(node, base: Path):
    """Return `node` with every `$image`/`$audio` path replaced by the base64-encoded file."""
    if isinstance(node, dict):
        if "$image" in node or "$audio" in node:
            key = "$image" if "$image" in node else "$audio"
            path = media_path(base, node[key])
            encoded = base64.b64encode(path.read_bytes()).decode()
            result = {k: v for k, v in node.items() if k != key}
            if key == "$image":
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                result["url"] = f"data:{mime};base64,{encoded}"
            else:
                result["data"] = encoded
                result.setdefault("format", "wav")
            return result
        return {k: expand(v, base) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v, base) for v in node]
    return node


def read_cases() -> list[dict]:
    """Load `public-test/cases.jsonl` and check each case against its suite."""
    base = CRATE / "public-test"
    rows = [json.loads(line) for line in (base / "cases.jsonl").read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs")
    for row in rows:
        spec = SPECS["suites"][row["suite"]]
        if row["check"]["type"] != spec["metric"]:
            raise ValueError(f"Unexpected metric: {row['id']}")
        expand(row["request"], base)  # Only checks the media; the job expands it again.
    return rows


def select_case(rows: list[dict], suite: str | None) -> dict:
    if suite not in SPECS["suites"]:
        raise ValueError("Choose one test: " + ", ".join(SPECS["suites"]))
    matches = [row for row in rows if row["suite"] == suite]
    if len(matches) != 1:
        raise ValueError(f"{suite} must contain exactly one case; found {len(matches)}")
    return matches[0]


def concurrency_value(value: str) -> int:
    value = int(value)
    if not 1 <= value <= 100:
        raise argparse.ArgumentTypeError("concurrency must be between 1 and 100")
    return value


def python_bundle(destination: Path) -> None:
    """Bundle this Linux Python's standard library, without site packages or downloads.

    Like the server binary, this runtime needs an ABI-compatible Arena worker.
    """
    if sys.platform != "linux":
        raise ValueError("Runtime packaging requires Linux")
    stdlib = Path(sysconfig.get_path("stdlib"))
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    def include(info):
        if set(Path(info.name).parts) & {"site-packages", "dist-packages", "__pycache__", "test", "tests", "ensurepip"}:
            return None
        return info

    with tarfile.open(destination, "w:gz", dereference=True) as archive:
        archive.add(Path(sys.executable).resolve(), "bin/python3")
        archive.add(stdlib, f"lib/{version}", filter=include)
        dynload = Path(sysconfig.get_config_var("DESTSHARED") or stdlib / "lib-dynload")
        if dynload.is_dir() and dynload != stdlib / "lib-dynload":
            archive.add(dynload, f"lib/{version}/lib-dynload", filter=include)
        # Arena's minimal image may lack expat/zlib/OpenSSL. Include the interpreter
        # and loaded stdlib extensions' dependencies, leaving glibc to the worker.
        linked = {str(Path(sys.executable).resolve())}
        linked.update(str(module.__file__) for module in list(sys.modules.values())
                      if getattr(module, "__file__", "") and str(module.__file__).endswith(".so"))
        dependencies = {}
        for binary in sorted(linked):
            output = subprocess.run(["ldd", binary], check=True, text=True, capture_output=True).stdout
            for name, path in re.findall(r"^\s*(\S+)\s+=>\s+(/\S+)", output, re.MULTILINE):
                if not re.match(r"lib(c|m|dl|pthread|rt|resolv|util|anl|nss_\w+)\.so", name):
                    dependencies[name] = path
        for name, path in sorted(dependencies.items()):
            archive.add(path, f"lib/{name}")


def emit(prefix: str, record: dict) -> None:
    print(prefix + json.dumps(record, ensure_ascii=False), flush=True)


def request(path: str, body: dict | None = None, timeout: float = 10) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run_request(row: dict, body: dict, request_index: int, timeout: float,
                start: threading.Event) -> dict:
    """Send one request and score its response. Never raises: a failure is a failed record."""
    start.wait()
    record = {"id": row["id"], "suite": row["suite"], "request_index": request_index, "passed": False}
    started = time.monotonic()
    try:
        payload = request("/v1/chat/completions", body, timeout=timeout)
        choice = payload["choices"][0]
        text = choice["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty or non-text answer")
        check = row["check"]
        spec = SPECS["suites"][row["suite"]]
        score = scoring.score_item(check["type"], text, check["expected"], check.get("options", "ABCDEFGHIJ"))
        if isinstance(score, tuple):
            edits, length = score
            score = edits / max(1, length)
            record.update(edits=edits, ref_length=length)
        passed = score <= spec["item_pass"] if spec["direction"] == "lower" else score >= spec["item_pass"]
        record.update(output=text, score=score, metric=spec["metric"], threshold=spec["item_pass"],
                      passed=passed, finish_reason=choice.get("finish_reason"), usage=payload.get("usage"))
        if not passed:
            record["error"] = "Accuracy threshold not met"
            if choice.get("finish_reason") == "length":
                record["error"] += "; output hit the token limit"
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def remote(run_one=run_request, summarize=None, prefix: str = "STAGE2") -> int:
    """Run inside the Arena job: start the server, send the requests, print the results.

    `performance.py` reuses this with its own `run_one` and `summarize`.
    """
    config = json.loads(Path("selection.json").read_text())
    row = config["case"]
    concurrency = config["concurrency"]
    total = ROUNDS * concurrency
    started = time.monotonic()
    env = dict(os.environ, GEMMA4_API_ADDR="127.0.0.1:8000", GEMMA4_API_KEY="")
    records = []
    server = None
    try:
        if not env.get("RNGD_MODEL_DIR"):
            raise ValueError("Arena must supply RNGD_MODEL_DIR")
        body = expand(row["request"], Path("public-test"))
        with open("server.log", "w") as log:
            server = subprocess.Popen(["./server"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        ready_deadline = started + config["startup_timeout"]
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("Server startup deadline exceeded")
            try:
                request("/v1/models", timeout=min(1, ready_deadline - time.monotonic()))
                break
            except (OSError, ValueError):
                time.sleep(0.2)
        print(f"Server ready after {time.monotonic() - started:.2f}s", flush=True)
        print(f"Sending {total} requests for {row['id']}, {concurrency} at a time", flush=True)

        # The pool keeps `concurrency` requests in flight: the first ones are released
        # together, and each later one starts as soon as a worker finishes.
        start = threading.Event()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            try:
                futures = [pool.submit(run_one, row, body, index, config["request_timeout"], start)
                           for index in range(1, total + 1)]
            finally:
                # `performance.py` reports each request's times relative to this instant.
                start.epoch = time.monotonic()
                start.set()  # Also release created workers if thread creation fails.
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                emit(prefix + "_RESULT ", record)
    except Exception as err:
        done = {record["request_index"] for record in records}
        for index in range(1, total + 1):
            if index not in done:
                record = dict(id=row["id"], suite=row["suite"], request_index=index, passed=False, error=str(err))
                records.append(record)
                emit(prefix + "_RESULT ", record)
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
            except ProcessLookupError:
                pass
        if Path("server.log").exists():
            print("SERVER_LOG_TAIL\n" + Path("server.log").read_text(errors="replace")[-8000:])
    passed = sum(record["passed"] for record in records)
    summary = {"id": row["id"], "suite": row["suite"], "concurrency": concurrency,
               "total": total, "passed": passed, "failed": total - passed}
    if summarize is not None:
        summary.update(summarize(records))
    emit(prefix + "_SUMMARY ", summary)
    return 0 if passed == total else 1


def local(description: str, select) -> int:
    """The steps `arena_test.sh` runs before submitting: `--prepare` or `--bundle-python`.

    `select` maps the `--test` value to the case to run.
    """
    parser = argparse.ArgumentParser(description=description)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--bundle-python", type=Path)
    parser.add_argument("--test")
    parser.add_argument("--concurrency", type=concurrency_value, default=1)
    args = parser.parse_args()

    if args.bundle_python:
        python_bundle(args.bundle_python)
        return 0

    row = select(args.test)
    print(f"Selected: {row['id']}; concurrency: {args.concurrency}", flush=True)
    config = {
        "case": row,
        "concurrency": args.concurrency,
        "startup_timeout": STARTUP_TIMEOUT,
        "request_timeout": REQUEST_TIMEOUT,
    }
    (args.prepare / "selection.json").write_text(json.dumps(config, indent=2) + "\n")
    return 0


def run(script: str, remote_main, local_main) -> None:
    """Entry point shared by both scripts; errors are reported as `script: message`."""
    try:
        sys.exit(remote_main() if sys.argv[1:] == ["--remote"] else local_main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"{script}: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stdout or "", error.stderr or "", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run("accuracy_test.sh", remote,
        lambda: local(__doc__.splitlines()[0], lambda name: select_case(read_cases(), name)))

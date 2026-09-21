#!/usr/bin/env python3
"""Capture the installed submitter's payload on loopback; never submit officially.

Run inside the existing development container. Each invocation requires a new
output directory. A synthetic token is passed only to the local receiver; no
login flow or saved credential is used. This checks CLI packaging, not the
competition server's contract, build, correctness, score or authentication.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading


def fingerprint(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def sources(root):
    paths = [root / "src/ops.rs"]
    for path in (root / "src/device").rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            paths.append(path)
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("Expected regular source files")
    result = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(paths)}
    assert 1 < len(result) <= 1000 and all(result.values())
    assert sum(map(len, result.values())) <= 10 * 1024 * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    output = args.output.absolute()
    output.mkdir()  # Never overwrite earlier observations.
    expected = sources(root)
    cli = Path(shutil.which("moa-submitter")).resolve(strict=True)
    version = subprocess.check_output([str(cli), "--version"], text=True).strip()
    assert version == "moa-submitter 1.2.0", version
    events, captures, errors = [], [], []
    marker = "LOCAL_PAYLOAD_CAPTURED_NO_OFFICIAL_SUBMISSION"
    dummy = "LOCAL_PREFLIGHT_ONLY_NOT_A_CREDENTIAL"

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, status, message):
            data = json.dumps({"message": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            events.append(["GET", self.path])
            self.answer(200 if self.path == "/api/health" else 404, "local only")

        def do_POST(self):
            events.append(["POST", self.path])
            try:
                assert self.path == "/api/submissions"
                assert self.client_address[0] == "127.0.0.1"
                assert self.headers.get("Authorization") == "Bearer " + dummy
                size = int(self.headers["Content-Length"])
                assert 0 < size < 20 * 1024 * 1024
                body = self.rfile.read(size)
                assert len(body) == size
                request = json.loads(body)
                assert set(request) == {"files"}
                decoded = {}
                for entry in request["files"]:
                    assert set(entry) == {"path", "contentBase64"}
                    name = entry["path"]
                    assert name not in decoded
                    decoded[name] = base64.b64decode(entry["contentBase64"], validate=True)
                assert list(decoded) == sorted(expected)
                assert decoded == expected
                captures.append(body)
                # Deliberately refuse: never synthesize an accepted submission ID.
                self.answer(400, marker)
            except Exception as exc:
                errors.append(type(exc).__name__ + ": " + str(exc))
                self.answer(400, "LOCAL_CAPTURE_VALIDATION_FAILED")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    assert server.server_address[0] == "127.0.0.1"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    environment = os.environ.copy()
    for name in list(environment):
        if name.lower().endswith("_proxy") or name in {"MOA_TOKEN", "MOA_API_URL"}:
            environment.pop(name)
    environment.update(MOA_TOKEN=dummy, NO_PROXY="127.0.0.1,localhost",
                       XDG_CONFIG_HOME=str(output / "isolated-config"))
    empty = output / "empty-cwd"
    empty.mkdir()
    cases = [("crate-cwd", root, []),
             ("nested-cwd", root / "src/device/sliding", []),
             ("explicit-source", empty, ["--source", str(root)]),
             ("outside-without-source", empty, [])]
    results = []
    try:
        for name, cwd, flags in cases:
            before = len(events), len(captures)
            command = [str(cli), "--server", endpoint, "submit", *flags]
            process = subprocess.run(command, cwd=cwd, env=environment,
                                     capture_output=True, text=True, timeout=30)
            log = process.stdout + process.stderr
            (output / f"{name}.log").write_text(log)
            observed = events[before[0]:]
            assert not errors, errors
            if name == "outside-without-source":
                assert process.returncode != 0 and "No baseline repository here" in log
                assert not observed and len(captures) == before[1]
            else:
                assert process.returncode != 0 and marker in log
                assert observed == [["GET", "/api/health"], ["POST", "/api/submissions"]]
                assert len(captures) == before[1] + 1
                (output / f"{name}.request.json").write_bytes(captures[-1])
            results.append({"name": name, "cwd": str(cwd), "command": command,
                            "exit_code": process.returncode, "events": observed,
                            "log": fingerprint(log.encode()), "passed": True})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert len(captures) == 3 and captures[0] == captures[1] == captures[2]
    assert sources(root) == expected
    assert not (output / "isolated-config").exists()
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "LOCAL_CLI_PAYLOAD_CHECK_PASS", "cli_version": version,
              "cli_binary": fingerprint(cli.read_bytes()),
              "source": str(root), "source_files": {n: fingerprint(b) for n, b in expected.items()},
              "source_count": len(expected), "source_bytes": sum(map(len, expected.values())),
              "payload": fingerprint(captures[0]), "three_requests_byte_identical": True,
              "source_unchanged_after_checks": True, "cases": results,
              "receiver": endpoint, "synthetic_token_only": True,
              "saved_config_used_or_written": False, "official_submission_performed": False,
              "server_contract_validated": False, "authentication_validated": False,
              "scope": "Installed CLI against a loopback receiver returning intentional HTTP400. "
                       "The three captured uploads contain the exact current allowed sources; "
                       "no official server request, score, rank or authentication test."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "source_count": len(expected),
                      "source_bytes": report["source_bytes"], "payload": report["payload"],
                      "cases_passed": len(results), "official_submission_performed": False}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pinned q020 source submission; the caller schedules it, this tool never does.

--check reads files/account/quota only and creates no run directory.
--submit permanently consumes its local attempt marker before the sole submit
subprocess. A timeout may leave the remote request running: never remove the
marker or retry automatically. Source files must remain frozen externally.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

CANDIDATE = "q020-b16-f4-mse-parallel"
MANIFEST_SHA = "6e618edcc3254e57bf9652d27dd673e631cdc2f77ffac5473673ac01f115d4c8"
SOURCE_COMMIT = "d52f0c08504d4827948abd8536b2d0011645fcce"
ELF_SHA = "ba8cc68d6610a05634319bdb5741017ba4fa79bb4902cc11c77f33d2b163f817"
SOURCE_FILES = 397
EARLIEST = datetime(2026, 10, 25, 21, 2, tzinfo=timezone.utc)
LATEST = datetime(2026, 10, 26, 11, 30, tzinfo=timezone.utc)
CONTAINER = "moa2026-round2"
CONTAINER_CANDIDATE = f"/round2/candidates/{CANDIDATE}"
RUN_NAME = "official-q020-20261026-c16"
TERMINAL = {"completed", "failed", "cancelled", "canceled", "error", "rejected"}


class Refused(RuntimeError):
    pass


def utcnow():
    return datetime.now(timezone.utc)


def sha256(path):
    if path.is_symlink() or not path.is_file():
        raise Refused(f"not a regular immutable input: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise Refused(f"input changed while hashing: {path}")
    return digest.hexdigest()


def verify_candidate(root):
    candidate = root / "candidates" / CANDIDATE
    if candidate.is_symlink() or (candidate / "src").is_symlink():
        raise Refused("candidate/src symlink is forbidden")
    manifest = candidate / "candidate.json"
    if sha256(manifest) != MANIFEST_SHA:
        raise Refused("pinned candidate.json SHA mismatch")
    data = json.loads(manifest.read_text())
    if data.get("source_commit") != SOURCE_COMMIT or data.get("binary_sha256") != ELF_SHA:
        raise Refused("pinned source commit or ELF metadata mismatch")
    hashes = data.get("source_hashes", {})
    if len(hashes) != SOURCE_FILES:
        raise Refused("expected exactly 397 source/config hashes")
    if {p for p in hashes if not p.startswith("src/")} != {"Cargo.toml", "Cargo.lock", "rust-toolchain.toml"}:
        raise Refused("unexpected config manifest entries")
    actual = set()
    for path in (candidate / "src").rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise Refused(f"nonregular src entry: {path}")
        if path.is_file():
            actual.add(path.relative_to(candidate).as_posix())
    expected = {p for p in hashes if p.startswith("src/")}
    if actual != expected:
        raise Refused(f"src inventory mismatch: missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    for name, expected_hash in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise Refused("invalid manifest path")
        if sha256(candidate / relative) != expected_hash:
            raise Refused(f"source/config SHA mismatch: {name}")
    elf = candidate / "target" / "release" / "server"
    if sha256(elf) != ELF_SHA or elf.stat().st_size != data.get("bytes"):
        raise Refused("frozen ELF SHA/size mismatch")
    return {"candidate": CANDIDATE, "source_commit": SOURCE_COMMIT,
            "manifest_sha256": MANIFEST_SHA, "binary_sha256": ELF_SHA,
            "verified_source_config_files": len(hashes), "unexpected_src_files": 0}


def cli(*args):
    return ["docker", "exec", "--workdir", CONTAINER_CANDIDATE,
            CONTAINER, "moa-submitter", *args]


def run_process(args, timeout):
    return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True,
                          timeout=timeout, check=False)


def output_bytes(value):
    return value.encode() if isinstance(value, str) else (value or b"")


def parse_auth(text):
    teams = re.findall(r"^Team:[ \t]*([^\r\n]*)$", text, re.M)
    stages = re.findall(r"^Stage 2:[ \t]*([^\r\n]*)$", text, re.M)
    if [v.strip() for v in teams] != ["BEEF"] or [v.strip() for v in stages] != ["advanced"]:
        raise Refused("whoami must report Team BEEF and Stage 2: advanced")


def parse_quota(text):
    found = re.findall(r"^Submissions in the last 24 hours:\s+(\d+) of (\d+)\s*$", text, re.M)
    if len(found) != 1:
        raise Refused("missing/ambiguous live quota line")
    used, limit = map(int, found[0])
    if limit != 4 or not 0 <= used <= limit:
        raise Refused("unexpected live quota")
    return used


def parse_submission(text):
    teams = re.findall(r"^Submitted as:\s+(\S+)\s*$", text, re.M)
    ids = re.findall(r"^Submission:\s+([0-9a-f]{8})\s*$", text, re.M)
    if teams != ["BEEF"] or len(ids) != 1:
        raise Refused("submission response ambiguous; marker retained, do not resubmit")
    return ids[0]


def parse_status(text, submission):
    if re.findall(r"^Submission:\s+([0-9a-f]{8})\s*$", text, re.M) != [submission]:
        raise Refused("status ID mismatch")
    if re.findall(r"^Team:\s+(\S+)\s*$", text, re.M) != ["BEEF"]:
        raise Refused("status team mismatch")
    states = re.findall(r"^Status:\s+([a-zA-Z ]+)\s*$", text, re.M)
    if len(states) != 1:
        raise Refused("status missing/ambiguous")
    return states[0].strip().lower()


def check_time(now):
    if not EARLIEST <= now <= LATEST:
        raise Refused(f"outside allowed submit window: {now.isoformat()}")


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path, data, exclusive=False):
    encoded = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    target = path if exclusive else path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(target, path)
    fsync_dir(path.parent)


def write_raw(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class Journal:
    def __init__(self, directory, runner, now):
        self.directory, self.runner, self.now = directory, runner, now
        self.tag = f"{now().strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.index = 0
        self.state = {"invocation": self.tag, "started_at": now().isoformat()}

    def save(self, **fields):
        self.state.update(fields, updated_at=self.now().isoformat())
        write_json(self.directory / "status.json", self.state)

    def command(self, name, args, timeout=45):
        self.index += 1
        prefix = self.directory / f"{self.tag}-{self.index:03d}-{name}"
        started = self.now().isoformat()
        error = None
        try:
            result = self.runner(args, timeout)
            code, out, err = result.returncode, output_bytes(result.stdout), output_bytes(result.stderr)
        except (subprocess.TimeoutExpired, OSError) as exc:
            code, out, err = None, output_bytes(getattr(exc, "stdout", None)), output_bytes(getattr(exc, "stderr", None))
            error = f"{type(exc).__name__}: {exc}"
        write_raw(prefix.with_suffix(".stdout"), out)
        write_raw(prefix.with_suffix(".stderr"), err)
        write_json(prefix.with_suffix(".json"), {"argv": args, "started_at": started,
            "finished_at": self.now().isoformat(), "returncode": code, "error": error,
            "stdout_sha256": hashlib.sha256(out).hexdigest(), "stderr_sha256": hashlib.sha256(err).hexdigest()})
        return code, out.decode("utf-8", "replace"), error

    def read(self, name, args, timeout=45):
        code, text, error = self.command(name, args, timeout)
        if code != 0:
            raise Refused(f"{name} failed: returncode={code}, error={error}")
        return text


def check_mount(root, text):
    mounts = json.loads(text)
    if not any(m.get("Destination") == "/round2" and Path(m.get("Source", "")).resolve() == root for m in mounts):
        raise Refused("container /round2 does not bind the explicitly supplied host root")


def monitor(journal, submission, now, sleep):
    deadline = now() + timedelta(hours=2)
    while now() < deadline:
        try:
            remaining = (deadline-now()).total_seconds()
            state = parse_status(journal.read("submission-status", cli("status", submission), min(45, remaining)), submission)
            journal.save(phase="accepted", submission_id=submission, remote_status=state)
            if state in TERMINAL:
                remaining = (deadline-now()).total_seconds()
                if remaining <= 0:
                    journal.save(phase="terminal", log_collected=False, log_error="collection deadline reached")
                    return 2
                code, _, error = journal.command("terminal-log", cli("log", submission), min(60, remaining))
                journal.save(phase="terminal", log_collected=code == 0, log_error=error)
                return 0 if code == 0 else 2
        except Refused as exc:
            journal.save(monitor_error=str(exc))
        sleep(min(30, max(0, (deadline-now()).total_seconds())))
    journal.save(phase="accepted_monitor_timeout", submission_id=submission)
    return 2


def execute(root, submit, runner=run_process, now=utcnow, sleep=time.sleep):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise Refused("--root must name the absolute host round2 directory")
    root = root.resolve()
    inspect = ["docker", "inspect", "--format", "{{json .Mounts}}", CONTAINER]
    if not submit:
        proof = verify_candidate(root)
        texts = []
        for args in [inspect, cli("whoami"), cli("status")]:
            result = runner(args, 45)
            if result.returncode != 0:
                raise Refused("read-only preflight command failed")
            texts.append(output_bytes(result.stdout).decode("utf-8", "replace"))
        check_mount(root, texts[0]); parse_auth(texts[1]); used = parse_quota(texts[2])
        print(json.dumps({**proof, "readonly": True, "quota_used": used, "quota_limit": 4,
                          "within_submit_window": EARLIEST <= now() <= LATEST}))
        return 0

    check_time(now())  # Early invocation never sleeps until the submission window.
    directory = root / "runs" / RUN_NAME
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or (root / "runs").is_symlink():
        raise Refused("run directory symlink forbidden")
    lock = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Refused("another invocation holds the submission lock") from None
        marker = directory / "submit-attempt.json"
        if os.path.lexists(marker):
            raise Refused("submission already attempted; marker retained, manual read-only follow-up only")
        journal = Journal(directory, runner, now)
        try:
            proof = verify_candidate(root)
            journal.save(phase="preflight", **proof)
            check_mount(root, journal.read("container-mount", inspect))
            parse_auth(journal.read("whoami", cli("whoami")))
            quota_deadline = min(now()+timedelta(minutes=10), EARLIEST+timedelta(minutes=10))
            quota_reads = 0
            while True:
                check_time(now())
                remaining = (quota_deadline-now()).total_seconds()
                if quota_reads and remaining <= 0:
                    raise Refused("no quota slot within bounded read-only poll window")
                # A late timer catch-up gets one fresh read, but cannot start another poll window.
                timeout = min(45, remaining) if remaining > 0 else 45
                code, text, error = journal.command("quota", cli("status"), timeout)
                quota_reads += 1
                if code == 0:
                    used = parse_quota(text)  # malformed successful output fails closed, never retries
                    journal.save(quota_used=used, quota_limit=4)
                    if used < 4:
                        break
                else:
                    # Only read-only quota transport failures may retry, within the same deadline.
                    journal.save(quota_read_error=error, quota_returncode=code)
                if now()+timedelta(seconds=20) > quota_deadline:
                    raise Refused("no quota slot within bounded read-only poll window")
                sleep(20)
            # Recheck immutable inputs and current identity immediately before the sole attempt.
            proof = verify_candidate(root)
            parse_auth(journal.read("whoami-final", cli("whoami")))
            check_time(now())
            args = cli("submit", "--source", CONTAINER_CANDIDATE, "--concurrency", "16")
            attempt = {**proof, "argv": args, "attempted_at": now().isoformat(),
                       "invocation": journal.tag, "rule": "never retry this marker, including ambiguous responses"}
            write_json(marker, attempt, exclusive=True)  # durable BEFORE starting the external side effect
            journal.save(phase="attempt_marker_durable")
            code, response, error = journal.command("submit", args, 180)
            try:
                submission = parse_submission(response)
            except Refused:
                journal.save(phase="ambiguous_submission", returncode=code, submit_error=error)
                raise
            journal.save(phase="accepted", submission_id=submission, returncode=code,
                         submit_error=error, response_complete=code == 0)
            # Even a timeout with a complete acceptance line only triggers read-only collection.
            return monitor(journal, submission, now, sleep)
        except Exception as exc:
            journal.save(error=str(exc), attempt_marker_exists=marker.exists())
            raise
    finally:
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    try:
        return execute(args.root, args.submit)
    except (Refused, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"REFUSED/STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

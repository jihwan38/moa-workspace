#!/usr/bin/env python3
"""No HTTP/model execution: check prepared inputs and copied-runner semantics."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def sha(raw):
    return hashlib.sha256(raw).hexdigest()

prepared = json.loads((HERE / "prepared-evaluations.json").read_text())
for item in prepared:
    dest = Path(item["evaluation"])
    config = json.loads((dest / "selection-preview.json").read_text())
    assert config["order"] == list(range(16))
    assert len({r["id"] for r in config["cases"]}) == 16
    assert sha((dest / "target/release/server").read_bytes()) == item["server_sha256"]
    assert sha((dest / "scripts/stage2/scoring.py").read_bytes()) == item["scoring_sha256"]
    requests = []
    for row in config["cases"]:
        body = row["request"]
        assert set(body) == {"model", "messages", "max_completion_tokens", "temperature"}
        assert body["max_completion_tokens"] == 768 and body["temperature"] == 0
        assert set(body["messages"][0]) == {"role", "content"}
        prefix, question = body["messages"][0]["content"][0]["text"].split("\n\n", 1)
        assert sha(prefix.encode()) == item["organizer_prefix_sha256"]
        assert row["provenance"]["split"] == "train"
        source = next(r for r in map(json.loads, (HERE / "selected-train.jsonl").read_text().splitlines())
                      if r["source_row"] == row["provenance"]["row"])
        assert question == source["question"] and row["check"]["expected"] == source["answer"]
        requests.append(body)
    raw = b"".join((json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"))+"\n").encode() for r in requests)
    assert sha(raw) == item["request_set_sha256"]
    (HERE / "shards").mkdir(exist_ok=True)
    shard = HERE / "shards" / f"shard{item['shard']}.jsonl"
    case_bytes = (dest / "public-test/cases.jsonl").read_bytes()
    if shard.exists(): assert shard.read_bytes() == case_bytes
    else: shard.write_bytes(case_bytes)

stage = Path(prepared[0]["evaluation"]) / "scripts/stage2"
sys.path.insert(0, str(stage))
spec = importlib.util.spec_from_file_location("research_accuracy", stage / "accuracy.py")
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)
assert accuracy.select_test("gsm8k", 16)[2] == list(range(16))
for suite, width in [("gsm8k", 15), ("mmlu", 16)]:
    try: accuracy.select_test(suite, width)
    except ValueError: pass
    else: raise AssertionError("invalid selection accepted")

class FakeServer:
    pid = 123456789
    returncode = 0
    def poll(self): return None
    def wait(self, timeout=None): return 0

def exercise(failed_index):
    rows = [{"id": f"fake{i}", "suite": "gsm8k", "request": {"fake_index": i},
             "check": {"type": "gsm8k", "expected": "#### 4"}} for i in range(16)]
    calls = []
    def fake_request(path, body=None, timeout=10):
        if path == "/v1/models": return {"data": []}
        calls.append(body["fake_index"])
        if body["fake_index"] == failed_index: raise OSError("intentional transport failure")
        return {"choices": [{"message": {"content": "#### 5"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            Path("selection.json").write_text(json.dumps({"cases": rows, "order": list(range(16)),
                "concurrency": 16, "test": "fake-completion-contract", "startup_timeout": 1, "request_timeout": 1}))
            output = io.StringIO()
            with patch.dict(os.environ, {"RNGD_MODEL_DIR": "FAKE_NOT_OPENED"}), \
                 patch.object(accuracy.subprocess, "Popen", return_value=FakeServer()), \
                 patch.object(accuracy.os, "killpg"), patch.object(accuracy, "request", side_effect=fake_request), \
                 contextlib.redirect_stdout(output):
                code = accuracy.remote()
        finally: os.chdir(old)
    records = [json.loads(l.split(" ", 1)[1]) for l in output.getvalue().splitlines() if l.startswith("STAGE2_RESULT ")]
    summary, = [json.loads(l.split(" ", 1)[1]) for l in output.getvalue().splitlines() if l.startswith("STAGE2_SUMMARY ")]
    assert sorted(calls) == list(range(16)) and len(records) == 16
    assert summary["accuracy_passed"] == 0 and summary["all_accuracy_passed"] is False
    expected = 16 if failed_index is None else 15
    assert summary["completed"] == expected and code == (0 if expected == 16 else 1)
    assert summary["accuracy_scored"] == expected and summary["accuracy_failed"] == expected
    assert summary["accuracy_unscored"] == 16 - expected
    for record in records:
        if record.get("completed"):
            assert record["output"] == "#### 5" and record["usage"]["completion_tokens"] == 7
    return {"intentional_failed_index": failed_index, "exit_code": code, "summary": summary}

report = {"evaluations_checked": len(prepared), "distinct_shards": 2,
          "request_identity_across_candidates": True,
          "model_or_http_executed": False, "mock_runs": [exercise(None), exercise(7)]}
(HERE / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))

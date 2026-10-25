#!/usr/bin/env python3
"""Read completed fixed GSM32 runs; never execute a model or change selection."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {
    "c081-physical32-clean": ["r1308", "r1310"],
    "q020-b16-f4-mse-parallel": ["r1309", "r1311"],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main():
    manifests = json.loads((HERE / "prepared-evaluations.json").read_text())
    selected = [json.loads(line) for line in (HERE / "selected-train.jsonl").read_text().splitlines()]
    dataset = json.loads((HERE / "dataset-manifest.json").read_text())
    assert sha((HERE / "selected-train.jsonl").read_bytes()) == dataset["selected_sha256"]
    assert [r["source_row"] for r in selected] == dataset["indices"]
    result = {"preregistration_commit": dataset["preregistration_commit"],
              "dataset_revision": dataset["revision"], "dataset_sha256": dataset["source_sha256"],
              "seed": dataset["seed"], "split": "train", "selected_rows": dataset["indices"],
              "official_score": False, "runs": [], "candidates": {}, "paired": []}
    per_candidate = {}
    request_hashes = {}
    for candidate, names in RUNS.items():
        accumulated = {}
        for shard, prefix in enumerate(names, 1):
            matches = list((ROOT / "runs").glob(prefix + "-*"))
            assert len(matches) == 1, (prefix, matches)
            run = matches[0]
            prepared = next(m for m in manifests if m["base_candidate"] == candidate and m["shard"] == shard)
            evaluation = Path(prepared["evaluation"])
            harness = json.loads((evaluation / "harness.json").read_text())
            config = json.loads((evaluation / "selection-preview.json").read_text())
            cases = config["cases"]
            assert config["order"] == list(range(16)) and len(cases) == 16
            assert [c["provenance"]["row"] for c in cases] == dataset["shards"][shard - 1]
            reqhash = sha(b"".join(canonical(c["request"]) for c in cases))
            assert reqhash == prepared["request_set_sha256"] == harness["request_set_sha256"]
            request_hashes.setdefault(shard, set()).add(reqhash)
            for filename, field in [("accuracy.py", "accuracy_helper_sha256"),
                                    ("scoring.py", "scoring_sha256"),
                                    ("entrypoint.sh", "entrypoint_sha256"),
                                    ("arena_test.sh", "arena_test_sha256")]:
                assert sha((evaluation / "scripts/stage2" / filename).read_bytes()) == prepared[field]
            assert sha((evaluation / "target/release/server").read_bytes()) == prepared["server_sha256"]
            spec = importlib.util.spec_from_file_location("organizer_scoring", evaluation / "scripts/stage2/scoring.py")
            scoring = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scoring)
            status = json.loads((run / "arena-status.json").read_text())
            tested = json.loads((run / "tested-candidate.json").read_text())
            assert tested["binary_sha256"] == prepared["server_sha256"]
            records = json.loads((run / "arena-records.json").read_text())
            outputs = [r["data"] for r in records if r["type"] == "STAGE2_RESULT"]
            summaries = [r["data"] for r in records if r["type"] == "STAGE2_SUMMARY"]
            assert len(summaries) == 1 and len(outputs) == 16
            summary = summaries[0]
            indexed = {r["id"]: r for r in outputs}
            assert len(indexed) == 16 and set(indexed) == {c["id"] for c in cases}
            assert status["status"] == "SUCCEEDED" and status["exit_code"] == 0
            assert summary["total"] == summary["completed"] == summary["accuracy_scored"] == 16
            assert summary["incomplete"] == summary["accuracy_unscored"] == 0
            for index, case in enumerate(cases, 1):
                row = indexed[case["id"]]
                assert row["request_index"] == index and row["completed"] is True
                assert row["metric"] == "gsm8k" and row["threshold"] == 1.0
                rescored = scoring.score_gsm8k(row["output"], case["check"]["expected"])
                assert rescored == row["score"] and (rescored >= 1.0) == row["passed"]
                usage = row["usage"]
                assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                assert 0 < usage["completion_tokens"] <= 768
                assert case["id"] not in accumulated
                accumulated[case["id"]] = {
                    "id": case["id"], "source_row": case["provenance"]["row"], "shard": shard,
                    "passed": row["passed"], "completed": row["completed"], "score": rescored,
                    "answer": scoring.extract_gsm8k(row["output"]),
                    "expected_answer": scoring.extract_gsm8k(case["check"]["expected"]),
                    "output_sha256": sha(row["output"].encode()),
                    "usage": usage, "finish_reason": row["finish_reason"], "seconds": row["seconds"],
                }
            assert sum(r["passed"] for r in outputs) == summary["accuracy_passed"]
            result["runs"].append({"run": run.name, "candidate": candidate, "shard": shard,
                "job": status["id"], "status": status["status"], "duration_seconds": status["duration_sec"],
                "source_commit": tested["source_commit"], "server_sha256": tested["binary_sha256"],
                "request_set_sha256": reqhash, "summary": summary,
                "files": {str(run / f): sha((run / f).read_bytes()) for f in
                          ["arena.log", "arena-records.json", "arena-status.json", "tested-candidate.json", "run.json"]}})
        assert len(accumulated) == 32
        per_candidate[candidate] = accumulated
        result["candidates"][candidate] = {
            "completed": sum(r["completed"] for r in accumulated.values()),
            "accuracy_passed": sum(r["passed"] for r in accumulated.values()), "accuracy_denominator": 32,
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in accumulated.values()),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in accumulated.values()),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in accumulated.values())),
            "request_latency_seconds_min": min(r["seconds"] for r in accumulated.values()),
            "request_latency_seconds_max": max(r["seconds"] for r in accumulated.values()),
        }
    assert all(len(hashes) == 1 for hashes in request_hashes.values())
    first, second = RUNS
    a, b = per_candidate[first], per_candidate[second]
    for index in dataset["indices"]:
        key = f"gsm8k.train.{index:06d}"
        assert a[key]["usage"]["prompt_tokens"] == b[key]["usage"]["prompt_tokens"]
        result["paired"].append({"id": key, "source_row": index, first: a[key], second: b[key],
             "accuracy_discordant": a[key]["passed"] != b[key]["passed"],
             "extracted_answer_equal": a[key]["answer"] == b[key]["answer"],
             "output_bytes_equal": a[key]["output_sha256"] == b[key]["output_sha256"]})
    result["paired_summary"] = {
        "both_correct": sum(a[k]["passed"] and b[k]["passed"] for k in a),
        "both_incorrect": sum(not a[k]["passed"] and not b[k]["passed"] for k in a),
        "control_only_correct": sum(a[k]["passed"] and not b[k]["passed"] for k in a),
        "candidate_only_correct": sum(not a[k]["passed"] and b[k]["passed"] for k in a),
        "same_extracted_answer": sum(r["extracted_answer_equal"] for r in result["paired"]),
        "same_output_bytes": sum(r["output_bytes_equal"] for r in result["paired"]),
        "same_prompt_usage_per_id": True,
    }
    result["limitations"] = [
        "Independent fixed32-example public TRAIN diagnostic, not official/private accuracy or proof of equivalence.",
        "Full outputs remain in hashed original records; this report retains only extracted numeric answers and output hashes.",
        "Request parity follows frozen client/case/binary manifests and identical observed prompt usage, not a separate network packet capture.",
        "Arena duration includes startup; completion lengths differ, so these quality runs do not estimate throughput speedup.",
        "Single execution of each frozen shard/model, no response-conditioned filtering or retry.",
    ]
    (HERE / "actual-c081-q020.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["candidates", "paired_summary"]}, indent=2))


if __name__ == "__main__":
    main()

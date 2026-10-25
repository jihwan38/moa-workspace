#!/usr/bin/env python3
"""Build frozen external GSM train shards; never submits or executes a server."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
ROUND2 = TREE.parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

def accuracy_copy(original):
    start = original.index("def select_test(")
    end = original.index("\ndef concurrency_value", start)
    selection = '''def select_test(suite: str | None, concurrency: int) -> tuple[str, list[dict], list[int]]:
    """Research-only fixed shard: each of16 distinct training rows exactly once."""
    if suite != "gsm8k" or concurrency != 16:
        raise ValueError("External GSM shard requires --test gsm8k --concurrency16")
    rows = read_cases()
    if len(rows) != 16 or any(row["suite"] != "gsm8k" for row in rows):
        raise ValueError("External GSM shard must contain exactly16 distinct GSM rows")
    return "external-gsm8k-train16", rows, list(range(16))

'''
    out = original[:start] + selection + original[end:]
    out = replace_once(out, '"request_index": request_index, "passed": False}',
        '"request_index": request_index, "passed": False, "completed": False}')
    out = replace_once(out, '        check = row["check"]',
        '        record.update(completed=True, output=text, finish_reason=choice.get("finish_reason"), usage=payload.get("usage"))\n        check = row["check"]')
    out = replace_once(out, '[-8000:]', '[-100000:]')
    out = replace_once(out, '    emit(prefix + "_SUMMARY ", summary)\n    return 0 if passed == total else 1',
        '''    completed = sum(bool(record.get("completed")) for record in records)
    scored = sum("score" in record for record in records)
    summary.update(completed=completed, incomplete=total-completed,
                   accuracy_scored=scored, accuracy_passed=passed, accuracy_failed=scored-passed,
                   accuracy_unscored=total-scored,
                   all_accuracy_passed=passed == total,
                   completion_status="complete" if completed == total else "incomplete",
                   accuracy_denominator_complete=scored == total,
                   external_train_diagnostic=True, official_score=False)
    emit(prefix + "_SUMMARY ", summary)
    return 0 if completed == total else 1''')
    return out

def load_selection(verify_source):
    manifest = json.loads((HERE / "dataset-manifest.json").read_text())
    raw = (HERE / "selected-train.jsonl").read_bytes()
    assert sha(raw) == manifest["selected_sha256"]
    selected = [json.loads(line) for line in raw.splitlines()]
    indices = random.Random(20261026).sample(range(manifest["source_rows"]), 32)
    assert indices == manifest["indices"] == [r["source_row"] for r in selected]
    assert manifest["split"] == "train" and len(set(indices)) == 32
    if verify_source:
        source = urllib.request.urlopen(manifest["source_url"], timeout=60).read()
        assert sha(source) == manifest["source_sha256"]
        rows = [json.loads(line) for line in source.splitlines()]
        assert len(rows) == manifest["source_rows"]
        assert selected == [{"source_row": i, **rows[i]} for i in indices]
    return manifest, selected

def build(candidate_name, manifest, selected):
    assert Path(candidate_name).name == candidate_name
    candidate = ROUND2 / "candidates" / candidate_name
    cm = json.loads((candidate / "candidate.json").read_text())
    binary = candidate / "target/release/server"
    assert sha(binary.read_bytes()) == cm["binary_sha256"]
    original = (TREE / "scripts/stage2/accuracy.py").read_text()
    scoring = (TREE / "scripts/stage2/scoring.py").read_bytes()
    upstream_scoring = subprocess.check_output(["git", "show", "8faa2422:./scripts/stage2/scoring.py"], cwd=TREE)
    assert scoring == upstream_scoring
    public = [json.loads(line) for line in (TREE / "public-test/cases.jsonl").read_text().splitlines()]
    template, = [r for r in public if r["suite"] == "gsm8k"]
    assert template["request"]["max_completion_tokens"] == 768 and template["request"]["temperature"] == 0
    prefix = template["request"]["messages"][0]["content"][0]["text"].split("\n\n", 1)[0]
    allrows = []
    for position, source in enumerate(selected):
        row = copy.deepcopy(template)
        row["id"] = f"gsm8k.train.{source['source_row']:06d}"
        row["request"]["messages"][0]["content"][0]["text"] = prefix + "\n\n" + source["question"]
        row["check"]["expected"] = source["answer"]
        row["provenance"] = {"dataset": manifest["repository"], "revision": manifest["revision"],
            "split": "train", "row": source["source_row"], "selection_position": position,
            "licence": "MIT", "seed": 20261026}
        allrows.append(row)
    outputs = []
    for shard in range(2):
        name = f"{candidate_name}-external-gsm-train32-s{shard+1}"
        dest = ROUND2 / "evaluations" / name
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copytree(TREE / "scripts", dest / "scripts")
        scripts = dest / "scripts/stage2"
        (scripts / "accuracy.py").write_text(accuracy_copy(original))
        entry = (scripts / "entrypoint.sh").read_text().replace(
            'all $mode tests passed', '$mode research batch completed; inspect separate accuracy fields')
        (scripts / "entrypoint.sh").write_text(entry)
        arena = (scripts / "arena_test.sh").read_text()
        arena = replace_once(arena, 'SUBMIT_TIMEOUT="${RNGD_SUBMIT_TIMEOUT:-70}"', 'SUBMIT_TIMEOUT=70')
        (scripts / "arena_test.sh").write_text(arena)
        rows = allrows[shard*16:(shard+1)*16]
        (HERE / "shards").mkdir(exist_ok=True)
        shard_path = HERE / "shards" / f"shard{shard+1}.jsonl"
        shard_bytes = b"".join(json_bytes(r) for r in rows)
        if shard_path.exists():
            assert shard_path.read_bytes() == shard_bytes
        else:
            shard_path.write_bytes(shard_bytes)
        (dest / "public-test/LICENSES").mkdir(parents=True)
        (dest / "public-test/cases.jsonl").write_bytes(b"".join(json_bytes(r) for r in rows))
        shutil.copy2(HERE / "LICENSE.gsm8k", dest / "public-test/LICENSES/gsm8k.txt")
        (dest / "target/release").mkdir(parents=True)
        shutil.copy2(binary, dest / "target/release/server")
        shutil.copy2(candidate / "candidate.json", dest / "candidate.json")
        shutil.copy2(HERE / "dataset-manifest.json", dest / "dataset-manifest.json")
        reqhash = sha(b"".join(json_bytes(r["request"]) for r in rows))
        hm = {"base_candidate": candidate_name, "kind": "external_gsm8k_train_research",
            "server_sha256": cm["binary_sha256"], "source_unchanged": True,
            "shard": shard+1, "source_rows": [r["provenance"]["row"] for r in rows],
            "request_set_sha256": reqhash, "total": 16, "concurrency": 16,
            "max_completion_tokens": 768, "temperature": 0, "hard_timeout_seconds": 70,
            "original_accuracy_helper_sha256": sha(original.encode()),
            "accuracy_helper_sha256": sha((scripts / "accuracy.py").read_bytes()),
            "scoring_sha256": sha(scoring), "organizer_prefix_sha256": sha(prefix.encode()),
            "entrypoint_sha256": sha(entry.encode()), "arena_test_sha256": sha(arena.encode()),
            "dataset_revision": manifest["revision"], "dataset_sha256": manifest["source_sha256"],
            "accuracy_failure_does_not_fail_execution": True, "server_log_tail_bytes": 100000,
            "no_response_based_filtering": True, "no_warmup": True, "official_score": False}
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["python3", str(scripts / "accuracy.py"), "--prepare", tmp,
                            "--test", "gsm8k", "--concurrency", "16"], check=True)
            config = json.loads((Path(tmp) / "selection.json").read_text())
            assert config["order"] == list(range(16)) and len(config["cases"]) == 16
            assert config["cases"] == rows
            shutil.copy2(Path(tmp) / "selection.json", dest / "selection-preview.json")
        assert sha((scripts / "scoring.py").read_bytes()) == sha(scoring)
        assert sha((dest / "target/release/server").read_bytes()) == cm["binary_sha256"]
        (dest / "harness.json").write_text(json.dumps(hm, indent=2) + "\n")
        outputs.append({"evaluation": str(dest), **hm})
    return outputs

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--verify-source", action="store_true")
    args = p.parse_args()
    manifest, selected = load_selection(args.verify_source)
    prepared = HERE / "prepared-evaluations.json"
    outputs = json.loads(prepared.read_text()) if prepared.exists() else []
    outputs += [row for candidate in args.candidate for row in build(candidate, manifest, selected)]
    for shard in [1, 2]:
        assert len({r["request_set_sha256"] for r in outputs if r["shard"] == shard}) == 1
    (HERE / "prepared-evaluations.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps([{k:r[k] for k in ["evaluation", "shard", "server_sha256", "request_set_sha256"]} for r in outputs], indent=2))

if __name__ == "__main__":
    main()

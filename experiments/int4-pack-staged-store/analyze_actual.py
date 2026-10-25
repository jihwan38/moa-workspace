#!/usr/bin/env python3
"""Replay the complete-call hidden pack gate; no FFN/model extrapolation."""
import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
ROOT = TREE.parent
ACTUAL = HERE / "actual-322987"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


status = read(ACTUAL / "arena-status.json")
assert status["id"] == 322987 and status["status"] == "SUCCEEDED"
log = (ACTUAL / "arena.log").read_text()
cases = re.findall(r"PACK_P512_CASE width=(\d+) case=(\d+) codes=(\d+) packed_bytes=(\d+) whole_owner_exact=true typed_raw_canonical_exact=true", log)
assert {(int(w), int(c)) for w, c, _, _ in cases} == {(w, c) for w in (3840, 15360) for c in range(5)}
assert len(cases) == 10
assert all(int(n) == 512 * int(w) and int(b) == 256 * int(w) for w, _, n, b in cases)
assert sum(int(x[2]) for x in cases) == 49_152_000
assert sum(int(x[3]) for x in cases) == 24_576_000
assert "PASS_PACK_P512 codes=49152000 packed_bytes=24576000 owner_rejection_checks=6" in log
assert "PASS_PACK_STAGED original_packed_bytes=24576000 staged_input_executed=false complete_FFN_or_model_claim=false" in log
build = read(HERE / "device-build.json")
inputs = read(ACTUAL / "source-hashes.json")
build_run = ROOT / "runs/a005-int4-pack-staged-device-build"
assert inputs == read(build_run / "source-hashes.json")
assert len(inputs) == 403
assert all(sha(TREE / rel) == digest for rel, digest in inputs.items())
elf = Path(build["artifact"]["path"])
assert sha(elf) == build["artifact"]["sha256"]
submitted = read(ACTUAL / "run.json")["submitted_artifact"]
assert sha(Path(submitted["path"])) == submitted["sha256"] == sha(elf)
elf_bytes = elf.read_bytes()
assert len(build["images"]) == 5 and not build["staged_input_embedded"]
for item in build["images"]:
    image = TREE / item["path"]
    assert sha(image) == item["sha256"]
    assert elf_bytes.count(image.read_bytes()) == 1

paired = []
pattern = r"PACK_STAGED_ABBA width=(\d+) case=(\d+) samples=16 candidate_us=(\[[^\]]+\]) original_us=(\[[^\]]+\]) input_same_function_control=(true|false) all_real_DMA_charged=true"
for w, case, cand, old, same in re.findall(pattern, log):
    cand, old = json.loads(cand), json.loads(old)
    assert len(cand) == len(old) == 2 and min(cand + old) > 0
    w, case = int(w), int(case)
    assert (same == "true") == (w == 3840)
    fraction = statistics.mean(cand) / statistics.mean(old)
    paired.append({"width": w, "case": case, "candidate_blocks_us": cand,
                   "original_blocks_us": old, "candidate_mean_us": statistics.mean(cand),
                   "original_mean_us": statistics.mean(old), "reduction_fraction": 1 - fraction,
                   "same_function_control": same == "true",
                   "hidden_20pct_gate": fraction <= 0.8 if w == 15360 else None})
assert {(x["width"], x["case"]) for x in paired} == {(w, c) for w in (3840, 15360) for c in (0, 4)}
assert len(paired) == 4
hidden = [x for x in paired if x["width"] == 15360]
passed = all(x["hidden_20pct_gate"] for x in hidden)
result = {
    "job_id": 322987, "status": status["status"], "duration_sec": status["duration_sec"],
    "strict_correctness_pass": True, "codes": 49_152_000, "packed_bytes": 24_576_000,
    "additional_original_packed_bytes": 24_576_000, "owner_rejection_checks": 6,
    "source_commit": build["source_commit"], "source_input_count": len(inputs), "ELF_sha256": sha(elf),
    "embedded_image_count": len(build["images"]), "staged_input_executed": False,
    "paired_blocks": paired, "hidden_per_case_20pct_gate_pass": passed,
    "hidden_candidate_mean_us": statistics.mean([v for x in hidden for v in x["candidate_blocks_us"]]),
    "hidden_original_mean_us": statistics.mean([v for x in hidden for v in x["original_blocks_us"]]),
    "actual_operation_trace_collected": "NPU_SPAN " in log,
    "raw_input_sha256": {p.name: sha(p) for p in sorted(ACTUAL.iterdir()) if p.is_file()},
    "scope": ["Pack-only ABBA includes real extra DM staging and final HBM store.",
              "The same H function runs in both arms; its difference measures repeat/owner noise, not an optimization.",
              "Raw-copy timings are retained in the raw log and are not GEMM timings.",
              "No quantizer, GeGLU, complete FFN, model accuracy or throughput result.",
              "Original H and failed four-row staged H resource gates remain HOLD."],
    "decision": "Hidden pack standalone correctness and measured-cost PASS; next complete quantize/pack/consume gate required." if passed else "Hidden speed HOLD; preserve strict correctness result separately.",
}
(HERE / "actual-result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("job_id", "strict_correctness_pass", "hidden_per_case_20pct_gate_pass", "hidden_candidate_mean_us", "hidden_original_mean_us")}))

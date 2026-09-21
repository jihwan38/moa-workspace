"""Read host-only V dumps against the unchanged safetensors oracle; stdlib only."""
import argparse
import collections
import json
import math
import re
import struct
from pathlib import Path


def read_expected(path, run):
    raw = Path(path).read_bytes()
    header_size = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + header_size])
    info = header[f"run{run}.sliding_project_qkv.expected.v"]
    assert info["dtype"] == "F32" and info["shape"] == [8, 256]
    start, end = info["data_offsets"]
    data = raw[8 + header_size + start : 8 + header_size + end]
    return list(struct.unpack("<2048f", data))


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def analyze(log, fixture):
    outputs = collections.defaultdict(dict)
    pattern = re.compile(r"DIAG_V_BF16 run=(\d+) offset=(\d+) actual_hex=([0-9a-f]+)")
    for run, offset, raw in pattern.findall(Path(log).read_text()):
        assert len(raw) % 4 == 0
        for local in range(len(raw) // 4):
            index = int(offset) + local
            value = struct.unpack("<f", struct.pack("<I", int(raw[4 * local : 4 * local + 4], 16) << 16))[0]
            if index in outputs[int(run)]:
                assert outputs[int(run)][index] == value, "conflicting duplicate dump"
            outputs[int(run)][index] = value
    result = []
    for run in sorted(outputs):
        assert sorted(outputs[run]) == list(range(2048)), f"run {run}: incomplete dump"
        actual = [outputs[run][index] for index in range(2048)]
        expected = read_expected(fixture, run)
        bad = [i for i, (got, want) in enumerate(zip(actual, expected))
               if not math.isfinite(got) or abs(got - want) > 0.04 + 0.01 * abs(want)]
        heads = []
        for head in range(8):
            a, b = actual[head * 256 : (head + 1) * 256], expected[head * 256 : (head + 1) * 256]
            dot, aa, bb = sum(x * y for x, y in zip(a, b)), sum(x*x for x in a), sum(y*y for y in b)
            heads.append({"head": head, "fail_count": sum(i // 256 == head for i in bad),
                          "actual_rms": rms(a), "expected_rms": rms(b),
                          "cosine": dot / math.sqrt(aa * bb) if aa * bb else None,
                          "least_squares_actual_over_expected": dot / bb if bb else None})
        result.append({"run": run, "fail_count": len(bad), "heads": heads,
                       "fail_indices": bad,
                       "fail_mod_histograms": {str(n): [sum(i % n == j for i in bad) for j in range(n)]
                                               for n in (4, 8, 16, 32, 64)},
                       "worst": sorted([{"index": i, "actual": actual[i], "expected": expected[i],
                                         "abs_error": abs(actual[i] - expected[i])} for i in range(2048)],
                                       key=lambda row: row["abs_error"], reverse=True)[:24]})
    assert result, "no diagnostic output found"
    return {"log": str(log), "fixture": str(fixture), "runs": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("fixture")
    args = parser.parse_args()
    print(json.dumps(analyze(args.log, args.fixture), indent=2, allow_nan=False))

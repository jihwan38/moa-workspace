#!/usr/bin/env python3
"""Compare every actual Rust-generated input byte stream with the CPU source."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", type=Path, required=True)
ap.add_argument("--dump", type=Path, required=True)
ap.add_argument("--log", type=Path, required=True)
args = ap.parse_args()
cpu = json.loads(args.cpu.read_text())
log = args.log.read_text()
assert "host_only=true" in log
assert "QKV_LAUNCH_BEGIN" not in log
assert "QKV_SUITE_RESULT completed=2 planned=2 total_failed_values=0 cache_failed_values=0 status=PASS" in log
checks = re.findall(r"^INPUT_CHECK case=(\d+) tensor=(\S+) bytes=(\d+) checksum=([0-9a-f]{8}) status=PASS$", log, re.M)
assert len(checks) == 38 and len({(c,k) for c,k,_,_ in checks}) == 38
indexed = {(int(c), k): (int(n), int(cs, 16)) for c,k,n,cs in checks}
records = []
for case in cpu["records"]:
    for key, want in case["sha256"].items():
        p = args.dump / f'case{case["case"]}.{key}.bin'
        data = p.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        assert got == want, (p, got, want)
        assert indexed[(case["case"],key)] == (len(data),case["checksums"][key])
        records.append({"case":case["case"],"input":key,"bytes":len(data),"sha256":got})
assert len(records) == 38
print(json.dumps({"host_only":True,"npu_launches":0,"cases":2,
                  "checksums_passed":38,"full_input_sha256_passed":38,
                  "dump_directory":str(args.dump),"records":records},indent=2))

#!/usr/bin/env python3
"""Parse frozen q021 LM-only diagnostics; no runtime or accuracy inference."""
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

HERE = Path(__file__).resolve().parent


def fields(line):
    out = {}
    for key, value in re.findall(r"(\w+)=([^ ]+)", line):
        if value in ("true", "false"):
            out[key] = value == "true"
        else:
            try:
                out[key] = float(value) if any(c in value for c in ".eE") else int(value)
            except ValueError:
                out[key] = value
    return out


def drift_stats(rows):
    return {
        "rows": len(rows),
        "values": sum(r["values"] for r in rows),
        "bit_differences": sum(r["bitdiff"] for r in rows),
        "relative_l2_min": min(r["relative_l2"] for r in rows),
        "relative_l2_max": max(r["relative_l2"] for r in rows),
        "max_absolute_difference": max(r["max_abs"] for r in rows),
        "same_top1_rows": sum(r["same_top1"] for r in rows),
        "top1_changes": [r for r in rows if not r["same_top1"]],
    }


def collect(run, job):
    path = HERE / f"r{run}-arena.log"
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    passes = [fields(l) for l in lines if l.startswith("PASS Q021_LM_")]
    assert len(passes) == 1
    final, = [fields(l) for l in lines if l.startswith("Q021_LM_FINISHED ")]
    result = {"run": run, "job": job, "log_sha256": hashlib.sha256(raw).hexdigest(),
              "pass": passes[0], "finished": final,
              "memory": [fields(l) for l in lines if l.startswith("Q021_LM_MEMORY ")],
              "startup": [fields(l) for l in lines if l.startswith("Q021_LM_STARTUP ")]}
    if final["mode"] == "model":
        drift = [fields(l) for l in lines if l.startswith("Q021_LM_DRIFT ")]
        assert len(drift) == 84
        result["eligible_b16_lm_drift_vs_q019_bf16_lm"] = drift_stats([r for r in drift if r["step"] >= 2])
        result["ineligible_scalar_and_default_off"] = drift_stats([r for r in drift if r["step"] <= 1])
        assert result["ineligible_scalar_and_default_off"]["bit_differences"] == 0
        steps = [fields(l) for l in lines if l.startswith("Q021_LM_STEP ")]
        stages = [fields(l) for l in lines if l.startswith("Q021_LM_STAGE ")]
        assert len(steps) == 30
        result["steps"] = steps
        # Step8 deliberately disables the q019 epilogue in BOTH arms. Exclude it
        # from the normal-q019 comparison; retain its timings separately.
        keep = [2, 3, 4, 5, 6, 7, 9]
        timings = {}
        for stage in ["whole_model", "logits.projection"]:
            means = []
            for arm in range(3):
                selected = [r for r in (steps if stage == "whole_model" else stages)
                            if r["pass"] == arm and r["step"] in keep
                            and (stage == "whole_model" or r["stage"] == stage)]
                assert len(selected) == len(keep)
                means.append(mean(r["model_ms" if stage == "whole_model" else "total_ms"] for r in selected))
            timings[stage] = {"q019_bf16_lm_ms": means[0], "candidate_first_ms": means[1],
                              "candidate_replay_ms": means[2],
                              "delta_replay_ms": means[0] - means[2],
                              "time_reduction_replay_fraction": 1 - means[2] / means[0]}
        result["diagnostic_timing"] = {"included_steps": keep, "timings": timings,
            "order": "complete control pass, candidate pass, candidate physical replay; not ABBA",
            "excludes": "history fill, active prompt quanta, output downloads, CPU selection and KV scans",
            "cold_scope": "first candidate projection occurs at step2; replay is warm"}
        cache = [fields(l) for l in lines if l.startswith("KV_ALL_BYTES_EXACT ")]
        assert len(cache) == 32
        result["cache_compared_bytes"] = sum(r["compared_bytes"] for r in cache)
        assert result["cache_compared_bytes"] == 2 * result["pass"]["replay_exact_cache_bf16_values"]
    return result


results = [collect(1303, 321550), collect(1304, 321551), collect(1305, 321554)]
report = {
    "source": "553c44c608ce8f2d7ae453313c5c30067798194c",
    "elf_sha256": "15ff8c81dc6db22a07069ec3790ad3b3f1d7cf79a5be13375d0cd18c08afa2c4",
    "all_three_pass": True,
    "exact_logit_values_reported": sum(r["pass"].get("replay_exact_logits", r["pass"].get("scalar_and_R32_exact_logits", 0)) for r in results),
    "exact_cache_bf16_values_reported": sum(r["pass"].get("replay_exact_cache_bf16_values", 0) for r in results),
    "task_accuracy_or_http_claim": False,
    "reference": "q019 MSE F4 FFN plus BF16 LM; both arms use identical FFN eligibility and teacher-forced inputs",
    "results": results,
}
(HERE / "actual.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
for r in results:
    print(json.dumps({k: r[k] for k in ["run", "pass", "eligible_b16_lm_drift_vs_q019_bf16_lm", "diagnostic_timing"] if k in r}, indent=2))

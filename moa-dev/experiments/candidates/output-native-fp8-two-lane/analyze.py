"""Development arithmetic probe; does not model RNGD's reduction tree."""
import importlib.util
import json
from pathlib import Path
import numpy as np

source = Path(__file__).parent / "../qkv-fp8-split/analyze.py"
spec = importlib.util.spec_from_file_location("fp8_probe", source)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
rng = np.random.default_rng(20260921)
weights = q.fp8(rng.normal(size=(64, 4096)).astype(np.float32))
cases = {"normal": rng.normal(size=4096), "zeros": np.zeros(4096)}
outliers = rng.normal(size=4096)
outliers[0] = 2.0**20
outliers[1024] = -(2.0**18)
cases["column_local_outliers"] = outliers
cases["column_dynamic_ranges"] = np.concatenate(
    [rng.normal(size=512) * 2.0**n for n in (-30, -20, -10, 0, 10, 20, 30, 40)]
)
results = []
for name, values in cases.items():
    x = q.bf16(values)
    reference_partials, candidate_partials, scales, bounds, errors = [], [], [], [], []
    for column in range(8):
        a, b = 512 * column, 512 * (column + 1)
        s, z, hi, lo, rebuilt = q.split(x[a:b])
        w = weights[:, a:b]
        reference_partials.append(w @ x[a:b])
        candidate_partials.append(((w @ hi) + (w @ lo) / np.float32(16)) * s)
        scales.append(float(s))
        bounds.append(float(s) * 2.0**-14)
        errors.append(float(np.max(np.abs(x[a:b].astype(np.float64) - rebuilt))))
        assert errors[-1] <= bounds[-1]
    baseline = np.sum(reference_partials, axis=0, dtype=np.float32)
    candidate = np.sum(candidate_partials, axis=0, dtype=np.float32)
    results.append({
        "case": name, "per_column_scales": scales,
        "per_column_input_max_abs_error": errors,
        "per_column_input_abs_error_bound": bounds,
        "f32_projection_max_abs_error": float(np.max(np.abs(baseline - candidate))),
        "bf16_projection_mismatches": int(np.count_nonzero(q.bf16(baseline) != q.bf16(candidate))),
        "outputs": 64,
    })
print(json.dumps({
    "seed": 20260921, "columns": 8, "column_width": 512,
    "limitation": "NumPy F32 sums do not reproduce RNGD native FP8 or inter-slice reduction order; projection probes are not the official post-norm/residual oracle.",
    "cases": results,
}, indent=2, allow_nan=False))

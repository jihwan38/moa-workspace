"""Independent CPU numeric/layout checks; no model fixtures, no RNGD claim."""
from pathlib import Path
import importlib.util
import json
import numpy as np

helper = Path(__file__).parents[1] / "qkv-fp8-split/analyze.py"
spec = importlib.util.spec_from_file_location("independent_fp8_math", helper)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)

# Every native packet contains four separate 16-element blocks. Packet reduction
# never crosses a block; the second contraction reduces all 240 block scalars.
positions = []
for packet in range(60):
    for block in range(4):
        for offset in range(16):
            positions.append((packet * 4 + block, packet * 64 + block * 16 + offset))
assert [h for _, h in positions] == list(range(3840))
assert all(b == h // 16 for b, h in positions)

def projection(x, q, scales):
    x = codec.bf16(x)
    s, _, hi, lo, rebuilt = codec.split(x)
    rows = q.shape[0]
    blocks = q.reshape(rows, 240, 16)
    # Float64 math isolates BF16-block rounding from hardware/F32 reduction.
    exact_blocks = np.sum(blocks.astype(np.float64) * x.reshape(240, 16), axis=2)
    hi_blocks = np.sum(blocks.astype(np.float64) * hi.reshape(240, 16), axis=2)
    lo_blocks = np.sum(blocks.astype(np.float64) * lo.reshape(240, 16), axis=2)
    split_blocks = (hi_blocks + lo_blocks / 16) * float(s)
    partials = codec.bf16(split_blocks)
    reference = np.sum(exact_blocks * scales, axis=1)
    candidate = np.sum(partials.astype(np.float64) * scales, axis=1)
    representation = np.abs(split_blocks - exact_blocks)
    rounding = np.abs(partials.astype(np.float64) - split_blocks)
    bound = np.sum(np.abs(scales) * (representation + rounding), axis=1)
    err = np.abs(candidate - reference)
    assert np.all(err <= bound + 1e-10)
    return {
        "max_input_reconstruction_error": float(np.max(np.abs(x - rebuilt))),
        "max_projection_absolute_error": float(np.max(err)),
        "triangle_bound_max": float(np.max(bound)),
        "bf16_output_mismatches": int(np.count_nonzero(codec.bf16(reference) != codec.bf16(candidate))),
        "rows": rows,
        "first_reference": float(reference[0]),
        "first_candidate": float(candidate[0]),
    }

rng = np.random.default_rng(20260921)
fp4_values = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
q = rng.choice(fp4_values, size=(32, 3840))
scales = codec.fp8(np.exp2(rng.uniform(-4, 4, size=(32, 240))))
samples = []
for exponent in [-10, 0, 10]:
    samples.append({"input_power": exponent, **projection(rng.normal(size=3840) * 2.0**exponent, q, scales)})

# Exact BF16 activations, FP4 weights 1, FP8 local scales 1. Native hi/lo
# reconstructs every input exactly, yet added BF16-block rounding doubles y.
x = np.zeros(3840, np.float32)
x[:16] = [1] + [2.0**-8] * 15
x[16:32] = [-1] + [-2.0**-8] * 14 + [0]
cancellation = projection(x, np.ones((1, 3840), np.float32), np.ones((1, 240), np.float32))
assert cancellation["max_input_reconstruction_error"] == 0
assert cancellation["first_reference"] == 2.0**-8
assert cancellation["first_candidate"] == 2.0**-7

print(json.dumps({
    "scope": "CPU exact block math plus modeled BF16 RNE/FTZ, not RNGD cast/reduction or oracle proof",
    "all_3840_input_positions_preserved": True,
    "all_240_blocks_preserved": True,
    "sample_cases": samples,
    "additional_bf16_block_rounding_counterexample": cancellation,
    "relative_error_guarantee": "None under cancellation",
}, indent=2))

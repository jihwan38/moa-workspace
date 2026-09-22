#!/usr/bin/env python3
"""Independent CPU diagnostics; not a model of NPU reduction/FTZ behavior."""
import json
from pathlib import Path
import sys
import torch


def quantize(x, mode):
    x = x.float()
    e5 = mode == "e5-three"
    dtype = torch.float8_e5m2 if e5 else torch.float8_e4m3fn
    exponent = 14 if e5 else 7
    scale = torch.exp2(torch.floor(torch.log2(x.abs().max().clamp_min(2.0 ** (-126 + exponent)))) - exponent)
    z = x / scale
    hi = z.to(dtype).float()
    if mode == "e4-pair16":
        lo = ((z - hi) * 16).to(dtype).float() / 16
        return (hi + lo) * scale
    lo = (z - hi).to(dtype).float()
    if e5:
        third = (z - (hi + lo)).to(dtype).float()
        return (hi + lo + third) * scale
    return (hi + lo) * scale


if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1])
    import generate_references as ref
    torch.set_num_threads(2)
    result = {"scope": __doc__, "torch": torch.__version__, "official_distribution_activations": [], "cancellation": []}
    modes = ["e4-pair16", "e4-time2", "e5-three"]
    for run in range(9):
        synth = ref.Synth("decoder_feedforward", run)
        x = ref.rms_norm(ref.H, synth.bf16("pre_ff_rms_weight", (ref.H,), ref.UNIT))(
            synth.bf16("residual", (ref.H,), ref.UNIT)).float()
        for mode in modes:
            q = quantize(x, mode)
            result["official_distribution_activations"].append({"run": run, "mode": mode,
                "different_elements": int((q != x).sum()), "max_abs_error": float((q - x).abs().max())})
    for small in (2.0**-22, 2.0**-23, 2.0**-32):
        x = torch.full((3840,), small)
        x[0], x[1] = 0.53125, -0.53125
        # Exact binary sums, FP4 unit weights, unit per-block scales.
        exact = float(x.double().sum())
        for mode in modes:
            q = quantize(x, mode)
            partial = q.reshape(-1, 16).sum(-1)
            hi = partial.bfloat16().float()
            lo = (partial - hi).bfloat16().float()
            result["cancellation"].append({"small": small, "mode": mode, "reference_f64": exact,
                "native_f32_blocks": float(partial.sum()), "bf16_blocks": float(hi.sum()),
                "two_bf16_blocks": float((hi + lo).sum())})
    p = torch.tensor([1 + 15/256, -1 - 14/256] * 480, dtype=torch.float32)
    hi = p.bfloat16().float()
    lo = (p-hi).bfloat16().float()
    result["block_rounding_counterexample"] = {"reference": float(p.sum()), "bf16_blocks": float(hi.sum()),
        "two_bf16_blocks": float((hi+lo).sum()), "scope": "Projection intermediate; not a measured full FFN final-output failure."}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

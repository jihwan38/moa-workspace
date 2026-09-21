#!/usr/bin/env python3
"""Diagnostic inputs; oracle imports unchanged frozen official reference functions.

Weights are synthesized by the same coordinate rule in probe_output.rs; only the
small BF16 inputs and expected outputs are uploaded. No quantizer is an oracle.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

H, Q = 3840, 4096
NAMES = [
    "signed_zeros", "column_row_boundaries", "amax_below_powers_two",
    "amax_at_powers_two", "amax_above_powers_two", "small_component_thresholds",
    "low_amax_floor", "correlated_cancellation_in_distribution_support",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bf_bits(values):
    return torch.as_tensor(values, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy().copy()


def bf_from_bits(bits):
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def weight_codes(case):
    r = np.arange(H, dtype=np.uint32)[:, None]
    q = np.arange(Q, dtype=np.uint32)[None, :]
    if case == 7:
        return np.broadcast_to(np.where(q % 256 < 2, 0x38, 0x77).astype(np.uint8), (H, Q)).copy()
    # Every value lies in the official FP8 signed exponent[7,14], mantissa[0,7] support.
    exponent = 7 + ((r // 120 + q // 256 + case) % 8)
    mantissa = (3 * r + q + q // 16) % 8
    sign = ((r // 8 + q // 32 + q % 3 + case) % 2) << 7
    return (sign | (exponent << 3) | mantissa).astype(np.uint8)


def inputs(case):
    q = np.arange(Q)
    row = np.arange(H)
    scale = bf_bits(np.ldexp(1.0, -(11 + row % 4)))
    gamma = bf_bits((1 + row % 7) / 8.0)
    residual = bf_bits((row % 5) / 8.0)
    x = np.zeros(Q, dtype=np.uint16)
    if case == 0:
        x[q % 2 == 1] = 0x8000
    elif case == 1:
        # Alternate entirely-zero and nonzero 256-column groups, then explicit
        # anchors on both sides of 256/512/2048 and head/cluster boundary 4095.
        values = np.where((q // 256) % 2 == 0, 0.0, ((q % 17) - 8) / 16.0)
        for col, value in [(0, .125), (255, -.25), (256, .5), (511, -.75),
                           (512, .25), (2047, -.5), (2048, .75), (4095, -.125)]:
            values[col] = value
        x = bf_bits(values)
        scale[[119, 120, 1919, 1920, 3839]] = bf_bits([2**-10, 2**-11, 2**-12, 2**-13, 2**-14])
    elif case in (2, 3, 4):
        for group in range(16):
            exponent = -7 + group % 8  # powers 2^-7 through 1, BF16 neighbors.
            anchor = int(bf_bits([2.0**exponent])[0]) + case - 3
            a = float(bf_from_bits(np.array([anchor], dtype=np.uint16)).float()[0])
            vals = np.array([a * ((i % 15) - 7) / 16 for i in range(256)], dtype=np.float32)
            vals[0] = a
            vals[255] = -a
            x[group*256:(group+1)*256] = bf_bits(vals)
    elif case == 5:
        for group in range(16):
            # amax=2^e -> s=2^(e-7). Include BF16 neighbors of both thresholds.
            exponent = -6 + group % 7
            values = np.zeros(256, dtype=np.uint16)
            values[0] = bf_bits([2.0**exponent])[0]
            for offset, power in [(1, exponent-13), (7, exponent-21)]:
                middle = int(bf_bits([2.0**power])[0])
                for j, delta in enumerate((-1, 0, 1)):
                    values[offset+j] = middle + delta
                    values[offset+j+3] = (middle + delta) | 0x8000
            # Repeat the small tails to expose their contribution through RMSNorm.
            values[16:] = values[1:13].repeat(20)
            x[group*256:(group+1)*256] = values
    elif case == 6:
        # Distinct signed values spanning all BF16 subnormals and normal values
        # strictly below 2^-119 (0x0400). amax floor branch; no large anchor.
        x = ((q * 37) % 0x400).astype(np.uint16)
        x[q % 2 == 1] |= 0x8000
        x[0] = 0x0001
        x[1] = 0x0080
        x[2] = 0x03ff
        gamma[:] = 0x3f80
        residual[:] = 0
    elif case == 7:
        x = bf_bits(np.full(Q, 2.0**-22))
        x[0::256], x[1::256] = 0x3f80, 0xbf80
        scale[:] = 0x3a80  # 2^-10, within official row scale support.
        gamma[:] = 0x3f00
        residual[:] = 0x3f00
    else:
        raise ValueError(case)
    return {"x": x, "o_weight_scale": scale, "post_attn_rms_weight": gamma, "residual": residual}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-scripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.official_scripts.resolve()))
    import generate_references as official
    from fixture_prng import checksum
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    tensors, records = {}, []
    for case, name in enumerate(NAMES):
        data = inputs(case)
        codes = weight_codes(case)
        weights = torch.from_numpy(codes).view(torch.float8_e4m3fn).float()
        x = bf_from_bits(data["x"])
        scale = bf_from_bits(data["o_weight_scale"])
        gamma = bf_from_bits(data["post_attn_rms_weight"])
        residual = bf_from_bits(data["residual"])
        raw = x.float() @ weights.T
        scaled_f32 = raw * scale.float()
        projected = official.scaled_projection(x, weights, scale)
        mean_square = projected.float().pow(2).mean()
        inverse_rms = torch.rsqrt(mean_square + official.EPS)
        normalized_f32 = projected.float() * inverse_rms * gamma.float()
        normalized = official.rms_norm(H, gamma)(projected)
        expected = (residual + normalized).to(torch.bfloat16)
        assert torch.equal(projected, scaled_f32.to(torch.bfloat16))
        assert torch.equal(normalized, normalized_f32.to(torch.bfloat16))
        intermediates = {"x": x.float(), "weights": weights, "raw_dot": raw,
                         "scaled_f32": scaled_f32, "projected_bf16": projected.float(),
                         "mean_square": mean_square, "inverse_rms": inverse_rms,
                         "normalized_f32": normalized_f32, "normalized_bf16": normalized.float(),
                         "expected": expected.float()}
        assert all(bool(torch.isfinite(t).all()) for t in intermediates.values())
        arrays = {k: np.ascontiguousarray(v).view(np.uint8) for k, v in data.items()}
        arrays["o_weight"] = codes.reshape(-1)
        checks, hashes = {}, {}
        for key, array in arrays.items():
            checks[key] = int(checksum(array))
            hashes[key] = sha(array.tobytes())
            tensors[f"case{case}.checksum.{key}"] = np.array([checks[key]], dtype=np.int64)
            if key != "o_weight":
                tensors[f"case{case}.input.{key}"] = array
        tensors[f"case{case}.expected"] = expected.float().numpy().copy()
        ranges = {key: {"min": float(t.min()), "max": float(t.max())}
                  for key, t in intermediates.items() if key != "weights"}
        rec = {"case": case, "name": name, "finite_intermediates": True,
               "sha256": hashes, "checksums": checks, "ranges": ranges,
               "expected_bf16_distinct": len(torch.unique(expected)),
               "expected_first_bits": f"0x{int(expected.view(torch.uint16)[0]):04x}"}
        records.append(rec)
        print(json.dumps(rec), flush=True)
        if case == 7:
            assert bool((raw == 0.2325439453125).all())
            assert bool((expected == 0.609375).all())
    metadata = {"purpose": "O-only boundary diagnostic; not official performance/score",
                "cases": len(NAMES), "atol": .05, "rtol": .01,
                "oracle": "Frozen official scaled_projection + RMSNorm + BF16 residual",
                "torch": torch.__version__, "cpu_threads": 4,
                "weight_rule": "probe_output.rs::weight_codes mirrors generate_suite.py::weight_codes",
                "official_sources": {str(p.relative_to(args.official_scripts)): sha(p.read_bytes()) for p in [
                    args.official_scripts / "generate_references.py", args.official_scripts / "reference/gemma4.py",
                    args.official_scripts / "fixture_prng.py"]}, "records": records}
    fixture = args.out / "boundary.safetensors"
    save_file(tensors, str(fixture), metadata={"purpose": metadata["purpose"]})
    metadata["payload"] = {"file": fixture.name, "bytes": fixture.stat().st_size,
                           "sha256": sha(fixture.read_bytes())}
    (args.out / "cpu-references.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU arithmetic screen, not an NPU emulator or a replacement benchmark oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

torch.set_num_threads(4)
torch.set_grad_enabled(False)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def round_away(x):
    # furiosa-opt-std 0.8.1 float::float_to_fixedpoint uses f64::round,
    # whose tie rule is away from zero, not round-to-nearest-even.
    return x.sign() * torch.floor(x.abs().double() + .5).float()


def quantize(x):
    blocks = x.float().reshape(-1, 16)
    maximum = blocks.abs().amax(-1).clamp_min(2.**-120)
    exponent = (maximum.view(torch.int32) & 0x7f800000).view(torch.float32)
    scale = exponent / 64
    normalized = blocks / scale[:, None]
    hi = round_away(normalized).clamp(-127, 127)
    lo = round_away((normalized - hi) * 128)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert (lo.abs() <= 64).all()
    combined = (hi * 128 + lo).to(torch.int32)
    reconstructed = combined.float() * (scale[:, None] / 128)
    return combined, scale, reconstructed.reshape(x.shape)


def elementary():
    integer = round_away(FP4 * 2).to(torch.int8)
    assert torch.equal(integer.float() / 2, FP4)
    assert 16 * 127 * 12 * 128 + 16 * 64 * 12 == 3133440 < 2**24 < 2**31
    # All finite BF16 encodings, grouped with nearby encodings. Separately mix
    # outliers and cancellation below; this sweep is not all possible blocks.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).float()
    values = values[torch.isfinite(values)]
    assert len(values) % 16 == 0
    _, _, reconstructed = quantize(values)
    sweep = {'finite_bf16_values': len(values), 'exact_reconstructions': int((values == reconstructed).sum()),
             'scope': 'Adjacent BF16 bit patterns within each block; not all 16-element combinations.'}
    boundary_results = []
    for name, x in [
        ('zero', torch.zeros(64)),
        ('signed_half_ties', torch.tensor([1., -1., 15/256, -14/256] * 16)),
        ('outlier_and_residual', torch.tensor([1., -1.] + [2.**-22] * 62)),
        ('bf16_maximum', torch.tensor([torch.finfo(torch.bfloat16).max, -torch.finfo(torch.bfloat16).max] + [0.] * 62)),
        ('smallest_bf16', torch.tensor([2.**-133, -2.**-133] * 32)),
    ]:
        x = x.to(torch.bfloat16).float()
        _, _, restored = quantize(x)
        assert torch.isfinite(restored).all()
        boundary_results.append({'name': name, 'elements': x.numel(), 'max_abs_error': float((x-restored).abs().max()), 'exact': bool(torch.equal(x, restored))})
    return {'fp4_codes_exact': 16, 'signed_integer_weight_range': [-12, 12], 'i32_combination_bound': 3133440,
            'finite_bf16_sweep': sweep, 'boundary_cases': boundary_results,
            'universal_bf16_equivalence_claimed': False}


def fixture_screen(reference_dir, fixture):
    sys.path.insert(0, str(reference_dir))
    import generate_references as ref
    original = load_file(str(fixture))
    records = []
    for run in range(3):
        synth = ref.Synth('decoder_feedforward', run)
        residual = synth.bf16('residual', (ref.H,), ref.UNIT)
        pre = synth.bf16('pre_ff_rms_weight', (ref.H,), ref.UNIT)
        post = synth.bf16('post_ff_rms_weight', (ref.H,), ref.UNIT)
        x = ref.rms_norm(ref.H, pre)(residual)
        dots = {}
        projection_quantization = {}
        for name, rows, width in (('up', ref.L, ref.H), ('gate', ref.L, ref.H), ('down', ref.H, ref.L)):
            if name == 'down':
                # Match the candidate's erf expression and BF16 boundaries.
                up, gate = dots['up'].float(), dots['gate'].float()
                x = (up * gate * (torch.erf(gate * 0.70710678118) + 1.) / 2).to(torch.bfloat16)
            digits, scale, restored = quantize(x)
            projection_quantization[name] = float((restored-x.float()).abs().max())
            weight = synth.f4(name + '_weight_packed', (rows, width))
            local_scale = synth.f8(name + '_weight_scale', (rows, width//16), ref.LOCAL_SCALE_EXP, signed=False)
            # Integer block dot computed in F32: every product and 16-element
            # integer sum is exactly representable under the proven <2^24 bound.
            block_dot = ((weight * 2).reshape(rows, width//16, 16) * digits.float()[None, :, :]).sum(-1)
            assert float(block_dot.abs().max()) <= 3133440
            block_dot *= scale[None, :] / 256
            block_dot *= local_scale
            projected = block_dot.sum(-1).to(torch.bfloat16)
            raw = float(ref.prng.f32_uniform(f'run{run}.global.{name}_global_scale', 1, *ref.GLOBAL_SCALE)[0])
            dots[name] = (projected.float() / raw).to(torch.bfloat16)
            del weight, local_scale, block_dot
        normalized = ref.rms_norm(ref.H, post)(dots['down'])
        output = ((residual + normalized).to(torch.bfloat16) * ref.LAYER_SCALAR).to(torch.bfloat16).float()
        expected = original[f'run{run}.decoder_feedforward.expected']
        error = (output - expected).abs()
        allowed = .01 + .01 * expected.abs()
        assert torch.isfinite(output).all()
        record = {'run': run, 'checked': output.numel(), 'failures': int((error > allowed).sum()),
                  'max_abs_error': float(error.max()), 'max_error_over_tolerance': float((error/allowed).max()),
                  'projection_activation_max_abs_error': projection_quantization}
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = {'scope': 'CPU mathematics of proposed C13 arithmetic against unchanged fixture outputs. Does not emulate compiler mapping, NPU conversion rounding or F32 reduction tree.',
              'elementary': elementary(), 'torch_version': torch.__version__}
    result['fixture_sha256'] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    result['reference_sources'] = {str(path.relative_to(args.reference_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [args.reference_dir/'generate_references.py', args.reference_dir/'fixture_prng.py', args.reference_dir/'reference/gemma4.py']}
    result['runs'] = fixture_screen(args.reference_dir, args.fixture)
    result['result'] = 'PASS' if all(r['failures'] == 0 for r in result['runs']) else 'FAIL'
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result['result'], flush=True)

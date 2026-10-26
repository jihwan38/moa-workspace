#!/usr/bin/env python3
"""Read-only coordinate, byte-identity and CPU arithmetic proof; no device claim."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parents[2]
OLD_SHA = 'd6eb545ab2a6ba951a23f1527fde51a1965c8adfb95f737883add2da4761ed79'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def f(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def rn(x):
    return f(bits(x))


def mask(k):
    z = ((k ^ 0x243f6a8885a308d3) + 0x9e3779b97f4a7c15) & ((1 << 64) - 1)
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
    return ((z ^ (z >> 31)) >> 63) << 31


def function(s, name):
    start = s.index('fn ' + name)
    brace = s.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[start:end]


def staged_offset(j, u):
    # Independent direct physical linearization of [P4,H/64,H%8,H/8%8].
    return j * 960 + (u // 64) * 64 + (u % 8) * 8 + (u // 8) % 8


def low_packets(stride):
    # Explicit Time/Packet iteration matching the declared low-stage axes.
    for j in range(4):
        for block in range(15):
            for upper in range(8 // (2 * stride)):
                for lower in range(stride):
                    low3 = 2 * stride * upper + lower
                    yield [(j, block * 64 + 8 * lane + low3) for lane in range(8)]


def reference(a):
    a = a[:]
    for stride in (1, 2, 4, 8, 16, 32):
        old = a[:]
        for block in range(0, len(a), 64):
            for pair in range(0, 64, 2 * stride):
                for k in range(stride):
                    lo = block + pair + k
                    hi = lo + stride
                    a[lo] = rn(old[lo] + old[hi])
                    a[hi] = rn(old[lo] - old[hi])
    return [rn(x * .125) for x in a]


def staged_transform(a):
    assert len(a) % 64 == 0
    # Real physical matrix transposition, followed by new owners at each stage.
    x = [None] * len(a)
    for block in range(0, len(a), 64):
        for row in range(8):
            for col in range(8):
                x[block + col * 8 + row] = a[block + row * 8 + col]
    for stride in (1, 2, 4):
        y = [None] * len(a)
        for block in range(0, len(a), 64):
            for upper in range(8 // (2 * stride)):
                for lower in range(stride):
                    first = 2 * stride * upper + lower
                    for lane in range(8):
                        lo = block + first * 8 + lane
                        hi = lo + stride * 8
                        y[lo] = rn(x[lo] + x[hi])
                        y[hi] = rn(x[lo] - x[hi])
        assert None not in y
        x = y
    a = [None] * len(a)
    for block in range(0, len(a), 64):
        for row in range(8):
            for col in range(8):
                a[block + row * 8 + col] = x[block + col * 8 + row]
    for stride in (8, 16, 32):
        old = a[:]
        for block in range(0, len(a), 64):
            for upper in range(64 // (2 * stride)):
                for lower in range(stride):
                    lo = block + 2 * stride * upper + lower
                    hi = lo + stride
                    a[lo] = rn(old[lo] + old[hi])
                    a[hi] = rn(old[lo] - old[hi])
    return [rn(x * .125) for x in a]


def main():
    old_path = ROOT / 'src/ops_int4_hadamard64.rs'
    new_path = ROOT / 'src/ops_int4_hadamard64_staged.rs'
    old, new = old_path.read_text(), new_path.read_text()
    assert sha(old_path.read_bytes()) == OLD_SHA
    preserved = {}
    for name in ('sign_masks', 'half', 'stage_8', 'stage_16', 'stage_32'):
        body = function(old, name)
        assert body == function(new, name), name
        preserved[name] = sha(body.encode())
    rotate = function(new, 'rotate_input')
    rotate = rotate.replace('    let x = to_staged(device, x);\n', '')
    rotate = rotate.replace('    let x = from_staged(device, x);\n', '')
    assert rotate == function(old, 'rotate_input')
    assert 'unsafe' not in new and '.reshape' not in new

    # Enumerate physical ownership and prove complete coverage without assuming
    # that equal extents imply equal physical layouts.
    seen = bytearray(512 * 3840)
    for c in range(2):
        for g in range(4):
            for r in range(64):
                local_seen = bytearray(4 * 960)
                for j in range(4):
                    p = 256 * c + 4 * r + j
                    for u in range(960):
                        h = 960 * g + u
                        canonical = p * 3840 + h
                        staged = staged_offset(j, u)
                        assert seen[canonical] == local_seen[staged] == 0
                        seen[canonical] = local_seen[staged] = 1
                        # Independent inverse of the staged flattened offset.
                        jj, v = divmod(staged, 960)
                        b, v = divmod(v, 64)
                        low3, high3 = divmod(v, 8)
                        assert (jj, 64 * b + 8 * high3 + low3) == (j, u)
                assert all(local_seen)
    assert all(seen)

    stages = []
    for stride in (1, 2, 4):
        written = bytearray(4 * 960)
        packets = 0
        for packet in low_packets(stride):
            for side in (0, 1):
                offsets = [staged_offset(j, u + side * stride) for j, u in packet]
                assert offsets == list(range(offsets[0], offsets[0] + 8))
                for (j, lo), offset in zip(packet, offsets):
                    assert lo // stride % 2 == 0
                    assert lo // 64 == (lo + stride) // 64
                    assert written[offset] == 0
                    written[offset] = 1
            packets += 1
        assert all(written) and packets == 240
        stages.append({'stride': stride, 'half_packets_per_slice': packets,
                       'pair_values_per_slice': 1920, 'contiguous_packet': True})

    vectors = ROOT / 'experiments/int4-hadamard64-diagnostic/independent-proof/vectors.json'
    v = json.loads(vectors.read_text())
    for row in v['vectors']:
        a = [f(b ^ mask(row['base_channel'] + i)) for i, b in enumerate(row['input_f32_bits'])]
        got = staged_transform(a)
        assert [bits(x) for x in got] == row['rotated_f32_bits']
        assert [bits(x) for x in got] == [bits(x) for x in reference(a)]
        bf16 = [((bits(x) + 0x7fff + ((bits(x) >> 16) & 1)) >> 16) & 0xffff for x in got]
        assert bf16 == row['rotated_bf16_bits']

    # Four complete local rows per960 for every H group, varied normal BF16 values.
    full_shard_values = 0
    for g in range(4):
        for j in range(4):
            a = []
            for u in range(960):
                word = (u * 2654435761 + j * 2246822519 + g * 3266489917) & 0xffffffff
                bf = ((word >> 16) & 0x8000) | ((123 + word % 9) << 7) | ((word >> 9) & 127)
                a.append(f((bf << 16) ^ mask(960 * g + u)))
            assert [bits(x) for x in staged_transform(a)] == [bits(x) for x in reference(a)]
            full_shard_values += len(a)
    dependencies = {name: sha((ROOT / name).read_bytes()) for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml')}
    plan = json.loads((ROOT / 'experiments/int4-hadamard64-device/plan.json').read_text())
    assert dependencies == plan['dependencies']
    return {
        'candidate_source_sha256': sha(new_path.read_bytes()),
        'preserved_original_module_sha256': OLD_SHA,
        'byte_identical_functions': preserved,
        'entry_body_change': 'exactly forward and inverse real transpose calls',
        'mapped_input_values': len(seen),
        'low_stages': stages,
        'frozen_vectors': len(v['vectors']),
        'frozen_vector_sha256': sha(vectors.read_bytes()),
        'additional_full_shard_values': full_shard_values,
        'dependencies': dependencies,
        'scope': 'CPU source/coordinate/RN32 proof only; no lowering, allocation, device correctness or speed claim'
    }


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))

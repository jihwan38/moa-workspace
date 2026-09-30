# Finite BF16 scale-producer identity

`original-scale-fusion-audit.py/json` preserve the D-E145 CPU analysis byte-for-byte. `scale-proof.py` changes only its output handling, checks the original result, and supports `--check`; `provenance.json` records both copies and the adapter. No device build or NPU execution is involved.

For every finite stored BF16 value, the original `floorPow2(max(abs(x), 2^-119)) * 2^-7` equals `max(reinterpret_f32(bits_f32(x) & 0x7f800000) * 2^-7, 2^-126)` bit-for-bit under the modeled F32 arithmetic. All 65,280 finite encodings pass, including signed zero, subnormals, the lower clamp boundary, both signs and maximum finite values. The result also survives flushing an intermediate subnormal product because the final floor is the smallest normal F32.

The pointwise map is monotone, so it commutes with nonempty H960 and K4 Max reductions. This proves the scalar value identity for finite BF16 groups. It does not prove the new device pipeline's lowering, Switch ownership, register state, all native arithmetic or performance; the actual graph and unchanged numerical suites remain required.

The one-pass producer removes two Main task boundaries and intermediate local/global-max DM round trips. It newly evaluates the scale multiplication on 512×960=491,520 input values instead of 128 valid group scalars. Direct Sub broadcast also has its own fetch, bitmap, Switch, collect and VRF costs. These counts are logical work, not a hardware latency estimate.

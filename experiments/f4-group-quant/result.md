# CPU helper gate: PASS

`host::f4_group960::quantize_blocked_e4` accepts current blocked E4 bytes and
returns packed F4 bytes, per960 F32 scales, and unscaled fixed-code E4 bytes for
the device reference. It does not accept or alter the retained row dequantizer.

The compiled Rust helper passed an independent Python `Fraction` oracle:

- All 254 finite E4 patterns at all 19 reachable scale exponents: 4,826 exact
  code decisions. Both nonfinite E4 patterns are rejected at all 19 exponents.
- Scale selection and E4 decoding checked for every finite E4 pattern; all
  16 F4 codes checked, including signed zeros.
- Two 480×1920 fixtures cover two row blocks and two K960 blocks:
  1,843,200 values, 1,920 group scales, and 921,600 packed bytes. The second is
  a true same-data row reversal plus within-group K reversal; restored codes
  and scales agree. Input files are unchanged.
- Both NaN signs and four invalid geometries are rejected. Rust unit tests
  separately exercise all nearest-code midpoint ties and sign/zero behavior.

The oracle computes scale using rational while-loop comparisons and nearest
code by rational distance; it does not reuse the helper's integer-unit formula.
Results and packed fixture hashes are in `cpu-proof.json`.

Reproduce inside the SDK container from `/round2/f4-group-quant`:

```sh
rustc --edition=2024 --test src/host/f4_group960.rs -C opt-level=2 -C debuginfo=0 -o experiments/f4-group-quant/cpu_tests
experiments/f4-group-quant/cpu_tests
rustc --edition=2024 experiments/f4-group-quant/cpu_driver.rs -C opt-level=2 -C debuginfo=0 -o experiments/f4-group-quant/cpu_driver
python3 experiments/f4-group-quant/cpu_proof.py --driver experiments/f4-group-quant/cpu_driver --output experiments/f4-group-quant/cpu-proof.json
```

This is a CPU quantizer/packing proof. Supplied-weight error, device LUT and
partial-scale arithmetic, complete FFN speed, and task quality remain separate
gates. The supplied current-E4 baseline is already a quantized reconstruction;
error against it must not be labeled total error against original NVFP4 weights.

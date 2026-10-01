# E153 boundary packages built, not run

First whole job141557 passed45 comparisons but FFN298332 cycles exceeded both controls294978/294929: paired +1.145435%. The positive condition for expanded boundary NPU was not met. No E153 boundary Arena job was submitted.

- Original raw up/gate8-case wrapper/host/generator/reference files are byte-identical to E142/E141. Entire E153 MLP prefix, exact diagnostic/host image identity and original CPU checksum records pass. The original wrapper calls the actual new sharded quantizer through its compatibility adapter. Independent emitted graph review passes. **No245760-output E153 hardware result exists**, and this raw boundary does not invoke pre-RMS.
- Original full-FFN regression input/reference and13input crosscheck are preserved. The full host embeds the exact complete E153 production image including changed ops/RMS/MLP. Build verification passes. **No3840-output E153 hardware result exists.**
- Source and whole emitted graph preserve original RMS arithmetic through normalized F32 DM; local BF16 cast coverage was reviewed. These facts do not constitute an NPU observation of intermediate RMS values. No general BF16 or model-wide guarantee is claimed.

See `boundary-build-summary.json`, `boundary-results.json` and `adoption/raw-graph-review.json`. All boundary evidence is frozen. m03 target is released, slot2 was never used by this candidate, and references/tolerances were not changed.

```sh
python3 moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/probe.py verify
python3 moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/full-ffn-stress/run.py verify e153
```

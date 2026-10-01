# E152 boundary packages built, not run

The first whole screen (job141424) passed45 comparisons but FFN regressed10.05543% against its bracket. Root stopped expanded validation. No E152 boundary Arena job was submitted.

- Original E142/E141 raw up/gate wrapper, host, generator, reference and quantizer-audit source remain byte-identical. The full frozen E152 MLP prefix is compiled into a separate diagnostic image. Exact/host image identity and all original8-case CPU checksum records pass. **No245760-output hardware result exists for E152.**
- The original constructed full-FFN regression reference and13input checks are preserved. The new host embeds the exact E152 whole production FFN image. Build/verification pass. **No3840-output hardware result exists for E152.**
- E152 leaves E150 down arithmetic unchanged in source; earlier raw-down results are historical evidence and are not counted as a new E152 run.

`boundary-build-summary.json` records actual image hashes; `boundary-results.json` records the built-not-run scope. m03 target is released. Builders/source/evidence are frozen, with no new tolerance, reference or input changes. The initial preparation lookup for unmerged E151 reference files was corrected to committed E150's identical original files before any full-host build; no kernel change resulted.

Build-only rechecks:

```sh
python3 moa-dev/experiments/candidates/ffn-gate-weight-stages/probe.py verify
python3 moa-dev/experiments/candidates/ffn-gate-weight-stages/full-ffn-stress/run.py verify e152
```

# E149 boundary suite: built and verified, not run on NPU

The original E137 raw nine-case diagnostic and E47 whole-O case7 packages were built from frozen E149 `e149-ingress-v1`. Both build verifiers pass. All46 original production source files are bound; the raw wrapper only appends its module to lib.rs and adds the original diagnostic sources. The raw host checks27 input checksums and retains the exact original E137 fixture. The whole host checks five input checksums and contains exactly the production O image. CPU reference checks pass.

First whole job141183 passes45 comparisons but O regresses34.616647% (57414 versus E147 bracket42650). Root ended expanded validation. **Boundary NPU jobs:0; boundary NPU output comparisons:0.** The planned34560 raw exact values and3840 whole tolerance comparisons were not executed and must not be counted as passing. No reference, tolerance or input was changed.

`oracle-build-v1.json`, `stress-build-v1.json` and their evidence directories preserve the completed builds. `status.json` binds those records, actual first-screen result and the distinct raw/whole image digests. The m03 target is released.

Recheck the completed build scope with:

```sh
python3 moa-dev/experiments/candidates/o-k1024-ingress-switch/boundary-suite/oracle.py verify
python3 moa-dev/experiments/candidates/o-k1024-ingress-switch/boundary-suite/full_stress.py verify
```

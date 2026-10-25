# Complete B16 fitted-scale micro ready, no actual result yet

Functional source `844c738b` atop original measured `ae7b102f`. The candidate,
intended dense-code graph and current E4 epilogue device modules remain
byte-identical, with hashes in `unchanged-source.json`. Static quantization is
the independently proved MSE helper8e7ce99e; each packed nibble expands to the
same fixed E2M1 E4 byte while fitted scales remain separate.

Host check passed11.18s and release build passed46.25s. All recorded source
hashes match the frozen build. Artifact:

```
experiments/b16-f4-mse/artifacts/test_b16_f4_mse-844c738b
SHA256 f05fb10ffca07bad3b7f5a35fcda9d8e631ed17876c91dae196be821554f73e2
4,947,416 bytes
```

Arguments `0 2` cover provided layers0/5 and `2 4` cover6/47, as in the previous
ceil-max gate. Tests retain all eight states, true permutations, two matched
feedback steps, sampled F64 partial/final oracles, immutable data and complete
ABBA. Root owns actual dispatch. Successful CPU proof and device compilation do
not prove the new non-power-of-two scale arithmetic or task quality.

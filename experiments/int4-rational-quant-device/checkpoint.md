The isolated B16×H3840 prototype is stopped at resource proof. Functional source
`5f3a1a0b` compiled successfully in a newly empty output directory; final source
only corrects the comment “per PE” to “per logical slice.” No executable device
probe, Arena run, model route, or official candidate was built or changed.

The complete resource estimate is **19,765 cycles**, versus archived power2
quantization's **9,312** (a different quantization contract). This narrowly meets
the preregistered ≤20,000 screen; it is not a measured speed result. Inputs,
amax, metadata, real broadcasts, sign restoration, code stores and scale stores
are included. Packing and the I4 consumer are excluded.

The critical comparison/reduction is 462 scheduled cycles. The five SRAM
redistributions and fragmented scalar store dominate the graph; removing
floating division does not remove that movement. VRF has 7,744 live bytes per
logical slice and an 8,192-byte allocation envelope. The earlier exponent VRF
is retired before the four core VRFs coexist. Only four explicit DramIO owners
occur; no DramStack is allocated.

This mapping uses all **512 logical slices** for 16 rows×four H960 shards×eight
thresholds. It cannot be extrapolated to P512, which already has 512 real rows.
The independent quotient/remainder CPU oracle passed three tests; final probe
typecheck passed. The executable probe source covers valid normalized mantissa
pairs, exponent/floor/FTZ classes, exact positive/negative half-ties, permutation,
separately poisoned outputs and immutable inputs. None of those device assertions
has run. Model approximation quality and complete quantize/pack/FFN cost remain
missing gates.

The initial resource attempt failed solely on a 4-byte-stride constant HBM
transfer. The retained repair gives each immutable odd threshold an initialized
32-byte record, without changing code-selection arithmetic. Failed typechecks,
the failed resource run, successful checks and the complete schedule are retained
in the paths and hashes listed by `resource-result.json`.

# E154: replicated scale output repairs E153 quantizer supply

Source origin is E151/8adbbae, explicitly composed with frozen E153 sharded-v1 and a
replicated-Max/scale output change. E151 remains the separate performance control.
Actual start b336c45c20bcf95815d8f66cfea4d9b8b8d99922, branch opt/ffn-sharded-scale-replicated.
This is a repair of an existing ownership combination, not a new architecture family.

E153 first141557 passes45 but regresses1.145435%; its frozen current-image readiness
diagnostic runs separately. Prepare one supported candidate in parallel: keep eight
global scalar copies through original power-of-two scale arithmetic and load each
quantization VRF locally, removing canonical-to-eight scalar TDMA. Every reshape
keeps all256physical/8valid positions; final all512 supply uses an explicit anonymous
replica view and typedDMA. Preserve originalSubsum, bounded480B term copies, late
packed23040B, originalRMS arithmetic and E151 native/down. Include added repeated
scalar writes/compute and verify actual source/destination/mask/VRF coverage.

M06 already tried this scalar topology in a different source context, with SFR868
failures also present after reverting to canonicalMax. No claim that its old timeout
or all target issues are fixed. New exact/whole tests are required. NPU selection and
adoption wait for actual evidence; no byte-to-cycle or rank promise.

## Adopted frozen result

Source/tag `3f384063f5231e00ad286f365209806e5043fc83` / baseline-e154. Three independent E151 brackets improve
FFN by paired median2.424948%, each faster than both controls. Standard9, raw8
245760 BF16exact(signed-zero equivalent), fullFFN3840exact pass with original references.
All46 inputs/three actual images/uploads are bound. QKV/O remain E151. The raw oracle
excludes pre-RMS; original emitted RMS prefix/local-cast mapping and completeFFN
validation retain that scope. Stop parameter tuning; official submission is separate.
[Final gate](adoption/independent-final-review.json) · [Adoption](adoption-decision.json).

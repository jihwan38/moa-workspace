Fixed H64 source/CPU/host checkpoint

Plan386c8d2e, helper/main62878a85, gate repairafec8a05. The original427 source/config inputs and locked dependencies remain byte-identical; only two new host source files are added. Three unchanged runtime entries acquire the prefix/current-production hidden. Five unchanged production startup entries are expected in a later device artifact, for eight total; no device artifact exists yet.

Pure-std CPU a003 passed8 tests plus all16 frozen root vectors:1024 signs,1024 F32 and1024 BF16 bit patterns. a004 plain cargo check passed0.34s. Earlier a001/a002 results are retained. The vector dense_exact_f64 field is never used as an exact oracle. Root Fraction proofs are retained verbatim.

Independent source review found a zero-reference relative-error edge in62878a85: infinity could compare as <=0.75*infinity. afec8a05 fails closed before JSON/gate evaluation for zero/nonfinite reference or ratio and adds a regression. This was a source-review finding, not an actual-data failure. The rotation or quantizer arithmetic did not change.

The future fixed diagnostic has18 expected records:1 setup,12 per-projection DOT,4 STEP,1 final. Each of9 arms uses6144 total owner-associated pairs (not6144 unique weight rows),47,185,920 terms. There are1536 sampled weight rows reused across4 order/step combinations. All dots are pre-global F64, not the F32 device reduction or BF16 whole-FFN output. Two steps share only current q020 feedback; DOWN input is its production GeGLU. Both row orders must match pair bits.

Numerical gates remain rotation-only L2<=0.01 and rotated A4W4 L2<=0.75*original for every projection/step/order. Any failure is HOLD without tuning. All F32/BF16 range assumptions fail closed. No device rotation, Arena run, model-quality result, runtime measurement or promotion is claimed. Official q020/final unchanged.

Root separately authorized the single fresh device build after final source/CPU/host reviews. a005 completed in45.54s at source24d9752e, captured ELF SHA c591b27a69a73e3b6d9a58e596d6ea2fc6612200fb5e8290cbc889f4965cf5fd (4,859,280bytes). audit_build.py verifies429 inputs,427 originals unchanged and all8 images equal prior actual323213 and the new empty-cache output; q020 control exact. This is build/provenance proof only; sampled numerical gate remains unexecuted here.

Root executed actual323407 (7.689726249s). Strict18-record replay passes and all12 fixed sampled numerical gates pass. This is CPU pre-global sampled-dot evidence only; no rotated device kernel or model/speed promotion. Raw permutation counter has mixed units, corrected explicitly in actual-result.md/json. Batch independently audited records,429inputs,8images and count/scope.

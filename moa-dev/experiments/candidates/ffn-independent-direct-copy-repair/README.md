# E141 independent direct six-row FFN with bounded input copies

**Correct complete candidate, +3.151638% FFN regression; not adopted.** Whole job111571 passes45comparisons and the original production up/gate oracle job111574 passes245760 exact BF16 outputs. Current runtime remains E140. Actual branch start `9427666ba1864569cec98faa1efce373f6da502b`. Compile source base and separate performance control are adopted E140 `c56b737c25e3d9c19472209fa97355b3a2852a08` / `baseline/e140.json`. The MLP explicitly originates from frozen E135 (`be9bb5a3...`) with exactly three up-input term copies changed to bounded480B Sub streams. All other E135 MLP bytes, six-row geometry, native arithmetic and weighted-down path remain unchanged. Other45 compile files come from E140, including its current O image.

`direct-six-row-copy-v1/recipe.json`, `source-contract.json`, `source.diff` and `execution-plan.md` record the full producer/consumer path, costs, exact source origins, and three independent adoption/tuning/research decisions before compilation. The E135 audit establishes an actual descriptor overlap with a live ring32 Switch bitmap in both old whole and projection-oracle images; it does not prove the runtime timeout instruction or promise restored liveness. Original E135 failures remain unchanged.

The complete design removes112.5MiB of up/gate FP8 materialization write and read each, retaining independent packed HBM/DM and blockscale inputs, ten six-row native/lookup streams, scale/reduction order, GeGLU and E140 weighted native down. Eight additional lookup preparations relative to E140 and all input preparation/transfer/lifetime costs remain part of whole measurements. QKV/O images must equal actual E140. No additive speedup or ranking claim follows from bytes removed.

Whole execution uses the original3inputs and E140-before/candidate/E140-after hosts. A separate production up/gate oracle reuses the exact eight E135 input/truth/wrapper files and245760 expected BF16 outputs, preserving finite scope. Root owns final adoption; no runtime or official submission changes occur here.

The whole result is E140314403 → candidate324186 → E140314159 cycles, bracket314281. All three standard FFN comparisons and unchanged QKV/O passed; actual uploaded archive and stage hashes were verified. Only one performance allocation ran, with the original3inputs. QKV/O timing variation is not an optimization because their images are identical to E140.

The original whole and production oracle timed out on their first launch. With only the three repaired copies, both now complete correctly. The repaired whole retains packed base17664B/slice and bitmap19200..19232B/slice; three writes end19104, safely before the bitmap. This strongly supports the identified copy-corruption explanation in these images. It does not establish the old runtime timeout instruction, arbitrary-input correctness or a general SDK fault.

The projection oracle reuses the original eight dyadic input/truth/wrapper files byte-for-byte, including historical `E135_*` log markers. It checks8×2×15360 outputs before GeGLU against original BF16 truth, with signed zero equivalence and all nonfinite outputs rejected. No full-model or arbitrary-BF16 equivalence is claimed. No extra6 or separate full-FFN stress was run because the complete candidate regressed; root's prepared `full-ffn-stress/` is not passing evidence.

A tool-only interruption is preserved in `package-e141-direct-six-row-copy-v1.json` and `evidence/package-attempt-v1.log`: host compilation and all3image checks passed, but a provisional independent checker counted separate down copies together with up copies. The packer correctly rejected that failed gate. Correcting the checker scope did not change source, image or host. `package_ready.py` rechecks the identical artifact and creates immutable `package-v2-e141-direct-six-row-copy-v1.json` without rebuilding.

`decision.json` separates the decisions: reject this measured prototype for adoption; stop parameter tuning; perform the root-selected bounded same-image E140/E141 readiness profile in a separate series after integration. Removing112.5MiB FP8 write/read each did not produce a whole gain, and the effect of eight extra lookup preparations remains unmeasured. The profile may select an existing direct30/F32blockpartial combination if its actual critical dependencies justify it. E103's old numerical failure has no preserved actual descriptor evidence, so this result does not diagnose its cause. The direct family is not globally exhausted.

Reproduce current evidence without NPU:

```
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/independent-audit.py e141-direct-six-row-copy-v1 --check
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/package_ready.py e141-direct-six-row-copy-v1 --check
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/screen.py check e141-direct-copy-screen
python3 moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/probe.py --check
```

`normalized/` contains stage-compatible candidate/before/after reports for root integration. Original sources, compilation and host failures, graph/schedule checks, raw logs and exact uploaded archives are preserved. Source/image/runtime and official submission remain unchanged by this candidate.

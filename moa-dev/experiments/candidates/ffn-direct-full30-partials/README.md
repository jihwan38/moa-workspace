# E142 full30 direct native with F32 block partials

Frozen complete FFN validated; ready for root adoption review. No live runtime or official submission was changed by these candidate records. Actual start `21d6618e8880ea25f6fb65ac2f2f2e0a51625677`. Source base/other45files and independent performance control are adopted E140 `c56b737c25e3d9c19472209fa97355b3a2852a08` / `baseline/e140.json`. MLP combines frozen E141 with only `project_rows_with_scale_contraction` replaced by the actual frozen E103 helper. `source-contract.json`, `source.diff` and `full30-v1/recipe.json` bind both origins; outside-helper E141 bytes are unchanged.

D-E141 profile111627 validated57comparisons and actual137/151marker mappings. First direct native began15434cycles earlier, but finalnative completed10888later and GeGLU10810later. This selects a concrete repeated-preparation trade, not a claim that overlapping54k or66k windows can be removed. E103 had true projection mismatches and a later timeout; its old unsafe copies exist but missing finaldescriptors leave their victim/cause unresolved. Its correctness is not inherited.

One full30 directFP4lookup/native per projection computes F32 block16 dots and applies inputscale. F32 partials `[Row30,H/16=240]` are committed and reread by five six-row scale/reduce streams. The result remains `(dot*xscale)*blockscale` then F32 H reduction, with no intermediate BF16 cast. Native/lookup streams10→2, all Main streams10→12. Added partial write and read are28.125MiB each; each producer owns28800B/slice. No up/gate fullFP8materialization or jointweight assembly is introduced. Existing480Binputcopies, TRF/scalereload, M03localF32gather/GeGLU and weightednativeDown/fulltail remain unchanged.

The frozen independent audit passed actual partial/source/destination bounds, scale pairing, prefix bitmap and register lifetimes before packaging. Exact compile passed with static150905cycles; the measured FFN image is289280B, SHA`c8185c19b42382a90658e426f7c4d07f03d46b5de582c04ac454fbb0458e82ed`. QKV/O device images match E140 exactly. Static cost is not an NPU measurement.

Three independent Arena allocations of identical candidate/control stage bytes all passed45comparisons each:

| Arena job | E140 before | E142 | E140 after | FFN change vs bracket |
| --- | ---: | ---: | ---: | ---: |
|111706|313548|302889|313520|−3.395166%|
|111719|315456|301615|314319|−4.214997%|
|111721|314498|302796|313835|−3.619259%|

Paired median improvement is3.619259%; aggregate candidate median302796cycles. These are repeated measurements of the original three inputs, not nine distinct inputs. Uploaded archives, raw logs and actual E140 source/package identity were rechecked. See [performance summary](adoption/performance-summary.md) and the immutable [first screen reports](normalized-first/).

Extra6 job111737 passed30comparisons on original run3..8, making nine distinct standard inputs overall. The extended host contains byte-identical QKV/O/FFN images; its timings are excluded from the performance comparison. Production projection oracle111704 passed245760 selected first-BF16 up/gate outputs across eight unchanged historical cases exactly (signed zero equivalent); it calls the actual quantizer/projection helper with original BF16 CPU truth. Whole-FFN stress111736 passed3840outputs against the unchanged public reference/tolerance on the original E134 tiny-value regression fixture, using the actual standard E142 image. These finite cases do not prove arbitrary BF16 or end-to-end model correctness. See [oracle verification](validation/verification-summary.json), [extra6 result](adoption/extra-results.json) and [full-FFN stress](full-ffn-stress/result-summary.md).

Three decisions remain separate in [decision.json](decision.json): the frozen positive is ready for root adoption; further row/tile tuning stops; subsequent major research must use the changed complete path, with retained F32 partial traffic and existing down preparation still present. The gain being below5% does not block adoption. No new architecture or global direct-family success is claimed. Historical E103 arithmetic failures and E135 timeouts remain preserved, with their distinct evidence scope.

Reproduction: `independent-audit.py e142-full30-v1 --check`, `package.py e142-full30-v1 --check`, `validation_performance.py --check`, `validation_extra.py verify`, `probe.py --check`, and `validation_verify.py --check`. The final combined verifier binds the separate whole, extra6, oracle, stress and source/image evidence; root owns live adoption and the baseline checkpoint.

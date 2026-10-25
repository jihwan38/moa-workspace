# Initial MSE B16 model diagnostic

Initial artifact `e17b5c4e339b8a86e9fb90b509e0818b74d752b6686e2911b7a88a09c7761d8a`, source `2509e19f`, passed r1260/job321361 (40.143s) and r1261/job321364 (34.698s). This is a default-OFF structural/replay diagnostic, not public task accuracy. The initial artifact was captured before identity edits; explicit physical owning-buffer identity is being rebuilt with `--fresh-package`. The subsequent fresh-package identity jobs below now confirm this storage identity; the original artifact and its provenance remain separate.

The same-candidate replay compared 18,612,224 logits exactly; unchanged scalar/Automatic and P128 checks add 786,432 and 524,288 respectively, explaining the printed 19,922,944 total. Replay cache comparison covers 2,952,790,016 BF16 values (5,905,580,032 bytes on one side). Independent scalar/R32 guards compared 9,175,040 logits exactly. Own-output greedy validation and seeded stochastic replay passed. These checks establish internal determinism, dispatch scope and cache behavior; they do not establish agreement with the original model.

Matched teacher-forced rows use the same logical owner/step schedule as the previous PowerOfTwo diagnostic. All 71 logged original top1 IDs agree across jobs; no serialized cross-job all-logit baseline proof was performed.

| Eligible F4 rows (60) | PowerOfTwo | MSE dyadic |
| --- | ---: | ---: |
| Original top1 differences | 37 | 29 |
| Mean relative L2 | 0.393830 | 0.229906 |
| Maximum relative L2 | 1.119454 | 0.821447 |
| Maximum absolute difference | 28.1875 | 21.4296875 |

MSE improves relative L2 on 53/60 paired rows; 13 regain the original top1 and 5 lose it. This remains substantial drift. The old-path fallback after divergent history has 5/8 top1 differences (PowerOfTwo: 1/8), so this result does not imply monotonic improvement at every later step. No quality gate or activation conclusion follows.

Seven eligible completed model calls average 37.069839ms original versus 34.158954ms candidate and 34.194183ms replay (about 7.8% less). Their 48 FFN epilogue spans average 13.699923ms versus 10.900410/10.874292ms. These timers exclude logit download, selection and cache scans; they are not HTTP throughput measurements.

Atomic shadow preparation took 23.458s in the model job and 25.194s in the guards job. Separately measured read/encode/upload totals were 6.364/15.690/0.584s and 8.987/14.717/0.623s; remaining setup/accounting time was 0.820/0.866s. Both jobs allocated exactly 4,282,122,240 extra HBM bytes (4,246,732,800 packed weights plus 35,389,440 group scales), leaving 9,997,525,504 free after the shadow. The existing E4 model remains resident. Startup prints all 144 projections with `MseDyadic`; no dense reference is constructed during startup.

`analyze_mse.py` reproduces `actual-mse-r1260-r1261.json`, including full per-projection/per-stage records, paired counts and log hashes. PowerOfTwo source/artifact and its durable report remain unchanged in `b16-f4-model`.

## Fresh-package identity confirmation

Source `2d0bb5a8`, automatically captured ELF `6ab8428702954445c9fb2773e7b04d964d9abbe208048effc0911f121d40d3e4`, passed r1275/job321403 (44.681s) and r1276/job321404 (37.127s). Both logs explicitly report physical/admission32, query64, full_pages256, context2048, 40 sliding and 8 full layers, and 12,884,901,888 bytes verified from every immutable owning K/V Buffer. This confirms actual storage, beyond source constants or requested admission limits.

The same strict replay/cache/guard counts pass again. All 71 printed drift records are identical to the initial MSE job, including original/candidate top1 and error statistics. The fresh completed-call means are 37.206753ms original versus 34.290466/34.350823ms candidate/replay; FFN spans are 13.681084 versus 10.894000/10.842572ms. Startup takes 25.895/26.063s and allocates the same 4,282,122,240 additional bytes. `analyze_mse.py --identity` reproduces `actual-mse-identity-r1275-r1276.json` and asserts both exact identity lines plus identical initial/fresh drift records.

A separate `b16-f4-mse-quality` research branch may now explicitly prepare MseDyadic and enable the existing B16 scope for public quality evaluation. This OFF diagnostic tree remains unchanged; the six public task gates are still required before any performance/adoption claim.

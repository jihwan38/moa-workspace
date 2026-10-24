BF16 QKV32 temporal reuse, source50039c0, SDK0.9.0.

Root Arena318031 (`runs/r264.../arena.log`) and318032 (`runs/r265.../arena.log`)
passed every R64/P128 output bit against the same-size H960 baseline for allfour
Qs/Ps/Qf/Pf shapes, on dyadic and random inputs. Independent full dyadic CPU
oracles and sampled F64 dot oracles passed. This validates temporal reuse and
P128's different lane/time permutation; it does not establish model accuracy.

Matched warm milliseconds, five launches plus one identical output read /5:

| Batch | Projection | Existing B8 | TRF32 |
| --- | --- | ---: | ---: |
| R64 | Qs | .2007 | .1881 |
| R64 | Ps | .1692 | .1623 |
| R64 | Qf | .2440 | .2226 |
| R64 | Pf | .1591 | .1605 |
| P128 | Qs | .3167 | .2938 |
| P128 | Ps | .2776 | .2643 |
| P128 | Qf | .4206 | .3832 |
| P128 | Pf | .2876 | .2812 |

R64 Pf has no measured win; root retains that baseline. Qs/Ps/Qf adoption is
root-coordinated. No further numerical approximation or weight copy was added.

Separately, root combined the proven native FP8 TRF temporal reuse with its R64
local-quantization module: Arena318028 checked245760 output bits exactly and
measured .6255ms→.5435ms. That combination belongs to root's source tree; the
standalone baseline source in this worktree is unchanged.

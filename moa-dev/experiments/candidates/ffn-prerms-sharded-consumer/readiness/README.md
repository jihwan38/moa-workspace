# E153 / E151 actual FFN readiness

Job 141612 completed all 57 original comparisons in seven serial stages. `verify.py e153-readiness-v1 141612 --check` rechecks the downloaded archive, original three-image hosts, both 46-file source manifests, exact name-only FFN edits, source/marker mapping and all 142 operations per trace. Runtime remains E151; E153 is not adopted.

| Boundary (task-relative endpoint median) | E151 | E153 |
|---|---:|---:|
| Pre-RMS BF16 ready | 11,822.5 | 10,386 |
| Quantization scale ready | 56,832 | 16,233 |
| Quantization scale VRF ready | 59,751.5 | 53,072 |
| First E5 term begins | 59,752.5 | 53,073 |
| Third term copy ready | 67,129.5 | 64,502 |
| Ring seed ready | 72,658.5 | 80,273 |
| Activation TRF ready | 99,981.5 | 107,264 |
| Gate native begins | 99,982.5 | 107,265 |
| Up native begins | 143,887.5 | 146,889 |
| GeGLU ready | 186,317.5 | 188,626 |
| Last down native ready | 260,458.5 | 265,147 |
| Output store ready | 295,152 | 297,293 |

E153 quant-scale DMA168 spans task-relative16,234–52,432; StoVrf172 spans52,434–53,072. The paired same-clock scale-ready→VRF-ready interval has median37,235 cycles, versus2,256 in E151. These spans include waits and overlap with gate-weight ingress. Removing this scalar TDMA dependency is a concrete repair question;37,235 is not an obtainable speedup. E153 already starts its first term6,679.5 cycles earlier by endpoint medians, yet its late packed copy232, seed262 and ring279 lead to a later gate native. Late packed destination remains23,040 bytes; its whole-DMA span does not measure exclusive transfer service.

Both observed down second natives start two cycles after their respective weight transfer finishes in all six run/cluster observations. No weighted sum of these spans is used. Stage schedules and moved dependencies can change under a new source.

Original whole hosts in this allocation measured E151296,324 / E153299,012 / E151294,082, or+1.290299% versus the bracketing median. Diagnostic info/trace medians were292,888/295,931 for E151 and296,382/298,162 for E153. These host/process scopes are recorded separately. This is one diagnostic allocation, not another independent adoption repetition.

The table shows medians across three inputs and two clusters; differences between endpoint medians are not medians of paired differences. `profile-analysis.json` separately stores within-run/cluster interval statistics and all raw observations. Inactive slice/cluster work is not inferred to be exclusive computation from a marker duration.

Decisions: reject E153 performance adoption; stop parameter tuning and extra repetitions; use this fixed evidence to evaluate the supported E154 replicated-Max/direct-scale-VRF repair with a new whole-kernel test. No ranking or aggregate savings claim follows from this diagnostic. `screen.py` retains a historical docstring typo (`E157`); actual package/control/source bindings are E151 throughout.

Reproduce:

```sh
python3 moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/readiness/analyze.py e153-readiness-v1 141612 --check
python3 moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/readiness/verify.py e153-readiness-v1 141612 --check
```

# E148 fixed-image readiness diagnostic

This single allocation compares the original E147 and frozen E148 O images with original three-input whole hosts and separate `info`/`trace` diagnostic processes. It also observes the unchanged current FFN (exact E142 source/image) using a separately names-patched copy of the same control diagnostic host. No production source, instruction, constant, address, I/O or marker position changes.

The original E148 screen passed45 comparisons but regressed3.2882057%. This diagnostic asks whether scalar preparation, direct residual VRF writes, coefficient/TRF readiness or either weight stage explains the complete-path change. A measured interval is not an exclusive stall or a promised saving. Current FFN is observed once; E139 is not a paired current control.

`plan.json` records scope and three separate decisions before execution. `package.py` binds all46 source inputs, all3 original device images, original fixture/runner and complete ordered O/FFN marker correspondence. The original control O image is identical to E137native-v3, so its preserved schedule is reused with its original schedule-source snapshot; the actual control host/source manifest remains E147. The FFN image/source are identical to E142full30-v1.

Commands from this directory:

```sh
python3 package.py profile-v1 --target-dir /tmp/moa-c09-validation-probe/target
python3 screen.py prepare e148-readiness-v1 --build profile-v1
python3 screen.py submit e148-readiness-v1
python3 screen.py collect e148-readiness-v1
python3 analyze.py e148-readiness-v1 JOB
python3 verify.py e148-readiness-v1 JOB
```

The9 stages expect63 original comparisons: three original whole stages, O control/candidate info/trace, and current FFN info/trace. Extra processes share one allocation and are not independent performance repetitions. `landmarks.json` maps semantic roles to actual schedule IDs; graph IDs are a separate namespace.

## Measured result (job141134)

All63 comparisons passed, all9 stages completed, and downloaded upload bytes matched the prepared package. `analyze.py --check`, `verify.py --check` and `independent-review.py --check` pass. Whole O was42271 /44266 /42470 cycles (E147 /E148 /E147): **+4.4736314%** versus the42370.5-cycle bracket. The earlier independent standard screen was+3.2882057%. E148direct-v1 is not adopted; this bounded diagnostic is complete and further parameter tuning is stopped by root.

The following are median Task-relative endpoints over the3 runs×2 clusters. Subtracting two endpoint medians is not the median of paired per-run differences. Actual same-run intervals are separately stored in `profile-analysis.json`.

| O landmark | E147 begin→end | E148 begin→end |
|---|---:|---:|
| input HBM |99→2002.5|98.5→1729|
| quantization done |22572.5→23268.5|10030→10723|
| activation TRF |23269.5→23860.5|10730→12004|
| first weight |2071.5→15290.5|4660→17864|
| first native |23861.5→26305.5|17866→20087|
| second weight |16291.5→30182.5|17875→31158|
| second native |30184.5→32623.5|31160→33601|
| final store |41891.5→42751.5|42521→43382|

The candidate advances quantization and the first native, but both of its natives start exactly2 cycles after their weight transfer ends in every run/cluster. The control first native starts1 cycle after TRF readiness; its second native also starts2 cycles after the second weight ends. This identifies the weight readiness boundary in the frozen candidate, without attributing an exclusive bus-stall fraction or predicting the saving of another mapping. Its added bitmap marker is1816→3763; the trace does not establish that removing that marker alone would recover the whole regression.

Current FFN uses a shared activation TRF load274 for gate native283 and up native374. Its trace median is302225 cycles (diagnostic info301385; original whole304766). Pre-RMS finishes at11849; the canonical input gather ends53179 while gate weight HBM runs4333→51776. Input terms finish64573; broadcast72498→95910 and TRF96265→99480 precede gate native99481→120491. Up native is141429→162431. GeGLU ends217157; down TRF ends254921 and down native starts254922. The paired GeGLU-end→down-TRF-end median is36829 cycles; down weight was already ready (end213604). This conditional readiness window is neither a sum of operation times nor an estimate of achievable whole-FFN speedup.

No further NPU run, production source change, adoption or official submission was performed by this diagnostic. Root selects subsequent research separately. Original build/package/raw log/archive and marker mappings remain preserved.

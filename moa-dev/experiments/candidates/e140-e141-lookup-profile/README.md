# D-E141: E140 / E141 whole FFN readiness

One fixed diagnostic allocation111627 passed57 comparisons. Original standard hosts bracketed E141 with E140:313898 →325748 →313736 cycles, or+3.801897% versus313817. E141 remains unadopted. Its separate8-case245760-output oracle passed; the prior E135 timeout is not an arithmetic failure of the repaired path.

Two logging hosts reuse the E139 transform. All three generated images match the original measured images byte-for-byte; only FFN marker-name storage is patched afterward. Instructions, constants, addresses, I/O, stack and marker numbers are unchanged. The seven processes are whole control before, control info/trace, candidate info/trace, candidate whole, control after. Uploaded archives, complete raw logs, all137/151 operations on each of three runs and both cluster clocks are verified. Host build invokes the SDK, but no device source or arithmetic changes and no new exact-source kernel variant is introduced.

| Event, median task-relative cycles | E140 | E141 |
|---|---:|---:|
| Activation TRF ready |125937|110503|
| First gate native start |125938|110504|
| Last gate native end |145627|140340|
| First up native start |169952|158322|
| Last up native end |190452|201340|
| GeGLU ready |229473.5|240283.5|
| Down native start |267831.5|277479|
| Final output store end |313693.5|324496|

E141 starts its first native15434cycles earlier but finishes its final up native10888cycles later. Its ten-native envelope grows64649→91078cycles; GeGLU is10810cycles later. The gate direct fused native intervals are about5200cycles each versus roughly3400 in E140. The first four up intervals are about8293/8693/8312/8309cycles and the final one5176. These Main markers include lookup, native contraction, scale/reduction and any waits; they cannot identify lookup service time alone.

Gate lookup tables are ready before their consumers. Except the first gate tile, block-scale VRF is the latest marked producer and native starts1cycle later. E141's first up native starts1cycle after its table finishes; subsequent up table preparations overlap preceding native intervals and block-scale VRF becomes the latest marked producer. The table-marker interval union is66420.5 versus12320cycles, but neither this difference nor overlapping Main/Sub durations is reclaimable latency. Without resource counters, contention is not a proven physical cause.

This evidence supports testing a coupled two-full30-row direct-stream path with separate F32 block-partial scale/reduction. It removes eight lookup/table contexts but reintroduces28.125MiB logical partial writes and the same amount of reads, plus separate Main scale/reduce work. Fresh actual bounds, unchanged first-BF16 oracle and complete FFN comparison are mandatory. E103's insufficiently preserved failing descriptors cannot establish the same defect or predict this new combination's accuracy. No new kernel is implemented here.

Historical E139 labels were independently rechecked: its schedule120/153 is actual up-weight ingress/decode,307/385 gate; it ran up before gate. E140's copy correction changed the emitted order, and both sides here execute gate before up. No historical correction is needed. Diagnostic info/trace cycles are recorded separately (E140312372/314603, E141322684/324846); they are not promotion measurements. Table cells are separate medians, so subtracting them need not equal a median per-observation duration.

Recheck:

```sh
python3 moa-dev/experiments/candidates/e140-e141-lookup-profile/analyze.py e141-lookup-readiness-v1 111627 --check
python3 moa-dev/experiments/candidates/e140-e141-lookup-profile/verify.py e141-lookup-readiness-v1 111627 --check
```

`profile-analysis.json` preserves every event and source/producer mapping; `decision.json` separates adoption, further tuning and research. `verification-initial.json` preserves the first correct-data verification before fixing its copied console-only120/130 marker-count text; the current verifier prints the actual137/151 counts. Source images, raw measurements and parsed results did not change. No extra profile allocation, live adoption or official submission is performed.

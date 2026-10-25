# PowerOfTwo model diagnostic: state gates pass, quality remains unproven

Root jobs321339/321340 (r1256/r1257) ran the uniquely frozen PowerOfTwo ELF
`2a2ffd0cb64d4e9b8993070e5a758934797147c47eb7ef66e8670df533eca088`, source72786a5b.
Model case finished in37.595s and unaffected-width case in27.872s. This source
retains the diagnostic-only ring dependency and is not an official export.

Strict checks passed: **18,612,224** same-candidate replay logits,
**786,432** unchanged scalar/Automatic logits and **524,288** interleaved P128
logits, totaling the printed19,922,944. The printed `replay_exact_logits` field
includes those last two unchanged-route checks. Candidate replay also compared
**5,905,580,032 cache bytes** (2,952,790,016 BF16 values, one side), including
untouched ring cells and all reserved pages. Separate fresh-history scalar/R32
checks passed **9,175,040 logits**. Own-output device greedy and exact-replay
seeded sampling passed. These are implementation/state checks, not task accuracy.

Original-model drift is large. Among60 rows that actually dispatch F4, **37 top1
tokens changed**; relative L2 averaged0.39383, range0.07960–1.11945, max absolute
logit error28.1875. The eight old-epilogue fallback rows run after their histories
have already diverged; one top1 changes there and their meanL2 is0.25143. They do
not isolate an old-path arithmetic defect. The three pre-divergence scalar/
Automatic outputs are exact. Across all71 observed rows,38 top1 selections differ.
This PowerOfTwo candidate is not established as quality-preserving or suitable
for activation. The separately gated fitted-scale candidate needs its own result.

For the seven matched eligible calls, original model time averages37.134ms,
candidate34.394ms and replay34.420ms (about7.3% reduction). The corresponding48
FFN epilogue spans average13.691ms versus10.909/10.854ms. These completed model
timers exclude host logit readback/selection and cache comparisons; this is not an
HTTP throughput or SLO measurement. Approximate histories differ by design.

All144 shadow projections prepare successfully. Additional measured HBM is
exactly **4,282,122,240B**: packed weights4,246,732,800 plus group scales35,389,440.
Both jobs start with14,279,647,744B available after current group/projection
scratch and retain9,997,525,504B after publication. Reported device capacity is
51,002,675,200B. This observed allocation is consistent with the separately
compiled physical32 pools; no stale server artifact was used for this unique ELF.

Shadow startup is **16.78s**, including8.72–8.83s original E4 readback,
5.75–5.90s CPU encoding and0.75–1.05s upload. Another1.25–1.30s falls outside the
three measured spans and remains unattributed. This is measured startup, unlike
the earlier exhaustive CPU encode-and-audit micro timer. Idempotent preparation
and rejection during active checkpoints pass. No cancellation stress was run;
partial-construction cleanup remains a bounded source-lifetime argument.

`analyze_power2.py` reproduces `actual-power2-r1256-r1257.json`, which retains
every startup projection, stage time, drift row, memory observation and log hash.
The old source and ELF remain frozen; default remains OFF.

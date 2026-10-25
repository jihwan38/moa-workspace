# Approximate full-H B16 LM shadow

Base: 394d60a290e6137c88938300468050128de39be6 (current c079 model).
Owner: round2_path; root alone dispatches Arena. No host/serving selection.

The current exact control is ops_b16_lm_storegroup::logits_batch: original BF16
embedding and full H3840 dot, BF16 dot boundary, F32 softcap division/tanh/multiply,
BF16 output, real W32→W128 final-store grouping. Preserve the older BF16 H960
partitioned strict HOLD (2,982/25,165,824 output differences; 0/384 selections).
No prior native-FP8 LM shadow was found in source/commit searches or e2e records.

One startup returned-output kernel quantizes each original vocabulary row with
s = 2^floor(log2(max(amax,2^-112)))/128 and E4=cast_E4(weight/s), the established
power-of-two row policy. Keep the original BF16 embedding for token lookup and
control. W×H E4 is 1,006,632,960 bytes; W F32 scales 1,048,576; total persistent
increment 1,007,681,536 bytes (<1.1 GB). Runtime temporary quant input is 61,440
bytes plus 512 padded scale bytes; existing B16 logits ABI is unchanged.
The older actual full-model allocation gate retained 3.515 GB spare after joined
FFN weights, so this is plausible, not current-model capacity proof. The standalone
probe records actual allocation deltas and preserves original and quantized tables.

Runtime original final-RMS BF16 input uses the established B16 power2 quantizer.
The FP8 core broadcasts all16 H3840 rows once to a 61,440-byte logical TRF
(Lane B%8, Element B/8×H). Each original W32 weight tile becomes122,880 bytes
versus245,760 BF16. One full-H native dot per band replaces the two B8 BF16 dots.
Multiply F32 accumulator by token scale then vocabulary scale before original BF16
dot boundary. Softcap operations, output bytes and downstream selection unchanged.
This is approximate to the original embedding/input arithmetic; no exact claim.

Resource gate: compare current exact full-H control and complete FP8 core; inspect
actual TRF lifetime, spills/hidden HBM, DM fit and scale traffic. Candidate must
credibly target >=25% complete LM reduction, including separate input quantization,
all multiplications and canonical output. No geometry sweep.
If viable, provided embedding + actual finalnorm weights; non-dyadic normalized
fixtures, original quantizer independent byte/scale oracle, F64 intended quantized
dots, full-logit errors and all384 greedy policy comparisons, candidate-first poison,
permutation and source immutability. Startup cost excluded from steady timing but
reported separately. Do not turn approximation differences into correctness passes.

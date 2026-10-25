# q021 scoped LM composition: model gates passed

Frozen default-OFF source553c44c6 and ELF15ff8c81dc6d were run by root:
r1303/job321550 (`model 512`), r1304/job321551 (`model 1024`), and
r1305/job321554 (`guards 512`). All three completed successfully.
`collect_actual.py` derives the detailed counts, timing and drift in `actual.json`
from the retained unmodified logs. Independent source review by round2_path
passed. No task accuracy or HTTP performance conclusion follows from these gates.

Both arms retain q019 MSE F4 FFN and identical teacher-forced inputs. Only the
terminal canonical B16 LM changes from the original BF16 embedding to the
previously tested q018 FP8 shadow. Step8 explicitly exercises identical legacy
FFN fallback in both arms, then step9 re-enters MSE FFN. Scalar, prompt and R32
keep the original LM. The default-OFF Automatic call is also tested separately.

Across both model runs and the guard run,55,836,672 logit values passed strict
equality checks. This count comprises candidate physical replay plus unchanged
scalar/default-OFF/prompt routes, and9,175,040 scalar/R32 guard values; it does
not claim eligible FP8 LM outputs equal BF16 LM. Both before replay overwrites
the control storage and after replay, all reserved KV bytes match exactly:
11,811,160,064 reported BF16 values, or23,622,320,128 compared bytes across both
runs. These are repeated comparison counts, not unique allocation capacity.
Each variant's device greedy selection matches its own downloaded logits;
seeded stochastic selection also reproduces on identical candidate replay.

Eligible B16 outputs differ from q019 BF16 LM:

| Actual history | Rows | Relative L2 range | Maximum absolute difference | Same reported top1 |
|---|---:|---:|---:|---:|
| 512 | 81 | 0.007962–0.035111 | 1.1875 | 77/81 |
| 1024 | 81 | 0.009083–0.041618 | 1.123047 | 77/81 |

The six scalar/default-OFF rows across these two runs are bit-exact. All
eligible values are finite. The eight changed top1 coordinates are preserved
in `actual.json`; no threshold was introduced or relaxed after these results.
These synthetic token histories are model execution checks, not GSM/task labels.

For the seven normal-MSE eligible steps2/3/4/5/6/7/9, mean forward time changes
34.528→33.225ms at history512 and34.983→33.646ms at history1024 (3.77–3.82%).
The corresponding LM projection stage changes2.375→1.167ms and2.384→1.169ms.
This compares a complete control pass with warm candidate replay, not ABBA;
first candidate use is separately retained in JSON. History fill, prompt
interleaves, downloads, sampling checks and KV scans lie outside these timers.
The first candidate LM call incurs extra setup, so its timings are not mixed
into the warm replay mean. These numbers do not establish serving latency.

Actual shadow allocation is1,007,732,480 bytes. Available HBM falls from
9,997,525,504 to8,989,793,024 bytes, and idempotent preparation leaves it unchanged.
The complete model cases end with8,982,948,864 available. Original BF16 embedding
and all original FFN weights remain retained. Missing-shadow and active-prompt
preparation guards passed in the frozen default-OFF probe.

Root authorized a separate isolated quality-candidate activation only after all
three passes, together with the byte-identical q020 parallel CPU startup helper.
That activation is a new source/build and requires fresh public/task evaluation.
The default-OFF oracle source and ELF above remain the evidence for these results.

# Matched c077 OFF versus c078 late-admission recovery

The completed matched diagnostic gains **4.08% throughput** through better
decode batching: the same6,112 measured decode rows require22 fewer rounds.
Prompt chunk/layer counts remain identical. First-output latency increases,
and the long already-visible stream has larger individual gaps. This is a
specific overlap diagnostic, not an official workload or a blanket SLO result.

## Provenance and scope

| | OFF | ON |
|---|---|---|
| Run / job | r1148 /320874 | r1150 /320894 |
| Source | f87085c1034f51cfe71661279dc9f59eecdbf9a1 | 5994a2973303df48e289f014eefd99cf761bc06a |
| ELF SHA256 | 6ff1b597c7674e3cace7638356bea66beb040bf299f493d2956c72b1d6369333 | e912924dbfa672d83001a12f9c82e3eff6751c5cff6dd219b67e9d6191d1e423 |
| Terminal result | SUCCEEDED /54.2997s | SUCCEEDED /52.6216s |

Both clients are byte-identical: `performance.py`, `performance_test.sh`,
`entrypoint.sh` and `scoring.py` hashes are recorded in `comparison.json`.
Each runs16 genuine warm requests, then96 measured requests at concurrency16.
Only measured HTTP index1 requests128 output tokens; every other request uses64.
All96 pass and produce the same6,208 total tokens in both runs. Suite, prompt
length and output length distributions match. No expected-answer routing is used.

The source diff changes only `startup_cohort.rs`, its late-admission lifecycle
tests and the Worker constructor/post-admission hook. Model arithmetic, LM,
weights, allocation policy, receipt3750ms budget, credit90ms and150ms soft gap
cap are unchanged. OFF disables recovery; ON enables recovery plus late joining.
Therefore this pair measures the combined recovery policy, not late joining
alone against an otherwise identical fixed-membership recovery server.

## Completed measurements

| Measured metric | OFF | ON |
|---|---:|---:|
| Window, s | 41.282881 | 39.663236 |
| Generated tokens/s | 150.377100 | 156.517738 |
| TTFT p50 /p90, ms | 1288.132 /3578.337 | 3766.382 /3807.943 |
| Estimated TPOT p50 /p90, ms | 88.246 /89.872 | 46.489 /56.037 |
| Individual TTFT >4000ms | 6 | 9 |
| Individual estimated TPOT >100ms | 0 | 0 |
| 2048-token subgroup TTFT p90, ms | 5641.660 | 4717.812 |

Holding genuine first outputs shifts waiting from decode to TTFT. Aggregate
TTFT p90 remains below4s, but the long-input subgroup and individual outliers
remain above it. No per-request4s guarantee follows from aggregate validity.

## Rounded counters explain the window difference

The first idle checkpoint in each log follows exactly16 completed warm
requests. The following counts and costs subtract that checkpoint from the
final cumulative `SCHED_TOTAL`.

| Measured work | OFF | ON |
|---|---:|---:|
| Decode rounds | 488 | 466 |
| Actual decode rows | 6112 | 6112 |
| Mean live decode width | 12.525 | 13.116 |
| Full16 decode rounds | 90 | 338 |
| Scalar decode rounds | 4 | 60 |
| Prompt preparations /completed chunks | 173 | 173 |
| Prompt layer dispatches | 8304 | 8304 |
| Prompt owner-count chunks1/2/3/4 | 31/79/1/62 | 9/31/0/133 |

All prompt work uses the same four-quarter cost class. More four-owner chunks
does not by itself prove less padding: one-owner512 and two-owner512 can already
fill the same dense work. There is no reduction in prompt layer count here.

| ON minus OFF elapsed accounting | Difference, ms |
|---|---:|
| Complete decode dispatch | −1509.14 ±6.03 |
| Complete prompt dispatch | −152.64 ±109.92 |
| Sum of those disjoint categories | −1661.78 ±115.95 |
| Actual HTTP measurement window | −1619.65 |

Bounds reflect only rounding of means to0.01ms. The observed window difference
is within that accounting bound. Prompt sub-timers show preparation−3.04ms,
layers−179.52ms and finalization−4.63ms; they are a decomposition of complete
prompt work and must not be added to it. Aggregate timing differences do not
establish a hardware-frequency change or a new kernel speedup.

Using each OFF width's measured mean cost, the changed width/count mix explains
**−1542.924ms** of decode work. Applying the observed ON within-width means adds
**+33.784ms**, leaving the actual rounded−1509.140ms. Thus the gain is explained
by changed batching rather than uniformly faster calls. This is an arithmetic
decomposition, not a simulated execution or proof of a counterfactual schedule.

## Joins remove repeated separate groups, but long prefills remain

OFF forms only the initial warm and first measured cohorts. ON forms seven
snapshots including warm; their initial→final member counts are
`16→16,16→16,15→15,15→15,12→16,14→16,16→16`.
Five join events append six fresh request IDs. Unlike historical c076 r1130,
there is no repeated11+5 snapshot sequence.

For snapshot4 (zero-based, including warm), old decode completions52/53/54/64
free slots. New members join while the existing deadline has2818.447,
2510.199 and965.630ms left:

- ID77:256 remaining tokens;
- IDs78/79:1024/2048 remaining;
- ID80:512 remaining.

Snapshot5 adds ID95 with768 tokens and ID96 with512 while2678.709/2635.618ms
remain. Every join preserves the exact old deadline. These are real admission
events, so they directly resolve the previous ambiguity about whether replacement
requests were admitted before an earlier snapshot's release.

Four snapshots still expire with unfinished long prompts:

| Snapshot | Ready /members | Remaining tokens | Unfinished worker IDs and remaining |
|---|---:|---:|---|
| 2, no joins | 12/15 | 1536 | 33:256,40:640,42:640 |
| 3, no joins | 12/15 | 2560 | 52:896,53:896,54:768 |
| 4,12→16 | 13/16 | 2048 | 69:512,70:640,79:896 |
| 5,14→16 | 14/16 | 768 | 81:256,94:512 |

All these unfinished owners have2048-token prompts. The first two failures to
finish by the barrier occur without any late joins, so they cannot be blamed on
joining alone. Added long member79 still has896 tokens at release; its arrival
was admitted with2510ms of the original deadline remaining, not a renewed budget.
The snapshot trace establishes unfinished ranges and previous decode service.
It does not expose the exact128/256/384/512 tail composition: no `SCHED_CHUNK`
records were captured. The total173 chunks cannot support a claim that long
prefix fragmentation was eliminated.

## Existing streams and lifecycle evidence

The unique observed `(prompt_tokens=256, completion_tokens=128)` tuple links
HTTP index1 to OFF worker20 and ON worker17. This is a uniqueness-based join,
not an assumption about HTTP/Worker ordering. ON worker17 is passed through at
both ends of a3037.455ms recovery snapshot. Its128 actual content events continue
with maximum gap174.175ms, ruling out holding that old stream for the barrier.
Six other old decoders complete inside the two joined snapshots, proving actual
service rather than eligibility alone.

| Unique long stream | OFF | ON |
|---|---:|---:|
| Completion from measured phase start, ms | 11852.598 | 11646.729 |
| Content-event gap p90, ms | 95.519 | 148.650 |
| Maximum content-event gap, ms | 157.387 | 174.175 |
| Gaps >150ms | 1 | 7 |

Completion improves by206ms, but tail gaps get worse; the soft cap is not a hard
per-gap promise. Content events are not generally identical to token boundaries,
although this fixture records128 events and128 generated tokens.

The reader finds110 unique once-participated members, no re-holds, no membership
removal/extension other than explicit append, no deadline extension, no ready
release with unfinished members, and no incomplete snapshot. Deadline overshoot
reaches18.814ms in the second no-join deadline snapshot, consistent with the
existing atomic-work limitation; the joined snapshots overshoot0.847/0.230ms.
Independent lifecycle validation belongs to the owning agent's report.

Historical fixed-membership c076 r1130 had591 total decode rounds and repeated
11+5 groups, whereas c078 has529 including warm. Its143.427TPS result is secondary:
the LM path and run timing differ, so this is not a matched late-join-only gain.
The matched OFF/ON result supports the observed batching mechanism, with the
TTFT and gap tradeoffs above. No parameter sweep, source change, extra Arena run
or production-adoption claim follows from this analysis.

Reproduce with `analyze.py r1148-c077-overlap-r96-submit r1150-c078-overlap-r96-submit`
then `compare.py`. The old fixed-membership reader retains its default behavior;
its optional `parse_recovery=False` lets this reader supply the join-aware state
machine without misinterpreting joins as releases.

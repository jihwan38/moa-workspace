# M08 / Arena105767 independent trace interpretation

`analyze_profile.py m08-pair4-v1` and `--check` passed on the collected result. The analyzer independently maps all22 native engine markers to each frozen schedule, checks weight/table/TRF→dot and dot→reshape→gather→scale lineage, then validates every marker in3runs×2clusters on each side. All12 run-cluster observations (264 engine intervals) are present; each run's maximum cluster Task duration exactly matches its reported kernel cycles. Absolute clocks from different clusters or launches are never subtracted.

The package→screen build→actual uploaded files→raw log digest chain is bound in the analysis. The raw-span-derived per-operation observations also exactly match `screen.py`'s saved observations. A frozen E104 trace was accepted as a parser check; deleting a weight marker or altering a Task-cycle value was rejected. No additional NPU execution was requested by this reviewer.

## Observed outcome

Standard benchmark host (`FURIOSA_OPT_PROFILE=info`) whole O:

- E122 before51,418; M0858,095; E122 after50,906 cycles.
- Bracketing control median51,162; candidate+6,933cycles / **+13.551073%**.
- Collected seven-stage screen:57 basic correctness comparisons passed. This is one allocation and the standard three inputs, not extended correctness validation or official leaderboard evidence.

Trace diagnostic medians retain all six observations per side:

| Window | E122 control | M08 |
|---|---:|---:|
| Weight DMA interval |26,769|27,190|
| Weight end→dot |2|2|
| Lookup/dot/16-column reduce |12,019|12,011|
| F32 gather |1,201|8,222|
| Dot end→Task end tail |9,384.5|16,751.5|

Each launch still reports weight as its latest local dot prerequisite. Weight supply did not show an improvement under the changed source order. Weight windows overlap in range, so the modest difference in their medians is not a robust bandwidth-regression estimate.

Gather is the clear observed cost increase: baseline range1,171–1,233 vs candidate7,735–8,535 cycles. Corresponding run/cluster-index candidate-minus-control deltas are6,564–7,364cycles (median6,991). Main dot duration is nearly unchanged. This is consistent with the descriptor change from32×480B to1,920×8B and with the longer post-dot tail. It does not establish an exact hardware packet-service constant or justify summing profile interval medians into the whole-host regression.

The independent whole-host result and the trace agree on the practical decision: close M08 pair4-v1 without adoption or a layout/tile sweep. A smaller HBM source-address jump did not deliver a measured ingress benefit here, while the direct gather's fragmentation was costly. The result rejects this complete producer/consumer combination; it does not prove all row interleaving is futile, that baseline HBM is saturated, or that direct8B gather is universally unsupported.

Generated evidence:

- `moa-dev/experiments/candidates/m08-output-pair4-ingress/analyze_profile.py` SHA256 `7f70793542c25b78df7540d224ee7df5f600d6d166fc1d0f67bb05b204323165`.
- `profile-analysis-m08-pair4-v1.json` SHA256 `417d11b58c879407ea6c57cd1dd72b8e3936d224b4fa369d0169620db99d7319`.
- `marker-source-map-m08-pair4-v1.json` SHA256 `6e1d2d846b30dad5495fa5b9699fe3ac7116a8df55b996ded842458d61237ec3`.

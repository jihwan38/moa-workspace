Independent read-only batch_kernels review passed. The diagnostic forces R64
at partial widths and passes only use_head_rows=false; prefix512/prefix768 and
mixed512 retain precedence. The probe eligibility prediction mirrors full-window
selection, including inactive groups at width7. Both phases refill every request
causally from0 into bijective, disjoint four-page maps; positions and selected-row
indexing match. Each mode compares199 whole vocabulary rows. No persistent mode,
cache-order or reference-route issue was found. Timed model work excludes host
logit download; actual NPU/model gate remains required.

# Packed prompt model gates

The 15 parent-dispatched runs below passed: 106,954,752 all-vocabulary BF16 values and 35,106,324,480 raw KV bytes compared exactly. Tail384 covers all 11 probe cases; ThreeOwner512 covers cases 6–9, including each doubled-owner orientation across position2048. Every run also repeats a reversed-owner reset. Case6 of each family interleaves48R64 decode steps. No Worker or HTTP throughput result is implied.

Each run checks real histories, final-owner output selection, all allocated full-attention pages and sliding rings across48layers, disjoint sentinels, post-prefill decodes, active-ownership guards, and NaN-poisoned inactive-quarter reset where applicable. Only real quarters perform heads/attention/KV commits; all padded dense execution remains charged as512rows.

| Route / case | Run / Arena | Current packed control (ms) | Candidate (ms) |
|---|---|---:|---:|
| Tail384 / case6_owners1_start1664 | r844 / 319297 | 223.131 | 139.701 |
| Tail384 / case5_owners3_start1920 | r847 / 319301 | 227.966 | 146.590 |
| Tail384 / case2_owners2_start256 | r848 / 319303 | 218.482 | 134.648 |
| Tail384 / case3_owners2_start896 | r852 / 319305 | 217.912 | 134.651 |
| Tail384 / case0_owners1_start0 | r869 / 319348 | 216.601 | 130.870 |
| Tail384 / case1_owners3_start0 | r870 / 319349 | 217.056 | 130.764 |
| Tail384 / case4_owners1_start896 | r871 / 319350 | 220.034 | 136.077 |
| Tail384 / case7_owners2_start1920 | r872 / 319352 | 224.864 | 141.220 |
| Tail384 / case8_owners2_start1920 | r873 / 319353 | 226.145 | 141.678 |
| Tail384 / case9_owners3_start256 | r874 / 319354 | 223.496 | 139.667 |
| Tail384 / case10_owners1_start1920 | r875 / 319356 | 224.309 | 140.501 |
| ThreeOwner512 / case6_double0_start1792 | r857 / 319313 | 238.197 | 155.948 |
| ThreeOwner512 / case7_double1_start1920 | r866 / 319344 | 236.712 | 152.527 |
| ThreeOwner512 / case8_double2_start1920 | r867 / 319345 | 242.065 | 155.620 |
| ThreeOwner512 / case9_double0_start1920 | r868 / 319346 | 236.862 | 152.670 |

Timings are sequential paired completed prepare+48layers+finish, excluding downloads, the deliberately injected interleaves, and cache verification. Controls independently recompute the same prefixes in the same model; Tail384 uses current Tail256+P128 as its stronger control, and ThreeOwner512 uses Tail384+P128 in addition to twoTail256 and fourP128. These are model diagnostics, not ABBA or HTTP measurements.

Build sources and ELF SHA256:

- Tail384: source `9b1e19cd9977244490a160a2c15dd7bebaa1f0be`; ELF `15b0a9415a8d5acef90dc247d0df966608a6cee7430061fde73a94cdc025218e`.
- ThreeOwner512: source `180202c32670bb5f48792d6d59ee3f37298b5231`; ELF `41b6c454ccafe300bfc899f0d58474885c65d6e27251c066319065c867d1ecd2`.

`collect_model_evidence.py` rehashed every submitted ELF and raw log, required successful Arena status and zero-mismatch markers, and verified each complete source manifest. All tracked files match the build commit; the untracked generated Cargo.lock is verified against the recorded run hash and preserved separately. Later submission metadata commits contain documentation, so the build commit above is explicitly distinguished from each recorded submission commit. `model-results.json` contains all per-run checks, source/log hashes, reset timings and totals; both whole source manifests are retained.

The isolated Worker export is `00ac271` → `8e6acce` → `38493b8`, based on root `b078362`. Its111hosttests and independent round2_path review pass. Full real512 selection precedes padded384/256, active ranges restore by allocation identity through cancellation/compaction, and completion validates all outputs before advancing any owner. Production composition and HTTP/public-task gates remain parent-owned.

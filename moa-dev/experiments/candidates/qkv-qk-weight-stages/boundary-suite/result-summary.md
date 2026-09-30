# E146 v4 boundary validation passed

Raw9 job137026 passed all27 comparisons /73728 BF16 outputs exactly (signed zero equivalent, nonfinite0). All63 actual input checksums and SHA256 hashes match the original reference; the downloaded uploaded archive matches the frozen host and fixture.

Full2 job137031 passed16384 outputs under the original atol0.04/rtol0.01 and8380416 untouched cache values. All38 actual input hashes match the original E53 host/reference. The host contains the exact selected E146 whole QKV image `e0245eaced58992518dfecaf6b0447c107a93f942303d15b7997716fcabe371f`,213760B, once.

The raw diagnostic uses a separate141056B image `a4943e8628f93fd38494bf80fefadfcf52b5c006f67c2f48b866b182ea15b724`. Its48 source inputs retain the46 actual production inputs except the documented lib-module append. Entire production projection SHA is `c4c596906051b9930af44e5806d05b6de2692c7c926c7c619ca0256b69e0da7f`. Raw9 wrapper, Rust host and F64 generator are byte-identical to the E144 suite. No historical NPU result is inherited.

`raw_oracle.py --check`, `check_full.py`, and `verify_preparation.py --check --require-selected` passed. [Digest-bound summary](result-summary.json), [raw result](raw-results-v1.json), [full result](result-137031.json). These are finite changed-boundary correctness observations. Performance, extra6 and adoption are separate decisions; this does not assert generic BF16 correctness or reachable PRNG support. Compiler target m03 and Arena slot2 are released.

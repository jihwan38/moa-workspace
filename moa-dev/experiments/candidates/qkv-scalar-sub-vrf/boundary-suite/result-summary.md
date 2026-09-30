# E147 scalar-v1 boundary validation passed

Raw9 job137109 passed27 comparisons /73728 BF16 outputs exactly (signed zero equivalent, nonfinite0). All63 input checksum/SHA256 checks match the original independent reference. The actual uploaded tar matches the frozen host and fixture.

Full2 job137116 passed16384 outputs under original atol0.04/rtol0.01 and8380416 untouched cache values. All38 input hashes match the original E53 host/reference. The host contains the selected E147 whole QKV image `fd2e252603b4de38389d8720edd78d6ae1b5c82514f6869934a5ebbb46c5f5fb`,219904B, once; the downloaded uploaded package matches its build.

The separate raw diagnostic image is145920B, SHA `39a0603465f5e2f48f864891192751997558351fedef7538445fe279e947a7b8`. Its48 inputs retain all46 selected production inputs except the documented lib-module append. Entire production projection is39440B, SHA `e21e207081935b441530e78c56ffbc1731f484c68844fc692ee283815a82f3cd`. Wrapper, Rust host and F64 generator are byte-identical to the original E144 suite. No old NPU result is inherited.

`raw_oracle.py --check`, `check_full.py`, and `verify_preparation.py --check --require-selected` passed. [Bound summary](result-summary.json), [raw result](raw-results-v1.json), [full result](result-137116.json). These finite cases exercise the new scale producer/Sub supply and unchanged raw/head/cache boundaries; they are not a generic BF16 or reachable-PRNG guarantee. Performance/adoption remain separate. m03 and Arena slot2 are released.

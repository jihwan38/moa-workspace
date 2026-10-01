# E148 boundary status: packages verified, no boundary NPU run

The first complete-kernel screen passed45 comparisons but O regressed3.2882%. Therefore raw9 and whole case7 were built and host-verified only; neither was submitted to Arena. See [status](status.json) and [build summary](build-summary.json). The preparation record below remains scoped to its original checks.

# E148 unchanged O boundary validation

Prepared from the frozen E137 O suite; no result is inherited. `preparation.json` records byte identity of the original raw wrapper/host/reference generator and the whole-case host/CPU checker. The raw nine cases retain original BF16/F64 reference values and require 34,560 BF16 outputs to match exactly, treating signed zero as equal. They check K/row/cluster boundaries, the global maximum and active fourth term. The whole E47 case7 retains the original fixture and public atol0.05/rtol0.01 check on3,840 outputs through RMS and residual. These finite tests do not prove arbitrary-input or end-to-end model correctness.

`oracle.py build --build-label LABEL` requires the selected production graph gate and binds all46 original compilation inputs, the complete production projection, and its whole image. It only appends the original diagnostic module and host. The raw image is distinct from the production image and its timing is excluded. `full_stress.py build --build-label LABEL` adds only the original host and requires its embedded image to equal the selected actual whole-O image. Original input checksums, generated raw fixture identity, actual uploaded archive contents and logs are checked again on collection and `--check`.

Both tools use the independently reserved `/tmp/moa-m03-validation-probe/target`; run their builds sequentially. Each supports `build`, `verify`, `submit`, `collect` and `--check`. Submission is a separate action and requires the parent's measured-candidate/slot authorization. No official submission occurs in this suite. The original E137 control images do not need rebuilding.

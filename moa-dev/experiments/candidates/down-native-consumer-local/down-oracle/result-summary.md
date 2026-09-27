# E140 local-v4 isolated down result

Arena111393 passes all nine original cases: 34,560 exact BF16 outputs. `probe.py --check` verifies the frozen source, image, input and uploaded archive.

This helper executes `project_down_tile` and `finish_down`, excluding upstream `prepare_ffn_fp8_split` and final `down_global_scale`. Its success does not validate the incorrect local-v4 whole FFN. The subsequently identified upstream copy overrun and local-v5 whole recovery are recorded in `../prefix-overrun.md` and `../local-down-decision.json`. This is diagnostic evidence, not an adoption gate for the control-only prefix repair.

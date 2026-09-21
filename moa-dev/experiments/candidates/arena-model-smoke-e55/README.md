# E55 real-model smoke on Arena

The exact frozen `gemma4` ELF from the [model-host build](../../../../docs/experiments/e55-model-host-build.json)
was submitted with `remote_entrypoint.sh`. The wrapper sets
`GEMMA4_MAX_NEW_TOKENS=1`, invokes the prompt `Hi`, and uses the model directory
already supplied by Arena. It changes no model or kernel source.

[Job71173](../../../../docs/experiments/e55-arena-model-smoke.json) terminated
after70.093479245 seconds with scheduler reason `timeout`, exit code−1.
Only the wrapper's begin marker appeared; neither `model loaded` nor a generated
token appeared. In the actual [CLI](../../../../furiosa-opt-gemma4-12B/src/bin/gemma4.rs),
both `Device::new` and `load_model` precede that log. This result does not tell
which operation took the time, whether checkpoint loading would eventually
succeed, or whether an optimized kernel is incorrect. It is not an inference
latency or throughput measurement.

The downloaded submission archive contains only `gemma4`, the wrapper, and
`job.yaml`. Its ELF SHA256 is
`7e801ed65016b5d0faa71da3fd3f3f9d012db393adc47863ad8e4665532678eb`, exactly the
previously inspected E55 model host. The archive, authoritative terminal status,
and full log are retained under the ignored artifact directory
`moa-dev/experiments/artifacts/arena-model-smoke-e55/`; their hashes are recorded
in the linked result.

Reproduction from the workspace root, with those local artifacts present:

```bash
./moa-dev/moa mkdir -p /tmp/moa-e55-model-smoke
docker cp moa-dev/experiments/artifacts/e55-model-host-build/gemma4 moa2026-dev:/tmp/moa-e55-model-smoke/gemma4
docker cp moa-dev/experiments/candidates/arena-model-smoke-e55/remote_entrypoint.sh moa2026-dev:/tmp/moa-e55-model-smoke/remote_entrypoint.sh
./moa-dev/moa furiosa-arena submit /tmp/moa-e55-model-smoke/remote_entrypoint.sh /tmp/moa-e55-model-smoke/gemma4 --name e55-model-smoke --entrypoint remote_entrypoint.sh --timeout 70
```

The follow-up [job71198](../../../../docs/experiments/e55-arena-model-progress.json)
used `progress_entrypoint.sh` and the separate
[fixed-prefix host](../model-fixed-prefix-e55/README.md). It completed in18.422
seconds: device initialization ended at0.317s, model loading at16.058s, and the
14-token `Hi` chat prefix produced262,144 finite logits. The existing46 model
inputs were preserved; only a new diagnostic host was added. Its ten text images
and measured Stage1 image identities were separately verified.

This is the first completed real-model text diagnostic, with no original-model
comparison, sampling, image/audio input, or position512 page transition tested.
Its timing is a single uncontrolled observation. It does not identify the cause
of the earlier silent CLI timeout, prove a cold-load time, or establish accuracy
and official throughput. The final reference file stayed in the remote job;
the subsequent paired diagnostic generates a fresh original-model reference.

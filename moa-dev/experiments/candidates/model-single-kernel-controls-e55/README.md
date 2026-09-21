# Single-kernel model comparisons around E55

Each control starts with the frozen E55 source and restores one kernel kind to
its original implementation. Before packaging, [prepare_package.py](prepare_package.py)
parses the actual host ELF registry and requires the target image to equal the
original while the other nine equal E55. All hosts reuse the unchanged
[four-prefix diagnostic](../model-prefix-suite-e55/probe_model_prefix_suite.rs).
These source controls isolate a whole kernel implementation, not an individual
rounding operation inside it.

QKV, sliding O, and FFN controls compare against the original host. If the newly
emitted original reference hash equals job71239's hash, their error statistics
can be compared with the earlier E55 statistics against that same reference.
The full-O control compares directly against E55 to test whether restoring its
residual loop changes any of the four complete output vectors. Model execution
results, including reference hashes, must be collected before drawing either
conclusion.

The [runner](remote_entrypoint.sh) performs full model/metadata identity checks,
emits four reference vectors, compares the control's four vectors, then checks
weight stat/header and metadata hashes again. Every case starts with a fresh
Workspace and Decode. The70-second Arena limit bounds the two model executions.
The remote reference JSON remains in the job directory; its SHA256 is logged.

The [control collector](../../collect_model_kernel_control.py) reopens the actual
downloaded job archive, validates its embedded kernel images, and calls the
[suite collector](../../collect_model_prefix_suite.py) with this pair's package
manifest. It relabels the legacy original/E55 summary fields as reference/control.
It checks terminal exit0, complete ordered events, finite whole-vector counts,
statistics consistency, uploaded and executed hashes, asset identity, and exact
job configuration. It does not independently recompute statistics from remote
raw logits. Numerical differences remain observations with no accuracy threshold
or acceptance verdict. These four short, high-margin prefixes do not establish
answer quality, long-context, multimodal, or official performance results.

Example preparation and collection, from the repository root:

```bash
python3 moa-dev/experiments/candidates/model-single-kernel-controls-e55/prepare_package.py \
  --control sliding-o --reference original \
  --host moa-dev/experiments/artifacts/model-control-sliding-o-original-e55/probe_model_prefix_suite
python3 moa-dev/experiments/collect_model_kernel_control.py \
  --status STATUS.json --log LOG.txt --archive UPLOADED_JOB.tar \
  --package moa-dev/experiments/candidates/model-single-kernel-controls-e55/sliding-o-package-manifest.json
```

Preparation refuses to overwrite an existing package or manifest. It launches
no NPU job; Arena submission is a separate explicit command recorded with each
result. The live E55 source and accepted binaries remain unchanged.

The [completed four pairs](../../../../docs/experiments/e55-model-single-kernel-controls.md)
are recorded separately from build preparation. The [collector checks](collector-tests.json)
accept all four actual pairs and reject eight control-specific mutations. Replaced
ELF tests also update the declared and logged hashes, then require the precise
target/non-target image rejection; an unrelated hash mismatch cannot satisfy
those tests. The shared suite collector retains its one-positive/twenty-negative
checks against job71239.

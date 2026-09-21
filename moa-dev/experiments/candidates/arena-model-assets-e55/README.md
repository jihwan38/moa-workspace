# Arena model asset discovery and identity

The [presence result](../../../../docs/experiments/e55-arena-model-assets.json)
comes from completed Arena job71164. `RNGD_MODEL_DIR` was set, and the third
candidate used by [load::model_dir](../../../../furiosa-opt-gemma4-12B/src/host/load.rs),
`gemma-4-12b-it-NVFP4`, contained four readable regular files:
`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and
`chat_template.jinja`. The optional `chat_template.json` was absent there.

`probe.sh` checks only fixed file names and file-type/readability metadata. It
does not read model contents, enumerate arbitrary directories, dump environment
values, or launch an NPU kernel. Absolute configured paths are not printed. The
same three candidates under `/models`, `/data`, and `/workspace` yielded no
readable regular matches. There were60 checks, four positive and56 negative.
This establishes availability in that Arena job, not checkpoint compatibility.

`fingerprint.py` subsequently selects the same first existing weight candidate
as the Rust loader. It reads the safetensors length prefix and bounded JSON
header, hashes the four fixed metadata files, and attempts a full weight-file
SHA256. Its40-second monotonic budget is checked around8MiB chunks. If the
budget expires, the complete-file hash is explicitly null; a partial hash is
never presented as a checkpoint identity. Blocking filesystem reads can still
exceed that cooperative deadline. File identity/stat changes invalidate hashes.
The header check covers JSON/descriptor structure, not all loader tensor names,
dtypes, shapes, or quantization semantics.

The actual commands below reproduce the presence probe and the historical
Python attempt from the workspace root:

```bash
docker cp moa-dev/experiments/candidates/arena-model-assets-e55/probe.sh moa2026-dev:/tmp/e55-model-asset-presence.sh
./moa-dev/moa furiosa-arena submit /tmp/e55-model-asset-presence.sh --name e55-model-asset-presence --entrypoint e55-model-asset-presence.sh --timeout 10
docker cp moa-dev/experiments/candidates/arena-model-assets-e55/fingerprint.py moa2026-dev:/tmp/e55-model-fingerprint.py
./moa-dev/moa furiosa-arena submit /tmp/e55-model-fingerprint.py --name e55-model-fingerprint --entrypoint e55-model-fingerprint.py --timeout 65
```

The [Python attempt71191](../../../../docs/experiments/e55-arena-model-fingerprint-python.json)
failed immediately with exit127 because remote `python3` was not found. No model
file was read and no fingerprint was obtained. The script is retained for local
inspection; it is not a working Arena fingerprint solution. The standalone
[Rust implementation](rust-inspector/src/main.rs) builds without Python or the
Furiosa SDK and retains the bounded hash and file-change checks. Neither program
writes model files.
Presence status, full log, and downloaded upload archive are retained under the
ignored `moa-dev/experiments/artifacts/arena-model-assets-e55/` directory, with
hashes in the tracked result. Authentication state and model files are not copied
into the repository.

The [loader header contract](loader-header-contract.json) enumerates1,149
required tensors and9,304,785,568 bytes of file payload. This differs from HBM
upload bytes because the vision position embedding stays on the CPU and layer
scalars are padded during upload. The contract is based on the unchanged
original/E55 loader and fixed axes. The [local Python checker](validate_header.py)
generates the same explicit mapping and completed32 synthetic checks without
reading or allocating a model payload.

Actual SDK0.8.1 `from_safetensors` checks shape and raw byte length, then
reinterprets bytes; it does not check dtype. Therefore677 required tensors have
an additional semantic dtype audit, whereas472 use explicit dtype checks in the
loader. The inspector reports necessary header conditions separately from
successful model loading. All header tensors, including extras, must have valid
shape/dtype byte lengths and continuous offsets covering the file payload.
The144 F32 global scale values are not read or checked for finite/nonzero values
by this header audit. Tokenizer semantic correctness and model accuracy are also
outside its scope.

The actual standalone ELF submitted to Arena has880,872 bytes and SHA256
`1fb587ee436c5eb27099fe6177b4b04911b9b6ea09699de4a781c41bb932cf11`.
It embeds contract SHA256
`8e2b9346ff1f4ee3d58581326c85bb89f2dca5c97cb6c5e2b1fd9c3260754055`.

```bash
docker cp moa-dev/experiments/artifacts/arena-model-assets-rust-inspector/moa-model-asset-inspector moa2026-dev:/tmp/moa-model-asset-inspector
./moa-dev/moa furiosa-arena submit /tmp/moa-model-asset-inspector --name e55-model-asset-identity --entrypoint moa-model-asset-inspector --timeout 65
```

[Actual Rust job71224](../../../../docs/experiments/e55-arena-model-identity.json)
completed in5.599s. The full9,304,966,064-byte weight file SHA256 is
`7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`.
All1,149 required tensor headers and intended dtypes passed; the complete header
has1,389 tensors including240 allowed extras. The three present metadata files
also have full hashes in the result. All four files were unchanged by the
before/open/after stat checks. Top-level `partial` reflects only the absent
optional `chat_template.json`; the weight hash is complete. This later snapshot
does not retroactively prove file identity during earlier model jobs.

Build provenance and reproduction are documented in the [Rust inspector README](rust-inspector/README.md).

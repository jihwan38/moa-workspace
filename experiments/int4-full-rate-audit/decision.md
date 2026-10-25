# Public SDK 0.9 INT4 throughput audit

Decision: **HOLD the assumption that the available public INT4 path supplies twice the FP8 arithmetic rate.** Hardware supports that peak, but this audit found no demonstrated legal public SDK 0.9 contraction that reaches it. The existing K64 implementation has a measured movement benefit; the already-rejected K128 configuration remains frozen. No build, NPU dispatch, SDK modification, or model change was made for this audit.

`capture.py` independently hashes the preserved experiments and downloads the pinned public files. All eight implementation files are byte-identical to installed crates. `manifest.json` records the SDK revision, file hashes, checks and recomputed trace sums. Version is 0.9.0, public revision `5ec376347e108689096928bc4be2c0e4176fd7c9`; this report is not a claim about another SDK release.

## Hardware, public interface, and actual compiler are separate evidence

| Layer | Verified fact | What it does not prove |
|---|---|---|
| Hardware | RNGD specifies 1024 INT4 TOPS and 512 FP8 TFLOPS at 1 GHz. The primary architecture paper describes 128 INT4 versus 64 FP8/INT8 inputs per dot-product engine. | That an arbitrary public SDK tensor mapping emits the full-rate configuration, or that our Arena job runs at the specified clock. |
| Public shape checks | `config_stream_adapter` and `config_contract_packet` permit 32-byte or 64-byte storage packets. For I4 these contain 64 or 128 values. | Complete backend legality: these validators do not check the private `RawInt5/Flit2` constraint encountered below. |
| Public packet documentation | The 64-byte internal packet is half occupied with PackSize=1 and fully occupied with PackSize=2. | A working I4 K128 example; the illustrated two-flit example is BF16. |
| Measured legal implementation | K64 I4 completes correctly and faster than its FP8 control; contraction/reduction/scale span is essentially unchanged. | Twice the arithmetic throughput, activation-quantizer quality, full FFN or model performance. |
| Tested K128 implementation | Explicit K1024 padding and P128 reuse reached backend lowering but failed with `RawInt5`, `Flit2`, `invalid collect_flits`. | Universal impossibility of all I4 configurations, or a physical hardware defect. |

Hardware references: [official RNGD specification](https://developer.furiosa.ai/latest/en/overview/rngd.html), [TCP ISCA 2024 paper, Section IV-C](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24).
Public interface: [stream-adapter verifier](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-lower/src/verify/engine/contraction/outer/stream_adapter.rs), [packet verifier](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-lower/src/verify/engine/contraction/packet.rs), [Outer documentation](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/docs/src/computing-tensors/contraction-engine/outer.md).

## What the prior actual experiment establishes

`r726-native-int4-bound-submit`, job318997, source `a4d4cf7`, ELF SHA256 `47b8a9fc0eb923531bc4968c81fa5264571f3d23482867855a3e4890c7894138`:

- All 15,728,640 BF16 output comparisons, 8,192 independent integer-dot samples, 256 signed packed-byte patterns and 117,964,800 weight-nibble mappings passed.
- Completed projection: FP8 **297.667–298.327 us**, I4 **203.813–205.452 us**. This includes scale reconstruction, BF16 conversion, transpose and output stores; excludes input quantization and static weight preparation.
- Both operands are already quantized signed codes. This is an arithmetic/backend bound, not a supplied-model accuracy result.

| Cluster-0 trace span, raw profiler units | FP8 | I4 K64 |
|---|---:|---:|
| Whole Task | 577,902 | 395,731 |
| Resident weight DMA | 190,907 | 96,709 |
| Sum of Switch/TRF ingress spans | 167,182 | 40,030 |
| Sum of contraction/reduction/scaling spans | 234,056 | 233,022 |
| Sum of output transpose spans | 32,108 | 31,414 |
| Sum of output HBM spans | 37,498 | 38,762 |

The spans overlap; they cannot be added into an elapsed-time decomposition. No clock conversion is assumed. I4 uses two P128 tiles where FP8 uses four P64 tiles. Its two arithmetic spans are 116,499/116,523; FP8's four sum to 234,056. The measured improvement is supported by smaller weight delivery and greater token reuse in TRF, not by a twofold decrease in arithmetic time. The documentation's half-packet explanation is consistent with this observation; it is not an independent measurement of individual MAC utilization.

Preserved inputs: `batch-kernels/experiments/batch-kpart/{int4-bound-result.md,int4-bound-actual-trace.json,int4-bound-manifest.json}`, actual log `runs/r726-native-int4-bound-submit/arena.log`, and `src/ops_native_i4_p512_bound.rs`.

## Public alternatives examined

| Proposed route | Source-backed result |
|---|---|
| Increase I4 packet to K128 | Already tested with explicit K960-to-K1024 zeros and P128 TRF reuse. Failure is preserved in `int4-up-packet128-rejected.rs` and `logs/int4-up-compile4.log`. No repeat is justified by a claim that padding was omitted. |
| Use 16 TRF lanes | `is_valid_lane_size` permits only 1, 2, 4, 8. No public 16-lane escape was found. |
| I4 weights with I8 activations | `ContractionWeight` implements pairs within I4/I5 or I8/I9 families, not across them. There is no direct public mixed-I4/I8 contraction trait implementation. |
| Zero-point-subtracted I5 staging | A supported I4-family operand, not a documented faster mode. Its extra arithmetic bit does not create a new published packet or lane option. |
| Public I32-buffer-to-I4 raw binding | Actual r1192 validates byte ownership and uses the same typed compiled K64 consumer. It changes binding/packing, not the CE instruction or accepted packet size. The one-live-slice probe is not a full-chip throughput experiment. |
| Change packet/time/lane fold | These implement reduction and output folding; no separate public full-rate INT4 engine entry point was found. Smaller lane count reduces spatial parallelism. |

Type sources: [contraction operand pair table](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-std/src/cast.rs#L350), [lane validation](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-lower/src/verify/mod.rs#L158), [lane fold](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-std/src/engine/contraction/lane.rs).

The pinned [register-file documentation](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/docs/src/computing-tensors/register-files.md#L143) marks I4-to-I5 TRF promotion as “not wired, so an `i4` weight stays `i4` today.” This is a documented implementation gap, **not proof of the cause** of the backend rejection. The public functional TRF-sequencer validator is also a [TODO stub](https://github.com/furiosa-ai/furiosa-opt/blob/5ec376347e108689096928bc4be2c0e4176fd7c9/furiosa-opt-std/src/engine/contraction/outer/trf_sequencer.rs). Functional widening to I32 in Rust does not establish NPU I8 promotion or its speed.

The failing check is located at `crates/npu-ir/tu-ops/src/renegade/feed_buffer_multi_shape/verify.rs:34`, absent from the inspected published crates. Public source therefore does not reveal the complete supported backend type/flit matrix. The defensible unresolved question is whether this compiler version has a legal public 128-I4-per-lane example, or whether that backend combination requires a compiler repair. No external contact was made.

Proceed only if a concrete supported public example or new backend evidence resolves that question. Until then, keep the K64 measured movement benefit and F4-to-FP8 compression work separate from an unproven 2x INT4 arithmetic promise. Do not retry K128, 16 lanes, or arbitrary private bindings under a new layout name.

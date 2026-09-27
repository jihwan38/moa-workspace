# M06 v4/v5: E140 copy finding의 적용 범위

2026-09-27 read-only 감사. 최초 확인 시 main `3ca6063f588ced038be621332825b5b2ea9fd655`, baseline E142였다. 새 compile/NPU/variant 또는 원본 후보·증거 수정은 없다. [재현 스크립트](m06-copy-retrospective-20260927.py)는 자신의 위치에서 저장소를 찾고 [같은 폴더의 JSON](m06-copy-retrospective-20260927.json)에 source/build/schedule/bin SHA와 계산 결과를 저장한다. 원본 ignored binary artifacts가 있는 checkout에서 다음 명령을 사용한다.

```sh
python3 docs/research/m06-copy-retrospective-20260927.py
python3 docs/research/m06-copy-retrospective-20260927.py --check
```

현재 작업 디렉터리가 달라도 스크립트를 실제 경로로 호출하면 같은 저장소와 출력 파일을 사용한다. JSON의 evidence 경로는 저장소 기준 상대 경로다.

**결론: E140의 480B copy 수리만으로 M06 전체의 timeout을 설명하거나 재개할 근거는 부족하다.** v4는 그 세 DMA를 제거한 실행형이며, v5에만 같은 위험 source 형태가 있다. v5에는 복원 scale을 손상시킬 구체적인 조건부 배치가 있지만, 원본 actual DMA descriptor가 보존되지 않아 E135처럼 실제 1440B write와 live bitmap overwrite를 확정할 수 없다. v4/v5 모두 Arena 첫 FFN launch timeout일 뿐, 실패 operator trace/후속 SRAM 값은 없다.

기본 경로는 [ops.rs](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/sharded-packed-v4/ops.rs)의 227–233행 canonical residual → H480 RMS input DMA → `normalize_sharded_inter_slice` → `mlp.rs:397` quantizer → `broadcast_ffn_quantized:442` packed terms를 ring32 seed로 전달/확산 → `project_up_and_gate:81` → M03 GeGLU/down이다. v4/v5는 이 새 sharded-RMS/quantizer 경로를 공유한다. 이를 이름만 보고 고장 원인으로 단정할 수 없다.

## 확인된 서로 다른 source 경로

원본 후보: [M06 기록](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/README.md).

- **v4 [mlp.rs](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/sharded-packed-v4/mlp.rs), 510–559행**: `[FfnTerm3,H%480]` packed allocation에 세 Main VE `commit_view`가 직접 기록한다(521/537/559행). 각 source stream은 H480 =60×8B commit이며 논리 offset0/480/960; 1440B를 정확 채우는 계약이다. Schedule에는 해당 3개의 `DmaStos`가 **없다**. Main O162/O175/O192 및 core index-write가 존재한다. 따라서 E140에서 수리한 `hi/lo/third.to_dm_view` 오류를 v4 원인으로 적용하면 틀린다. Graph/vISA가 없어 actual commit descriptor까지 검증했다는 뜻은 아니다.
- **v5 [mlp.rs](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/sharded-dma-terms-v5/mlp.rs), 510–563행**: 별도 480B hi/lo/third 및 Sub sum을 복원하고 557–562행에서 `[FfnTerm=1#{!}3,H%480]` mutable tile로 typed DMA3개를 쓴다. E140/E135에서 실제 1440B read/write가 확인된 source 구조와 같다. Schedule O195/O200/O207도 이 세 DMA를 확인한다. 그러나 이 schedule에는 DMA의 address/stride/count descriptor가 없고 mutable-view output tensor 자체도 address table에서 생략되어 있다.

## 기존 schedule에서 확정되는 배치, 조건부로만 계산되는 overrun

Schedule Sram address/size는512slice 전체 표기이며, 기존 E135 graph/schedule 교차검증과 같은 convention으로512로 나눈 per-slice 주소다. 아래 시간은 **static schedule**이며 NPU profile이 아니다.

| 항목 | v4 | v5 |
|---|---:|---:|
| packed allocation | [26880,28320) | [28928,30368) |
| ring32 bitmap,32B/slice | [0,32) | [27392,27424) |
| ring32 bitmap lifetime |34902–49972 |29394–60065 |
| final input restore scale,32B/slice |[30464,30496) |[30464,30496) |
| scale distribution DMA |O224,35621–40373 |O203,30606–35358 |

v5 IF the same1440B copy extent was emitted:

| copy | 480B source base | destination write | past packed | static time |
|---|---:|---:|---:|---:|
| hi O195 |24576 |[28928,30368) |0 |28926–29394 |
| lo O200 |26368 |[29408,30848) |480 |30113–30581 |
| third O207 |26880 |[29888,31328) |960 |35358–35826 |

- None of those destination ranges overlaps the ring32 bitmap27392. **E135와 같은 bitmap overwrite는 아니다.** Third의 source overread는 bitmap을 읽을 수 있지만, 읽기와 control bitmap 덮어쓰기는 다르다.
- Third conditional write includes restored x_scale30464–30496 **after** scale DMA completion35358 and **before** x_scale StoVRF O22436958–37225 / O22937225–37492. This is a concrete potential numeric-input hazard. Lo's earlier conditional overwrite is overwritten by the later scale DMA. It is the **up input restore scalar**, not the later down_global_scale.
- v5 actual down_global_scale loads only at O549126758–127245 into12544–12576; it is not the same live scalar overlap demonstrated for E140local-v4.
- These conditional ranges are not recovered descriptor evidence and do not prove the timeout mechanism, payload bits, or SDK fault. The v5 source pattern deserves repair before future reuse, but that is weaker than attributing the historical liveness failure.

## Evidence and API boundary

[v4 build](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/build-m06-sharded-packed-v4-repro.json) / [v5 build](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/build-m06-sharded-dma-v5-repro.json), [v4 schedule](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/evidence/m06-sharded-packed-v4/schedule.json) / [v5 schedule](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/evidence/m06-sharded-dma-v5/schedule.json), and ignored `artifacts/m06-sharded-input-pipeline/{label}/exact-kernel.bin` all remain and hashes match. Whole artifacts and container `/tmp/moa-c02-c03/{label}` contain schedules/binaries but no graph/vISA. Image metadata parser and raw bitmap constant search do not decode TDMA packet address/count. Both exact binaries uniquely contain the expected512×32B ring32 table; this only proves embedded constant contents, not its eventual DM state.

Installed SDK0.8.1 `src/tensor/memory.rs:1042–1062` marks `DmTensorView::to_dm_view` as a primitive, checks dimension/alignment and invokes `dst.inner.transpose`. Mutable view offset/base_map is preserved by `src/tensor/view.rs:173–186`. This explains why padded destination/window contracts matter. The Rust reference transpose's live-cell clipping is **not** proof the NPU lowering clips a DMA packet; historical actual E140/E135 descriptors show why the emitted boundary must be inspected. Installed memory.rs SHA `fec623ddf8fadd5cf31b32beb747243c2c463c48e4f73bb29bb2b6ef5d1aaff1` (matching `/tmp/moa-opt-std-release/...`).

[README](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/README.md)의 87–132행, [decision](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/decision.json), [104566 raw log](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/evidence/arena-104566.log) / [104588 raw log](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/evidence/arena-104588.log) preserve v4 E119 and v5 E120 first FFN launch timeouts, unchanged QKV/O12 PASS and45 successful control comparisons each. No candidate FFN arithmetic/performance result exists.

**Priority implication:** preserve M06 as unresolved. Do not call both failures a now-fixed480B bug or interpret v5 as a clean isolation that exonerates v4's direct packing. A future sharded pre-RMS plan must justify savings against current E142 complete readiness/costs and have a supported producer/consumer contract; merely applying E140's three-copy repair does not localize v4. No new experiment is selected by this note.

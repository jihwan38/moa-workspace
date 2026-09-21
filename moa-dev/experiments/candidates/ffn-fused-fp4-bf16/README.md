# FP4를 중간 FP8 DM 없이 BF16으로 materialize

**사용하지 않는다. Root의 관련 fused-conversion 경로 E13/Arena 70162가 NPU에서 실패했다:** FFN run0 within-tolerance 30.96%, max error 0.63281, 이후 run1 launch timeout. 이 폴더의 일반 vector-dequant 전체 후보 자체를 NPU 실행한 것은 아니지만 같은 fused decode 경로의 실측 실패 때문에 안전한 대안으로 취급하지 않는다. API 허용과 CPU 값 변환 증명이 실제 fused hardware 동작의 정확성을 보장하지 않았다.

상태: 독립 FFN 컴파일 성공. 일반 vector-dequant 경로에서는 정적 makespan이 **204460 → 228739**로 11.88% 증가하여 **채택하지 않는다**. NPU 미검증이다. 이미 full BF16 unscaled weight를 저장하는 별도 block-scale 후보에 이 구현을 공유했으나, 이후 위의 NPU 실패가 확인되어 FP4→FP8 DM→BF16 LUT DM의 검증된 두 단계 변환으로 되돌리기로 했다.

현재 SDK 0.8.1은 `fetch_table_lookup::<bf16>()`를 FP4에 직접 적용하지 못한다. FP4 table은 FP8 출력만 지원하며, `PositionFetchTableLookup`에서 두 번째 `fetch_table_lookup`을 호출할 수도 없다. 대신 공개 API의 다음 순서가 허용되고 실제 컴파일에 성공한다.

```rust
.fetch::<m![L % 32], m![H]>()
.fetch_table_lookup::<f8e4m3>()
.fetch_cast::<f32>()
.collect::<m![L % 32, H / 8], m![H % 8]>()
.cast::<bf16, m![H % 8 # 16]>()
.commit_trim::<m![H % 8]>()
.commit()
```

원래 full-matrix FP8 materialization을 이 BF16 materialization으로 바꿨다. Up/gate/down의 이후 local-scale F32 multiplication과 BF16 dequant 경계는 유지한다. `numerical-proof.json`은 SDK `F4E2M1_TO_F8E4` table의 16개 코드 전체를 대조해 F32→BF16이 모두 exact이고 -0도 보존함을 확인한다. 그러므로 이 새 cast 자체는 FP4 값을 반올림하지 않는다. 하드웨어 전체 정확성은 별도 검증 대상이다.

정적 DMA union은 132947로 같지만 Main union이 168570→191130으로 늘었다. Up/gate의 full unpack 각 9863→17543, down unpack 9263→16463으로 총 22560 cycles가 늘어난다. BF16 중간 저장 폭이 FP8의 두 배이고 FP32 확장·BF16 cast가 추가된 경로이므로 중간 tensor 제거만으로 속도가 좋아진다고 판단하지 않는다.

SDK 근거는 Docker의 `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src/` 기준이다.

- `cast.rs:43–90`: `TableLookupCast`의 FP4→FP8와 FP8→BF16 구현만 존재.
- `engine/mod.rs:261–274`: lookup 이후 fetch-cast 허용, lookup 재진입은 불허.
- `engine/mod.rs:312–323`: collect 뒤 cast 허용.
- `engine/fetch_adapter.rs:177–217`: main-context lookup, paired FP4 decode 뒤 F32 fetch-cast 설명.
- `scalar.rs:918–940`: 16개 FP4 값과 실제 FP8 lookup table.

관련 QKV/O 조사도 같은 SDK로 확인했다. `cast.rs:35–42`에서 `FetchCast<f32>`는 FP8에 구현되어 있지만 `FetchCast<bf16>`는 FP8에 없다고 명시한다. F32 입력 contraction도 허용되지 않고(`cast.rs:307–326`), float contraction은 같은 dtype끼리만 가능하다(`cast.rs:344–377`). 따라서 FP8 weight와 BF16 activation의 기존 경계를 유지한 채 direct `fetch_cast::<bf16>()`로 LUT를 지우는 후보는 만들지 않았다.

`mlp.rs`, `change.patch`, `mlp.f8-materialized.rs`, `numerical-proof.json`, `static-comparison.json`에 코드와 근거를 보존했다. Scratch `/tmp/moa-ffn-fused-fp4-bf16`, 별도 target `/tmp/moa-ffn-fused-fp4-bf16-target`에서 FFN 하나만 컴파일했다. [Schedule](../../artifacts/ffn-fused-fp4-bf16/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-fused-fp4-bf16/compile.log). live source, 공유 target, NPU, Git에는 변경하지 않았다.

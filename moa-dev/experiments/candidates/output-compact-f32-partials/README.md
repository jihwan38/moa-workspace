# O FP32 column partial 압축: 두 compile 실패를 보존한 설계 검토

**채택 가능한 후보가 없다.** E31 원본 O를 별도 scratch에서 exact compile해 재현했지만, 2행과 4행 FP32 압축안 모두 transpose 제약으로 실패했다. `projection.rs`는 마지막 **실패한** 4행 안이다. NPU를 실행하지 않았고 후보 binary/schedule은 생성되지 않았다. 원본 커널과 실패 파일을 혼동하지 않아야 한다.

검토 기준은 `artifacts/e31-switch-gather-rms/src`다. 원본 O makespan은 직접 확인한 **24,997 cycles / 52 instructions**이며, 재생성한 88,576B 커널 SHA256 `e9d3402c62fd866f417053e618e7a7af1d5cb60485c04cc292f3a702f5996cde`가 E31 고정 커널과 일치한다. 이 기준은 이전 E26, E29 또는 O row-interleave 후보의 추정값이 아니다.

## 호출·산술·메모리 경계

E31 `ops.rs::sliding_attention_output`은 `project_output` → `normalize_output_fused` → `add_output_residual` → 같은 HBM residual write를 호출한다. 이번 파일은 `projection.rs:283`의 `project_output` 본문만 수정하며 QKV, quantizer, channel scale, RMSNorm, residual helper 및 ops는 원본과 같다. 새 unsafe reshape는 없다.

O 입력은 Qs4096을 512값씩 8개 열로 나눈다. row는 `h=1920*c+60*g+r`이고 `c=0..1`, `g=0..31`, `r=0..59`다. physical slice는 `8*g+column`이다. 각 column이 가진 입력의 runtime power-of-two scale `s[column]`은 서로 다를 수 있다. 같은 weight row에 대한 FP8 hi/lo DPE 결과를 `a`, `b`라 하면 원본 수치 순서는 다음과 같다.

```text
각 column: p = F32(F32(a + F32(b * 1/16)) * s[column])
8개 column의 F32 inter-slice sum → BF16
→ channel scale → BF16 → RMSNorm/gamma → BF16 → residual add → BF16
```

`project_output:302–318`은 pair 합과 column scale까지 수행해 F32 DM에 저장한다. `320–330`은 새 Main/VE pass에서 8개 column만 합한다. 중간 DM 저장은 BF16 반올림 경계가 아니다. 이 F32 boundary를 없앤다고 BF16 cast까지 제거해서는 안 된다.

원본 partial은 `DmTensor<f32,...,OutputColumns,m![H%60,1#8]>`다. 논리값은 row당 하나지만 physical packet은 8 F32이므로 slice당 **60×32=1,920B**, 512 slices 전체 **983,040B**다. 실제 schedule의 T29 size와 일치한다. compact H60이면 240B/slice, 전체 122,880B가 된다. 이 메모리 계산은 설계상의 절약량이며 성공한 후보의 실측값이 아니다.

## 한 pass로 합치지 않은 이유

SDK0.8.1 실제 구현을 확인했다. 아래 경로의 기준은 Docker의 `furiosa-opt-std-0.8.1/src/engine/vector/`다.

| API/typestate | 확인한 구현과 제약 |
| --- | --- |
| `op/mod.rs:219`, `FpBinaryOp::alu` | `AddF`/`SubF`는 `FpFma` ALU를 사용한다. |
| `tensor/vector_tensor_pair.rs:1235`, `vector_fp_zip` | pair 합 역시 이 `AddF` 자원을 사용한다. 기존 직결 시도는 backend에서 `no available alu for op GlobalReduce(GlobalReduceAddF)`로 실패했다. [기존 로그](../output-native-fp8-two-lane/attempt-1.log)를 그대로 근거로 삼았다. |
| `op/mod.rs:414`, `ClipBinaryOpF32::alu` | `ClipBinaryOpF32::Add`는 별도 `ClipAdd` ALU다. |
| `tensor/vector_tensor_pair.rs:1353`, `vector_clip_zip` | Way8 F32 pair를 ClipAdd로 합치는 frontend API가 있다. |
| `stage/markers.rs:300–351` | Clip→InterSliceReduce는 허용되지만 Clip→Fp는 허용되지 않는다. ClipAdd로 pair를 합친 **뒤** 원본 Mul1 column scale을 적용할 수 없다. |
| `tensor/vector_tensor.rs:505–530` | IntraFirst/Way8/commitable stream의 inter-slice sum은 가능하다. frontend type 성립이 backend ALU 배치 성공까지 보장하지는 않는다. |

따라서 `hi + lo/16`를 ClipAdd로 바꾼다는 API 치환만으로 원본 순서를 보존하는 한 pass가 완성되지는 않는다. 먼저 각 pair에 column scale을 적용하면 `a*s + (b/16)*s`로 식이 바뀐다. power-of-two scale이어도 중간 overflow/underflow나 cancellation까지 포함한 임의 입력의 F32 동치를 보장하지 않는다. 또한 pair를 8-column sum 뒤로 옮겨도 서로 다른 `s[column]`을 먼저 적용해야 하며 column 합과 pair 합의 순서를 바꾸게 된다. 이번 범위에서는 이 재결합을 구현하지 않았다. 별도 ALU가 보인다는 이유로 한 pass의 장치 지원·정확성을 주장하지 않는다.

## 시도한 compact layout과 정확한 실패

1. **2행 F32/8B commit**: 동일 pair/scale stream의 F32 값을 `Time H/2%30`, `Packet H%2#8`로 transpose해 padding을 trim한다. 다음 pass도 두 row를 동시에 읽고 같은 8-column Add를 적용한다. 마지막 BF16은 2값=4B여서 8B write 정렬을 만족시키기 위해 기존 4행 packet으로 합치려 했다. Rust type checking은 통과했지만 마지막 transpose가 다음 오류로 실패했다.

   ```text
   Transpose row evidence (H % 4) is not present in Time (H / 2 % 30): Unconsumed([H % 2])
   ```

   낮은 H bit가 이미 Packet에 있는데 target `H%4`를 Time 쪽 행축으로 요구한 결과다. 원본·후보의 row/column 30,720개 논리 좌표는 각각 유일하며 `r=2*(r/2)+r%2`로 복원되지만, 그 수학적 mapping만으로 SDK transpose의 분해 제약을 만족하지 못한다. 소스는 `projection.two-row.failed.rs`, 상세 evidence는 `attempt-1-evidence.json`에 남겼다.

2. **4행 F32/16B commit**: 추가 한 변형을 허용받아 `Time H/4%15`, `Packet H%4#8`로 첫 transpose를 바꾸었다. column sum 뒤 BF16 4값=8B를 직접 commit하면 마지막 transpose가 필요 없다. 그러나 이번에는 첫 F32 transpose 자체가 거부되었다.

   ```text
   Transpose: output packet beyond max_in_rows (2) must be `1 # n` pure padding, got: H / 2 % 2 # 4
   ```

   이 경로의 실제 compiler 제약은 **max_in_rows=2**였다. 검토 중 F32 4행을 가능하다고 추정한 내용은 이 결과로 철회한다. 임의로 padding을 유효 row처럼 reshape하지 않았고 더 컴파일하지 않았다.

다음 검토가 필요하다면 2행 F32 저장을 유지하면서 final BF16 packet의 low-row 축과 transpose 행축을 명시적으로 분리하거나, 2 BF16값과 실제 write padding을 포함한 8B DM packet으로 저장해 기존 typed DMA가 유효 row만 모으게 하는 방법이 있다. 모두 **미구현·미컴파일 설계안**이다. 기존 2-row FFN 실패 이력도 있으므로 첫 NPU screen만으로 끝내지 않고 추가 입력과 원본 출력 비교가 필요하다.

## 병목과 재현 범위

`baseline-profile.json`에 실제 E31 노드를 기록했다. weight DMA는 13,512 cycles, pair+scale contraction은 1,238 cycles, fresh column sum은 **390 cycles**다. 그 뒤 epilogue layout copy는 469 cycles다. 따라서 현재 두 pass 중 두 번째 pass 자체는 전체 24,997 cycles의 약1.56%이며, 중간 DM 8배 압축이 O 전체 8배 가속을 뜻하지 않는다. 전체 schedule의 allocation/overlap 변화까지 exact compile과 NPU로 확인해야 한다.

raw 로그·원본 schedule·원본 binary는 [artifacts/output-compact-f32-partials](../../artifacts/output-compact-f32-partials)에 있다. Docker scratch `/tmp/moa-output-compact-f32-partials`, 독립 Cargo target, `CARGO_HOME=/tmp/moa-extra-seeds-e31-rms-extra/cargo-home`을 사용했다. Cargo source는 E31 frozen snapshot이며 `source-manifest.json`은 마지막 실패 변형의 파일 hash를 기록한다. 빌드 명령과 로그 hash는 `evidence.json`에 있다. live source, main target, Git index/commit과 NPU는 변경하지 않았다.

# FFN pre-RMS shard 유지 후보 검토

두 후보는 public kernel 계약과 residual/FFN 후단을 보존하지만, **단순 layout 변경이나 bit-exact 변경이라고 설명하면 안 된다.** 첫 후보도 EPS addition의 ALU와 pass 경계가 바뀐다. 두 번째 `ops.gather.rs`는 여기에 8개 partial mean의 FP32 합산 방식까지 바꾼다. 명백한 값 누락·padding 승격·residual 덮어쓰기는 소스에서 발견하지 않았다.

Root 보고: inter-slice 후보 exact compile 162,735 cycles, gather 후보 164,216 cycles, E53 기준 162,209 cycles다. 각각 +0.324%, +1.237%여서 둘 다 NPU 미실행/미채택이다. 이 review는 schedule 비용을 독립 재계산하거나 compile/NPU를 실행하지 않았으며 소스/API/dataflow만 검토했다.

## 비교 원본과 호출 범위

원본은 `../../artifacts/e53-bf16-output-clean-source/`다. 후보 `ops.rs.original`/`mlp.rs.original`이 이 snapshot의 해당 파일과 byte-identical임을 확인했다.

| 파일 | 검토 SHA256 |
| --- | --- |
| ops.rs.original | `f2ad258f736dd8f3ebb64b4dc8c7b0e15579bded22e41a40ee355f4674f42c6a` |
| mlp.rs.original | `331088a000ca972d8fdad97c76d9d6ff093a3712bc2cda95e9a3b15fa05856e1` |
| ops.rs: inter-slice | `ac46cb8167eb9f62243fb4b2c7f51571b67187793a8c939bb4f812ba9c888d15` |
| ops.gather.rs | `a19a15cf9bfe3d7b6729c0cd8457e5a114ccb4a5af0b75b5f1f5053afe5f706b` |
| mlp.rs: 두 후보 공통 | `b93ec3e481a776b23638151c4ff298db67fe1221dbb118b092f080a110e8ad7b` |

`ops::decoder_feedforward`의 `#[device(chip=1)]`, 인자 순서·dtype·mapping·mutability와 반환 계약은 그대로다([ops.rs](ops.rs:202)). 이 public entry의 실제 호출자는 공식 `src/bin/test_kernels.rs:603`과 전체 모델 `src/host/runtime.rs:552`다. 후자는 sliding/full attention layer 뒤의 공통 `mlp` 호출이므로 전체 모델에서는 양쪽 decoder layer의 FFN에 영향을 준다. host/fixture/reference/harness는 후보 범위 밖이다.

`shared::mlp::feedforward_sharded`는 private(crate) input 타입만 `[Slice,H]`에서 `[DownEpilogueSlices,H%480]`으로 바뀐다([mlp.rs](mlp.rs:168)). 전체 E53 src를 검색하면 호출자는 `decoder_feedforward`와 같은 파일의 compatibility wrapper `feedforward` 두 곳이다. wrapper는 기존 `[Slice,H]` signature를 유지하고 BF16 typed DMA로 H480 shard를 만든 뒤 새 helper를 부른다([mlp.rs](mlp.rs:150)). 현재 src에 wrapper 자체의 호출자는 없으므로 Stage1 경로는 이 추가 변환을 거치지 않는다.

`decoder_feedforward` 이전의 다른 ops 전체, post-FFN norm부터 파일 끝까지의 suffix, MLP의 hierarchical DMA statement부터 파일 끝까지의 suffix가 각각 byte-identical이다. 따라서 QKV/O 계산이나 weight decode/scale/GeGLU/down projection을 이 patch가 직접 바꾸지 않는다. `rmsnorm.rs`/`residual.rs`를 편집하지 않고 이미 존재하던 helper를 다른 위치에서 호출한다. `ops.gather.rs`는 첫 후보와 비교해 pre-RMS helper 이름 한 곳만 다르다.

## 숫자 경계: 세 RMS 경로

관련 구현은 동결 [shared/rmsnorm.rs](../../artifacts/e53-bf16-output-clean-source/src/device/shared/rmsnorm.rs:9)의 `normalize`, `normalize_sharded_inter_slice`(:203), `normalize_output_fused`(:107)다. 세 함수의 local `mean_square` statement 전체가 문자 단위로 같음을 확인했다. 각 shard는 **480개 x²의 F32 합을 H=3840으로 나눈 값**을 만든다. `/480` 뒤 평균8을 내는 변경은 없다. EPS는 같은 `src/lib.rs:25`의 F32 `1e-6`이다.

| 단계 | E53 기존 generic normalize | 첫 inter-slice 후보 | 두 번째 gather 후보 |
| --- | --- | --- | --- |
| BF16 입력 분할 | 함수 안에서 H480×8 typed DMA | ops에서 같은 typed DMA | 첫 후보와 같음 |
| local mean | BF16→F32, 제곱, local sum, `/3840`, F32 DM | 같음 | 같음 |
| 8 partial 합 | `vector_inter_slice_reduce(Add)` | 같은 API·input/output mapping | Switch로 유효8개를 한 slice Time축에 모아 `vector_intra_slice_reduce(Add)` |
| EPS/sqrt | Way8 `vector_clip(Add,EPS)` → F32 DM → 별도 pass Way4 Sqrt → F32 DM | Way4 `vector_fp_binary(AddF,EPS)` → Sqrt를 같은 pipeline에서 실행 → F32 DM | global sum을 F32 DM에 commit한 뒤 별도 pass의 AddF→Sqrt → F32 DM |
| normalize/gamma | F32 DivF(rms) 뒤 Mul0(gamma) | 같은 순서·타입 | 같은 순서·타입 |
| 첫 BF16 cast | normalized F32 DM을 단일 slice로 Switch gather한 뒤 cast | shard별 normalized pipeline 마지막에서 cast | 첫 후보와 같음 |

설치 SDK0.8.1의 `engine/vector/op/semantics.rs:167–175,206–210`은 AddF와 Clip Add 모두 `a+b`로 정의한다. **Clip Add가 값을 clamp하는 변경은 아니다.** 하지만 `engine/vector/op/mod.rs:230–236,439–447`의 실제 ALU 선택은 AddF=`RngdAlu::FpFma`, Clip Add=`RngdAlu::ClipAdd`로 다르다. `vector/tensor/vector_tensor.rs:1309–1334,1760–1790`에서도 전자는 Way4 FP stage, 후자는 Way8 Clip stage다. 같은 host 수식만으로 서로 다른 NPU ALU의 rounding/FTZ/exception 처리나 fused-pass 결과가 bit-identical하다고 증명할 수 없다. AddF→Sqrt는 두 연산을 같은 pipeline에 놓는 것이며 단일 수학 intrinsic 또는 FMA로 바꾸었다고 부르지 않는다.

첫 후보의 inter-slice reduction mapping은 원본과 같으므로 source-level 합산 구조는 유지한다. 두 번째 후보는 global reduction의 실행 단위와 묶음이 달라 FP32 addition tree/order 변화가 가능하다. `normalize_output_fused`의 8 partial gather 뒤 EPS/Sqrt를 별도 pass에 둔 이유도 중요하다: local intra-reduce는 FP math stage 뒤에 있어 같은 pipeline 앞단에 EPS를 옮기면 총합에 EPS를 한 번 더하는 수식이 되지 않을 수 있다. 후보는 기존 helper를 그대로 사용해 이 구분을 보존한다.

BF16 cast는 **gamma 적용 뒤 한 번**이라는 논리 경계를 유지한다. 원본의 F32 commit→Switch gather는 그 사이 산술이 없으므로, 해당 F32 값들이 같다면 element-wise BF16 cast를 shard 쪽으로 옮기는 것은 수식상 같은 경계다. 그러나 앞선 RMS ALU/pass 변경까지 포함한 실제 숫자의 동일성을 이 관찰로 대신할 수 없다. 낮은 RMS/EPS 부근, BF16 midpoint 근처, 큰 유한 값에서의 F32 square overflow/작은 값 underflow는 별도 수치 검증 대상이며 모든 입력 보장은 없다. 일반 finite 입력에서 RHS는 원래와 같은 `gamma*x/sqrt(sum(x²)/3840+EPS)`다.

## 좌표·복제·padding

`H=3840`, `DownEpilogueSlices=m![1#32,H/480]`이므로 유효 shard 8개와 shard당480값이다. 원래 원소 h는 `(shard=h/480, local=h%480)`로 일대일 대응한다. 1#32는 physical padding이며 논리32개 복사로 해석하지 않는다. Cluster는 기존 `m![1#2]`, 단일 유효 cluster다.

`DmTensor::to_dm`은 SDK `tensor/memory.rs:955–965`에서 dtype을 유지하며 mapping 검증 후 transpose/copy한다. 새 ops의 `residual.to_dm` 및 MLP assembly는 BF16→BF16이므로 이 변환 자체에 cast/산술은 없다. 기존 normalized F32 전체 gather를 제거하지만 **full-H assembly 자체가 없어지는 것은 아니다.** [mlp.rs:183](mlp.rs:183)의 typed DMA가 8개의 BF16 shard를 H3840으로 모으면서 두 cluster의 16개 anchor에 복제한다.

그 뒤 원래 그대로인 `CustomBroadcast { ring_size:16 }`이 anchor를 각16-slice ring에 복제한다. 2 clusters×256 slices가 전체 H 벡터의 실제 copy를 가진 후, 기존 view가 up/gate row mapping `L/32%240#256`에 맞춰 cluster당 마지막16개 copy를 사용하지 않는다. 이미 실제 값이 있는 복사본을 버리는 방향이며 uninitialized padding을 유효 값으로 승격하지 않는다. 유효 weight row는 여전히 `2×240×32=15360`이다.

RMS scalar의 `Dummy8→H/480` unsafe reshape도 기존 helper의 그대로다. inter-reduce 후 8개 logical position에 같은 전체 RMS가 복제되어 있고 양쪽 shape/wire order가 같은 경우의 이름 변경이다. gather helper는 8개의 유효 partial만 Time으로 모은 후 scalar를 typed DMA로 다시 배포한다. 새 입력 path에 unchecked reshape를 추가하지 않았다. 컴파일 성공은 SDK layout lowering의 수용 여부이며 실제 device copy/lifetime 정확성까지 증명하지 않는다.

## residual 및 출력

입력 `residual_hbm`은 처음 BF16 DM `residual`로 읽는다([ops.rs:219](ops.rs:219)). 새 shard `rms_input`은 readonly borrow를 통한 별도 typed conversion이며 정규화된 x로 원래 residual을 덮어쓰지 않는다. x의 ownership은 `feedforward_sharded`로 이동하지만 residual은 마지막까지 별도 binding으로 남는다.

post-FFN `normalize_output_fused` → `add_and_scale_sharded(&x,&residual,layer_scalar)` → HBM write는 원본과 byte-identical([ops.rs:237](ops.rs:237)). 기존 [shared/residual.rs:189](../../artifacts/e53-bf16-output-clean-source/src/device/shared/residual.rs:189)는 residual을 H480 shard로 읽고 `BF16(x+residual)`을 먼저 commit한 다음 layer scalar를 곱해 다시 BF16으로 commit한다. 두 BF16 경계 및 연산 순서가 유지된다. HBM mutation은 기존 마지막 `to_hbm_view` 한 곳이며 cache나 다른 HBM tensor를 변경하는 경로를 추가하지 않았다.

후보의 실효 목적은 normalized F32 전체 gather 및 추가 scalar pass를 줄이는 것이다. 실제 전체 makespan은 다른 연산의 overlap/lifetime에 좌우되므로 일부 단계 비용 감소를 전체 개선으로 표현하지 않는다. 현재는 두 후보 모두 compile-only이고 수치·NPU 정확도 및 성능 통과 주장은 없다. 검토일 2026-09-21. 이 파일 외 candidate/live source, Git, build, NPU는 변경하지 않았다.

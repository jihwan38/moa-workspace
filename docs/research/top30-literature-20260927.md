# E134 이후 실행 구조 선택에 쓴 문헌·SDK 근거

2026-09-27 열람. 비교 기준은 E134 source `4f8fe7f`, 설치 SDK0.8.1이다.
아래 세 원리는 [선택한 전체 경로](top30-paths-20260927.md)를 뒷받침한다.
논문의 타 장치 성능 수치나 제거한 논리 바이트를 RNGD의 예상 이득으로 옮기지 않는다.

## 1. 큰 weight의 dtype 변환 대신 작은 activation을 분해한다

[Multi-Scale Dequant](https://arxiv.org/html/2605.13915v1) §§3.1,3.8–3.9
(arXiv v1, 2026-05-13)는 activation을 여러 저정밀 항으로 나눠 같은 weight에 곱하고
복원하는 구조를 제안한다. 이 논문의 FP8 오차 상한은 Table4에서 TBD이며, MXFP4 정의도
Gemma의 E2M1과 다르다. 따라서 적용할 것은 **변환 대상의 교체**이고 수치 보증은 아니다.
[FP8 형식 논문](https://arxiv.org/pdf/2209.05433) (2022-09-12)의 E4M3 유한 최대448,
최소 subnormal2^-9는 설치 `src/float/f8e4.rs`에서도 확인했다.

RNGD `ContractionWeight`의 부동소수점 조합은 E4×E4, E5×E5, BF16×BF16이다.
E4 weight와 BF16 activation의 native mixed contraction을 가정할 수 없다.
E136의 가설은 실제 BF16 activation을 weighted E4 네 항으로 분해하고, 원본 E4 weight와
native dot한 F32 항들을 복원하여 weight→BF16 lookup 경로를 없애는 것이다.
[Outer 문서](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html)의
Broadcast 절은 TimeBroadcast가 같은 fetch packet을 재사용한다고 명시한다. 항 수가 늘면
contraction 작업은 늘지만, 그것만으로 HBM weight를 항 수만큼 다시 읽는다고 세면 틀린다.

독립 CPU 검토에서 normalized BF16 유효숫자 격자19,328개에 대한
`[1,16,2^22,2^27]` 증폭은 모든 cast 입력을448 이내로 유지했고 최대 복원오차는2^-37였다.
이는 논문 결과가 아닌 별도 계산이다. E47 상쇄 반례는 복원하지만, scalar 복원이 정확해도
native F32 합산순서·runtime scale 선곱·최종 RMS 오차까지 같아지는 것은 아니다.
`hi*s`가 overflow한 뒤 음수 residual이 상쇄하지 못하는 극단 사례도 있다.
그래서 실제 pre-RMS 입력부터 분해·broadcast·F32 복원·channel scale·BF16·기존 후단까지
포함하고 과거 반례를 실제 이미지에서 검증해야 한다.

## 2. TRF에 둘 operand와 출력 lane을 함께 바꾼다

[Furiosa GEMV 설계 예제](https://developer.furiosa.ai/furiosa-opt/book/quick-start/kernel-design.html),
[Outer의 Lane/Operand 절](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/outer.html),
[register files 문서](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html)는
resident operand와 stream operand의 역할, Lane1/2/4/8, plain DM→TRF를 확인하는 직접 근거다.
Q8행×3840=30,720B/slice, KV4행=15,360B/slice의 **원본 E4 weight를 Sub→TRF**에 두고
작은 activation을 stream하면 lane을 output row로 쓸 수 있다. C10이 실패한 Main
BF16 lookup→StoTRF edge를 통과하지 않는 것이 이번 조합의 구체적인 차이다.

대가는 weight 전체 Sub 적재, activation4항 생성·전파, TRF 수명/공유 bank 사용,
term 합산 뒤 channel scale을 위한32KiB F32 row handoff다. TRF 용량에 들어간다는 것은
lowering 성공이나 bank stall 감소의 증명이 아니다. O의 단순 Row15 time-broadcast안은
term reducer의 inner-time8-slot 제약에 걸리고, 순서를 바꾼 표현은 activation15회 재읽기를
추가한다. 따라서 먼저 QKV의 Q8/KV4 구조를 검증하고 O에 무료로 전이하지 않는다.

[SPECTRA](https://arxiv.org/html/2609.24847v1) §3.1.1 (2026-09-21)는 FPGA의8×8 array를
GEMV dot lanes로 재구성한다. 참고할 것은 batch1의 유효 출력 병렬성이며, RNGD가 동일한
2D systolic padding 손실을 갖거나 FPGA의 개선율을 회수할 수 있다는 주장은 아니다.

## 3. decode producer와 fused 소비자를 연결하되 projection assembly를 없앤다

[Gemma3/AMD FusedDQP](https://arxiv.org/html/2602.06063v1) §3.2.1 (2026-02, v1)는
decode와 projection을 같은 compute tile에 배치한다. Q4NX/offline 재배열은 MOA의 고정
입력에 그대로 쓸 수 없다. [Ascend W4A16](https://arxiv.org/html/2601.16536v1) §§3,4.2
(2026-01-23, v1)은 vector→cube global-workspace 전달 비용을 드러낸다. RNGD DM은
on-chip이므로 이것을 HBM 왕복이라고 부르면 안 된다. 공통점은 producer가 만든 값을
consumer가 요구하는 형식과 위치로 전달하는 **전체 경계 비용을 포함**해야 한다는 것이다.

E135는 up/gate 각각의 compact FP4 DM에서 `fetch_table_lookup<E5>`→native dot→
input scale→block scale→F32 H reduction을 직접 연결한다. E131의 packed joint 복사와
scale assembly를 추가하지 않으면서 E134의 전체 decoded up/gate weight DM을 제거하는
조합이다. Native10회는 유지되지만 lookup 준비도10개 stream에 생기므로 비용이 사라지는
것만 세지 않는다. 실제 emitted graph의 row6 view와8B F32 commit, TRF/VRF 수명,
후단 canonical gather를 먼저 검사하고 whole FFN에서 비교한다.

설치 SDK의 `engine/vector/op/mod.rs`는 intra-slice Add를 `ReduceAccTree`로 내리며,
`engine/vector/tensor/vector_tensor.rs:510–524`는 IntraFirst→widen→inter-slice reduction을
지원한다. 현 E134 down에서도 사용한다. 이것은 과거 E47의 pair `AddF` 경로와 다르다.
반대로 term intra-reduce 뒤 ordinary FP Mul로 되돌아가는 stage API는 없으므로 QKV의
channel scale 후처리 비용을 숨기면 안 된다. `cluster_tile`과 `begin_interleaved`도 공개
API 이름만으로 target lowering을 보증하지 않는다. 보존된 실패는 그대로 적용한다.

## 그대로 채택하지 않은 최근 대안

| 직접 확인한 1차 자료 | 이 작업에서의 적용 한계 |
| --- | --- |
| [LiquidGEMM](https://arxiv.org/html/2509.01229v1) §§4–5, 2025-09-01 | operand 방향과 decode pipeline의 공동 설계는 참고한다. LiquidQuant offline 변환, CUDA PTX/TMA는 현재 제출 입력·SDK에 없다. |
| [FASQ](https://arxiv.org/html/2605.04084v1) §4.1/Table1, arXiv v1 | 자기 batch1 실험에서 direct compute가 LUT보다 빠르다. PQ codebook/weight representation이 다르며 이 결과가 RNGD fixed lookup 제거의 성능 증명은 아니다. |
| [LUT-GEMM](https://arxiv.org/html/2206.09557v4), ICLR2024; [T-MAC 저자 코드](https://github.com/microsoft/T-MAC/) | activation 의존 LUT와 bitplane weight 표현을 요구한다. `fetch_table_lookup<OutD>`는 dtype conversion이며 runtime table handle이 아니다. 설치SDK에서 같은 경로를 표현할 근거가 없다. |
| [SAIL](https://arxiv.org/html/2509.25853v1), 2025-09-30 | C-SRAM/새 ISA를 포함한 하드웨어 제안이다. 현재 RNGD 소프트웨어 후보로 세지 않는다. |

이번 선택의 새로운 근거는 논문 목록 자체가 아니라 **activation 정밀도 계약, native weight의
TRF 소유권, independent decode→fused reduction의 전달 경계**를 실제 API와 결합한 데 있다.
순위나 큰 개선 폭은 예측하지 않으며 결과는 각 후보의 frozen source/whole 측정에서 판단한다.

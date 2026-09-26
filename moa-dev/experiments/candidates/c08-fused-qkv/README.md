# C08 — QKV 계산 구조의 빠른 비교

**2026-09-27 정정:** [원 E105 image를 재현한 DMA 감사](../c08-joint-ingress/README.md)에서
K/V weight와 scale의 padded destination tile 전송 범위 오류를 확인했다.
각 K/V weight는 원본의4배 byte volume을 읽고 마지막 source end가46,080B 초과한다.
Q ingress와 세 output gather의 대응·범위는 통과했다. 아래 정확도 실패 기록은
유효하지만 이를 공동 QKV 구조 자체의 실패로 해석하지 않는다. 수정본 E124는 별도 평가한다.

출발 소스는 E104(`7f891f9c48e670fe014b2ac5bd2c92bae3bac433`)다.
E104는 비교군이며 두 후보의 설계 제약이 아니다. 작은 개선을 장시간 조정하는 대신
서로 다른 QKV 구조를 기본 정확도와 앞뒤 대조군으로 먼저 판별한다.
[실측 요약](performance-summary.json) · [정적 일정](static-screen.json).

**종료: 두 후보 모두 미채택, runtime은 E104 유지.** E105는 정확도 실패,
E106은 기본15개 비교 PASS지만 QKV104604→136334 cycles(+30.33%)다.
E106 뒤쪽 대조103544보다도 느리다. 후보마다 한 allocation에서 판별한 결과이며,
명백한 실패/회귀여서 반복·추가 입력·공식 제출을 진행하지 않는다. [최종 판정](decision.json).

## 두 구조와 실제 실행 경로

**E105: Q/K/V 통합.** [fused.rs](fused.rs)의 `project_qkv_fused`가 Q를2048행씩 둘로
나눈 뒤 K/V와 함께4개 행렬로 배치한다. Slice마다 각 행렬의4행을 보유하고, 한 BF16
입력 TRF를 공유하는 lookup→dot→channel scale pipeline에서16행을 계산한다.
Q/K/V HBM tensor를 별도 DM allocation 대신 합친 DM의 각 tile에 직접 전송한다.
출력은 typed DMA로 기존 head 배치에 모아 원래 RMS/RoPE/cache 경로로 보낸다.

Arena job74530에서 Q/K/V의9개 비교가 모두 실패했다. 앞뒤 E104의30개 비교와 후보의
변경하지 않은 O/FFN6개 비교는 통과했다. 실패한 QKV cycles는 성능 집계에서 제외한다.
정적 일정은100091 cycles이며, K와V weight load 각각26475 cycles가 걸리고 마지막
weight load가 끝난 뒤 통합 dot이 시작한다. 이 관측은 이 구현을 더 조정할 근거가
약하다는 판정에 사용한다. 정확도 실패의 내부 원인을 확인했다는 뜻은 아니다.
`v1`의 DM view API compile 오류와 수정 후 실행한 `v2`를 모두 보존한다.

**E106: 8열 분할.** [column_split.rs](column_split.rs)는 E105를 수정한 코드가 아니라
E104에서 따로 출발한다. `ops::sliding_project_qkv`의 H480 단위 pre-RMS 결과를 받아,
입력 전체를 gather·ring broadcast하는 단계를 제거한다. 각 행의 H3840 dot을8개 slice의
H480 dot으로 나누고 F32 inter-slice reduce로 합친다. 입력은 BF16, weight lookup 출력도
BF16이며 channel scale 이후 처음 BF16으로 반올림한다. 합산 순서는 E104와 다르다.
Arena job74554는 앞뒤 대조 포함45개 비교를 통과했다. 정적 일정69536 cycles에서
Q weight load는19372, K/V는각9961 cycles다. 이는 자원별 소요 시간이며 겹침을
무시해 합산한 값을 실행 latency로 사용하지 않는다.

- Q: cluster `Qs/2048`, slice `(Qs/64%32, H/480)`, local `(Qs%64, H%480)`.
- K/V: cluster `Ps/1024`, slice `(Ps/32%32, H/480)`, local `(Ps%32, H%480)`.
- 각 cluster의32개 행 그룹 × 8개 열 그룹을 모두 사용한다. F32 합산 후 각8개 slice 중
  한 slice가 Q64행 또는 K/V32행을 출력한다. Typed DMA로 기존 head 배치에 모은다.
- K/V는 같은 H480 입력 TRF를 공유한다. Head RMS, RoPE, cache 위치와 공개 인자는 유지한다.

두 후보 모두 `src/ops.rs`와 `src/device/sliding/projection.rs`만 실제로 바꾼다.
빌더의4-file replacement 목록에는 byte-identical `rmsnorm.rs`, `rope.rs`도 포함된다.
O/FFN device image는 E104와 동일함을 빌드에서 확인한다. 이 사실로 새 QKV의 정확도를
승계하지 않는다. 정적 cycles는 NPU 실측이나 공식 점수가 아니다.

## 재현과 중단 기준

```bash
python3 moa-dev/experiments/candidates/c08-fused-qkv/prepare.py
python3 moa-dev/experiments/candidates/c08-fused-qkv/prepare_columns.py
python3 moa-dev/experiments/candidates/c08-fused-qkv/build.py c03-columns8 --label NEW-BUILD
python3 moa-dev/experiments/candidates/c08-fused-qkv/screen.py prepare NEW-BATCH NEW-BUILD
python3 moa-dev/experiments/candidates/c08-fused-qkv/screen.py submit NEW-BATCH
# Arena terminal 상태 뒤:
python3 moa-dev/experiments/candidates/c08-fused-qkv/screen.py collect NEW-BATCH
```

`c03-` 이름은 기존 provenance builder의 QKV 파일 선택 규약이다. 과거 C03 후보가 아니다.
빌드는 별도 crate/package를 사용하고 채택한 runtime 소스를 덮어쓰지 않는다.
공유 Cargo target을 쓰는 full build는 직렬 실행한다. 기존 build/batch label은 재사용하지 않는다.

각 batch는 E104 → 후보 → E104의 세 benchmark process를 한 Arena allocation에서 실행한다.
각 process는 공식 기본3입력·15개 비교를 수행한다. 같은 allocation의 앞뒤 대조를 독립 반복으로
세지 않는다. Collector는 실제 업로드 archive를 다시 받아 hash를 대조하고 완전한 로그를 읽는다.
명백한 회귀/실패는 기본 screen에서 중단하며, 유망한 개선만 반복·추가 입력 검사로 진행한다.

[freeze.py](freeze.py)가 실패한 시도까지 정확한 변경 소스와 로그를 내용 hash로 보존한다.
나머지 source는 기록한 E104 commit과 Cargo lock에서 재구성한다. Binary/schedule은 ignored
artifacts에 두고 build 기록에 hash를 남긴다. [results.py](results.py)는 E105/E106 기본 screen을
원장 형식으로 추출한다. 새 batch를 추가하면 GROUPS와 catalog도 명시적으로 갱신한다.

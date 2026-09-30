# 현재 baseline: E144

현재 실행 소스는 **E144 / `baseline-e144` / `3586c41`**다. QKV weight를 H3840
행 단위로 DM에 적재하고 Main Switch에서 H960 조각을 native K4 계산 위치로 분배한다.
입력의 weighted E4 네 항을 Q/K/V가 공유하며 F32 Term/K4 합산, canonical head gather,
공통·channel scale 후 첫 BF16 변환, 기존 head RMS/RoPE/cache를 수행한다.
E142의 direct FP4 full30/F32 partial FFN과 E137 계열 native O 이미지는 유지한다.

E142 앞뒤 대조를 둔 독립3회 QKV 감소율은 **7.4094%,8.0340%,7.9907%**, paired 중앙
**7.9907%**다. 30MiB weight 총량은 같고 추가 Switch 및 입력 준비·gather·scale 비용을
포함했다. 논리 byte 제거만으로 개선을 추정한 결과가 아니다.

표준9입력, raw Q/K/V73,728 BF16 exact(±0 동등), 전체 QKV16,384출력 허용오차,
비대상 cache8,380,416원소 보존 검사가 통과했다. source46과 실측 이미지를 연결했다.
[채택 결정](../moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption-decision.json) ·
[독립 검증](../moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption/independent-final-review.json).
유한 사례 검증이며 임의 입력이나 전체 모델 정확성을 보장하지 않는다.

공식 제출 준비 중이다. 제출 전 관측 최고점은 E142 `e1f58b9e`,6.526159×,2026-09-30
**58/79위**다. 현재 기록은 [진행 현황](../docs/experiments/current-work.md)을 따른다.

| 찾는 내용 | 파일 |
|---|---|
| 현재 적용한 17개 구성요소: 목적·설정·호출 위치·의존 관계 | [methods.md](methods.md) |
| 현재 구성표와 기준 commit | [e144.json](e144.json) |
| 이전 E142 구성표 | [e142.json](e142.json) |
| 이전 E140 구성표 | [e140.json](e140.json) |
| 이전 E137 구성표 | [e137.json](e137.json) |
| 이전 E134 구성표 | [e134.json](e134.json) |
| 이전 E128 구성표 | [e128.json](e128.json) |
| 이전 E127 구성표 | [e127.json](e127.json) |
| 이전 E122 구성표 | [e122.json](e122.json) |
| 이전 E104 구성표 | [e104.json](e104.json) |
| 이전 E79 구성표 | [e79.json](e79.json) |
| 이전 E74 구성표와 고정 비교군 | [e74.json](e74.json) |
| 이전 E55 구성표와 고정 비교군 | [e55.json](e55.json) |
| 현재 채택 구성표를 가리키는 포인터 | [current.json](current.json) |
| 현재 작업 브랜치·구조 실험의 종료 상태 | [current-work.md](../docs/experiments/current-work.md) |
| 빨라짐·미확인·느려짐·정확도 실패별 과거 결과 | [최적화 결과 분류](../docs/experiments/optimization-results.md) |
| 최신 감사의 실제 실행 결과·DMA 오류·다음 판단 기준 | [9월 27일 실행](../docs/research/top10-execution-20260927.md) |
| 다음 큰 구조 변경 후보·논문 근거·조기 종료 조건 | [9월 23일 C09–C14](../docs/research/structural-candidates-20260923.md) |
| 이전 최적화 후보와 C01–C08 진행 이력 | [E55 이후 조사](../docs/research/optimization-candidates-20260922.md) |
| 무엇을 추가·제거·변경했는지 남기는 양식 | [조합 실험 양식](../docs/experiments/combination-template.md) |
| Source 부모·variant 파일·성능 비교군을 별도로 지정하는 실제 JSON 입력 | [소스 조립 recipe](../docs/experiments/combination-template.md#json을-실제-소스-선택에-연결하기) |
| 실제 단독/조합 비교를 채운 예시 | [E58 + E59 → E60](../docs/experiments/combinations/e60-down-padding.md) |
| C01 scale-DPE의 단독·조합과 미채택 이유 | [E65 + E66 → E67](../docs/experiments/combinations/e67-scale-dpe.md) |
| C05의 두 dataflow·정확도·반복 측정 | [E73 / E74 / E75](../moa-dev/experiments/candidates/c05-streamed-decode/README.md) |
| E74에서 함께 대체한 F2/F3/F4 | [E74 조합 구성](../docs/experiments/combinations/e74-streamed-partials.md) |
| C04의 열 분할·방송·gather 9개 조합 | [E76–E85](../moa-dev/experiments/candidates/c04-down-transfer/README.md) |
| E79에서 변경한 F1과 보존한 계산 경계 | [E79 조합 구성](../docs/experiments/combinations/e79-down-broadcast.md) |

## 실제 호출과 적용 구성

[host/runtime.rs](../furiosa-opt-gemma4-12B/src/host/runtime.rs)의 `Decode::sliding`은
QKV → attention → O → `self.mlp` 순서로 실행한다. `Decode::full`도 같은 `self.mlp`를
사용하고, 이 함수가 `ops::decoder_feedforward`를 호출한다. 따라서 FFN의 공유 코드를
바꾸면 sliding/full 양쪽 layer에 적용된다. 아래 세 진입점은
[ops.rs](../furiosa-opt-gemma4-12B/src/ops.rs)에 있다.

| 진입점 | 현재 데이터 흐름과 방법 ID |
|---|---|
| `sliding_project_qkv` | BF16 x → 분할 RMS **Q1** → H960 seed/ring·weighted E4 네 항 **Q2** → full-H row ingress·Main Switch **Q3** → native Term/K4 합산 **Q4** → F32 head gather·공통/channel scale·첫 BF16 **Q5** → head RMS/RoPE **Q6** → Q 출력·지정 K/V cache 슬롯 쓰기 |
| `sliding_attention_output` | BF16 attention 결과 → consumer-local weighted E4 4항 **O4** → 2-cluster/16-column의 두60행 native stage **O1** → F32 Term/K16 합산·입력scale·channel scale 뒤 BF16 **O2** → H480×8 post-RMS·residual **O3/R1** → residual HBM 갱신 |
| `decoder_feedforward` | residual·pre-RMS → E5M2 세 항 입력 **F6** → 2-cluster/full30 native up/gate·F32 block partial·6행 scale/합산 **F1/F2/F3/F4** → cluster-local120행 GeGLU → weighted E5 3항·L-half 내부 ring32·direct FP4 native down **F1/F6** → 두 H partial의 F32 합 **F3** → H480×8 global scale/post-RMS/residual/layer scale **F5/R1** → residual HBM 갱신 |

QKV/O는 [sliding/projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
FFN은 [shared/mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)가 중심이다.
공유 RMS를 바꾸는 실험은 두 호출 경로와 각각의 image를 확인한다.

**이하 문단은 E79/E104의 역사적 설계 기록이다.** E122는 down의 L ownership과 cluster 합산 경로를 바꿨다. E127은 O의 두 stage 공급을, E128은 native up/gate scale·합산 fusion을 추가했다. E134는 weighted native down, E137은 consumer-local four-term native O를 채택했다. 현재 구현은 [methods.md](methods.md)를 따른다.

E79 FFN은 전체 BF16 weight를 DM에 저장하지 않는다. FP8 DM은 유지하고 lookup의
BF16 출력을 DPE로 바로 보내16원소별 F32 dot을 저장한다. 이후 별도 VE에서 scale과 합산을
수행한다. Block dot을 BF16으로 반올림하는 C01 scale-DPE와 다르다. 과거 후보·회귀·컴파일
실패를 새 구성의 검증과 섞지 않는다. E79는 down 입력을 leader에만 DMA한 뒤 Main
ring32에서 각 column의4개 row group에 방송한다. 계산 배치·후단 유지가 이후
layout·decode·dot·scale·tile·통신의 동시 재설계를 제한하지 않는다.

## 이전 E79 검증과 성능 기준

[E79 동일 batch package 3회](../docs/experiments/e79-c04-eight-ring32.json)의
job 중앙값들에서 구한 FFN 중앙값은 **353136 cycles**다.
[E85 동시기 E74 대조](../docs/experiments/e85-c04-e74-control.json)의 **369997**보다
**4.56% 감소**했다. 앞/뒤 E74의 job 중앙값 관측 범위와도 겹치지 않는다.
QKV/O image는 E55/E74와 같으므로 이 두 커널의 cycle 차이를 FFN 변경 효과로 해석하지 않는다.

기본45개 비교, [추가30개 비교](../docs/experiments/e79-c04-extra-correctness.json),
[projection 출력138240개](../docs/experiments/e79-c04-probe-correctness.json)를 통과했다.
[정확한 source/build 연결](../moa-dev/experiments/candidates/c04-down-transfer/source-build-audit.json),
[compiler 방송 bitmap 확인](../moa-dev/experiments/candidates/c04-down-transfer/broadcast-proof.json).
QKV/O 경계 근거는 동일 image의 기존 검사를 승계하지만 새 FFN은 별도로 검사했다.
모델 전체 성능이나 모든 입력의 정확성을 보장하지 않는다.
[공식 E79 제출 `df79b0b9`](../docs/experiments/e79-official-submit-20260922.json)은 **5.8470×**로
완료됐다. QKV103764 / O51033 / FFN354648, 기본15개 비교 PASS다.
[리더보드 snapshot](../docs/experiments/leaderboard-e79-20260922.json)은60팀 중41위다.
Arena 반복, 공식 채점과 리더보드 snapshot을 구분한다.

E55 당시 자료와 E57–E72의 각 동시기 대조는 그대로 남긴다. 새로운 기준값을 과거
실험의 비교군으로 소급 적용하지 않는다.

## 조합을 시험하는 순서

1. 비교군은 동결한 구성표(`e55.json`, `e74.json` 등)로 식별하고 후보의 실제 출발점과 설계를 남긴다.
   **추가 / 제거 / 파라미터 변경 / 대체 구현**을 구별한다. E55 구성요소의 유지 여부를
   적되, 기존 구조를 유지하는 것을 목표로 삼지 않는다.
2. Decode·dot·scale·tile·통신처럼 서로 영향을 주는 선택은 함께 재설계할 수 있다.
   각 중간 단계가 비교군보다 빨라야 다음 조합을 시험할 수 있는 것은 아니다.
   독립적으로 분리 가능한 변경은 **C / C+A / C+B / C+A+B**나 최종 후보에서 하나씩
   제거하는 비교로 기여와 상호작용을 확인한다. 이 형식을 모든 설계의 선행조건으로 삼지 않는다.
3. `methods.md`의 연결 항목을 따라 layout·scale 좌표·자료형·반올림 경계를 함께
   검토한다. 방법 ID는 구현된 on/off 옵션이 아니다. 과거 patch는 당시 base에 대한
   차이이므로 파일 전체를 순서대로 덮어쓰면 앞서 적용한 변경을 잃을 수 있다.
4. source diff와 실제 compile 입력을 고정한다. 최소 compile과 schedule에서 가설이
   성립하지 않으면 그 상태로 닫고 다음 후보로 간다. 유망한 후보만 정확한 image의
   package로 기본 정확도·성능 screen을 수행한다. 채택할 후보의 반복 측정은 동일
   benchmark stage payload(binary/fixture/runner)의 독립 allocation 최소3회를 기본으로
   하며, 바깥 batch 순서 차이를 기록하고 산술·배치 변경에 맞는 경계 검사를 추가한다.
   5% 미만이면 추가 튜닝을 제한하되 재현되는 작은 이득의 채택 검증은 진행할 수 있다.
   결과가 겹치면 개선 미확인으로 유지한다. Compile-only 종료와 단회 중단은 별도
   상태로 남긴다. [짧은 판별 기준](../docs/research/structural-candidates-20260923.md)을 따른다.
5. 채택 전에는 **조합 전체**의 정확도와 성능을 다시 확인한다. 단독 PASS를 조합 PASS로,
   각 개선율의 합을 조합의 실측값으로 대체하지 않는다. 새 채택본은 `eNN.json`과
   새 Git tag로 남기고 `current.json`을 갱신한다. 과거 `e55.json`과 실험 원본은 보존한다.

[E60 조합 기록](../docs/experiments/combinations/e60-down-padding.md)은 위 형식을
기존 실측으로 채운 예시다. 단독으로 작은 이득처럼 보인 scale padding도 packed
padding과 결합한 결과까지 보고 판단한다.

## 기록 확인과 재생성

저장소 루트에서 실행한다. 첫 명령은 두 안내 문서를 생성하고 두 번째는 최신 여부를
확인한다. `--check-live`는 현재 compile 입력이 채택 baseline과 같은지 추가 확인한다.
실험 중 소스를 바꿨으면 `--check`만 사용하고 후보의 소스 해시는 별도로 보존한다.
`--check-live`에는 crate의 `Cargo.lock`도 있어야 한다. 새 clone에서는 아래의 별도
lock을 crate에 복원한 뒤 사용한다.

```bash
python3 moa-dev/experiment_index.py
python3 moa-dev/experiment_index.py --check
python3 moa-dev/experiment_index.py --check --check-live
```

도구는 태그의 commit, 동결한 46개 compile 입력의 해시, 구성요소의 코드 참조,
성능 JSON의 package 동일성·job 중앙값, 후보와 대조의 fixture/runner 일치, 생성 문서의
일치를 검사한다. `--check-live`는 추가·삭제된 source 파일도 검사한다. 모델이나 NPU를
실행하지 않는다. Cargo.lock은 crate에서 Git 제외되므로 태그 검사는
[별도로 보관한 lock](../moa-dev/locks/gemma4-Cargo.lock)을 사용하고,
`--check-live`는 실제 crate의 lock까지 대조한다.

`baseline/`에 실행 코드 사본을 추가하지 않았다. 위 commit으로 소스를 복원하고 기록된
lock을 사용한다. 큰 binary·전체 schedule·원시 runtime log는 기존 ignored 경로의
로컬 자료이며, 이 폴더는 원격 백업 여부를 바꾸지 않는다.

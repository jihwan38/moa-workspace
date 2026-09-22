# 현재 baseline: E79

여기는 **채택한 코드의 구성과 검증 근거**를 보관하는 폴더다. 현재 실행 소스는
`furiosa-opt-gemma4-12B/src/`에 있고, 현재 채택 source는 `baseline-e79` 태그의
`2735b5899f41195dedeab1e076c82317e24efcb2`다. Git의 `baseline` 브랜치와 이 폴더는
별개다. 이전 E55는 `baseline-e55` / `873f64ed94229c88cd18d2731bdcca7eea0d6365`로 보존한다.
이전 E74는 `baseline-e74` / `9e2058e`로 보존한다.
고정 commit은 **비교군의 신원**을 식별한다. 후보를 E55나 E74 구조에서 시작하거나
그 주변만 탐색해야 한다는 제약은 없다. 후보의 실제 출발 commit과 설계를 별도로 기록한다.

| 찾는 내용 | 파일 |
|---|---|
| 현재 적용한 15개 구성요소: 목적·설정·호출 위치·의존 관계 | [methods.md](methods.md) |
| 현재 구성표와 기준 commit | [e79.json](e79.json) |
| 이전 E74 구성표와 고정 비교군 | [e74.json](e74.json) |
| 이전 E55 구성표와 고정 비교군 | [e55.json](e55.json) |
| 현재 채택 구성표를 가리키는 포인터 | [current.json](current.json) |
| 빨라짐·미확인·느려짐·정확도 실패별 과거 결과 | [최적화 결과 분류](../docs/experiments/optimization-results.md) |
| 다음 최적화 후보·우선순위·검증 조건 | [E55 이후 C01–C07](../docs/research/optimization-candidates-20260922.md) |
| 무엇을 추가·제거·변경했는지 남기는 양식 | [조합 실험 양식](../docs/experiments/combination-template.md) |
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
| `sliding_project_qkv` | BF16 x → 분할 RMS **Q1** → 2-cluster ring32 복제 **Q2** → 8-way 행 배치 **Q3** → BF16 입력·weight lookup dot **Q4** → F32 channel scale fusion **Q5** → head RMS/RoPE **Q6** → Q 출력·지정 K/V cache 슬롯 쓰기 |
| `sliding_attention_output` | BF16 attention 결과 → 2-cluster/16-column dot **O1** → F32 합산·channel scale 뒤 BF16 **O2** → H480×8 post-RMS·residual **O3/R1** → residual HBM 갱신 |
| `decoder_feedforward` | residual·pre-RMS → 2-cluster up/gate **F1** → FP4→FP8 DM→BF16 stream/DPE→compact F32 partial **F2** → 별도 F32 scale·합산 **F3**, 8/12행 scale tile **F4** → GeGLU·leader DMA/ring32 입력 방송·down **F1** → H480×8 global scale/post-RMS/residual/layer scale **F5/R1** → residual HBM 갱신 |

QKV/O는 [sliding/projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
FFN은 [shared/mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)가 중심이다.
공유 RMS를 바꾸는 실험은 두 호출 경로와 각각의 image를 확인한다.

**현재 FFN은 전체 BF16 weight를 DM에 저장하지 않는다.** FP8 DM은 유지하고 lookup의
BF16 출력을 DPE로 바로 보내16원소별 F32 dot을 저장한다. 이후 별도 VE에서 scale과 합산을
수행한다. Block dot을 BF16으로 반올림하는 C01 scale-DPE와 다르다. 과거 후보·회귀·컴파일
실패를 새 구성의 검증과 섞지 않는다. E79는 down 입력을 leader에만 DMA한 뒤 Main
ring32에서 각 column의4개 row group에 방송한다. 계산 배치·후단 유지가 이후
layout·decode·dot·scale·tile·통신의 동시 재설계를 제한하지 않는다.

## 현재 검증과 성능 기준

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
[공식 E79 제출 `df79b0b9`](../docs/experiments/e79-official-submit-20260922.json)은 접수되어
채점 중이다. Arena 측정과 공식 점수·순위를 구분한다.

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
4. source diff와 실제 compile 입력을 고정하고 정확한 image를 포함한 package를
   확인한다. 기본 정확도 screen을 통과한 후보를 대조군과 반복 측정하고, 산술·배치
   변경에 맞는 추가 입력·경계 검사를 수행한다. 기본 운영은 동일 package 최소 3회이며,
   결과가 겹치면 개선 미확인으로 유지한다. 큰 회귀의 단회 중단은 그 사실을 적는다.
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

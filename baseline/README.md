# 현재 baseline: E55

여기는 **채택한 코드의 구성과 검증 근거**를 보관하는 폴더다. 현재 실행 소스는
`furiosa-opt-gemma4-12B/src/`에 있고, 기준점은 `baseline-e55` 태그의
`873f64ed94229c88cd18d2731bdcca7eea0d6365`다. Git의 `baseline` 브랜치와 이 폴더는
별개다. 실험의 출발점은 움직일 수 있는 브랜치 이름 대신 위 commit으로 고정한다.

| 찾는 내용 | 파일 |
|---|---|
| 현재 적용한 15개 구성요소: 목적·설정·호출 위치·의존 관계 | [methods.md](methods.md) |
| 같은 내용을 도구에서 읽을 구성표와 기준 commit | [e55.json](e55.json) |
| 현재 채택 구성표를 가리키는 포인터 | [current.json](current.json) |
| 빨라짐·미확인·느려짐·정확도 실패별 과거 결과 | [최적화 결과 분류](../docs/experiments/optimization-results.md) |
| 무엇을 추가·제거·변경했는지 남기는 양식 | [조합 실험 양식](../docs/experiments/combination-template.md) |
| 실제 단독/조합 비교를 채운 예시 | [E58 + E59 → E60](../docs/experiments/combinations/e60-down-padding.md) |

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
| `decoder_feedforward` | residual 읽기·기존 pre-RMS → 2-cluster up/gate **F1** → 두 단계 weight decode **F2** → 16원소 dot 뒤 F32 scale **F3**, 8/12행 tile **F4** → GeGLU·down → H480×8 global scale/post-RMS/residual/layer scale **F5/R1** → residual HBM 갱신 |

QKV/O는 [sliding/projection.rs](../furiosa-opt-gemma4-12B/src/device/sliding/projection.rs),
FFN은 [shared/mlp.rs](../furiosa-opt-gemma4-12B/src/device/shared/mlp.rs)가 중심이다.
공유 RMS를 바꾸는 실험은 두 호출 경로와 각각의 image를 확인한다.

**현재 FFN은 전체 FP8·BF16 중간 weight를 DM에 저장한다.** Scale 곱·합산을 두 번째
DPE로 옮기는 방식과 중간 weight 저장을 없애는 staging은 현재 채택 구성에 없다.
과거 E23/E33과 E48/E50/E56의 관련 시도도 결과 분류에 남겼다. 현재 상태를
“모든 dataflow 최적화를 적용한 코드”로 해석하지 않는다.

## 현재 검증과 성능 기준

[채택 당시 동일 package 3회](../docs/experiments/e55-fused-qkv-repeats.json)의
job 중앙값들에서 구한 중앙값은 QKV **102,878**, O **51,249**, FFN **380,339 cycles**다.
공식 입력의 15개 비교를 매 job 통과했다. 별도 추가 입력 30개 비교와 QKV 상쇄 진단도
통과했다. O 경계 검사는 같은 O image를 실행한 기존 E53의 근거를 승계한다.
[빌드·검증 연결](../docs/experiments/e55-source-build-audit.json),
[QKV 경계 비교](../docs/experiments/qkv-boundary-comparison.md),
[O 경계 비교](../docs/experiments/output-boundary-comparison.md).

최근 메모리 실험의 [동시기 E55 대조 3회](../docs/experiments/e55-memory-layout-control-repeats.json)는
QKV **103,858**, O **51,445**, FFN **379,073**이다. **E57–E64의 변화율은 이 대조군을
사용한다.** 같은 코드도 측정 시점에 따라 cycles가 달라지므로 유리한 기준값을 골라
비교하지 않는다. E55의 추가·경계 검사 통과를 소스가 달라진 후보에 자동 승계하지 않는다.
이는 Arena 개발 측정이며 공식 점수·순위나 모든 입력의 정확도 보장은 아니다.

## 조합을 시험하는 순서

1. `e55.json`의 commit과 적용 ID 전체를 새 조합 기록의 출발점으로 복사한다.
   **추가 / 제거 / 파라미터 변경 / 대체 구현**을 구별한다. 유지한 방법도 ID로 남긴다.
2. 우선 한 변화 A를 기준 C와 비교한다. 다른 변화 B도 같은 기준에서 확인하고,
   가능한 조합은 **C / C+A / C+B / C+A+B**로 비교한다. 최종 후보에서 A 또는 B를
   뺀 측정도 남기면 각 방법의 기여와 상호작용을 확인하기 쉽다.
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

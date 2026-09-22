# MOA 2026 최적화 실행 계획

2026-09-22 현재 작업 코드는 **E79**, 브랜치는 `opt/c04-down-transfer`다.
목표는 공식 평가 최소3위 진입이며 아직 달성하지 못했다. 현재 코드는
[baseline 구성표](../baseline/e79.json), 이전 E55는 [고정 비교군](../baseline/e55.json)으로 보존한다.

**Baseline은 비교군이다. E55/E74 주변만 탐색하거나 현재 방법을 유지해야 한다는 제약은 없다.**
후보는 다른 출발점·layout·decode·dot·scale·tile·통신 구조를 사용할 수 있다.
서로 영향을 주는 선택은 조합 전체로 설계하고 측정한다. 중간 단계 하나가 느려도
그것만으로 조합을 배제하지 않는다. 공개 인터페이스, 허용 수정 범위와 정확도 계약은 지킨다.

## 현재 채택 결과

[C04 비교](../moa-dev/experiments/candidates/c04-down-transfer/README.md)에서 열 분할·방송·gather
9개 조합을 NPU에서 실행했다. **E79: E74 F32 partials + down leader DMA/Main ring32 방송**을
채택했다. 같은 package의 독립3 jobs에서 E74 **369997→353136 cycles, −4.56%**이며
기본45개·추가30개 비교와 projection 출력138240개가 통과했다. QKV/O image는 동일하다.
공식 제출 [`df79b0b9`](experiments/e79-official-submit-20260922.json)는 **5.8470×**로 완료됐다.
공식 FFN354648 cycles,15개 비교 PASS이며 [리더보드 snapshot](experiments/leaderboard-e79-20260922.json)은60팀 중41위다.
[구성 변경](experiments/combinations/e79-down-broadcast.md), [E85 대조](experiments/e85-c04-e74-control.json).
4열30행 E81은 단회−1.65%로 후보에 남고, 나머지는 회귀했다. E81의 반복·추가 진단은 미실행이다.

아래 C05는 이전 E74를 채택했을 때의 기록이다.

[C05 비교](../moa-dev/experiments/candidates/c05-streamed-decode/README.md)에서 두 구조를 구현했다.
E73은 lookup·dot·scale을 row tile마다 연결했고, E74는 lookup·dot을 전체 행에 stream한 뒤
compact F32 partial을 저장하고 별도 VE에서 scale·합산을 수행한다.

| 구성 | FFN 중앙값 cycles | 동시기 E55 대비 | 결과 |
|---|---:|---:|---|
| E75: E55 고정 대조 | 377804 | — | 각 독립 job의 control_before |
| E73: 직접 tile fusion | 416500 | +10.24% | 미채택 |
| E74: F32 partial stream | 369246 | **−2.27%** | 채택 |

세 독립 Arena jobs의 동일 package에서 비교했다. Control_after도 별도 보관한다.
E74의45개 기본 비교·30개 추가 입력 비교·네 full-shape 진단의138240개 출력은 모두 PASS다.
QKV/O image는 E55와 같다. 메모리 사용 감소는 compiler schedule 모델로 확인한 것이며
runtime telemetry가 아니다. [빌드·소스 귀속](../moa-dev/experiments/candidates/c05-streamed-decode/source-build-audit.json).

E55 공식 제출 `3ab72d66`은 **5.6665×**로 완료됐다.
[E74 공식 제출 `01ba2ec2`](experiments/e74-official-submit-20260922.json)도 **5.7805×**로 완료됐다.
공식 cycles는 QKV102721 / O51230 / FFN369335이다.
Arena의2.27% FFN 개선을 공식 점수나 현재 순위로 환산하지 않는다.

## 실제 실행 경로

세 public device 진입점은 [ops.rs](../furiosa-opt-gemma4-12B/src/ops.rs)의
`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`다.
`host/runtime.rs`의 sliding/full layer가 같은 FFN 구현을 호출한다.

- QKV: BF16 pre-RMS → ring32 복제 → FP8 weight의 BF16 lookup/DPE → F32 channel scale
  → BF16 row 복원 → head RMS/RoPE → Q와 지정 KV cache slot 쓰기.
- O: BF16 activation과 BF16 weight lookup/DPE → column partial F32 합산·channel scale
  → BF16 → H480×8 post-RMS·residual 쓰기.
- FFN: residual/pre-RMS → ring16 복제 → FP4→FP8 DM → BF16 lookup을 DPE에 연결
  → compact F32 block dot 저장 → 별도 F32 scale·합산 → BF16 up/gate → GeGLU
  → leader DMA/Main ring32 입력 방송 → 같은 방식의 down과8-column F32 합산 → global scale/post-RMS/residual/layer scale.

FFN의 weight dot은 up/gate32행·down60행을 stream한다. Scale VRF만8/12행으로 나눈다.
최종 projection까지 F32 partial을 유지하지만 F32 합산 순서는 E55와 다르다.
현재 값·호출 위치·coupling은 [methods.md](../baseline/methods.md)에 있다.

## 다음 탐색

1. **C06: O의 작은 전송·scale/RMS/residual 의존성을 검토한다.** 현재 입력·weight·scale·후단
   배치의 실제 비용을 확인하고 BF16 경계를 포함한 전체 조합으로 검증한다. C04에서는 E79를
   채택했지만4열30행 E81을 다른 decode/scale 구조와 결합하는 후속 탐색도 가능하다.
2. **C05 후속: decode와 전달·소비 일정을 함께 탐색한다.** 이번 Sub reduction은 SDK ISA
   lowering 내부 오류로 실패했다. 이전 E56의8/16행 HBM staging 오류도 미해결이다.
   해당 경로를 다시 쓸 때는 실제 shape의 nibble·row/cluster/chunk 경계와 DPE consumer까지
   검사한다. 소유 tensor 두 개를 만들었다는 이유로 physical double buffer라고 부르지 않는다.
3. **QKV/O도 병행 가능한 탐색 대상이다.** C03의 한 head 배치가 느렸다는 사실로 다른
   head/weight/입력 전달 구조를 배제하지 않는다. C06 O의 scale/RMS/residual 전달은
   전송 비용과 BF16 경계, shared caller 영향을 함께 확인한다.
4. **C07 native FP8 및 정밀도 보정은 설계가 성립하면 비교한다.** 과거 hi/lo 구현의
   정확도 PASS·회귀·실패 범위를 읽고, BF16 입력 전체 범위의 동치를 가정하지 않는다.
   C01/C02의 특정 구현이 느렸다는 이유로 다른 layout/packet/보정 조합까지 닫지 않는다.

이는 고정 순서나 후보 수 제한이 아니다. 관측된 병목과 검증 가능성을 근거로 순서를 바꾼다.
후보의 실제 출발점과 대체한 구성은 [조합 양식](experiments/combination-template.md)에 남긴다.
독립적으로 분리할 수 있을 때 단독/조합 또는 최종 설계에서 하나씩 제거하는 비교를 수행한다.

## 검증과 기록

1. 실제 source·SDK/lock·compiler·host/fixture/runner와 image를 동결한다. Compile 실패 뒤
   이전 binary를 실행하지 않는다. Shared target의 빌드는 직렬로 한다.
2. 변경하지 않은 공식 기본 harness로 정확도와 cycles를 검사한다. 성능 가능성이 있으면
   동일 package를 대조군과 최소3개의 독립 Arena jobs에서 비교하고 추가/경계 검사를 한다.
   한 job 안의3개 입력이나 여러 프로세스를 독립 job으로 세지 않는다.
3. 기본·추가·진단·모델 전체 측정은 구별한다. 모델 전체 정확도나 임의 입력 동치는
   별도 근거 없이 주장하지 않는다. QKV cache 부수효과, scale/block 좌표와 cast 경계도 확인한다.
4. 빠름·미확인·회귀·정확도 실패·compile 실패를 모두 남긴다. 관측 범위가 겹치는 작은
   개선을 확정하지 않는다. 채택 source commit/tag·recipe·current pointer를 함께 갱신한다.
   이전 baseline과 실패 원본을 덮어쓰지 않는다. 검증한 후보는 공식 제출로 별도 평가한다.

[실험 원장](experiments/README.md), [성능별 분류](experiments/optimization-results.md),
[후보 연구 기록](research/optimization-candidates-20260922.md),
[메모리 배치 E57–E64](research/e55-memory-layout-audit.md),
[C01](../moa-dev/experiments/candidates/c01-scale-dpe-e55/README.md),
[C02/C03](../moa-dev/experiments/candidates/c02-c03-e55/README.md).

[규칙·평가 계약](research/moa-rules-and-target.md), [RNGD/SDK](research/rngd-furiosa-opt.md),
[국내외 대회·논문](research/competition-techniques.md),
[mixed dtype DPE](research/mixed-dtype-dpe-audit.md),
[lookup table API 감사](research/fp8-lookup-table-sharing-audit.md)를 참고한다.
SDK에 없는 prefetch/barrier/table handle을 가정하지 않는다. 다른 플랫폼의 허용 오차나
메모리 명령을 RNGD에 그대로 적용하지 않는다.

```bash
python3 moa-dev/experiment_index.py --check --check-live
python3 moa-dev/experiments/candidates/c05-streamed-decode/verify.py
```

Arena70초/동시 queued+running2개 정책에 맞춰 작업을 제출한다. 실제 업로드 archive를
내려받아 fingerprint를 확인한다. 큰 binary·원시 log·schedule은 ignored artifacts에,
source·빌드/업로드 해시·검사 결과·판정은 Git에 보관한다.

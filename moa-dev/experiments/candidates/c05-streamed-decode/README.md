# C05: BF16 weight 행렬 대신 F32 block dot 저장

`opt/c05-streamed-decode`에서 두 FFN 데이터 흐름을 구현하고 비교했다.
**E74는 동시기 E55보다 FFN cycles가 2.27% 낮아 현재 작업 코드에 반영한다.**
E55는 고정 비교군이며 후보 설계의 제약이 아니다. 이번 후보는 편의상 E55의 공개
진입점·QKV/O·FFN 후단을 재사용했지만, decode와 dot의 경계 및 저장 자료를 바꿨다.
앞으로 layout·tile·통신까지 함께 바꾸는 후보도 허용한다.

| 구성 | FFN job 중앙값들 | 중앙값 | E75 대비 | 판정 |
|---|---|---:|---:|---|
| E75: E55 control_before | 377324 / 378268 / 377804 | 377804 | — | 고정 대조 |
| E73: tile마다 lookup→dot→scale→합산 | 415242 / 417425 / 416500 | 416500 | +10.24% | 회귀 |
| E74: 전체 lookup→F32 block dot 저장→scale/합산 | 369421 / 369246 / 368268 | 369246 | **−2.27%** | 개선 |

Jobs **73965 / 73983 / 73984**는 byte-identical package의 독립 Arena 실행이다.
각 job 안에서 E55 → E73 → E74 → E55를 별도 프로세스로 실행하며 공식 기본 host,
fixture, runner를 바꾸지 않았다. 같은 job의 여러 프로세스를 독립 반복으로 세지 않는다.
뒤쪽 E55는 378510 / 379987 / 377290으로 별도 보관한다. 표의 기준은 앞쪽 E55만이다.
E74와 앞/뒤 E55의 **job 중앙값 관측 범위**는 겹치지 않는다. 이는 세 번의 관측이며
통계적 신뢰구간이나 모든 환경에 대한 보장은 아니다.

[반복 원본과 요약](performance-summary.json),
[E73](../../../../docs/experiments/e73-c05-direct-repeats.json),
[E74](../../../../docs/experiments/e74-c05-partials-repeats.json),
[E75](../../../../docs/experiments/e75-c05-control-repeats.json).
별도 최초 E73 job73954의 415268 cycles는 같은 batch package가 아니므로 반복에 합치지 않았다.

## 바뀐 데이터 흐름

호출은 `ops::decoder_feedforward → feedforward_sharded → project_up_and_gate /
project_down_sharded`다. Sliding/full 모델 경로가 같은 FFN을 사용한다.

E55는 `FP4 HBM → FP4 DM → FP8 DM → BF16 전체 weight DM → row-tiled dot/scale/reduce`다.
E74 [partials/mlp.rs](partials/mlp.rs)는 다음 순서다.

1. 기존 전체 packed weight 전송과 FP4→FP8 DM decode를 수행한다.
2. FP8→BF16 lookup 출력을 바로 BF16 DPE에 연결한다. 각 16원소 block dot을 F32로
   유지하고 `transpose`에서 두 값씩 묶어 compact F32 partial DM에 저장한다.
   **전체 BF16 weight 행렬과 BF16 partial 반올림은 없다.**
3. `reduce_eight_rows_f32`는 up/gate의 32행을 8행씩 처리한다. FP8 block scale을
   Sub에서 F32 VRF로 준비하고 Main에서 F32 partial×scale 및 block 합산을 한다.
   Down은 60행을 12행씩 처리하며 local F32 합산 뒤 8-column F32 합산을 한다.
4. 최종 projection에서 BF16으로 변환한다. GeGLU, global scale, RMS, residual,
   layer scale과 공개 함수 signature는 유지했다. F32 합산의 순서는 달라진다.

Up/gate 240 blocks×32 rows, down 120 blocks×60 rows의 partial을 저장한다.
8/12행 scale tile의 VRF는 각각 slice당 7680/5760B다. Weight dot은 전체 행을 한 번에
stream하고 scale 처리만 tile로 나누므로 tile을 더 키우는 것이 유일한 탐색 방향은 아니다.

E73 [direct/mlp.rs](direct/mlp.rs)는 BF16 weight DM과 partial DM 둘 다 없앤다.
대신 FP8→BF16 lookup을 4 up + 4 gate + 5 down tile에서 반복한다.
E74는 projection당 한 번, 총 세 번이다. 큰 중간 배열을 덜 쓴다는 사실만으로
더 빠르지는 않았다. 두 설계를 조합 전체로 먼저 비교했고, 중간 변경의 단독 이득을
실험의 선행조건으로 요구하지 않았다. `*-upgate`/`*-down` 파일은 생성한 분해 후보이며
이번에 따로 compile하거나 실행하지 않았다.

## 저장량과 일정

[analyze.py](analyze.py)는 실제 schedule을 다시 읽어 아래 값을 계산한다.
**컴파일러 모델이며 NPU 실측 시간 귀속이나 메모리 계측값이 아니다.**

| 항목 | E55 | E73 direct | E74 F32 partials |
|---|---:|---:|---:|
| 전체 정적 cycles | 162209 | 168523 | 157822 |
| Main interval union | 125274 | 101355 | 108315 |
| DMA interval union | 135607 | 143987 | 135607 |
| Main/DMA overlap | 104921 | 82828 | 92298 |
| SRAM 물리 페이지 lifetime union peak, bytes | 250052608 | 172433408 | 147263488 |

두 후보 schedule에서 전체 BF16 weight tensor가 사라진 것도 검사한다.
세 행렬의 FP8+BF16 중간 write는 논리506.25MiB였다. E74의 FP8+F32 partial write는
210.9375MiB다. Read, padding, 다른 tensor는 제외한 계산이며 실제 버스 트래픽과 다르다.
E74의 SRAM peak 모델은41.1% 감소했다. 전송량·메모리 감소율을 속도 개선율로 바꾸지 않는다.
DMA 일정 자체는 그대로이고 overlap도 줄어 실측 개선은2.27%였다.

## 정확도와 빌드 증거

- 반복 batch의 기본 비교180개와 별도 E73 최초 screen15개가 모두 PASS다.
  E74 자체는 독립3 jobs에서45개 기본 비교를 통과했다.
- [추가 run3..8](../../../../docs/experiments/e74-c05-extra-correctness.json), job73993:
  30개 비교 PASS. 세 device image가 기본 빌드와 같고 fixture/host의 추가 입력만 다르다.
- [Full-shape projection 진단](../../../../docs/experiments/e74-c05-probe-correctness.json),
  job73999: 네 case×up/gate/down, **138240 BF16 출력 모두 CPU oracle과 정확히 일치**했다.
  모든 FP4 코드와 행 identity를 담은 packed 입력, block/shard 끝 경계, cancellation,
  zero input을 사용한다. Oracle은 독립 FP4 값 표와 binary fraction의 F64 합으로 만들고
  block·전체 값이 F32로 정확히 표현 가능한지 CPU에서도 확인한다.
  NaN으로 채운 출력 전체를 검사한다. Signed zero는 수치 동등성으로 비교한다.
  [host](probe_stream.rs), [wrapper](stream_probe.rs)는 동결한 E74 projection 소스를 직접
  호출한다. 별도 diagnostic image이며 성능 측정에 넣지 않았다.
- [검증기](verify.py)는46개 compile 입력, 성공·실패 빌드 로그, 정확히 한 번 embedded된
  target image, 기본/추가 image 일치와 업로드된 package를 확인한다.
  QKV/O image는 E55 그대로다. 모델 전체 추론이나 full-model 정확도는 이번에 실행하지 않았다.

E74 FFN image: `317a37c8c74d72a4c4f79030ffb91f407d0c7b26c9afb38e5fd371a11889a460`.
[23파일 CLI payload 확인](official-payload-preflight.json) 후 공식 제출
**`01ba2ec2`**를 접수했다. [공식 채점 추적](../../../../docs/experiments/e74-official-submit-20260922.json).
Arena 개선율과 공식 점수는 별도로 기록한다.

## 실패 경로와 다음 탐색

- F32 partial을 transpose4로 저장하려던 첫 compile은 허용 최대2행 제약으로 실패했다.
  [원본](attempts/partials-transpose4.rs)을 보존하고 transpose2로 수정한 것이 E74다.
- [partials-sub/mlp.rs](partials-sub/mlp.rs)는 dot과 합산을 Main/Sub로 나누는 구조다.
  공개 API 타입 검사 이후 NPU ISA lowering 내부 오류(`sfr_impl.rs:868:33`)로 실패했다.
  Host package를 만들거나 NPU에 제출하지 않았다. API에 메서드가 있다는 사실만으로
  해당 NPU 실행 경로가 지원된다고 주장하지 않는다.
- Projection 진단의 초기 두 build는 device 함수 이름과 지역 binding 충돌,
  SDK BF16의 없는 `NAN` 상수 때문에 실패했다. 수정 전 소스와 build 기록도 남겼다.
- E56의8/16행 HBM decode staging·buffer 재사용 오류는 이번에 해결했다고 주장하지 않는다.
  E74는 그 경로를 쓰지 않는다. 다음에는 **C04의 down column/activation 전달 구조와
  decode·dot을 함께 재설계**할 수 있다. E55나 E74 주변으로 탐색 범위를 제한하지 않는다.

재현은 workspace root에서 한다. 빌드는 shared target을 사용하므로 직렬 실행한다.
기존 build label은 덮어쓰지 않는다. 큰 binary/schedule/raw log는 ignored artifacts에 있다.

```bash
python3 moa-dev/experiments/candidates/c05-streamed-decode/prepare.py
python3 moa-dev/experiments/candidates/c05-streamed-decode/build.py partials --label c05-new-label
python3 moa-dev/experiments/candidates/c05-streamed-decode/analyze.py
python3 moa-dev/experiments/candidates/c05-streamed-decode/results.py --check
python3 moa-dev/experiments/candidates/c05-streamed-decode/verify.py
```

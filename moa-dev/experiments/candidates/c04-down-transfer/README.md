# C04: down 열 분할·입력 전달·weight 배치 결합

`opt/c04-down-transfer`에서 E74의 실제 소스 `9e2058e`를 출발점으로 사용했다.
E55/E74는 고정 비교군이다. 2/4/8열, row padding 제거, Main/Sub 방송, 입력 선행 gather를
조합해 비교한다. E74의 8열 배치를 유지하는 것을 탐색 조건으로 삼지 않았다.

현재 가장 좋은 **E79: 8열 + Main ring32 방송**은 독립 세 Arena jobs에서
E74 **369997 → 353136 cycles, −4.56%**다. 기본45개와 추가30개 비교를 통과했다.
전체 projection 경계 진단138240개 출력도 통과해 **E79를 작업 코드에 채택했다.**
공식 제출 **`df79b0b9`**도 **5.8470×**로 채점 완료됐다.
[공식 기록](../../../../docs/experiments/e79-official-submit-20260922.json): QKV103764 / O51033 /
FFN354648 cycles,15개 비교 PASS. [리더보드 snapshot](../../../../docs/experiments/leaderboard-e79-20260922.json)은
BEEF60팀 중41위다. E74의5.7805×에서 점수는 올랐지만 순위는 유지됐다.
QKV/O image는 같으므로 전체 공식 점수 차이를 FFN 변경 효과만으로 해석하지 않는다.

## 실행 경로와 바뀐 배치

`Decode::sliding / Decode::full → self.mlp → ops::decoder_feedforward →
feedforward_sharded → project_up_and_gate → geglu → project_down_sharded`다.
공식 진입점, RMS, up/gate와 GeGLU의 소스는 E74와 동일하며 후보는 down 내부를 바꾼다.
[prepare.py](prepare.py)가 고정 commit에서 후보를 생성하고
[layout-proofs.json](layout-proofs.json)에 좌표·VRF/TRF 크기를 남긴다.

GeGLU 출력은 cluster0의 128개 유효 slice에 BF16 120개씩 있다. E74는 이를 바로
두 cluster의 32 row groups×8 columns에 복사한다. Down의 한 slice는 activation1920개,
weight60×1920개, scale60×120개를 사용한다.

[E79 소스](eight-ring32/mlp.rs)의 전달 경로는 다음과 같다.

1. `to_dm`은 `[Dummy256/32,1#4,L/1920]`의 leader에만 입력을 전달한다.
   32-slice ring마다 첫8개 slice가 서로 다른 column의1920개 BF16 값을 받는다.
2. Main `CustomBroadcast { ring_size:32 }`가 같은 column의 네 row group에 복제한다.
   출력 배치는 `[Dummy256/32,4,L/1920]`이며 별도 BF16 DM에 저장한다.
3. 복제 축을 `DownRowsByColumns=[H/60%32,L/1920]`로 rename한 뒤 Sub가 TRF에 올린다.
   Physical slice `s`의 source는 `32*(s//32)+s%8`이며 activation index는
   `1920*(s%8)+local_element`다. Rename 자체는 데이터를 전송하지 않는다.
4. FP4→FP8 DM→BF16 lookup/DPE→F32 partial 저장→12행씩 F32 scale/reduce→8열 F32
   합산→BF16 projection은 E74와 같다. Weight/scale 좌표와 합산 순서를 바꾸지 않는다.

원래 activation은30720B다. TDMA 목적지의 논리 payload는1966080B에서491520B로75%
줄고 Switch가1966080B의 복제 결과를 만든다. 이는 **activation TDMA 목적지 payload**의
감소이며 전체 트래픽·메모리·실행 시간75% 감소가 아니다. Bitmap load와 Switch 비용도 든다.

[broadcast-proof.json](broadcast-proof.json)은 실제 compiler graph의256개 bitmap을 위
source 식과 대조하고 graph 재컴파일 image가 측정 image와 같은지 확인한다.
과거 [E43](../ffn-down-activation-broadcast-four/README.md)에도 같은 방송 자체가 있었지만
E55 계열의 정적 분석 뒤 NPU에는 실행하지 않았다. 이번 E79는 **E74의 F32 partial 구조와
결합해 실제로 검증한 구성**이다. 방송 알고리즘을 새로 발명했다는 의미가 아니다.

## 다른 조합

| ID | 배치와 입력 전달 | 변경 이유 |
|---|---|---|
| E76 | 4열, 32행, 60/64 row groups, Main ring16 | 연속 weight/scale segment를 늘리고 입력은4개 row group에 방송 |
| E77 | E76 + Main ring32 | Leader 수를 줄이는 대신8개 row group에 방송 |
| E78 | 2열, 16행, 120/128 row groups, Main ring16 | Weight/scale segment를 더 길게 하고8개 row group에 방송 |
| E79 | 8열, 60행, 32 row groups, Main ring32 | E74의 계산 배치와 입력 방송을 결합 |
| E80 | E76 + Sub ring16 | 방송을 weight decode와 다른 실행 경로에 배치 |
| E81 | 4열, 30행, 64 row groups, Main ring16 | Padding 없이512개 slice 전부 활용 |
| E82 | E81 + Main ring8 | 방송 수신자 수와 TDMA leader 수의 절충 |
| E83 | E81의 계산 + full-L gather 후 직접 TDMA 복제 | 작은 GeGLU 조각을 먼저 연속 버퍼로 모음 |
| E84 | E74 계산 + full-L gather 후 직접 TDMA 복제 | 열 분할 변경 없이 gather 가설을 분리 |

4열32행은 scale VRF `8×240×4=7680B`, activation TRF7680B이고, 2열16행은
scale VRF `4×480×4=7680B`, TRF15360B다. Padding은 typed copy로 제거한 다음 high H 축을
rename한다. 4열30행은6행 tile, scale VRF5760B이며 모든 slice를 쓴다. 6개 BF16 값의
불완전한4값 commit을 피하기 위해 column 합산 결과를 F32로2개씩 저장하고 H480×8로
모은 뒤 BF16로 바꾼다. BF16 변환 전에 추가 산술은 없다. 2/4열에서는 F32 합산 순서가
달라지므로 좌표 증명만으로 정확하다고 판단하지 않는다.

`eight-ring32-sub/mlp.rs`는 생성만 한 분해 후보다. 별도 compile/NPU 결과로 세지 않는다.
E80의 Sub broadcast는 compile과 기본 정확도는 통과했지만 반복 측정에서 느렸다.
C05에서 실패했던 Sub VE reduction과는 다른 연산이며 그 실패를 해결했다고 주장하지 않는다.

## 측정과 정적 비용

[screen.py](screen.py)는 E55 → E74 → 후보들 → E74를 별도 프로세스로 실행한다.
공식 기본 host·fixture·runner를 바꾸지 않고, 원본 파일을 stage 이름으로 업로드한 다음
각 directory에 복원한다. 모든 batch는70초 제한이다. 한 job의 stage들을 독립 반복으로
세지 않으며 다른 batch package끼리 묶어 반복 수를 늘리지 않는다.

E78/E79/E80는 동일 package의 jobs74142/74154/74161에서 비교했다.

| 구성 | FFN job 중앙값 | 중앙값 | E74 대조 대비 |
|---|---|---:|---:|
| E85: E74 before | 369997 / 370189 / 368838 | 369997 | — |
| E78: 2열 ring16 | 374383 / 375135 / 376410 | 375135 | +1.39% |
| E79: 8열 ring32 | 354091 / 353136 / 352762 | 353136 | −4.56% |
| E80: 4열 Sub ring16 | 399701 / 401167 / 399126 | 399701 | +8.03% |

뒤쪽 E74는370799/369439/370203으로 별도 보관한다. E79의 job 중앙값 범위는
앞/뒤 E74 모두와 겹치지 않는다. 이는 세 번의 관측 범위이며 통계적 신뢰구간은 아니다.
QKV/O image는 동일하므로 그 cycles 차이를 이 FFN 변경 효과로 해석하지 않는다.

[analyze.py](analyze.py)의 [static-summary.json](static-summary.json)은 compiler 모델이다.

| 항목 | E74 | E79 |
|---|---:|---:|
| 전체 정적 cycles | 157822 | 160203 |
| DMA interval union | 135607 | 131718 |
| Main interval union | 108315 | 112418 |
| Sub interval union | 19378 | 19378 |
| SRAM 물리 페이지 lifetime union peak, bytes | 147263488 | 147263488 |

정적 전체 시간은 오히려 늘었지만 NPU에서는 반복 개선됐다. 실측 전 정적 makespan만으로
후보를 폐기하지 않았다. TDMA→Switch 이동과 실제 성능의 방향은 일치하지만 compiler
duration으로 실제 엔진별 시간을 측정한 것은 아니다. NPU 병목 전체의 원인을 확정하지 않는다.

그 밖의 단회 screen은 E76 **378738(+2.57%)**, E77 **377277(+2.17%)**,
E81 **364496(−1.65%)**, E82 **381171(+2.85%)**, E83 **412179(+11.60%)**,
E84 **378015(+2.35%)**다. 각각 같은 batch의 E74 before와 비교했다. E81은
후속 후보로 보존하지만 독립 반복·추가 진단 없이 개선을 확정하거나 채택하지 않는다.
[전체 측정·비교군](performance-summary.json), [E79 반복](../../../../docs/experiments/e79-c04-eight-ring32.json),
[E85 E74 대조](../../../../docs/experiments/e85-c04-e74-control.json).

## 재현과 검증 범위

[build.py](build.py)는 E74를 archive하고 후보 MLP 하나만 교체한다. Compile 입력46개의
해시를 container에서 확인하고 exact FFN compile과 전체 host build가 성공해야 package를
만든다. Exact image가 host에 정확히 한 번 포함되는지, QKV/O image가 고정 비교군과 같은지
검사한다. 추가 입력 빌드도 세 device image가 기본 빌드와 동일하다.

E79 FFN image SHA256: `fe1ee4a0883d9fbdf5878d317dfb467a2d4336fbe96e43b087bc4f87f318307e`.
[verify.py](verify.py)는 소스·빌드 로그·image·실제 업로드 tar·성공 로그를 다시 검사한다.

[probe_stream.rs](probe_stream.rs)의 독립 CPU oracle은 모든 FP4 code, 행 identity,
zero, 상쇄, 모든1920-column 경계의 앞/뒤 block을 검사한다. 실제 후보의 projection 함수를
호출하는 별도 diagnostic image이며 성능 반복으로 세지 않는다. 기본/추가 fixture와
허용오차를 변경하지 않는다. [추가 job74163](../../../../docs/experiments/e79-c04-extra-correctness.json)는
30개 비교 PASS, [projection job74173](../../../../docs/experiments/e79-c04-probe-correctness.json)은
138240개 출력 모두 정확히 일치했다. Signed zero는 수치 동등성으로 비교한다. 여섯 batch
jobs의 총495개 기본 비교에는 모든 후보와 대조군이 포함된다. E79 자체는45개다.
모델 전체 실행은 이번 실험의 검증 범위에 포함하지 않는다.

```bash
python3 moa-dev/experiments/candidates/c04-down-transfer/prepare.py
python3 moa-dev/experiments/candidates/c04-down-transfer/build.py eight-ring32 --label c04-new-label
python3 moa-dev/experiments/candidates/c04-down-transfer/analyze.py
python3 moa-dev/experiments/candidates/c04-down-transfer/results.py --check
python3 moa-dev/experiments/candidates/c04-down-transfer/verify.py
```

Full builds는 shared target을 쓰므로 직렬 실행한다. Label과 실측 원본을 덮어쓰지 않는다.
큰 binary/schedule/raw log는 ignored artifacts에 남긴다. `run_queue.py`는 이번에 지정한
작업들의 기록된 제출만 재개하며, 요청 결과가 불확실한 제출은 자동으로 재시도하지 않는다.

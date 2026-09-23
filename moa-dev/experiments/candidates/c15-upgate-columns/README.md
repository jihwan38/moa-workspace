# C15: up/gate 입력 column 소유권 재설계

`opt/c15-upgate-columns`. E104에서 up/gate 전체의 activation 분배, weight/scale
전송, TRF, native DPE, F32 column 합산과 최종 row gather를 함께 교체했다.
GeGLU/down/epilogue는 E104 그대로다. FFN 전체를 재설계했다고 세지 않는다.

**기본 NPU 정확도 PASS, FFN +14.93%로 미채택.** Arena job79300 하나 안에서
E104335618 → C15 385621 → E104335432 cycles(각 process 3입력 중앙값).
앞뒤 비교군 중앙값335525에 대한 차이다. 모든 process의15개 비교 PASS,
후보 FFN3입력11520출력 PASS. 확대 정확도·독립 반복·공식 제출은 하지 않았다.
[실측](../../../../docs/experiments/e110-c15-upgate-columns.json), [종료 판정](decision.json).

## 실행 경로와 수정

`ops::decoder_feedforward → feedforward_sharded → prepare_ffn_fp8_split →
project_up_and_gate → project_column_rows → cast_gathered_rows → geglu →
project_down_sharded` 순서다.

전역 scale의 E5M2 세 항 quantizer는 유지했다. 입력 H3840을 두 column이 나눠
보유하고, cluster는 L의 두 절반, slice는128 row groups×2 columns가 된다.
각 slice가60 output rows×1920 inputs를 계산한다. Ring32의 두 column anchor에
입력을 보내고 같은 column끼리만 방송한다. FP4 weight는 FP8 DM으로 변환하고
native DPE로16원소 block별 F32 partial을 저장한다. 12행씩 block scale을 곱해
local H를 합산한 뒤 두 column을 F32로 합산한다. Canonical L/60 배치로 DMA
한 뒤 첫 BF16 변환을 수행한다. Down/GeGLU 산술 경계는 바꾸지 않았다.

- [v1](four-column-v1/mlp.rs): 4 columns×960 inputs,120 rows. 60#64 scale 행을
  전체 fetch 후 cast하여 생긴8-lane packet과4#8 collect가 충돌했다.
- [v2](four-column-v2/mlp.rs): fetch부터4-block packet을 지정해 첫 오류를 수정.
  다음 HBM→DM scale DMA에서240 blocks를4×60#64로 대응시키는 축 분해가 실패.
  60-byte column 시작 주소도8-byte 배수가 아니다.
- [v3](two-column-v3/mlp.rs): scale 열을120 bytes로 정렬하고 padding을 제거하기
  위해2 columns로 변경. Scale VRF는12×120 F32=5760B,5회 처리한다. 첫 두 오류를
  SDK 전체의 불가능성으로 판단하지 않고 producer/transfer/consumer 계약을 고쳤다.
  세 버전은 같은 C15 가족의 구현 변형이다. 마지막 recipe가 실제 실행 구성을 정의한다.

## 줄어든 비용과 늘어난 비용

[정적 일정](static-cost.json)의 전체 예상치는157045→175862(+11.98%)다.

| 구간 | E104 | C15 v3 |
|---|---:|---:|
| 입력 ring 방송 |11783|6023|
| up/gate 각각 FP4 weight DMA |24612|29427|
| up/gate 각각 scale DMA |4880|9774|
| up/gate 각각 native dot |5684|5684|
| up/gate 각각 FP4→FP8 |9263|9263|

입력/TRF 논리 크기는11520→5760B/slice지만 weight/scale 총량은 그대로다.
HBM 행에서 H 구간을 잘라 읽어 FP4 subrow960B, scale subrow120B가 되면서
전송 일정이 늘었다. F32 column reduce/gather도 추가됐다. Main interval union은
94944→89264로 줄고 DMA union은126267→145493으로 늘었다. 중첩되는 일정이므로
차이를 합산해 실측 원인을 확정하거나 대역폭·stall 비율로 해석하지 않는다.
NPU 전체 비교가 이 완성 조합의 회귀를 확인했다.

공식 [메모리 성능 안내](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)의
정렬·주소 배치 제약을 실제 SDK0.8.1 compile 결과와 대조했다. 다른 column 개수의
추가 sweep은 하지 않는다. 재시도하려면 분할 weight/scale 전송 비용을 줄일 구체적인
배치와 consumer를 함께 제시해야 한다. 입력 복제 축소 자체가 불가능하거나 모든
column 배치가 느리다는 결론은 아니다.

실행 source와 비교 package는 [recipe](two-column-v3/recipe.json)에 분리 기록했다.
`build-*-repro.json`은 source/log/image/package hash를 담고, `evidence/`에는 실패
소스·컴파일 로그와 성공 일정·실측 로그를 동결했다. QKV/O image는 E104와 같다.

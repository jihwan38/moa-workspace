# FFN direct layout, 원본 four-row contraction 복원

상태: 독립 컴파일 성공, 정적 makespan **562630 cycles**. **Arena job 70029에서 실제 NPU 정확성 모두 통과**, FFN 중앙값 **1139366 cycles**. 최초 baseline 4264297 대비 3.7436×다. [실측 로그](../../../logs/e5-ffn-four-row-job-70029.log).

## 왜 다시 four-row인가

Arena job 70000(two-row만 변경)과 69992(two-row + direct column layout)의 FFN 오차 통계가 모든 입력에서 동일했다. 따라서 공통으로 바뀐 two-row contraction/reduction/padded commit/tiny output DMA 경로가 원인 후보로 좁혀졌다. 어느 연산의 lowering가 문제인지는 아직 확정하지 않았다.

이 후보는 다음을 함께 적용한다.

- 직접 `DownRowsByColumns`에 FP4 packed weight와 FP8 local scale을 적재한다.
- down FP4 전체 행렬을 한 번 load/LUT한 뒤 four-row view를 사용한다.
- local scale 적용, contraction, 8-way reduction, BF16 cast, `commit_view`를 **4개 row 단위**로 실행한다.
- 원본과 같은 `commit_trim::<m![H % 120 = 4]>()`와 output offset `4*i`를 사용한다.
- two-row용 padding tensor와 작은 output TDMA copy를 완전히 제거한다.

direct layout에서는 four-row local scale가 slice당 4×120 F32 = 1920 bytes라 VRF 8192-byte 제한 안에 든다. 원본처럼 전체 L을 한 slice에서 dequant할 경우 four-row scale는 4×960 F32 = 15360 bytes라 같은 방법을 쓸 수 없다. 따라서 단순히 기존 two-row를 four-row로 바꾸는 것이 아니라 column 분할 배치와 함께 적용해야 한다.

## 정적 비교

| 항목 | 공식 원본 | 실패한 direct two-row | 이 후보 |
| --- | ---: | ---: | ---: |
| makespan | 1918472 | 616787 | 562630 |
| DmaEngine union | 1066334 | 290287 | 263227 |
| MainContext union | 872451 | 335894 | 319304 |
| SubContext union | 129898 | 41011 | 33121 |
| instructions | 1742 | 1326 | 876 |

two-row 대비 tiny output DMA 27060 cycles를 제거했고, down contraction은 원본과 같은 30회/22560 cycles로 돌아왔다. down scale dequant는 30회/66030 cycles다. 정적 비율은 실제 성능이나 정확성 통과를 의미하지 않는다.

적용 파일은 `mlp.rs`다. `change-from-baseline.patch`는 공식 원본에서, `change-from-previous.patch`는 직전 실패한 direct two-row에서 적용하는 패치다. up/gate, GeGLU, global scale, post norm/residual 처리 및 device signature는 변경하지 않았다.

Scratch: `/tmp/moa-ffn-direct-four-row`, 독립 target: `/tmp/moa-ffn-direct-four-row-target`.

- [정적 스케줄](../../artifacts/ffn-direct-four-row/decoder_feedforward.json)
- [컴파일 로그](../../artifacts/ffn-direct-four-row/compile.log)
- 상세 context/source 합과 SHA256: `static-comparison.json`

추가 진단으로 이전 후보를 일반 Cargo CPU backend에서 실행했으나, 원본 QKV 9개 출력 비교 PASS 후 변경하지 않은 attention-output에서 `carve: piece must be contained in self: Unconsumed([H = 480 # 245760])` panic이 나 FFN에 도달하지 못했다. 이 CPU 실행으로 FFN 정확성을 판단하지 않는다. 하드웨어 테스트가 최종 기준이다.

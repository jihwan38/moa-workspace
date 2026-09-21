# Exact FFN schedule과 실측 회귀 원인

2026-09-21, 동일 compiler0.8.1·Cargo.lock·frozen e6 source를 사용하는 독립 scratch에서 `cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule ... --dump-graph ... --dump-summary ...`를 실행했다. `--exact`로 FFN 하나만 compile했고 다른14개 entrypoint는 제외됐다. Raw schedule/graph/summary는 `../../artifacts/ffn-blockscale-schedules/`, 집계 코드와 결과는 `summarize_schedule.py`, `static-comparison.json`에 있다.

| 정적 schedule 항목 | e6 전체 전개 | up/gate block dot | up/gate + down block dot |
| --- | ---: | ---: | ---: |
| 최종 makespan | 392312 | 403875 | 357707 |
| MainContext interval union | 319304 | 238814 | 194114 |
| VectorEngine interval union | 264270 | 183780 | 139080 |
| DmaEngine interval union | 263227 | 288367 | 313507 |
| SubContext interval union | 33121 | 71911 | 110701 |
| Compiler peak memory bytes | 987885568 | 959270912 | 936222720 |

Main/Vector의 감소가 전체 지연 개선으로 연결되지 않았다. up/gate의 기존 dequant command는 `2×61815`, contraction command는 `2×18375`, 합160380 cycles다. 새 combined command는30개×2663=79890 cycles로 **50.2% 줄었다**. 이는 local-scale 논리 곱 횟수의1/16과 다른 값이다. dump의 combined command에서는 Mul ALU만 따로 분리한 실측 cycle을 얻을 수 없다.

추가된 비용은 **매 tile의 `fetch_table_lookup::<bf16>()` table setup**이다. LIR graph에서 `table_lookup` label의4096-byte DRAM 상수가3개→33개로 증가한다. 기존의 full FP4→FP8 decode3개 외에 up/gate 각15회 helper가 별도의 FP8→BF16 table을 요구한다. 각 추가 setup은:

- DRAM table→DM:838 cycles.
- DM table→복제된 lookup SRAM:1293 cycles.
- Schedule tensor의 복제 table storage:8MiB. 이는 매 번8MiB를 HBM에서 읽는다는 뜻은 아니다. DRAM 원본은4KiB다.

따라서 DMA는 정확히30×838=25140 cycles, Sub는30×1293=38790 cycles 늘었다. 상수 weight 영역도12288→135168 bytes로120KiB 증가했다. 이 비용은 block reducer 자체의 table이 아니라 **FP8 decoder lookup** 비용이다.

## 완료 경로와 overlap

최종 scheduled completion path(makespan)는392312→403875로2.95% 길어졌다. Compiler RLIR 통계의 DMA만 실행되는 시간(`io_only_cycle`)은63044→147652, compute union은326121→253324다. 계산이 줄어든 구간에서 DMA가 더 드러나고, table setup이 DMA/Sub 자원을 추가로 점유한다.

큰 DMA command의 위치도 바뀐다:

| 명령 | e6 interval | up/gate block dot interval |
| --- | --- | --- |
| Down packed weight DMA | 189808..248110 | 215355..273657 |
| Down scale DMA | 248395..267393 | 282088..301086 |
| 마지막 residual HBM write | 391278..391712 | 402841..403275 |
| 마지막 explicit sync 종료 | 392312 | 403875 |

Down weight DMA 시작이25547 cycles 늦어지고 scale DMA 완료도33693 cycles 늦어진다. 이 exact timeline과 I/O-only 증가가 계산량 감소만으로 성능을 예측할 수 없는 이유다. 표는 compiler가 선택한 자원 제약을 포함한 schedule을 설명한다. 단순 tensor graph의 이론적 longest path나 Mul 연산의 독립 latency를 측정한 것은 아니다. `PeCore`에는 `DramReuse` 수명 marker가 포함되어 있으므로 긴 interval을 CPU 산술 busy time으로 계산하지 않는다.

root의 실제 Arena job70107은 공식3개 seed의 모든 출력 PASS였지만 FFN median이820226→921693 cycles로12.37% 증가했다. 정적 회귀와 방향은 같지만 절대치와 비율은 일치하지 않는다. Hardware cycle과 정적 schedule을 서로 대체하지 않는다.

## 다음 후보

Down까지 같은 방법을 적용한 `../ffn-blockscale-all/mlp.rs`는 exact compile에 성공했다. local120개 block의 FP32 합 뒤8-slice FP32 합을 수행하고 기존 BF16 경계를 지킨다. 정적 makespan은357707로 e6 대비8.82% 작지만 lookup setup은 더 많고 아직 NPU 정확성/성능을 확인하지 않았다.

더 직접적인 해법은 lookup 횟수를 줄이는 것이다. `../ffn-blockscale-two-cluster-eight-row/`는2 clusters×240 valid slices×32 rows 배치에서8-row helper4회씩, 총8회로 줄인다. VRF는7680 bytes다. 8개 scalar 출력을 한 번에 모으는 transpose가 `max_in_rows=4`로 거절되어, 계산 tile8개는 유지하고 최종 output을4개씩 두 packet으로 묶는 경로를 검증한다. Whole-matrix unscaled BF16 decode를 한 번만 materialize하는 변형은 추가 LUT 감소와 DM 용량/수명의 tradeoff를 확인해야 한다.

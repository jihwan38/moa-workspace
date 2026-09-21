# FP4 block scale을 up/gate/down dot 뒤로 이동

`mlp.rs`는 `../ffn-blockscale-after-dot/mlp.rs`에서 **down만 추가 변경**한 별도 후보다. Up/gate-only 파일은 덮어쓰지 않았다. Exact `ops::decoder_feedforward` NPU compile 통과, 정적 makespan357707 cycles. NPU 실행·정확도·실측 성능은 아직 검증하지 않았다.

Down의 `DownRowsByColumns = [H/120,L/1920]` 배치를 유지한다. 각 slice가 담당하는1920개 입력을16개씩120 block으로 나누어 BF16 weight×activation을 FP32 dot으로 만든다. FP32 local scale을 곱한 뒤 `vector_intra_slice_reduce<L>`로 local120개 block을 합하고, `vector_inter_slice_reduce<DownRows>`로8개 column slice를 합한다. **BF16 cast는 두 합이 모두 끝난 뒤**에 있어 원래 수치 경계를 보존한다. FP32 덧셈의 순서는 달라지므로 하드웨어 정확도 검사는 필요하다.

`vector_intra_slice_tag`로 IntraFirst 경로를 선택하고4-way에서 local reduction 후8-way로 widen하면, SDK0.8.1은 같은 Vector pipeline에서 inter-slice reduction을 허용한다. 별도 BF16 intermediate나 FP32 DM roundtrip을 추가하지 않았다.

정적 비교:

| 항목 | e6 원본 | up/gate만 | 전체 block scale |
| --- | ---: | ---: | ---: |
| Makespan | 392312 | 403875 | 357707 |
| Main | 319304 | 238814 | 194114 |
| Vector | 264270 | 183780 | 139080 |
| DMA | 263227 | 288367 | 313507 |
| Sub interval union | 33121 | 71911 | 110701 |

Down의 lookup30회가 추가되어 DMA/Sub setup은 증가한다. 계산 감소와 DMA overlap 개선으로 이 compiler schedule에서는 전체 시간이 줄지만 NPU 결과를 보장하지 않는다. 자세한 up/gate 회귀 원인은 [schedule 분석](../ffn-blockscale-after-dot/schedule-analysis.md)에 있다.

후보/source hash는 `source-hashes.json`, compile 상태/schedule hash는 `evidence.json`, 전체 집계는 `static-comparison.json`, compile 출력은 `compile.log`에 있다. Scratch는 컨테이너 `/tmp/moa-ffn-blockscale-down`이며 공식 source/harness/live binary나 Git을 수정하지 않았다.

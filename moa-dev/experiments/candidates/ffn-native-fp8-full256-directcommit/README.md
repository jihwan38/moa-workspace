# Full256 FFN의 FP8 weight/scale 중간 복사 제거

2026-09-21, compiler/std0.8.1. E35 job70485에서15개 검사를 통과한 full256 FFN을 기준으로 두 후보를 만들었다. **두 후보 모두 첫 exact compile에 성공했다. NPU 정확성·성능은 아직 검증하지 않았다.** 기존 full256 source와 두 후보를 각각 보존한다.

| source | 변경 | 정적 cycles | E35 대비 |
|---|---|---:|---:|
| 기준 E35 |30행 F8 DM→실제zero32행 DM 복사|324,369|—|
| `mlp.rs` |F4→F8 lookup을 valid30 destination에 직접 commit|207,944|−35.89%|
| `mlp.direct-scales.rs` |위 변경+scale HBM을 valid30 destination에 직접 DMA|191,836|−40.86%|

기준 E35의 실측 FFN median은811,219cycles다. 위 수치는 compiler 정적 schedule이며 같은 비율의 NPU 속도 향상을 보장하지 않는다. Source를 적용할 때 선택한 파일 하나를 `src/device/shared/mlp.rs`로 사용한다. `mlp.rs`는 weight-only로 동결했고 `mlp.direct-scales.rs`와 서로 다르다.

## 원래값과 실제zero tail 유지

E35는 각 cluster의256개 slice에32개 **live** row를 사용한다. Row0..29는 원본 데이터이고 row30/31은 실제0이다. `L=7680*cluster+256*row+slice`와15360행을 복원하는 output gather는 변경하지 않았다.

Weight-only는 각 up/gate의 FP8 `[FfnRow=32,H]`를 먼저 전체 `memset(0)`한다. 기존 F4 source30행을 같은 table lookup으로 decode한 뒤, 별도의 FP8 source DM에 쓰는 대신 zero-initialized buffer의 `tile<FfnRow,30,[FfnRow=30#{!}32,H]>(0)`에 직접 `commit_view`한다. Write 범위가 첫30행뿐이므로 실제초기화한 마지막2행은0으로 남는다. Lookup 결과의 dtype은 계속FP8이고 추가cast는 없다. Slice당115,200B의 중간 FP8 DM 두 개와61,875cycles씩의 DM 복사 두 개가 없어진다.

두 번째 후보는 local scale에도 같은 원칙을 적용한다. FP8 `[32,H/16]`를 전체 실제0으로 채운 뒤, 기존 HBM30행 view를 `to_dm_view`로 valid30 tile에 직접 읽는다. 중간30행 scale DM과4,275cycles씩의 추가 복사 두 개가 없어졌다. HBM으로부터 읽는 원래값과 목적지 논리인덱스는 동일하며 마지막2행의0도 유지한다.

두 후보 모두 네 개의 실제 whole-buffer memset은 그대로다. `#`나 `#{0}` 선언만으로0을 가정하지 않는다. SDK0.8.1 `DmTensorViewMut::memset`, `commit_view`, `HbmTensorView::to_dm_view`를 사용하며, destination tile의 bottom padding은 범위 밖 쓰기를 제한한다. Type/API의 의도와 exact compile 성공이 실제 NPU lowering 정확성을 대신하지는 않는다.

## 변경하지 않은 계산

E35의 `project_rows_with_scale_contraction`은 byte-identical이다. Native FP8 packet4, hi/lo pair, runtime scale 적용, BF16 block boundary, 두 번째 BF16 scale contraction을 바꾸지 않았다. `compact_full256_rows` 이후 전체 source와 `feedforward` 이후 quantizer/GeGLU/safe down/global scales도 byte-identical이다. Ops/residual도 같다. E35 자체에 존재하는 FP8 근사와 BF16 block 반올림에 **새 수치 변경을 더하지 않는 dataflow**를 의도한다.

`validate_layout.py`는 원본15360행, actual zero tail 모델, no-extra-cast, DPE helper 동일성, safe down을 포함한 suffix 동일성을 확인한다. 두 번째 파일도 인자로 검사할 수 있다. Output은 first30 row만 기존 Switch 경로로 모으므로 dummy2행이 downstream GeGLU나 normalization에 들어가지 않는다.

정적 schedule에서는 weight-only226instructions, weight+scale224instructions다. Weight direct commit은 각각 기존 lookup과 같은9,263cycles이고, weight32 전체 초기화는각15,626cycles다. Scale-direct의 HBM load는각6,565cycles이고, scale32 초기화는각1,226cycles다. Copy 제거 외에도 lifetime과 scheduling이 바뀌었으므로 makespan 차이를 제거된 DMA 시간의 단순합으로 해석하지 않는다.

## 동결 source와 재현

- `mlp.rs`: `fcab0ea8a694a07ab352dcd93c2ef3643181f840e1d3f081ab81ad6d2489678e`
- `mlp.direct-scales.rs`: `7c363a7f116ff9ebf4027f6a9957faa5cb3ce74aac85b3c5f7b6579ff628b267`
- `ops.rs`: `afc2d11998903a5afc9750abd7f01ec6b54103964752de86efae49aa4e206560`
- 필요한 residual: `83d001788e3c1ae073ec8720435569fde062a47e52c181ab859a2d3f047accfc`

`weights.patch`와 `weights_and_scales.patch`는 각각 E35 frozen source에 대한 diff다. Scratch `/tmp/moa-ffn-full256-directcommit`의 현재 MLP는 **두 번째 variant**다. Compile 전에 의도한 파일을 scratch `src/device/shared/mlp.rs`로 복사해야 한다.

```sh
python3 moa-dev/experiments/candidates/ffn-native-fp8-full256-directcommit/validate_layout.py
python3 moa-dev/experiments/candidates/ffn-native-fp8-full256-directcommit/validate_layout.py mlp.direct-scales.rs
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-full256-directcommit; export CARGO_TARGET_DIR=/tmp/moa-ffn-full256-directcommit/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-full256-directcommit/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/weights-and-scales-direct.json 2>&1 | tee logs/weights-and-scales-direct.log'
```

`evidence.json`에 source hashes와 baseline/variant 수치를, `schedule-summary.json`에 source-correlated instructions를 보존했다. Full schedules는 host `/tmp/moa-ffn-full256-weight-direct-schedule.json`과 `/tmp/moa-ffn-full256-weight-scale-direct-schedule.json` 및 Docker scratch에 있다. Live source, Git, NPU는 변경하지 않았다.

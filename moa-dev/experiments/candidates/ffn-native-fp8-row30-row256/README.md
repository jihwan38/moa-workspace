# E42 대조: 실제30행 + 검증된 row256 순서

**첫 exact compile PASS, static179,710 cycles/196 instructions. E42 job70576에서 공식3개 입력 구성의15개 출력 비교가 모두 PASS**했다. 실측 중앙값은 QKV98,728/O45,794/FFN456,849 cycles이며, FFN이 E37 best379,741보다 느려 **미채택**했다([실측 기록](../../../../docs/experiments/e42-native-row30-row256.json)). 전체 실행 binary SHA는 `5d4e26b9e44c24f6e8255858b9da490a26ae0f05bde0200dbda2d9a9a71c8533`다. 원본 로그는 `moa-dev/logs/e42-native-row30-row256-job-70576.log`다. 정적179,710과 실측456,849의 차이 원인은 단정하지 않는다. E41 `../ffn-native-fp8-row30/mlp.rs`의30행 계산·BF16 scalar padded commit을 그대로 두고, 입력 배치와 출력 gather만 E35/E39의 row256 순서로 복원했다. 성능 채택보다 실패 원인을 분리하기 위한 후보다.

후보 `mlp.rs` SHA: `b82b141e33f1afc14bd261b3505c4fff11d8431a7f43e473c9d1a8aeb2b085f9`.

## 대조의 이유와 한계

Parent가 제공한 결과에 따르면 E39(row256+tail-zero, job70551)는 all15PASS/FFN415457 cycles, E40(row2+tail-zero, job70552)는 Q/O PASS지만 FFN3회 모두 output0부터 non-finite였다. E40의432862 FFN cycles는 실패값이므로 성능 비교에 쓸 수 없다. 후속 E41(row2+row30, job70562)은 all15PASS/FFN426442로 확인됐다([기록](../../../../docs/experiments/e41-native-row30.json)). E37 best379741보다 느려 미채택했다. 따라서 row2 배치가 항상 실패한다는 설명은 성립하지 않으며 E40 실패 원인은 미확정이다.

이 후보는 row2 입력 배치·gather와 실제30행 scalar 저장을 분리한다. row256도 mapping뿐 아니라 compiler의 allocation·schedule을 바꾸므로, 한 번 통과하더라도 row2 Switch 등 특정 primitive가 원인이라고 단정할 수 없다.

## 유지한 경로와 변경한 경로

`project_rows_with_scale_contraction`은 E41과 byte-identical이다. 따라서 native FP8 pair/packet4,16항 block partial의 BF16 반올림, BF16 scale DPE, 최종 BF16 cast와 scalar `1#4` commit을 유지한다. 모든256 slice가 실제30행을 계산하며 dummy행·zero초기화·행 확장 copy는 없다.

`project_up_and_gate`의 네 HBM view를 다음 순서로 바꾼다.

```text
E41: [Dummy2, FfnSlice/2, FfnRow, FfnSlice%2, H 또는 H/16]
E42: [Dummy2, FfnRow, FfnSlice, H 또는 H/16]

E42 논리 행 L = 7680*cluster + 256*local_row + slice
```

원래 HBM의15360행을 `[2,30,256]`으로 나눈 정확한 wire order다. 행마다 H=3840 weight와 H/16=240 scale을 끝까지 읽는다. 계산은 행별 독립이며 축 순서 외의 연산은 바꾸지 않았다.

`compact_full256_rows`는 유효 BF16 scalar만 fetch하여 `Broadcast1 { slice1:256, slice0:1 }`로 모은다. `(row,slice)` 순서의 네 adjacent slice 값을8B packet으로 묶는다. 결과는 cluster별 `[FfnRow,FfnSlice]`=7680개 실제 BF16이며, `L%7680`으로 rename한 뒤 기존 `UpGateRows`에 typed DMA한다. 이 gather는 E39에서 입력 tile 제거/declared Row30/입력 scalar stride 변경을 제외하면 동일하다.

scalar DM의 row별 유효값은 byte `8*row`에 있으며 나머지6B는 padding이다. fetch/transpose는 논리 scalar만 선택하고 padding을 데이터로 바꾸지 않는다. `Slice`는 `1#256`이므로 gather 후 **실제1 slice/physical256 slots**이며, schedule의 gather tensor 전체 physical 크기는7,864,320B다. 실제값은 두 cluster 합30,720B다. 원형 typed DMA가 유효 slice만 소비한다.

`feedforward` 이하(runtime quantizer, GEGLU, global scale, 기존 safe BF16 down, 후단 호출) source suffix는 E41과 byte-identical이다. 실패한 down-scale 후보는 포함하지 않았다. inherited FP8 근사와 block BF16 반올림 외에 추가 산술·cast는 없다.

## 검증과 정적 비교

`validate_layout.py`가 전15,360행 bijection, HBM 연속 주소,3,840개8B packet 순서, poison padding 미선택, helper/suffix 동일성을 확인했다. backend 런타임 oracle을 대신하지 않는다.

| 정적 항목 | E41 row2/row30 | E42 row256/row30 |
|---|---:|---:|
| full FFN makespan |170,176|179,710|
| 첫 native DPE/각 projection |7,484|7,484|
| 둘째 BF16 DPE |717|717|
| output Switch gather |325|8,457|
| output typed DMA |530|1,366|

통과한 E39 row256/tail-zero의 static175,459보다도4,251 cycles(+2.42%) 크다. 메모리 초기화 제거가 이 schedule에서 전체 성능 향상을 보장하지 않는다. 후속 실측 FFN456,849로 정확도는 통과했으나 E37보다 느렸다.

schedule에서 O260→T125→O265, O331→T148→O337이 둘째 DPE→scalar저장→gather 관계다. scalar tensor는 두 cluster 합122,880B이고, gather는 해당 producer 종료 뒤 실행된다. `schedule-summary.json`에 tensor shape/size/lifetime과 단계별 cycles를 보존했다. raw schedule은 host `/tmp/moa-ffn-native-fp8-row30-row256-schedule.json`에 있다.

## 재현

기준 E41 scratch의 src/Cargo/toolchain을 별도 `/tmp/moa-ffn-native-fp8-row30-row256`에 복사하고 이 `mlp.rs`만 덮어썼다. `source-manifest.json`은 이 파일 외 src가 E41과 모두 동일함을 기록한다.

```bash
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-row30-row256; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-row30-row256/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-row30-row256/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/row30-row256.json 2>&1 | tee compile-1.log'
python3 moa-dev/experiments/candidates/ffn-native-fp8-row30-row256/validate_layout.py
python3 moa-dev/experiments/candidates/ffn-native-fp8-row30-row256/inspect_schedule.py /tmp/moa-ffn-native-fp8-row30-row256-schedule.json
```

Compiler0.8.1. 첫 시도 성공으로 추가 compile은 하지 않았다. kernel image SHA `ac49d840b4056f1e43de6ccd2f6e2b235e1d3085ffd1d7ae8e9266905bd4fbbf`. 준비일2026-09-21. 이 subagent는 live source/Git/NPU 실행을 변경하지 않았으며 root가 전체 build와 job70576 제출을 수행했다. 본 문서는 job70576의 후속 PASS 결과와 미채택 결정을 기록한다.

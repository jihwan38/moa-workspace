# Full256 FFN: 마지막2행만 실제zero 초기화

2026-09-21, compiler/std0.8.1. Base는 `ffn-native-fp8-full256-directcommit/mlp.direct-scales.rs` SHA`7c363a7f…`다. **두 번째 exact compile에 성공했다.** 정적 makespan은191,836→**175,459cycles**(−8.54%)이며248instructions다. **새 후보의 NPU 검증은 아직 없다.** Source는`07910fa5b11bd23c7da97c7c31c431db6629fe7db95cead2eaf5cdda482f1514`로 동결했다.

## 두 쓰기로 실제32행을 정의한다

SDK0.8.1 `src/tensor/memory.rs:1375–1377`은 `view_mut().tile(...).memset(...)`을 translator가 지원하지 않는다고 명시한다. 따라서 destination subview에 직접memset하지 않았다. 실제2행짜리 FP8 DM tensor를 따로 만들고 그 전체에 지원되는 `memset(0)`을 실행한다. 이 **초기화된 실제값**을 destination row30/31로 복사한다.

네 destination(up/gate weight, up/gate scale)에 각각 다음 두 쓰기가 있다.

- 원본row0..29: 기준과 같은 F4→F8 direct `commit_view` 또는 scale HBM direct `to_dm_view`.
- Dummy row30..31: 실제2-row zero tensor의 Sub fetch/collect→`commit_view`.

Destination 전체32행을 먼저0으로 채우는 작업은 없지만, 첫30행과 마지막2행의 합이 모든 live cell을 덮는다. Source/transport의 padding을0이라고 가정하지 않는다. Destination은 계속32개 live rows이고 slice도256개 모두 live다.

| per-slice destination | 원본값 쓰기 byte 구간 | 실제zero 쓰기 구간 | 전체 크기 |
|---|---|---|---:|
| FP8 weight |`[0,115200)`|`[115200,122880)`|122,880B|
| FP8 local scale |`[0,7200)`|`[7200,7680)`|7,680B|

각 구간은 비중첩이며 전체를 정확히 덮는다. `tile<FfnRow,2,[FfnRow=2#{!}32,...]>(30)`가 마지막2행의 실제주소를 선택한다. Source zero tensor의 local row0/1이 destination row30/31에 대응한다. Weight zero source는2×3840=7,680B, scale은2×240=480B/slice다.

Weight Sub copy는32개의 FP8 원소를 packet으로 사용한다. Scale은 packet마다16개의 실제 FP8 원소와 transport padding16개를 갖고, `commit_trim<16live>`가 실제16개만 저장한다. 이 padding은0으로 간주하지 않으며 destination/DPE에 복사하지 않는다. 모든 source live값은 실제memset 결과0이다. 새 dtype cast는 없다.

## Dependency 확인

`validate_initialization.py`는 두 쓰기의 주소 coverage, actual zero source 모델, 추가cast 없음, DPE helper와 `compact_full256_rows` 이후의 byte 동일성을 검사했다.

`inspect_dependencies.py`는 source 호출 순서를 추측하지 않고 frozen schedule의 `input_tensors/output_tensors`를 역추적한다. 네 first consumer 모두에 zero-source memset, tail copy, valid30 write가 ancestor로 연결되어 있고, 두 write가 끝난 뒤 읽기를 시작한다.

| destination | tail copy 완료 | valid30 write 완료 | 첫 consumer 시작 |
|---|---:|---:|---:|
| gate weight |3,535|42,193|61,188|
| up weight |9,018|90,715|94,742|
| gate scale |6,109|41,174|43,843|
| up scale |9,808|89,383|93,279|

Scope는 compiler schedule의 데이터 의존성과 시간이다. 이는 NPU가 실제로 올바르게 실행했다는 실측 증거가 아니다. Schedule에서 gate가 먼저 배치됐다는 사실을 source의 up-first 호출 순서와 혼동하지 않았다.

## 정적 비용과 한 번의 실패

기존 weight32 memset은각15,626cycles였다. 새 weight2 memset1,226+Sub copy1,227=2,453cycles씩이다. Scale32 memset각1,226은 scale2 memset326+copy385=711cycles씩으로 바뀌었다. Sub activity 감소와 전체 makespan 감소는 겹치는 실행과 scheduling 변화 때문에 일치하지 않는다.

첫 시도는 actual2-row zero DM을 TDMA로 destination offset30 tile에 복사했다. Backend가 `source_accessing_range_local.end <= input_tensor_range` 검증에서 거절했다. 이 오류를 피하려고 source padding을 실제값처럼 넓히지 않았다. 동일한 실제2-row source를 Sub의 typed stream copy로 옮긴 두 번째 시도는 통과했다. 첫source/log는 `mlp.attempt1.rs`, `attempt-1-dma-range-failed.log`에 보존했다.

## 변경 범위와 재현

`candidate.patch`는 고정 base SHA`7c363a7f…`에 대해 **네 buffer 초기화만** 바꾼다. 기존 valid30 directwrites, runtime quantizer/scales, FP8 pair/block reducer, BF16 boundaries, row mapping, output gather, GeGLU/down, ops/residual은 같다. 별도 row2 실험의 HBM views/output gather 변경과 겹치지 않는다.

```sh
python3 moa-dev/experiments/candidates/ffn-native-fp8-tail-zero/validate_initialization.py
python3 moa-dev/experiments/candidates/ffn-native-fp8-tail-zero/inspect_dependencies.py
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-tail-zero; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-tail-zero/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-tail-zero/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/tail-zero.json 2>&1 | tee logs/attempt-2.log'
```

Full schedule은 host `/tmp/moa-ffn-native-fp8-tail-zero-schedule.json` 및 Docker scratch에 있다. Candidate에는 source, patch, compile log, 실패 기록, compact schedule/dependency evidence와 hashes를 보존했다. Live source, Git, NPU는 변경하지 않았다.

후속Arena70551에서공식15개출력을모두통과했다(Q97459/O45220/FFN415457).
Native경로는개선됐지만채택E37의FFN반복중앙값379741보다느려교체하지않는다. 추가입력은미검증이다.
[실측](../../../../docs/experiments/e39-native-tail-zero.json).

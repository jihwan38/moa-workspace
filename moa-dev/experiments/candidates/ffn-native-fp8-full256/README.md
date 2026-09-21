# Native FP8 FFN: 256개 live slice 진단

2026-09-21, compiler/std0.8.1. `ffn-native-fp8-scale-contraction`(E25)을 기준으로 각 cluster의256개 slice를 모두 실제 데이터가 있는 slice로 바꾼 진단 후보다. **두 번째 exact compile에 성공했고 E35 job70485에서 공식15개 검사 모두 PASS했다.** FFN median811,219cycles(Q97,301/O45,799)다. 이 결과는 full256 실제zero layout에서 native FP8 retained packet4가 실행된다는 사례이며, padding만이 이전 timeout 원인이었다고 확정하지는 않는다. 정적 makespan324,369cycles는 E25의166,220보다 크게 느리다. 추가 초기화와 복사 때문에 성능 채택보다 timeout 원인 범위를 줄이는 데 목적이 있다.

동결한 `mlp.rs` SHA256: `e835f288722aac9b6528ff0fb3504d53bb87e4da10ba2fe625113db6c55926dc`.

## padding 가설의 근거와 한계

E22/E25/E28은 모두 native FP8 up/gate의 첫 FFN launch에서 timeout으로 끝났다. E32는 Sequential/pair를 없앤 두 lane1 dot으로 바꿔도 같은 launch timeout이므로 pair 제거만으로 해결되지 않았다. 어느 실험도 FFN의 실패 instruction을 알려 주지 않았다. E33 down scale-DPE 정확성 실패는 별도 실험이며 이 후보의 down에는 반영하지 않는다.

공통 원형은 `UpGateSlices=m![L/32%240#256]`이다. 두 clusters에서 각240개 slice가32행씩 담당하고16개는 padding이다. Runtime activation hi/lo와 scale은 `prepare_ffn_fp8_split`/`broadcast_ffn_quantized`가 이미256개 slice 전체에 실제 복제한 뒤240개 slice view로 제한한다. Weight/local scale은 typed DMA가240개 live slice만 채우며 끝16개 값을 보장하지 않는다. BF16 경로는 같은240개 row-slice 구조로 공식 테스트를 통과했으므로 **padding 자체가 잘못됐다고 볼 수는 없다.** Q/O의 native FP8 성공 경로는256개 slice에서 실제 행/열을 계산한다는 차이가 있으나, packet4와 다른 차이도 많다.

공식 [Mapping Expressions](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/mapping-expressions.html#padding)는 `#`를 임의 값인 top padding으로 구분하고, `#{0}`도 선언 자체가 fill을 수행하지 않는다고 설명한다. 따라서 기존240#256 buffer를256 live로 바꾸면서 끝16개를0이라고 가정할 수 없다. SDK0.8.1 `tensor/memory.rs:1369–1386`의 `DmTensorViewMut::memset`도 **live cells만 채우며 padding은 그대로 둔다**고 명시한다. 기존 padded tensor에 memset만 추가하는 안도 이 조건을 충족하지 않는다.

[Packet Reducer](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/packet-reducer.html)와 SDK `engine/contraction/packet.rs:35–63`의 검증은 입력 byte 폭·packet reduction을 다룬다. `outer/mod.rs:84–89`는 cluster/slice 크기와 flit 폭을 검사한다. 읽은 SDK/공식 문서에서 **240개 유효 slice와 FP8 retained packet4의 조합을 금지하거나 timeout 원인으로 지정한 근거는 찾지 못했다.** 이 조합의 runtime/compiler 문제가 있다는 주장은 아직 가설이다.

## 실제 live32 buffer와15360행 보존

새 축은 `FfnSlice=256`, `FfnRow=32`다. 한 cluster의 원래7680행을 **30실제행×256slices**로 배치한다.

```text
원래 row L = 7680*cluster + 256*local_row + slice
cluster=0..1, local_row=0..29, slice=0..255
```

HBM의 `[L,H]` view를 `[Dummy2,FfnRow=30,FfnSlice,H]`로 rename한다. 물리 linear index는 `(((cluster*30+row)*256+slice)*3840+h)`로 원래 `L*3840+h`와 정확히 같다. Local scale `[L,H/16]`도 마지막 길이만240이며 같은 증명이 적용된다. 이 reshape는 값을 이동하거나 padding을 live 데이터로 승격하지 않는다.

Up/gate 각각에서 다음 절차를 수행한다.

1. F4 weights30행을 모든256개 slice로 읽고 기존 F4→F8 lookup으로 decode한다. 각 slice의 weight source는30×3840개의 실제 FP8 값이다.
2. 별도의 FP8 `[FfnRow=32,H]` tensor를 만들고 **32행 전체**에 실제 `memset(0)`을 실행한다.
3. Source30행을 destination의 `tile<FfnRow,30,[FfnRow=30#{!}32,H]>(0)`로 typed DMA한다. Bottom padding은 write 범위 밖의 마지막2행에 쓰지 못하게 제한한다. 그 두 행은 이미 실제 memset으로0이다.
4. Local scale도 별도의 FP8 `[32,H/16]` 전체를 실제0으로 채우고 첫30행만 원래값으로 덮어쓴다.
5. 기존 E25의32행 DPE helper를 축 이름만 바꿔 호출한다. FP8 packet64→네16원소 block, Sequential hi/lo pair, hi+lo/16, runtime scale, BF16 block cast, 두 번째 BF16 scale contraction을 그대로 유지한다.

Weight와 scale의 두 dummy rows는 실제 live0이다. F4 dtype의 memset은 SDK가 지원하지 않으므로 F4 buffer를0으로 채웠다고 주장하지 않는다. Decoder가 만든 FP8 값을 그대로 옮기며 **추가 dtype cast는 없다**. Runtime quantizer, global scales, GeGLU, safe down, ops는 E25와 같다. E25에 이미 존재한 FP8 근사와 BF16 block 반올림까지 제거한 후보는 아니다.

마지막에는 각 slice의 output32행 중첫30행만 read tile로 선택한다. Switch가 `Time=[FfnRow=30,FfnSlice]` 순서로 single slice에 모으고, 인접한4개의 slice 값을 BF16×4=8B 단위로 commit한다. 그 결과 `row*256+slice` 순서의7680개 값이 연속되므로 `L%7680`으로 rename하고 기존 output layout에 typed DMA할 수 있다. Row30/31은 gather하지 않으므로 GeGLU나 이후 normalization에 들어가지 않는다. Dummy row를 제거하기 위해 새 FP32/BF16 cast를 넣지 않았다.

`validate_layout.py`는15360개 원본 행의 전수 일대일 대응, gather 순서, actual32-zero-fill→30-copy 모델, no-extra-cast, E25 DPE helper의 축 이름 외 동일성, feedforward 이후 suffix/ops의 byte 동일성을 확인한다. 이는 compiler/NPU가 해당 의미를 구현한다는 실측 증거는 아니다.

## 정적 결과와 실패 이력

첫 source는 `L=7680*c+30*slice+row`의 연속30행 배치였다. 전체zero-fill과30→32weight/scale 복사는 lowering을 통과했지만 output30 BF16의60B tail을 TDMA로 compact하는 단계가 `tail_size.is_multiple_of(min_align)`에서 거절됐다. `mlp.attempt1.rs`와 `attempt-1-alignment-failed.log`에 보존했다.

두 번째 source는 위의 row-interleaved 배치와8B Switch commit을 적용하여 exact compile에 성공했다. 추가 compile 시도는 하지 않았다.

| 단계 | 정적 비용 |
|---|---:|
| 전체 FFN |324,369cycles,228instructions|
| E25 기준 전체 FFN |166,220cycles|
| FP8 weight32 전체 memset |2×15,626cycles|
| FP8 local scale32 전체 memset |2×1,226cycles|
| 실제 weight30행 destination 복사 |2×61,875cycles|
| 실제 scale30행 destination 복사 |2×4,275cycles|
| Native FP8 첫 block dot |2×7,964cycles|
| 두 번째 BF16 scale dot |2×745cycles|
| Output first30 Switch gather |2×8,457cycles|

단계들이 일부 겹치므로 비용을 단순합해 makespan으로 해석하면 안 된다. Weight source FP8 payload115,200B/slice, destination122,880B/slice다. Local scale source7,200B, destination7,680B다. 큰 DM 복사와 새로운 lifetime 때문에 정적 회귀가 크다. 실제 네 memset과 네 유효범위 복사가 schedule에 남아 있음을 `schedule-summary.json`에 기록했다. Full schedule은 `/tmp/moa-ffn-native-fp8-full256-schedule.json` 및 Docker scratch에 보존한다.

성공하더라도 padding이 유일한 원인이라는 결론은 안 된다. Padding 외에도 real-row assignment, weight transfer, output gather, register/DM lifetime과 scheduling이 달라졌다. 다시 timeout이면 “240개 live slice를 전부256개로 바꾸면 해결된다”는 단순 가설은 약해지지만 retained packet4나 quantizer를 원인으로 바로 확정할 수도 없다.

## 재현

E25 scratch source/manifests를 새 `/tmp/moa-ffn-native-fp8-full256`로 복사하고 MLP만 교체했다. `ops.rs` SHA는 `afc2d11998903a5afc9750abd7f01ec6b54103964752de86efae49aa4e206560`, residual은 `83d001788e3c1ae073ec8720435569fde062a47e52c181ab859a2d3f047accfc`다.

```sh
python3 moa-dev/experiments/candidates/ffn-native-fp8-full256/validate_layout.py
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-full256; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-full256/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-full256/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/full256.json 2>&1 | tee logs/attempt-2.log'
```

`evidence.json`에 source/kernel binary/schedule SHA, 좌표, 초기화, 수치 경계 및 confounders를 기록했다. `candidate.patch`는 E25 frozen source 대비 변경이다. Live source, Git, NPU를 변경하지 않았다.

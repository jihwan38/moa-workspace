# Native FP8 packet4를 유지하는 pair-free FFN 진단

2026-09-21, compiler/std 0.8.1. E22/E25 첫 FFN timeout을 분리하기 위한 후보다. Native FP8 64-element packet의 네 16-element block 결과는 유지하되 **Sequential two-lane과 pair unzip/zip을 없애고 lane1 두 dot**으로 계산한다. 첫 exact compile에서 성공했고 static makespan은 **181,640 cycles**다. 성능 채택보다 실행 경로 분리를 위한 진단이며 **NPU는 아직 실행하지 않았다**.

기준은 first-party `ffn-native-fp8-upgate/mlp.postreduce-div.rs`다. quantizer, F4→F8 DM, local/global scales, up/gate row mapping, GeGLU/down을 유지한다. 완성 소스는 `mlp.rs`, 기준 대비 diff는 `candidate.patch`다. `ops.rs`는 E16 tail을 사용한 scratch snapshot이며 이 실험에서 수정하지 않았다.

## 실제 매핑과 타입 흐름

packed `[FfnTerm,H]`의 hi/lo component를 각 lane1 TRF로 읽는다. `contract_packet<H/16%4>`는 네 개 block dot을 남긴다. 그러나 `LaneMode::Interleaved`는 Lane을 output Packet으로 옮기므로 lane1이면 **output Packet은 `1#8`**이다. 네 block 축은 OutTime으로 보내 `Time=[row4,H/16]`가 된다. 기존 `Packet=H/16%4#8`을 그대로 쓰면 SDK lane-folder 규칙과 맞지 않는다.

Low dot은 이 scalar block stream에서 1/16을 곱하고, F32 transpose로 인접 block 두 개를 묶어 8B씩 commit한다. 결과 DM `[row4,block240]`는 padding 없는 F32이며 이를 그대로 compact VRF로 읽는다. low 결과를 BF16로 변환하지 않는다.

High dot도 lane1 Interleaved의 같은 scalar Time stream을 만든다. `vector_intra_slice_tag`로 시작하는 **Standalone** VE 경로에서 low VRF를 더하고 local scale VRF를 곱한다. 그 뒤 H block 축 전체를 intra-slice reduce하고, 허용된 후단 `vector_fp_div(1/s)`로 runtime scale을 복원한다. 마지막에 기존 BF16 row output을 만든다. `s*hi`와 `s*lo`를 먼저 각각 곱하지 않는다.

```text
lo native dot, packet4 유지 → lane1 Interleaved scalar blocks → /16
  → compact F32 block DM → low VRF
hi native dot, packet4 유지 → lane1 Interleaved scalar blocks
  → low 더하기 → local scale 곱하기 → block240 F32 sum
  → /(1/s) → 기존 BF16 projection output
```

이 경로에는 Group/Zipped typestate가 없으므로 `vector_intra_slice_reduce`를 같은 standalone pass에 연결할 수 있다. H3840은 60 packets ×4 blocks ×16 values로 전부 유지하며, block Time 인덱스는 `4*packet_time+packet_block`으로 0..239에 일대일 대응한다. 추가 BF16 block 반올림은 없다. FP8 입력 근사와 FP32 reduction 순서·NPU cast 차이까지 없다는 뜻은 아니다.

## 용량과 진단의 한계

4-row low VRF는 4×240×4=3,840B, local scale VRF도 3,840B, reciprocal VRF는 32B로 operand 합이 7,712B/slice다. 실제 compiler allocator도 이 배치를 통과했다. `vector_stash`를 추가할 여유가 있다고 가정하면 안 된다. two lane1 TRF는 각각 FP8 payload3,840B지만 schedule에서는 각각 physical32KiB/slice가 할당되어 주소0과32KiB에서 서로 겹치지 않는다.

E22/E25에서 실패한 native packet4 경로는 유지하면서 Sequential/pair를 제거했지만, buffering·실행 순서·register lifetime도 달라진다. 따라서 NPU가 성공해도 pair만 유일한 원인이라고 결론내리지 않는다. 다시 timeout이면 packet4, padded-slice 연결, 공통 quantizer 또는 다른 공통 경로를 추가로 분리할 근거가 된다. static regression 자체는 이 진단의 실패가 아니다.

## 재현

scratch `/tmp/moa-ffn-native-fp8-two-dot-diagnostic`는 E25 scratch의 source/manifests를 복사한 뒤 이 MLP만 교체했다. target/output은 모두 scratch 내부다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-two-dot-diagnostic; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-two-dot-diagnostic/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-two-dot-diagnostic/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/two-dot.json 2>&1 | tee logs/compile.log'
```

source/schedule hashes, 실제 TRF allocation과 dot stages는 `evidence.json`, compiler 출력은 `compile.log`에 있다. GeGLU 이후 quantizer까지 suffix가 E22 source와 byte-identical이고 up/gate에 Sequential/pair가 없는 것을 검사했다. 소스는 동결한다. live source, Git, NPU 제출은 하지 않았다.

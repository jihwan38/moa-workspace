# Two-lane QKV에서 K/V register를 따로 적재

**후속 E23 job70327에서 실패했다.** Q/K는 각각 3개 설정 모두 PASS, V는 3개 모두 FAIL이며 E20과 같은 오류 통계를 재현했다. 따라서 K/V TRF와 scale VRF를 각각 다시 적재하는 변경만으로 해결되지 않는다. 공유 register 재사용을 실패 원인으로 결론내릴 수 없다. 원본 소스는 그대로 동결한다. [원시 로그](../../../logs/e23-kv-reload-streamed-scale-job-70327.log).

2026-09-21, compiler/std 0.8.1. E20 canonical two-lane와 E21 row-interleaved two-lane가 모두 Q/K 3개 설정 PASS, V 3개 설정 FAIL을 보였다. 반면 E19의 두 별도 dot은 전부 PASS였다. 이 후보는 실패 원인 분리를 위해 **실패한 canonical 후보의 K/V register 공유만 제거**한다. Exact QKV compile PASS, static 53,305 → **53,366 cycles** (+61), 후속 NPU 검증은 위 실패 기록을 따른다.

`project_key_value`가 한 번 만들던 FP8 `x_trf`와 F32 runtime `scale_vrf`를 private `project_one_kv_matrix` 안으로 옮겼다. helper는 같은 immutable packed DM과 scale DM을 받고 K/V 호출 각각에서 register를 만든다. Q projection, quantization, pair-mode 산술, BF16 cast/scale 경계, KV weight/row/head layout, O suffix, ops는 변경하지 않았다. source 수준의 lifetime만 좁혔으며 별도 barrier는 넣지 않았다.

실제 schedule에서도 K/V 적재가 합쳐지지 않았다. V는 TRF91(23,058→25,027)과 scale VRF79(17,847→25,027), K는 TRF97(25,504→44,317)과 scale VRF83(18,381→44,317)을 사용한다. K/V TRF는 시간 구간이 겹치지 않아 같은 address 0을 재사용한다. Q는 별도 TRF94를 사용한다. K와 V dot+pair는 모두 기존과 같은 746 cycles다.

실제 소비 경로는 첫 O179→T92→O186/T95→O187/T96→O203/T100→O209/T103→normalize_value O255이며, 마지막 O317은 normalize_key O393으로 이어진다. 따라서 실행 순서는 V→Q→K다. 초기 문서가 source 호출 순서로 K/V를 반대로 붙인 것을 정정했다. “K가 먼저 읽고 V가 나중에 같은 register를 재사용한다”는 이전 가설의 전제는 잘못됐다. 이 후보는 양 projection의 register 적재를 구분하지만, E23 V 오류를 고치지 못했다. K register는 여전히 Q/norm/RoPE보다 일찍 prefetch된다. 실패한 baseline schedule에서도 명백한 주소·lifetime 중첩은 찾지 못했으며 compiler allocator bug라고 결론내리지 않는다. 원인을 quantization 오차로 단정하지도 않는다.

완성 소스는 `ops.rs`, `projection.rs`, baseline 대비 diff는 `candidate.patch`, hash와 contraction register 증거는 `evidence.json`, compiler 출력은 `compile.log`에 있다. source를 동결한다. scratch `/tmp/moa-qkv-fp8-kv-reload`에서 전용 target/output으로 아래 exact compile을 수행했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-fp8-kv-reload; export CARGO_TARGET_DIR=/tmp/moa-qkv-fp8-kv-reload/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-fp8-kv-reload/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/all-qkv.json 2>&1 | tee logs/compile.log'
```

live source, Git, NPU 제출은 하지 않았다. 정확성 확인은 root의 oracle 실행이 필요하다.

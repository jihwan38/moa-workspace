# Q/K two-lane 유지, V만 두 native FP8 dot으로 계산

**후속 [E27 job70382](../../../../docs/experiments/e27-qk-pair-v-two-dot.json)에서 전체 15개 검증 결과가 PASS였다.** 이 조합의 QKV median은 97,247 cycles이며 같은 job의 O/FFN은 52,543/386,729 cycles였다. 공식 세 설정의 검증이며 임의 입력의 정확성이나 기존 pair 실패의 유일한 원인을 증명하지 않는다.

2026-09-21, compiler/std 0.8.1. `qkv-fp8-direct-packed-quant`를 기준으로 **V projection만 E19의 두 별도 dot 방식으로 바꾼** salvage 후보다. Exact QKV compile PASS, **51,710 cycles**다. direct-packed all-pair 51,574보다 136 cycles 높다. 후속 공식 NPU 검증은 위 E27 PASS 기록을 따른다.

Canonical/row-interleaved two-lane QKV는 E20/E21에서 Q/K가 통과했으나 V가 실패했다. E23의 K/V register 재적재도 V 오류를 해결하지 못했다. 한편 E19의 Q/K/V 두 별도 dot은 모두 통과했다. 이 후보는 그 V 경로를 사용하면서 Q/K two-lane, row interleave, direct-packed quantizer를 유지한다. E19의 성공이 새로운 합성 후보를 검증한 것은 아니다.

## 변경 범위와 실행 경로

`ops::sliding_project_qkv`는 원래대로 BF16 input RMSNorm → `prepare_fp8_split` → Q와 K/V projection → 각 head RMSNorm → Q/K RoPE → Q output 및 K/V cache 기록을 수행한다. **V에도 기존 `normalize_value`의 RMSNorm이 적용된다.** ops와 normalization 함수는 수정하지 않았다.

Q의 전체 함수, K의 기존 `project_one_kv_matrix`, O/residual suffix, quantizer는 기준과 byte-identical하다. V만 새 `project_value_two_dot`를 호출한다. packed input `[hi(H3840),lo(H3840)]`의 in-slice `Dummy2` component 0/1을 `.tile`로 선택하고, 각각 Lane=`1`, Element=`H`인 독립 TRF에 적재한다. 같은 two-lane TRF를 V에서 공유하지 않는다.

V low dot은 native FP8 `weight × lo`의 H3840 전체를 FP32로 합산하고 1/16을 곱해 F32 DM에 저장한다. 이 값을 VRF에 적재한 뒤 high dot `weight × hi`에 더하고, 합에 runtime `s`를 곱한다. 그 다음 기존 BF16 projection cast를 수행한다. `s*hi`와 `s*lo`를 각각 먼저 계산하지 않으며, 기존 BF16 channel scale 경계도 유지한다.

E19 dot 본문에서 local row axis `Ps%4`를 `Ps/8%4`로 바꿔 기존 8-way row interleave를 유지했다. BF16 dot 이후 Switch → channel scale → head layout 변환은 K helper의 동일 tail을 사용한다. cluster=`Ps/1024`, slice=`Ps/32%32,Ps%8`, local=`Ps/8%4`로 모든 Ps2048 rows를 한 번씩 계산한다. 입력·weight·scale을 생략하지 않는다.

`validate_layout.py`는 Q/K/O/ops/quantizer의 불변, V dot과 E19 코드의 축 이름 변경 동치, V/K epilogue의 동일성, packed 7,680주소와 KV2,048 row bijection을 검사했다. 결과는 `layout-evidence.json`이다. 이 검사는 compiler lowering과 NPU 산술을 증명하지 않는다.

## Schedule에서 확인한 내용

V low/high dot은 각각 745 cycles다. low TRF94 lifetime은 26,628→28,116, high TRF96은 28,116→29,883이다. 서로 다른 적재이며 두 lifetime이 겹치지 않아 같은 physical address를 순차 재사용한다. Q/K는 별도의 two-lane TRF91/100을 사용한다.

컴파일러는 **V two-dot → Q pair → K pair** 순서로 계산했다. V dot은 27,371→29,883, Q pair는 30,848→32,074, K pair는 42,229→42,975다. 따라서 후속 oracle가 성공해도 “pair arithmetic만 V 실패의 원인”이라고 결론내릴 수 없다. 두-dot 경로 변경과 전체 instruction ordering 변경을 함께 포함한 우회 실험이다. FP8 근사 오차와 기존 failure 진단은 별도로 유지한다.

## 재현과 산출물

scratch `/tmp/moa-qkv-fp8-v-two-dot`는 direct-packed scratch의 source/manifests를 복사하여 만들었다. 전용 target/output만 사용했다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-fp8-v-two-dot; export CARGO_TARGET_DIR=/tmp/moa-qkv-fp8-v-two-dot/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-fp8-v-two-dot/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/all-qkv.json 2>&1 | tee logs/compile.log'
```

첫 Rust check에서는 소유권을 넘긴 `DmTensorView`를 재사용하여 E0382가 났다. K용 view와 V용 view를 원래 immutable DmTensor에서 각각 생성하여 해결했다. 실패 출력도 `attempt-1-moved-view.log`에 보존했다.

완성 소스는 `projection.rs`와 `ops.rs`, 기준 대비 변경은 `candidate.patch`, hash와 schedule 요약은 `evidence.json`, compiler 결과는 `compile.log`다. source를 동결한다. live source, Git, NPU job은 건드리지 않았다.

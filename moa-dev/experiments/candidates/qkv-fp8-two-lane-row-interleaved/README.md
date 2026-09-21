# Native FP8 two-lane QKV + 8-way row interleave

**NPU 검증 실패: E21 job 70300.** Q/K는 각각 3개 설정 모두 PASS였으나 V는 3개 모두 FAIL이다. V max absolute error는 [3.82812, 2.58789, 2.8125], tolerance 내 비율은 [94.78, 99.17, 94.63]%다. 측정 QKV median 99,568 cycles는 정확성 실패 때문에 채택 가능한 성능 수치가 아니다. 원본 소스는 변경하지 않고 실패 증거를 보존한다. 원인은 아직 확인되지 않았으며, E19 두 별도 dot의 PASS와 V에 집중된 오류를 근거로 register lifetime·pair/packing 경로를 분리 검증한다.
로그: [`e21-row-fp8-qkv-native-output-job-70300.log`](../../../logs/e21-row-fp8-qkv-native-output-job-70300.log).

2026-09-21, compiler/std 0.8.1. 동결한 `qkv-fp8-split-two-lane`과 first-party `qkv-row-interleaved`를 결합했다. **Exact compile 성공: 52,428 cycles**. canonical two-lane 53,305 대비 877 cycles 감소, canonical BF16 52,971 대비 543 cycles 감소다. 후속 NPU 검증에서 V가 실패했으므로 이 후보는 채택하지 않는다.

완성 소스는 `projection.rs`, `ops.rs`다. `candidate.patch`는 canonical two-lane 대비 projection만 변경한다. ops는 canonical two-lane와 같고, projection의 O/residual과 quantizer도 byte-identical하다. row interleave의 Switch→channel scale→head copy 구간은 검증된 `qkv-row-interleaved/projection.rs`에서 그대로 가져왔다. 공개 외부 구현은 사용하지 않았다.

## 정확한 행 배치와 복원

Q는 cluster=`Qs/2048`, slice=`Qs/64%32,Qs%8`, local row=`Qs/8%8`이다. 원래 row를 `q=2048*c+64*g+8*l+s`로 쓰면 physical slice=`8*g+s`, local=`l`에 들어간다. 32개 groups ×8 slices ×8 local rows ×2 clusters로 모든 Q4096 rows를 한 번씩 계산한다.

K/V는 cluster=`Ps/1024`, slice=`Ps/32%32,Ps%8`, local row=`Ps/8%4`다. `p=1024*c+32*g+8*l+s`, physical slice=`8*g+s`, local=`l`이며 l∈[0,4)다. 모든 Ps2048 rows를 한 번씩 계산한다. H3840의 순서는 바꾸지 않으며 typed HBM DMA가 각 weight row를 읽는다.

각 slice의 input은 동일한 `[hi(H),lo(H)]`와 같은 runtime scalar s다. 따라서 replicated input/scale을 새 row slice 이름으로 재해석해도 다른 값을 읽지 않는다. TRF Lane=`Dummy2`에서 두 FP8 내적을 병렬 계산하고 pair-mode가 `(hi_dot+lo_dot/16)*s`를 계산한다. 이 수식과 H reduction, BF16 cast 경계는 canonical two-lane와 같다. 바뀐 부분은 dot Time의 local row 좌표다.

BF16 dot 결과에서 Switch가 low Slice 8좌표와 local row Time을 교환한다. Q는 원래 `slice=Qs/8%256, local=Qs%8`로 돌아온 뒤 기존 scale을 곱한다. K/V는 4 local rows를 8로 padding하여 Switch하고, output slice=`Ps/32%32,Ps/8%4#8`, local=`Ps%8`로 만든다. 각 8-slice group의 앞 4개만 유효하다. scale도 이 유효 layout에서 8개 BF16 원소씩 처리한다.

K/V 마지막 view는 Cluster=`Ns/4`, Slice=`Ns%4,Ds/32,Ds/8%4#8`, Element=`Ds%8`이며 물리 순서를 보존한 축 이름 변경이다. typed DMA가 Ns head layout으로 모으면서 padding slices를 제외한다. padding 원소는 normalization의 Ds reduction에 들어가지 않는다.

`validate_layout.py`는 Q4096/KV2048 모든 row의 bijection, weight/scale 주소, Switch 출력, head 좌표와 K/V invalid slice 위치를 검사한다. O/quantizer/ops 불변과 검증된 row candidate의 epilogue byte equality도 검사했다. 결과 PASS다. 이 검사는 주소 대응이며 NPU 산술/oracle 검증을 대신하지 않는다.

## 비교와 한계

| QKV 구현 | static cycles |
| --- | ---: |
| canonical BF16 | 52,971 |
| row-interleaved BF16 | 53,190 |
| canonical native FP8 two-lane | 53,305 |
| **row-interleaved native FP8 two-lane** | **52,428** |

새 schedule에서 Q weight DMA는 13,383 cycles, K/V 각각 6,966이다. Switch 복원은 Q/K/V 각각 343 cycles다. 전체 makespan 차이는 이 duration들을 단순 합산한 수치가 아니며 context 중첩과 scheduling을 포함한다.

기존 row-interleaved BF16은 [E18 job70248](../../../../docs/experiments/e18-qkv-row-interleaved.json)에서 QKV 105,502 cycles를 기록했다. FP8 두 별도 내적은 [E19 job70261](../../../../docs/experiments/e19-qkv-fp8-split-two-dot.json)에서 oracle를 통과했다. 서로 다른 두 구현의 성공을 이 합성 후보의 정확성/성능으로 간주하지 않는다. [FP8 근사 오차 분석](../qkv-fp8-split/README.md)과 [two-lane pair-mode 설명](../qkv-fp8-split-two-lane/README.md)이 이 조합에도 적용된다.

## 재현

scratch `/tmp/moa-qkv-fp8-two-lane-row-interleaved`는 기존 two-lane scratch의 source/manifests를 복사하여 만들었다. 전용 target/output을 사용했고 live source/Git/NPU 제출은 수행하지 않았다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-fp8-two-lane-row-interleaved; export CARGO_TARGET_DIR=/tmp/moa-qkv-fp8-two-lane-row-interleaved/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-fp8-two-lane-row-interleaved/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/all-qkv.json 2>&1 | tee logs/compile.log'
```

exit 0, exact kernel 1개 성공. `evidence.json`의 source hashes가 실제 성공한 compile scratch와 일치함을 확인했다. source 파일은 동결했다.

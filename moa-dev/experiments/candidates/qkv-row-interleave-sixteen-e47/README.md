E47 QKV의 8-way 행 interleave를 **16-way로 바꾼 컴파일 후보**다. 첫 exact compile은 성공했고 정적 makespan은 **50,364 → 49,811 cycles(−553, −1.10%)**다. 후속 **E52 job70767은 공식 3입력의 all15 PASS지만 실측 개선이 없어 미채택**이다. 가중치 DMA 시간은 그대로이며, channel-scale DMA와 head gather 비용 감소가 관측된 이득이다. 실제 HBM bank 병렬성이나 성능 개선을 확인했다는 뜻은 아니다.

기존 후보의 projection 소스·README/evidence를 검색하고 [8-way 선행 실험](../qkv-row-interleaved/README.md)을 확인했다. QKV 16-way 변형은 찾지 못했다. O의 column partition이나 FFN row interleave와 구분해서 이 후보 하나를 만들었으며, 추가 수정·두 번째 컴파일은 필요하지 않았다.

## 변경 범위

동결 `e47-output-sixteen-columns/src/` 43개 파일 중 `device/sliding/projection.rs`만 바뀌었다. `project_query`, K helper, V helper의 row mapping·복원 layout과 KV row alias를 변경했다. `project_output`부터 파일 끝까지는 byte-identical이므로 E47 O16-column 경로와 QKV FP8 quantizer를 그대로 유지한다. ops·RMS·head norm·RoPE·cache scatter·FFN·harness·axes도 그대로다.

Q/K는 기존 pair dot, V는 기존 two-dot을 사용한다. H3840의 fetch32/contract64, packet/time reduction, hi+lo/16, input scale, BF16 dot 결과 경계를 변경하지 않았다. 복원 후 각 행에 `BF16→FP32 channel-scale multiply→BF16`를 적용하는 것도 같다. 16개 값으로 넓어진 scale 처리에서는 두 8값 stream, 네 4값 vector packet을 사용하며 새로운 reduction을 넣지 않는다.

[verify_layout.py](verify_layout.py)는 dot 본문이 row-axis 치환을 되돌리면 원본과 byte-identical임을 검사한다. Channel-scale 산술과 KV dispatch, O/quantizer suffix 불변 검사도 포함한다. 실제 compile 입력 43 source와 Cargo.toml/lock/toolchain의 46개 해시는 [manifest](compiled-source-manifest.json)에 있으며 host scratch와 모두 일치한다.

## 행 이동과 padding 복원

| 경로 | Cluster | dot Slice | dot local row | Switch 후 Slice | Switch 후 local |
|---|---|---|---|---|---|
| Q | `Qs/2048` | `[Qs/128%16,Qs%16]` | `Qs/16%8` | `[Qs/128%16,Qs/16%8#16]` | `Qs%16` |
| K/V | `Ps/1024` | `[Ps/64%16,Ps%16]` | `Ps/16%4` | `[Ps/64%16,Ps/16%4#16]` | `Ps%16` |

Q의 cluster-local 128행 그룹에서 `q=128*g+16*l+s`를 `slice=16*g+s, local=l`에 배치한다. K/V는 64행 그룹에서 `p=64*g+16*l+s`, `l∈[0,4)`다. 모든 256개 slice가 dot에서 유효하고 Q는 8행, K/V는 4행씩 계산한다. 각 행의 H3840개 원소는 논리 순서를 유지한다.

`InterTranspose{slice1:16,slice0:1,time0:1}`는 low slice `s`와 local row `l`을 교환한다. Q의 8행은 `8#16`, K/V의 4행은 `4#16`로 fetch Time에 padding을 유지한다. 결과는 `slice=16*g+l, local=s`다. 이 단계부터 cluster당 Q 128개, K/V 64개의 slice만 유효하다. 나머지는 실제 0이라고 가정하지 않고 계속 padding으로 표시한다.

복원된 유효 slice에는 연속 BF16 channel scale 16개가 들어간다. 같은 row mapping으로 scale을 적재하고 elementwise 곱을 적용한 뒤, 아래 head view로 **동일 physical wire order의 축 이름만** 바꾼다.

- Q: Cluster=`Ns/4`, Slice=`[Ns%4,Gs,Ds/128,Ds/16%8#16]`, Element=`Ds%16`.
- K/V: Cluster=`Ns/4`, Slice=`[Ns%4,Ds/64,Ds/16%4#16]`, Element=`Ds%16`.

마지막 typed `to_dm`은 기존 `Cluster=1#2`, `Slice=[1#32,Ns]`, Q Element=`[Gs,Ds]` 또는 K/V Element=`Ds`에 모은다. 유효 Ds256 좌표만 목적지에 존재하며 Switch padding을 Ds reduction에 포함시키는 reshape는 없다.

실제 [ops consumer](../../artifacts/e47-output-sixteen-columns/src/ops.rs)는 이 canonical 반환값에 Q/K/V 각각의 head normalization을 적용하고, Q/K에는 RoPE를 적용한다. 이어 Q 전체 write와 K/V `kv_offset` scatter를 실행한다. 이 호출·write 경로는 byte-identical이다. 이 후보에서 별도 cache side-effect 실행 검사를 한 것은 아니다.

[verification.json](verification.json)은 Q4096/KV2048 전체 행의 전후 bijection, weight/scale row 주소, 모든 physical Switch slot의 valid/padding 분류, head의 `Ns8×Gs2×Ds256` 또는 `Ns8×Ds256` 좌표가 각각 한 번씩 채워짐을 검사한 결과다. H3840의 unchanged packet/time 분해는 별도로 전수검사한다. 이 정수 좌표 증명은 SDK/NPU가 padding을 실제로 처리하는 동작의 검증을 대신하지 않는다.

## 정적 비용과 DMA descriptor

Baseline은 [E37 exact schedule](../qkv-e37-paired-value/baseline.json)과 [graph](../qkv-e37-paired-value/baseline-graph.json)다. 연결된 QKV image `91b232…`는 E37 clean-build image와 같고 동결 E47 host ELF에도 정확히 한 번 포함되어 있다. 후보 [schedule](../../artifacts/qkv-row-interleave-sixteen-e47/candidate.json)/[graph](../../artifacts/qkv-row-interleave-sixteen-e47/candidate-graph.json) 및 상세 node 비교는 [static-comparison.json](static-comparison.json)에 있다.

| 항목 | 8-way | 16-way | 차이 |
|---|---:|---:|---:|
| makespan | 50,364 | 49,811 | −553 |
| DMA interval union | 45,195 | 44,641 | −554 |
| Main interval union | 27,938 | 28,568 | +630 |
| Sub interval union | 9,803 | 9,809 | +6 |
| Vector interval union | 16,950 | 16,956 | +6 |
| Q weight DMA | 13,383 | 13,383 | 0 |
| K/V weight DMA, 각각 | 6,966 | 6,966 | 0 |
| Q channel-scale DMA | 792 | 651 | −141 |
| K/V channel-scale DMA, 각각 | 665 | 600 | −65 |
| Q head gather | 716 | 575 | −141 |
| K/V head gather, 각각 | 575 | 504 | −71 |
| Q/K/V Switch, 각각 | 343 | 551 | +208 |

Main/Vector 등 실행 구간은 겹치므로 합계를 elapsed cycles로 해석하지 않는다. 3개 channel-scale Main instruction과 scale VRF 준비도 각각 2 cycles 늘었다. Dot 비용은 동일하다. 전체 instruction 수는 171개로 같다.

Weight DMA는 세 행렬 모두 8개 source descriptor와 최내부 256B packet을 유지한다. Q graph node70의 첫 descriptor는 다음처럼 달라진다.

```text
8-way:  Qs_8=8:30720, H_256=15:256, Qs_64=4:245760,
        Qs_1=8:3840, Qs_256=2:983040, H_1=1:256
16-way: Qs_16=8:61440, H_256=15:256, Qs_128=2:491520,
        Qs_1=16:3840, Qs_256=2:983040, H_1=1:256
```

K/V도 local-row stride 30,720→61,440B와 low-slice 8→16 분해가 바뀐다. 이 정보만으로 transaction 수나 bank 충돌을 단정하지 않는다. Channel-scale load는 유효 slice마다 기존 BF16 8개/16B에서 16개/32B를 읽는다. 이와 head gather의 감소가 DMA 차이 −554를 설명하며, 더 긴 Switch와 달라진 scheduling을 포함한 전체 차이는 −553이다.

## 재현과 동결

첫 compile 전에 비어 있던 isolated target을 사용했다. installed compiler/SDK 0.8.1과 cache를 재사용했고 `/workspace/src`·`/workspace/target`·Git·NPU는 변경하지 않았다.

```bash
export CARGO_HOME=/home/moa/.cargo
export CARGO_TARGET_DIR=/tmp/moa-qkv-row-interleave-sixteen-e47/target
export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-row-interleave-sixteen-e47/target/furiosa-opt/kernel
export CARGO_NET_OFFLINE=true
cd /tmp/moa-qkv-row-interleave-sixteen-e47
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule schedules/candidate.json --dump-graph schedules/candidate-graph.json
```

[compile log](../../artifacts/qkv-row-interleave-sixteen-e47/attempt-1.log)는 exit0이다. [projection.rs](projection.rs)의 SHA-256은 `99b5b4b37d268b48e2725c643ba0e68531517d6832cc134ef1de68505de97324`, QKV image는 187,648B / `7331e194feaa01bece7d3f552e06c7c722d3e2f66a965afdcd832ecc2b612fb1`이다. 원본·patch·해시는 [evidence.json](evidence.json)에 연결했다.

독립 구현 단계에서는 full host executable과 NPU 실행을 수행하지 않았다. 이후 root가 동결 후보를 빌드·캡처해 [E52 job70767](../../../../docs/experiments/e52-qkv-sixteen-row.json)을 실행했다. Q/O/FFN은 94,674 / 45,029 / 379,774 cycles이며 공식 3입력의 15개 비교가 모두 PASS다. QKV는 E47 반복 중앙값94,662보다12 cycles(+0.013%) 높아 사실상 같고, 정적 1.10% 감소에 대응하는 실측 개선은 관측되지 않았다. E52 반복·추가 입력 검사는 하지 않았으며 **미채택**이다. Live E47 복구는 root가 관리한다.

[실측 요약](hardware-result.json)은 host ELF `0a52592829965bb9979749a2e4532e41700f66ee4f51c4854f1dd82d27332f7f`에 이 QKV image가 정확히 한 번 포함되고, E52 snapshot projection이 후보와 일치함을 확인한 기록이다. 기존 compile evidence/manifest는 당시 상태를 보존한다. 이 공식3입력 통과는 임의 입력의 padding·수치 정확성 또는 별도 non-target cache 보존 검증을 대신하지 않는다.

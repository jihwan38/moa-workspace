# QKV lane-parallel dot: 컴파일 성공, 정적 회귀로 보류

확인일 2026-09-21. compiler/std 0.8.1에서 요청한 weight-TRF lane 병렬화를 구현했다. **QKV static makespan은 61,994→70,492 cycles로 13.7% 증가하여 채택을 권하지 않는다.** 결과는 정적 schedule 비교이며 NPU 정확성/실측은 하지 않았다.

## 무엇을 바꿨는가

출발점은 두 cluster QKV + hier16 activation broadcast다. 원래 activation TRF Lane=1, weight stream의 Time=output row×H 구조를 다음과 같이 바꿨다.

1. 원래 row layout으로 FP8 weight를 DM에 읽는다.
2. 기존과 같은 table lookup으로 BF16 weight를 만들고 DM에 저장한다. BF16 값을 그대로 저장하므로 이 materialization 자체는 새로운 float cast가 아니다.
3. Q는 TRF Lane=`Qs%8`, Element=`H`; K/V는 Lane=`Ps%4`, Element=`H`에 weight를 적재한다. H=3840이므로 slice당 Q 61,440 bytes, K/V 각각 30,720 bytes다. std 0.8.1의 TRF payload 한도 65,536 bytes 안이다.
4. BF16 activation을 stream으로 보내 `contract_outer`에서 H%32 packet을 만들고, packet→scalar reduction과 H/32→1 time reduction을 수행한다. output row는 Lane에 남아 있다.
5. `contract_lane`은 Q의 여덟 row를 한 packet, KV의 네 row를 `Ps%4#8` packet으로 보존한다. 기존처럼 dot 결과를 BF16으로 cast한 뒤 원래 scale/gather 경로에 넘긴다.

Q/K/V scale, gather 및 `project_output`부터 파일 끝까지의 O suffix는 기존 후보와 byte-identical하다. `ops.rs`를 변경하지 않았다. 적용용 증분 패치는 `candidate.patch`, 완성 소스는 `projection.rs`다. 파일의 O suffix에는 cached O2-cluster/parallel residual 구현이 포함되지만 이 실험이 O를 변경한 것은 아니다.

## 왜 손해인가

| 단계 | 기존 Q | lane Q | 기존 K/V 각각 | lane K/V 각각 |
| --- | ---: | ---: | ---: | ---: |
| BF16 lookup + fused dot | 5,063 | — | 2,663 | — |
| BF16 decode/DM store만 | — | **5,063** | — | **2,663** |
| weight TRF load | — | 7,943 | — | 4,103 |
| lane-parallel dot만 | — | 505 | — | 505 |

dot 자체는 505 cycles로 짧아지지만 **decode만 수행하는 노드가 기존 decode+dot 융합 노드와 정확히 같은 길이**다. 이 compiler schedule에서는 기존 dot이 lookup/stream 시간 안에 포함되어 있어, dot row time을 줄인 이득보다 BF16 DM materialization 및 큰 weight TRF load가 크다. 원래 activation TRF는 Q와 공유 KV 각각 1,223 cycles였다.

전체 schedule은 overlap과 재배치의 영향도 받으므로 위 노드 시간을 더해서 makespan을 예측하지 않는다. 실제 max(end)는 기존 61,994 / 후보 70,492이고 instruction 수는 153 / 158이다. 이번 결과로 같은 BF16 materialization 구조를 O에 그대로 확장할 우선순위는 낮다. lookup→TRF 직접 적재 등 다른 구조의 성능/지원 여부를 검증한 것은 아니다.

## 정밀도와 안전성 범위

각 output row의 H%32 packet 및 H/32 time reduction 분할, contraction 이후 BF16 경계, FP32 channel-scale 곱 및 재BF16 cast는 유지했다. activation/weight의 stream/TRF 역할만 교환하고 rows를 Lane에 옮겼다. 유효 row를 합하거나 줄이지 않는다. KV의 padded 네 lanes는 BF16 commit_trim에서 제외한다. 기존 activation replica의 row-view reshape 외에 새 unsafe reshape를 추가하지 않았다.

유한 입력의 곱과 row별 합이 동일해야 한다는 구조적 근거는 있지만, operand 배치가 바뀐 device의 정확성 및 특수값 처리까지 컴파일만으로 검증했다고 주장하지 않는다. NPU oracle은 실행하지 않았다.

근거: [std 0.8.1 collect.rs의 to_trf/capacity](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/collect.rs), [contract_lane의 Lane→Packet 규칙](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/contraction/lane.rs), [contract_outer operand 및 layout 검증](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/contraction/outer/mod.rs), [Register Files 공식 설명](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html).

## 재현

Scratch `/tmp/moa-qkv-lane-parallel`와 전용 target/kernel directory를 사용했다. 첫 컴파일은 Collect에서 직접 commit할 수 없다는 API 오류로 실패했고, decode 단계에 `commit_trim::<m![H%16]>()`를 넣어 수정했다. 그 뒤 exact QKV 컴파일은 exit 0이며 미사용 `broadcast_sliding_heads` 경고만 나왔다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-qkv-lane-parallel; export CARGO_TARGET_DIR=/tmp/moa-qkv-lane-parallel/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-lane-parallel/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedules/sliding_project_qkv.json 2>&1 | tee logs/lane-parallel-fixed.log'
```

`evidence.json`에는 정확한 source/schedule hashes와 모든 관련 decode/TRF/dot 노드를 기록했다. raw schedule 및 최초 오류 로그는 scratch에 남겼다. 라이브 소스·Git·NPU 작업은 수행하지 않았다.

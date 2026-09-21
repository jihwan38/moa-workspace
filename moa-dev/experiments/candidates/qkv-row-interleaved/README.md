# QKV 가중치의 8-way 행 interleave

2026-09-21, SDK/compiler 0.8.1. E15 source snapshot에서 정확히 `ops::sliding_project_qkv`를 독립 컴파일했다. baseline **52,971 cycles**, 후보 **53,190 cycles**로 219 cycles / **0.41% 증가**했다. 두 컴파일 exit 0이다. 이후 root가 동일 projection SHA를 적용한 [**E18 job70248에서 전체 15 comparisons PASS**](../../../../docs/experiments/e18-qkv-row-interleaved.json), QKV median **105,502 cycles**를 측정했다. E15 반복 median 115,780 대비 **8.88% 감소**이며, E18 자체의 추가 반복 측정은 대기 중이다. 정적 지연 증가와 실측 개선이 반대로 나왔으므로 정적 값만으로 memory-bank 효과를 판정할 수 없음을 보여준다. `projection.rs`와 `candidate.patch`가 최종 Switch 복원 후보다. 라이브 소스·공유 target·Git·NPU 작업은 수행하지 않았다.

## 가중치 주소와 산술

기존 Query는 cluster=`Qs/2048`, slice=`Qs/8%256`, local row=`Qs%8`이다. 후보는 cluster를 유지하고 slice=`[Qs/64%32, Qs%8]`, local row=`Qs/8%8`로 바꾼다. cluster 내부 각 64행 블록에서 `q=64*g+8*l+s`를 `slice=8*g+s, local=l`에 배치한다. 기존의 `slice=8*g+l, local=s`와 비교해 작은 8×8 행 배치만 전치하며, 각 가중치 행의 H3840개 원소는 그대로 연속이다.

K/V는 cluster=`Ps/1024`, slice=`[Ps/32%32, Ps%8]`, local row=`Ps/8%4`이다. 32행 블록에서 `p=32*g+8*l+s`, `slice=8*g+s, local=l`이다. Q는 8행, K/V는 4행씩 기존과 같은 H3840 contraction을 수행한다. H의 packet/time/reduction 순서, FP8→BF16 lookup, FP32 contraction→BF16 경계는 유지한다. 가중치 데이터 자체를 임의로 재해석하지 않으며 typed HBM DMA가 해당 논리 좌표를 읽는다.

## 작은 BF16 값의 DMA 제약과 최종 복원

처음에는 scale도 interleave layout으로 직접 읽고 scaled output을 canonical layout으로 typed DMA 복원하려 했다. 그러나 scalar가 slice를 건너 이동하면서 2-byte stride가 생겨 DMA의 8-byte alignment 검사에 실패했다. 먼저 scale을 기존 layout으로 읽은 뒤 DM→DM transpose하는 두 번째 시도도 같은 제약에 실패했다. 실패 코드와 로그를 삭제하지 않았다.

최종 후보는 contraction BF16 결과를 **Main Switch InterTranspose**로 먼저 모은 뒤 scale을 적용한다. Q는 low Slice 8개와 local Time 8개를 교환하여 원래 `slice=Qs/8%256, local=Qs%8`로 복원한다. 그 뒤 scale load·multiply·BF16 cast·head reinterpret는 원래와 같다. 이 이동에는 산술과 추가 BF16 rounding이 없다.

K/V의 Time은 4개만 유효하므로 8개로 padding하여 같은 Switch를 사용한다. 결과는 slice=`[Ps/32%32, Ps/8%4#8]`, local=`Ps%8`이며, 각 8-slice 그룹의 4개만 유효하다. scale도 이 layout으로 읽으므로 각 유효 slice에 8개의 연속 BF16 값이 들어가 DMA alignment를 충족한다. Query에서 이미 쓰는 두 4-value vector packet의 elementwise scale pipeline을 K/V에도 사용한다. scalar별 `BF16 dot → FP32 scale multiply → BF16` 경계는 그대로다.

K/V scaled 결과의 physical slice를 `[Ns%4, Ds/32, Ds/8%4#8]`, local을 `Ds%8`로 **동일 wire order에서만** 이름 변경한다. 이후 typed DMA가 기존 head layout `Slice=[1#32,Ns], Element=Ds`로 모으면서 invalid slice padding을 제외한다. padded Time/slice를 head normalization의 Ds reduction에 포함하지 않는다. padding 처리와 Switch의 실장 동작은 root E18의 공식 3개 입력 구성에서 통과했다. 추가 입력 구성과 반복 성능 측정은 이 후보 단독 기준으로 아직 확인하지 않았다.

`validate_layout.py`는 Q4096/KV2048 모든 row의 bijection, 실제 weight/scale row 주소, Switch 결과와 head 좌표, K/V padding 위치를 정수 연산으로 전수 검사한다. O suffix가 baseline과 동일한지도 검사한다. 이 검사는 주소 증명이며 수치 oracle 또는 NPU 실행을 대신하지 않는다.

## 정적 비교

| 항목 | E15 baseline | 8-way 후보 |
| --- | ---: | ---: |
| 전체 makespan | 52,971 | 53,190 |
| DMA context union | 46,241 | 45,356 |
| Main context union | 30,813 | 31,844 |
| Query weight DMA | 13,512 | 13,383 |
| Key weight DMA | 7,030 | 6,966 |
| Value weight DMA | 7,030 | 6,966 |
| K/V channel-scale DMA, 각각 | 838 | 665 |
| K/V head copy, 각각 | 716 | 575 |

가중치 DMA 합은 27,572→27,315로 257 cycles 감소했다. 그러나 Switch 재배치와 달라진 scheduling이 전체 시간을 상쇄한다. context union이나 node duration을 합산해 전체 개선으로 해석하면 안 된다. 이 정적 결과만으로 bank-level NPU 성능을 확정할 수도 없지만, 당시 정적 비교만으로는 우선 채택하지 않았다. 이후 E18의 실제 QKV 개선이 확인되어 하드웨어 후보로 평가를 계속한다. 처음 제안된 full-256-way interleave는 추가 시도하지 않았다.

## 재현과 증거

Scratch는 Docker 내부 `/tmp/moa-qkv-row-interleaved`, target은 그 안의 `baseline-target`/`candidate-target`으로 분리했다. `CARGO_HOME=/tmp/moa-extra-seeds-e5-four-row-extra/cargo-home`을 사용했다. 최종 컴파일 명령은 다음과 같다.

```sh
CARGO_TARGET_DIR=/tmp/moa-qkv-row-interleaved/candidate-target \
FURIOSA_OPT_OUT_DIR=/tmp/moa-qkv-row-interleaved/candidate-target/furiosa-opt/kernel \
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
  --dump-schedule schedules/interleave8.json
```

Raw schedule과 로그는 Git-ignored `moa-dev/experiments/artifacts/qkv-row-interleaved/`, 요약은 `static-comparison.json`이다. `evidence.json`은 실제 compile scratch의 projection hash와 unchanged ops/norm/rope/axes hash를 저장한다. 기존 `broadcast_sliding_heads` 미사용 경고 이외에 최종 compile 경고나 오류는 없었다.

root는 적용한 실제 source/binary를 캡처하여 E18을 실행했고 Q/K/V 및 O/FFN 모든 output이 통과했다. 이후 추가 입력 구성과 반복 성능 측정이 필요하다. 우선 확인할 경계는 Q의 64-row block, K/V의 32-row block과 padded Switch Time, cluster 경계 및 RoPE half boundary다. 실측 성능은 동일 조건의 E15 반복 측정과 비교해야 한다.

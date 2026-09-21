# O projection column split: 8 / 4 / 2 비교

확인일 2026-09-21, compiler/std 0.8.1. root의 `output-two-cluster-column-split` 8-column 소스를 기준으로 4/2-column 두 변형을 isolated scratch에서 exact compile했다. **8-column을 유지한다.** 4/2 모두 컴파일은 성공하지만 static makespan이 각각 9.11%, 14.02% 증가했다. 이 두 후보의 NPU 정확성과 실제 성능은 측정하지 않았다.

| column groups | cluster당 row groups | local rows × columns | static cycles | activation DMA | weight DMA | lookup + dot + FP32 reduce | 추가 scalar compaction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8, 기준 | 32 | 60 × 512 | **32,842** | 1,318 | 13,512 | 5,063 | 0 |
| 4 | 64 | 30 × 1024 | 35,834 | 2,090 | 13,512 | 5,063 | 2,185 |
| 2 | 128 | 15 × 2048 | 37,447 | 3,632 | 13,512 | 5,063 | 2,185 |

8-column 기준의 schedule은 `moa-dev/experiments/artifacts/output-two-cluster-column-split/schedule.json`이다. 기준 소스/스케줄 hash가 원 후보의 evidence와 일치한다. 해당 기준은 이후 root의 [E14 job70174](../../../../docs/experiments/e14-headnorm-output-columns.json)에서 O oracle 통과, median 65,694 hardware cycles를 기록했다. 이 검증은 새 4/2-column 후보의 검증을 대신하지 않는다.

## 데이터와 정밀도

모든 변형은 두 cluster를 사용하며 원래 HBM weight `[H=3840,Qs=4096]`를 `[Dummy2,H%1920,Qs]`로 같은 physical order에서 재해석한다. `N` column groups에 대해 `G=256/N`, `R=1920/G`, `C=4096/N`이다.

- Cluster=`Dummy2`, Slice=`H/R%G,Qs/C`, Element=`H%R,Qs%C`.
- output row는 `h=1920*cluster+R*row_group+local_row`다.
- reduction column은 `q=C*column_group+local_column`이다.
- 모든 3840 rows와 4096 columns가 중복/누락 없이 대응한다. 가중치 총 15,728,640개를 모두 사용한다.
- slice당 FP8 weight payload는 세 변형 모두 30,720 bytes다. 입력 BF16 복제의 유효 payload는 512KiB / 1MiB / 2MiB로 늘어난다. 원 HBM activation은 항상 같은 4096개 값이다.

각 column group에서 FP8→BF16 lookup 및 FP32 contraction을 수행하고, FP32 `vector_inter_slice_reduce(Add)`로 group들을 합친 뒤 한 번 BF16으로 cast한다. 기존 channel scale은 이 BF16 결과 뒤에 그대로 적용한다. **FP32 합의 grouping은 8×512에서 4×1024 또는 2×2048로 바뀐다.** BF16 경계를 유지했어도 부동소수점 결합 순서가 달라 비트 동일성을 주장할 수 없다.

`ops.rs`는 root 기준과 byte-identical하다. `projection.*.rs`의 QKV 함수 prefix와 `HiddenRows` 이후 channel-scale/residual suffix도 기준과 byte-identical하다. final RMSNorm EPS, residual 순서, 외부 `#[device]` signature를 수정하지 않았다. `validate_layout.py`가 row/column 전수 대응과 변경 범위를 검사한다.

## scalar padding 처리와 최초 실패

local rows=30/15는 4의 배수가 아니므로 local row축을 `H/4`로 쪼개지 않았다. 요청한 경로대로 `contract_lane(H%R,1#8)` → FP32 inter-slice reduce → BF16 `1#16` cast → `commit_trim(1#4)`로 각 BF16 결과에 8-byte DM slot을 확보한다.

첫 4-column 시도에서는 `[H%30,1#4]`에서 canonical `[H%1920]`로 typed DMA가 padding까지 제거하게 했다. lowering이 `DmaCommand` destination stride `2/60 bytes`가 8-byte alignment를 만족하지 못한다고 거부했다. 당시 소스와 진단은 `failed-attempts/`에 보존했다. 그 폴더의 2-column direct-compact 소스는 같은 경로의 **미실행 초안**이다.

성공한 경로는 다음과 같다.

```text
분산 결과 [local rows, 1#4]
  → typed DMA: cluster별 단일 Slice의 [H%1920, 1#4]
  → TU transpose: Time=H/4%480, Packet=H%4#16
  → commit_trim(H%4): canonical BF16 [H%1920]
  → cluster Dummy2를 H/1920으로 복원
  → 기존 HiddenRows 배치, channel scale, RMSNorm, residual
```

typed DMA 동안 8-byte scalar slots를 유지하여 destination stride 문제를 없앤다. 모든 1920 rows가 연속된 뒤에만 factor4 compaction을 수행한다. 이 경로는 두 후보 모두 성공했으며 scalar compaction은 값을 더하거나 재반올림하지 않는 BF16 이동이다. 관련 API는 [std 0.8.1 memory](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/tensor/memory.rs), [transpose](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/transpose.rs), [commit](https://docs.rs/crate/furiosa-opt-std/0.8.1/source/src/engine/commit.rs)다.

## schedule에서 확인한 직렬 구간

activation DMA 다음 weight DMA가 실행되고, 그 weight 결과를 기다린 fused lookup/dot이 시작된다. 8/4/2의 weight DMA 시작은 3,159 / 3,931 / 5,473이고 완료는 16,671 / 17,443 / 18,985다. weight DMA duration은 동일하여, 더 긴 연속 HBM segment의 이득은 이 compiler의 static schedule에서 관찰되지 않았다.

fused lookup/dot 완료 뒤 cluster-local 결과 DMA는 469 / 504 / 575 cycles다. 4/2에는 compaction 2,185 cycles가 추가된다. 이어 기존 cross-cluster-to-HiddenRows DMA 808 cycles와 후속 계산이 동일하게 진행된다. 4-column의 전체 증가 2,992 cycles는 activation 증가 772 + gather 증가 35 + compaction 2,185와 정확히 같다. 2-column의 증가 4,605도 activation 증가 2,314 + gather 증가 106 + compaction 2,185와 같다.

이는 해당 schedule의 관측된 직렬 의존 구간에 대한 설명이다. context durations를 전부 합쳐 elapsed time으로 취급하지 않았고, 공개 schedule에 없는 하드웨어 stall이나 resource dependency를 추정하여 확정하지 않았다. 더 큰 segment가 실제 NPU 메모리 성능에 주는 영향은 실측 전에는 알 수 없다. 이 static 회귀 후보에 NPU job을 쓰는 우선순위는 낮다.

## 보존 파일과 재현

`projection.8.rs`와 `ops.rs`는 기준 snapshot, `projection.4.rs`/`projection.2.rs`는 성공한 완성 소스다. `columns-4.patch`/`columns-2.patch`는 기준 대비 projection만 변경한다. `evidence.json`에 schedule lifetimes, hashes, mapping, 실패 원인을 기록했다. raw schedule은 Docker scratch와 host `/tmp`에 보존하고 후보 폴더에는 요약을 넣었다.

각 후보를 `/tmp/moa-output-column-sweep/src/device/sliding/projection.rs`에 복사한 뒤 실행했다. full source/manifests는 root의 `/tmp/moa-output-column-split`에서 복사했으며 다른 agent의 live source를 변경하지 않았다.

```sh
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-output-column-sweep; export CARGO_TARGET_DIR=/tmp/moa-output-column-sweep/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-output-column-sweep/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedules/sliding_attention_output-4.json 2>&1 | tee logs/columns-4-padded-gather.log'
```

2-column은 source를 교체하고 dump/log의 `4`를 `2`로 바꾸어 실행했다. 두 명령 exit 0, exact kernel 1개 성공이다. 미사용 `broadcast_sliding_heads` 경고 외 오류는 없다. NPU/Git/live source 작업은 수행하지 않았다.

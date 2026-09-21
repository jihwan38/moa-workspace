# E53 FFN pre-RMS의 sharded 입력 경로

두 isolated exact compile 모두 성공했지만 전체 정적 비용이 증가해 NPU에 제출하지 않았다. Live E53 source/binary는 바꾸지 않았다. 기준 FFN image는 E37/E47/E53에서 같은 `61ad0c50…`이며 기존 schedule162,209를 사용한다.

`ops.rs::decoder_feedforward`는 원본 residual DM을 보존하고 H480×8 shard로 옮겨 pre-RMS를 수행한다. 변경한 private `feedforward_sharded`는 이 BF16 shard를 직접 두 cluster/ring16의 전체 H 입력으로 모아 복제한다. 기존 generic normalize의 F32 전체-H gather를 생략한다. 다른 private caller `feedforward`는 입력을 shard로 변환하는 wrapper를 유지한다. 공개 signature·QKV·O·FFN weight decode/dot·GeGLU·후단 residual은 그대로다.

| 정적 지표 | E53 | inter-slice 후보 | gather 후보 |
|---|---:|---:|---:|
| makespan | 162,209 | 162,735 | 164,216 |
| Main 합집합 | 125,274 | 124,248 | 122,240 |
| DMA 합집합 | 135,607 | 134,615 | 134,934 |
| Main/DMA 겹침 | 104,921 | 102,732 | 99,891 |
| Sub 합집합 | 19,378 | 19,378 | 19,378 |
| pre-RMS Main 작업량 | 4,407 | 3,381 | 1,373 |
| pre-RMS 시작–완료 | 3,675–8,349 | 3,675–7,323 | 30,150–32,165 |

첫 후보 `ops.rs`는 기존 `normalize_sharded_inter_slice`를 호출한다. 원래 RMS보다 Main1,026과 초기 activation DMA992를 줄였지만 Main/DMA 중첩도2,189 줄었다. 두 번째 `ops.gather.rs`는 기존 `normalize_output_fused`의8-partial Switch gather를 사용한다. RMS 연산량은 더 작지만 gate weight DMA가2,679–29,154에 먼저 배치되고 pre-RMS가30,150에 시작하여 입력 준비가 늦어졌다. Main/DMA 중첩이5,030 줄었다. 이 수치는 compiler 스케줄에 관한 것이며 실제 NPU 지연의 완전한 원인이나 성능 개선을 증명하지 않는다.

수치적으로 단순 복사 제거만 한 변경은 아니다. 첫 후보도 EPS addition이 ClipAdd→FpAdd로 바뀌고 sqrt pass가 합쳐진다. Gather 후보는8개 F32 partial의 합산 순서도 달라질 수 있다. BF16 cast는 좌표별 normalize 완료 뒤라는 산술 경계를 유지하지만 전체 후보의 bit-exact 동치를 주장하지 않는다. [독립 source/caller 감사](review.md).

[근거·해시](evidence.json)에 두 source 조합, schedule/graph/image를 기록했다. 공통 MLP는 `mlp.rs`이며 `ops.rs`와 `ops.gather.rs`를 서로 다른 후보로 구분한다. 각각의 원본 diff를 보존했다. HBM 입력/weight/fixture/oracle은 수정하지 않았다.

재현 scratch는 `/tmp/moa-ffn-sharded-input-rms-e53`다. 동결 E53 clean source에 해당 ops와 공통 MLP만 적용하고 SDK0.8.1, 고정 Cargo.lock, 독립 target/output 경로에서 다음 명령을 실행했다.

```bash
cargo furiosa-opt compile ops::decoder_feedforward --exact \
  --dump-schedule schedule.json --dump-graph graph.json
```

새 입력 계약이나 실제 DMA/스케줄 제어 근거 없이 이 두 변형을 다시 시도하지 않는다. 정적 회귀 때문에 screening에서 제외했으며 NPU 정확성 검사를 통과했다고 표시하지 않는다.

# E123 whole-v1 독립 검토 — 2026-09-27

검토 대상은 `moa-dev/experiments/candidates/ffn-direct-boundary-recompute/`의
`whole-v1/mlp.rs`, `whole_verify.py`, `evidence/e123-whole-v1/{graph,schedule}.json`이다.
소스·기록 읽기와 기존 verifier 재실행만 수행했다. 새 compile/Arena 실행이나 후보 수정은 없다.
현재 frozen whole-v1에서 projection/scale/row 매핑 또는 복사 범위의 구체 구현 오류는 발견하지 못했다.

## 소스와 실제 producer/consumer

- `build-e123-whole-v1-repro.json`의 source-origin 46개 입력을 E104 Git
  `7f891f9c48e670fe014b2ac5bd2c92bae3bac433` 및 pinned Cargo.lock에서 다시 읽어
  크기/SHA-256을 대조했다. 모두 일치했다. 변경 파일은 shared `mlp.rs`뿐이다.
- `v4/mlp.rs`는 E104 `mlp.rs`와 바이트 단위로 동일하다. whole-v1의
  `fn cast_gathered_rows`부터 EOF까지도 E104와 동일하다. 따라서 quantizer,
  GeGLU, down, global-scale/BF16 경계는 그대로이며, `src/ops.rs:212`의
  preRMS → FFN → postRMS → residual/layer-scalar 경로도 변경되지 않았다.
- `whole-v1/mlp.rs:30`은 up/gate FP4를 각각 compact DM으로 읽고 raw Main
  복사로 joint `[Projection=2, Row=30, H=3840]`를 조립한다. `:97`의 한 native
  stream은 lookup → 3-term contraction → block16 F32 partial → input restore
  scale 순서다. `:49`는 projection별 FP8 blockscale을 F32로 변환해 곱한 뒤
  H축을 F32로 합산한다. `:111`의 projection 선택과 `:114`의 gather 이후
  `:121`에서 BF16으로 변환한다. row ownership은 `l=7680*c+30*s+r`로 유지된다.

## whole graph의 index·scale·마지막 주소 확인

- FP4 assembly `RenegadeCommand#O59`는 `IO(2)` up → projection index 0,
  `#O64`는 `IO(3)` gate → index 1이다. `IndexAccess#O57/#O62`의 stride는
  115200 FP4 elements = 57600 B이다. 두 raw 복사의 fetch/commit은 각각
  1800×64 FP4 elements = 57600 B/slice이며, destination 구간은
  `[0,57600)`, `[57600,115200)` B로 정확히 나뉜다. graph tensor의 공통 base
  address만 보고 두 복사가 같은 위치에 쓴다고 해석하면 안 된다. index 연결을 포함해야 한다.
- joint producer `RenegadeCommand#O68`은 T79 FP4 joint + T65 activation TRF
  + T64 restore-scale VRF를 읽고 T84 F32 `[2,30,240]` partials를 쓴다.
  commit은 3600×4 F32 = 57600 B/slice이다. `IndexAccess#O70/#O125`는
  T84의 projection 0/1을 7200 F32 elements = 28800 B stride로 선택한다.
- up reduction `#O82/#O92/#O102/#O112/#O122`는 `IO(5)` up blockscale,
  gate reduction `#O137/#O147/#O157/#O167/#O177`는 `IO(6)` gate blockscale을
  사용한다. 각 partial/scale/output tile의 row index는 모두 0,6,12,18,24로
  일치한다. 각 tile은 6×240 F32 = 5760 B를 읽고 6 F32 rows를 쓴다.
  up partial 구간은 `[0,28800)`, gate는 `[28800,57600)` B이며, gate 마지막
  tile `#O177`은 `[51840,57600)` B로 joint 할당 끝과 정확히 일치한다.
  모든 reduction op에서 `FpMul0` 다음 `ReduceAccTree: LReduce(+) [axes=H]`,
  `commit_in_size: 8B (=2EA)`를 확인했다.
- FP4 ingress `DmaCommand#O53/#O54/#O214`는 각각 `IO(2)/IO(3)/IO(4)`를 읽는다.
  각각 논리 읽기 합계와 마지막 end-exclusive가 모두 29,491,200 B이며,
  public allocation 크기도 같다. scale ingress `#O72/#O127/#O212`의
  `IO(5)/IO(6)/IO(7)`도 합계·마지막 끝·할당이 모두 3,686,400 B이다.
- schedule에서 assembly 두 Main 실행은 25618–27682와 51778–53842,
  joint native는 77514–95777이다. 첫 scale/reduction 소비는 그 이후다.
  이 숫자는 정적 의존성 확인에만 사용했으며 NPU 지연이나 더할 수 있는 절감량으로 해석하지 않았다.

`python3 -B .../whole_verify.py e123-whole-v1`도 PASS했다. 단 이 verifier의
`:49`–`:67` 자동 계약 검사는 FP4 HBM ingress와 raw-copy sequencer까지이며,
projection index, 서로 다른 scale IO, 10개 reduction tile의 대응은 assert하지 않는다.
위 내용은 그 공백을 별도 graph 추적으로 확인한 결과다. 자동 PASS를 전체 numerical 검증으로 읽으면 안 된다.

## probe 결과가 입증하는 범위

`v4/boundary_probe.rs:26`은 FP4 코드와 BF16 입력으로 독립 F64 oracle을 계산하고,
선택한 block/final row가 F32로 표현 가능한지 확인한다. job 105009에서
`direct_fixed`와 `hybrid`는 각각 4 cases × 2 projections × 15360 rows =
122880 F32 row comparisons를 exact PASS했다. materialized는 첫 launch에서
timeout/exit 101, 비교 0개다. 전체 job FAILED이며 paired materialized PASS가 아니다.
README `:31`과 `verification-ffn-boundary-v4.json`도 이 차이를 보존한다.

추가 한계: host `:78`–`:84`는 up/gate에 같은 `bs`를 전달한다. 따라서 이 probe만으로는
두 projection의 blockscale 인자 swap을 검출할 수 없다. 위 whole graph의 서로 다른
`IO(5)`/`IO(6)` 연결 확인은 별도의 근거다. 정밀 oracle PASS를 전체 FFN 입력 일반성,
BF16 consumer 경계 또는 timeout 원인의 확정으로 확대하지 않았다.

## 전체 FFN 결과와 비용의 별도 근거

kernel agent가 수집한 `e123-whole-screen-results.json`/`...-summary.json`과
raw `moa-dev/logs/e123-whole-screen-105045.log`를 읽었다. raw SHA-256
`430b02073121bfe41e7d6cdcb74064f491059fac5877a39fdd9124ffa7f2092a`를 재확인했다.
job 105045는 SUCCEEDED이며 E104 before → E123 → M03 → E104 after의
4개 stage 모두 exit 0, stage당 15 comparisons PASS다. FFN은 stage당 동일한
표준 입력 3개이다. 총 FFN 비교 12개를 서로 다른 입력 12개로 세면 안 된다.

| FFN stage | Median cycles |
| --- | ---: |
| E104 before | 336282 |
| E123 | 329395 |
| M03 | 331848 |
| E104 after | 335390 |

E104 bracket 335836 대비 1.917900% 개선, M03 대비 0.739194% 개선이다.
이는 한 allocation의 screen 결과이며 반복·확장 검증이나 채택 근거로 확대하지 않는다.
Root가 5% 미만 screen 종료를 결정했으며 이번 리뷰로 추가 NPU 테스트를 요청하지 않았다.

Root가 발견한 frozen recipe의 assembly read 중복 나열은
`cost-accounting-correction.json` 및 README의 수정표로 명시됐다.
weight-only DM read는 E104 decode 56.25 + native 112.5 = 168.75 MiB,
E123 assembly 56.25 + joint native 56.25 = 112.5 MiB다. listed DM write도
168.75 → 112.5 MiB다. 실제 raw 복사 두 개, compact HBM ingress,
28.125 MiB F32 partials, scale/reduction과 후속 FFN 작업은 코드 안에 남아 있다.
이 논리 bytes를 bus 실측이나 동등한 runtime 절감으로 해석할 근거는 없다.

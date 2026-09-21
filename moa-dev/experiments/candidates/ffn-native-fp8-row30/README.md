# Native FFN 실제30행, scalar padding 저장

`ops::decoder_feedforward` exact compile이 첫 시도에 통과했다. 정적 makespan은 row2/fullmemset 기준 **182,868 → 170,176 cycles(-6.94%)**, 명령은224 →196개다. **E41 job70562에서 공식3개 입력 구성의15개 출력 비교가 모두 PASS**했다. FFN 실측은429,663/426,154/426,442 cycles, 중앙값 **426,442**다. E37 best379,741보다 느려 **미채택**했다. 기록은 [e41-native-row30.json](../../../../docs/experiments/e41-native-row30.json), 원본 로그는 `moa-dev/logs/e41-native-row30-job-70562.log`다. 정적170,176과 실측426,442의 차이는 관측값이며 원인을 단정하지 않는다.

기준은 `../ffn-native-fp8-full256-row-two/mlp.rs`(SHA `bf5e8c153ec6b226136cf0a9887267fedefd562a67b74ef34d47429960dbf602`)다. 후보 `mlp.rs` SHA는 `dff506dea523b17c7e32c4c200cda8c5138a800ea61e40ed805da1b1e3df3129`다. 다른 src 파일은 기준 scratch와 모두 동일하며 `source-manifest.json`으로 확인했다. 이 subagent는 live source/Git/NPU를 변경하지 않았고, 실측 제출은 root가 수행했다. E41 artifact의 MLP SHA는 후보와 같으며 통합 ops snapshot은 scratch와 다르다. 정적·실측 수치를 같은 실행 조건으로 간주하지 않는다.

## 데이터 경로와 제거한 작업

`feedforward → project_up_and_gate → project_rows_with_scale_contraction → compact_full256_rows → geglu` 경로 중 up/gate 저장 방식만 바꾼다.

- `FfnRow=32`를 실제행 수30으로 바꾼다. 두 cluster의256개 slice는 모두 유효하다.
- FP4 HBM `[Dummy2,FfnSlice/2,FfnRow,FfnSlice%2,H]`를 실제30행 DM으로 DMA한 뒤 FP4→FP8 LUT 결과를 `[FfnRow,H]`에 직접 commit한다.
- block scale도 실제30행 `[FfnRow,H/16]`로 HBM에서 직접 DMA한다. weight/scale 네 buffer의 전체 memset, dummy2행,30→32 copy와 tile이 모두 사라진다.
- 첫 native FP8 pair DPE는30행×3840입력을 처리한다. packet에 네 개16항 block dot을 유지하고, `hi + lo/16` 합과 동적 input scale 복원 뒤 각 block을 BF16으로 반올림한다. 기준과 같은 수식·같은 cast 위치다.
- 두 번째 BF16 DPE는30행×240 block partial/scale을 곱한다. 결과의 **기존 BF16 cast를 그대로 유지**하고 `commit_trim::<m![1#4]>()`로 각 scalar를8B slot에 저장한다.
- `compact_full256_rows`가 row별 유효 scalar만 fetch하고 두 slice를 모은다. row 두 개×slice 두 개의 실제 BF16 네 값을8B로 commit한다. 여기에 산술·추가 cast가 없다.

이후 GEGLU/global scales/down projection/postnorm/residual 경로와 runtime quantizer를 포함한 `feedforward` 이하 source suffix는 byte-identical이다. 실패한 down-scale 후보는 사용하지 않았다.

## 주소와 padding 증명

cluster `c∈[0,2)`, slice `s∈[0,256)`, local row `r∈[0,30)`의 원래 행은 다음과 같다.

```text
L = 7680*c + 60*floor(s/2) + 2*r + (s%2)
```

원래 HBM의 연속행 순서를 `[2,128,30,2]`로 factorize한 순서와 정확히 같다. 모든15,360행을 중복·누락 없이 덮으며 H=3840과 scale axis H/16=240은 그대로 연속이다. 한 slice당 FP8 weight115,200B와 FP8 scale7,200B 전부 유효값이다. 초기화가 필요한 tail 행 자체가 없다.

둘째 DPE 결과는 `Element=[FfnRow,1#4]`다. row `r`의 유효 BF16 scalar는 byte offset `8*r`에 있고, 나머지6B는 transport padding이다. 다음 fetch의 `Time=[FfnRow], Packet=[1#16]`도 논리 scalar 수1을 유지한다. padding을 row/data로 rename하거나 유효값으로 승격하는 reshape는 없다.

Switch 후 group `g=floor(s/2)`의 stream 순서는 `(r=0,s%2=0),(0,1),(1,0),(1,1),...`이다. transpose는 `[FfnRow/2] × [(FfnRow%2,FfnSlice%2)#16]`로 실제 네 BF16을 묶고, trailing padding만 trim한다. 최종60개 BF16은 원래 연속행 `[60*(128*c+g),...,60*(128*c+g)+59]`와 같다. 이어지는 원형 typed DMA가 실제 group만 canonical `UpGateRows`로 옮긴다.

`validate_layout.py`는 전15,360행과3,840개8B packet을 확인한다. scalar slot의 padding을 poison sentinel로 채운 모형에서 모든 fetch/gather의 선택값이 실제 scalar임도 확인했다. 이는 좌표 모형과 source 불변성 검사이며 compiler/backend의 런타임 정확도 oracle은 아니다.

## 정적 결과

| 항목 | row2/fullmemset | 실제30행 |
|---|---:|---:|
| Full FFN makespan |182,868|170,176|
| 명령 수 |224|196|
| weight 전체 memset (각 projection) |15,626|없음|
| scale 전체 memset (각 projection) |1,226|없음|
| FP8 첫 DPE (각 projection) |7,964|7,484|
| block scale BF16 LUT |1,463|1,388|
| block scale TRF load |2,183|2,063|
| BF16 둘째 DPE |745|717|
| row2 Switch gather |325|325|
| weight HBM DMA |29,427|29,427|
| scale HBM DMA |6,565|6,565|

Main/Sub/DMA는 겹쳐 실행되므로 표의 개별 감소량을 합쳐 makespan 감소로 해석하면 안 된다. 이 후보도 weight HBM DMA 비용은 그대로다. 결과 scalar DM은 각 projection에서512×30×8=122,880B이며, 실제 정보는512×30×2=30,720B다. 행 padding을 없애는 대가로 중간 scalar의 저장 간격이 커진다.

공개 schedule에서 둘째 DPE O262→T125→gather O267 및 O335→T148→gather O347의 실제 dependency와 저장 shape를 확인했다. 각각 둘째 DPE 종료 이후 gather가 시작된다. `inspect_schedule.py`와 `schedule-summary.json`에 해당 tensor의 크기/shape/소비 관계를 기록했다.

## 수치 조건

행끼리는 각각 독립 dot이므로 row32 중 마지막 두 zero행을 제거하는 것이 실제30행의 연산 순서나 값을 변경하지 않도록 설계했다. native helper의 산술/cast는 source 비교로 유지됨을 확인했다. 최초 block BF16 반올림과 두-term FP8 input 근사는 기준 후보에서 상속되며, 원래 BF16 FFN과 모든 입력에 대해 동일하다는 보장은 없다. 이번 변경은 BF16 반올림을 추가하거나 cast를 이동하지 않는다. 새 scalar 저장/gather를 포함한 root 통합 커널은 E41의 공식3개 입력 구성에서 PASS했다. 이것이 모든 입력이나 다른 compiler allocation에서의 정확성을 보장하지는 않는다.

## 재현과 근거

격리 scratch는 Docker `moa2026-dev:/tmp/moa-ffn-native-fp8-row30`다. row2 scratch의 `src`, Cargo 파일과 toolchain을 복사하고, 이 폴더의 `mlp.rs`만 덮어쓴 후 다음 명령으로 컴파일했다.

```bash
./moa-dev/moa bash -c 'set -o pipefail; cd /tmp/moa-ffn-native-fp8-row30; export CARGO_TARGET_DIR=/tmp/moa-ffn-native-fp8-row30/target; export FURIOSA_OPT_OUT_DIR=/tmp/moa-ffn-native-fp8-row30/target/furiosa-opt/kernel; cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule schedules/row30.json 2>&1 | tee /tmp/moa-ffn-native-fp8-row30/compile-1.log'
python3 moa-dev/experiments/candidates/ffn-native-fp8-row30/validate_layout.py
python3 moa-dev/experiments/candidates/ffn-native-fp8-row30/inspect_schedule.py /tmp/moa-ffn-native-fp8-row30-baseline-schedule.json /tmp/moa-ffn-native-fp8-row30-schedule.json
```

- compiler/kernel version: Furiosa Opt0.8.1. `compile-1.log`는 dead_code warning 두 개 외 오류가 없다.
- kernel image SHA: `638a4a3e396e1a29344201a0f8656bd396fdad137104e9b9dd42267e7424da88`.
- SDK 실제 구현: `/tmp/moa-opt-std-release/furiosa-opt-std-0.8.1/src/engine/commit_adapter.rs:145`의 `commit_trim`, `:232`의 `config_commit_trim` 위임; `src/engine/fetch.rs:129`의 `fetch`, `:218`의 `config_fetch` 위임. 이번 scalar padded commit/fetch mapping은 exact backend lowering까지 통과했다.
- 공식 [padding 문서](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/mapping-expressions.html#padding)와 [tensor semantics](https://developer.furiosa.ai/furiosa-opt/book/mapping-tensors/tensor-semantics.html)의 논리값/padding 구분을 따른다.
- 확인일:2026-09-21. raw schedule은 Docker scratch와 host `/tmp`에 두고 저장소 후보에는 summary와 hashes만 남겼다.

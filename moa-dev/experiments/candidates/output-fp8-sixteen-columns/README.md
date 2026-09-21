# E37 O projection: 16×256 column 분할

E37의 O core를8×512에서16×256으로 나눈 단일 후보다. exact `ops::sliding_attention_output`은 **24,997 → 24,621 cycles(−376,−1.50%)**로 줄었다. baseline1회와 후보1회 모두 compile PASS이며 추가 variant는 만들지 않았다. root의 E47 동일 package3회(jobs70664/70681/70688)는 모두 공식15 comparisons PASS다. O job median44,309/44,663/44,588의 중앙값은44,588로 E37 반복 기준45,748보다2.535% 낮고 range는0.794%다. 추가 run3..8 job70694도 30 comparisons PASS였으며 root가 E47 O16을 채택했다. 공식 run0..2와 합쳐 서로 다른 9개 입력 구성에서 각 kernel을 검증했다. 반복 측정은 새 입력 구성을 추가하지 않는다.

[projection.rs](projection.rs) SHA256 `db048b93d9b768bb4cb9cb6f8fd603783789d4a5eb609e42a00a23c69a330718`. [kernel image](../../artifacts/output-fp8-sixteen-columns-static/sliding_attention_output.bin)는87,040 bytes, SHA256 `00d8a969edf26d5a34fbf9bc5430cdd889f60fa58f122ebcbd7aa2208f584353`다. [candidate.patch](candidate.patch), [evidence.json](evidence.json), schedule/graph/log와 검증 결과를 함께 동결했다. live source, Git, NPU는 변경하지 않았다.

## source 범위와 배치

[E37 snapshot](../../artifacts/e37-kernel-specific-rms/provenance.json)의 source43개와 manifest3개를 새 scratch `/tmp/moa-output-fp8-sixteen-columns`에 복사했다. baseline projection SHA `b58d2ce261f1d12d2a219fd4edc2a715b163b4bf033d5e973ca3d64ba2df08e5`에서 `project_output` 내부와 private O quantizer/layout만 수정했다. Q/K/V 관련 모든 함수, 공개 `project_output` signature, `ops.rs`, H480×8 channel-scale/RMS/residual epilogue, FFN, 공식 harness는 byte-identical이다. [source manifests](candidate-source-manifest.json)가 실제 compile scratch의 차이를 기록한다.

O 입력4096개를256개씩16 column으로 분할한다. 각 cluster는1920 output rows를 담당하며 `h=1920*c+120*g+r`이다. c∈[0,2), g∈[0,16), r∈[0,120), q=256*k+t에서 k∈[0,16), t∈[0,256)다. physical slice는 `16*g+k`, local weight는 `[r,t]`다. 즉2clusters×16rowgroups×16columns=512개 slice를 전부 사용한다. FP8 weight량은 slice당120×256=30,720B로 기존60×512와 같다.

각256-input column에서 runtime amax → power-of-two scale → FP8 hi/lo를 만든다. native FP8 paired dot은120 local rows 각각에 대해256 columns를 합산한다. hi+lo/16 후 해당 column의 input scale을 곱한 F32 partial을 DM에 저장하고, 별도 stream에서16개 column slice를 F32로 합산한 다음 BF16로 cast한다. 기존과 같은4-row transpose를30번 적용해240B의120 BF16 결과를 commit한다. `OutputRows=[H/120%16,1#16]`의 padding은 그대로 두고 cluster 축을 `H/1920`으로 rename한 뒤 typed DMA가 기존 H480×8 epilogue 배치로 옮긴다.

[verify_layout.py](verify_layout.py)는3840 rows,4096 columns,61,440 physical row/column tiles를 검사한다. 독립적인 row·column bijection의 곱이15,728,640 weights를 정확히 덮는다. graph에서도 `GReduceAddF [ratio=16]`, F32 fetch→BF16 cast, `commit_in_size:8B`, `commit_projection:0+[30:4,4:1]`를 확인했다. [verification.json](verification.json)은 PASS다.

## payload와 정적 비용의 교환

| slice당 payload | E37 8×512 | 후보16×256 |
|---|---:|---:|
| BF16 입력 DM |1,024B|512B|
| packed hi/lo FP8 및 TRF 유효 입력 |1,024B|512B|
| hi F32 VRF |2,048B|1,024B|
| input-scale VRF |32B|32B|
| FP8 weight DM |30,720B|30,720B|
| F32 partial(1#8 padding 포함) |1,920B|3,840B|
| reduction BF16 결과의 physical local 저장량 |120B|240B|

TRF 유효 payload가 절반이라는 것은 고정 TRF bank 예약량도 절반이라는 뜻은 아니다. schedule의 bank 크기는 양쪽에서 같다. unique HBM 입력/weight 총량도 같다. rowgroup 복제와 로컬 처리량이 달라지는 비교다.

[weight-dma-mapping.json](weight-dma-mapping.json)에 실제 DMA command를 보존했다. 논리적으로512열 stripe를256열로 줄였지만 **baseline도 이미 `Qs_256=2:256`과 최내부256B 전송으로 분해**된다. 후보는 그2-way 내부축을 없애고 `Qs_256=16` slice축으로 옮긴다. 실제 차이는 destination row stride512→256, local rows60→120이며 “물리 HBM burst가512B에서256B로 절반”이라고 해석하지 않는다.

| instruction duration | E37 | 후보 | 차이 |
|---|---:|---:|---:|
| 입력 HBM→DM |1,318|933|−385|
| 입력 DM copy |947|691|−256|
| weight HBM→DM |13,512|13,383|−129|
| native pair Main |1,238|1,242|+4|
| column reduction Main |390|522|+132|
| H480 epilogue gather |469|446|−23|
| 전체 makespan |24,997|24,621|−376|

amax/hi/lo Main은각409→345, hi-VRF Sub327→295, TRF Sub391→327이다. 전체 Main duration 합5252→5196, DmaLoad16468→15954, DmaStos2298→2019다. 이 duration들을 합쳐 makespan 절감으로 계산하면 안 된다. weight load 완료17346→16709, pair 완료18584→17951, reduce 완료18974→18473, gather 완료19448→19072를 거쳐 변경하지 않은 epilogue가 정확히376 cycles 먼저 시작한다. 마지막600-cycle sync는 같다.

## 수치 범위와 검증 한계

quantizer의 절대값 mask, amax floor2^-119, scale계수1/128, hi/lo계수1/16과16은 그대로다. finite BF16 입력에 대해 scale s∈[2^-126,2^120], |x/s|<256이고 RN E4M3에서 lo 입력의 절댓값은128 이하다. 각256-group은 기존512-group의 부분집합이므로 amax와 scale이 기존보다 커지지는 않는다. 하지만 **scale 그룹 자체가 달라지고 dot/F32 inter-slice 합산 순서도8→16으로 바뀌므로 bitwise 동치나 모든 출력에서 오차 감소를 주장하지 않는다.** 기존 FP8 근사·cancellation·외부 scale의 overflow 한계도 남는다.

finite FP8 weight 최대448에서256개 hi dot의 보수적 절댓값 상계는29,360,128, lo dot은14,680,064다. 이는 scale 적용 전 dot의 범위일 뿐 이후 모든 연산의 overflow-freedom을 증명하지 않는다. BF16 cast는 기존처럼 column input-scale 적용과 F32 column 합산 뒤에 있으며 channel weight-scale, post-RMS, residual 순서는 그대로다.

정적 compile/shape 검사 뒤 root의 첫 hardware screen이 통과했다. [E47 기록](../../../../docs/experiments/e47-output-sixteen-columns.json)과 [job70664 로그](../../../logs/e47-output-sixteen-columns-job-70664.log)의 결과는 Q94,662/O44,309/FFN377,274, 공식15 comparisons PASS다. run0..2 O cycles는49,067/43,476/44,309이다. 첫 screen의3.15%는 후속3회 중앙값 기준2.535%로 갱신했다. [반복 측정 기록](../../../../docs/experiments/e47-output-sixteen-columns-repeats.json)의 aggregate는 Q94,662/O44,588/FFN377,564이다. [추가 검증 기록](../../../../docs/experiments/e47-output-sixteen-columns-extra-correctness.json)의 job70694는 run3..8의 30 comparisons PASS다. O는 공식 3개와 추가 6개, 총 9개 입력 구성의 34,560개 출력값을 기존 tolerance로 검사했다. 추가 입력의 latency는 진단값이며 공식 점수나 반복 성능 중앙값에 합치지 않는다. `verification.json`의 hardware 미검증 표기는 최초 정적검사 시점이며 후속 측정은 `evidence.json.hardware`에 기록한다.

후속 [correctness coverage 감사](correctness-audit.md)는 실제 run0..8 입력checksum과CPU activation 재구성으로검사한경계및아직없는stress cases를구분한다. 새NPU/compile은추가하지않았다.

[제출 artifact 검증](repeat-artifact-verification.json): 다운로드된 실제3개tar의binary/fixture/entrypoint/job.yaml이모두같고,frozen O kernel87,040bytes가제출ELF offset1,181,309에정확히한번포함됨을확인했다. captured source43개+manifest3개도exactcompile입력과같다.새build/NPU실행은없었다.

[채택 source/build 감사](../../../../docs/experiments/e47-source-build-audit.json)는 source 43개가 동결 E47과 같고 전체 host executable을 측정 package와 byte-identical하게 재빌드했음을 기록한다. QKV/O/FFN device image 각각은 공식 host와 추가 입력 host에 정확히 한 번씩 들어 있다. root는 이 검증과 추가 입력 PASS 후 O16을 채택하고 E37 QKV/FFN을 유지했다. 이는 9개 입력 구성의 검증이며 모든 가능한 입력에 대한 수치 증명은 아니다.

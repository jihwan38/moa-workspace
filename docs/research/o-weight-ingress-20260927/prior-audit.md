# O weight ingress 선행 실험 감사 — 2026-09-27

범위: main `48bea3d`의 E122를 기준으로 기존 source/recipe/graph와 저장된 Arena 로그만 읽었다. 새 compile·host build·Arena·live source 변경은 없다. E122 O는 E53/E104 image `359473814839effc16ade242bad3c598a93610601bc650f550ff44228898d88b` 그대로다. 모든 아래 `candidates/…` 경로의 실제 prefix는 `/home/jihwan38/moa-workspace/moa-dev/experiments/`다.

재계산 자료: `docs/research/o-weight-ingress-20260927/prior-audit.py`, `docs/research/o-weight-ingress-20260927/prior-audit.json`. 34개 report의 50 report-runs, 42개 unique job에서 원시 로그 SHA-256과 해당 O stage의 median·samples를 다시 확인했다. 혼합 batch는 `STRUCTURAL_BEGIN candidate_N`을 잘라 각 후보만 읽었다. M01/M02의 `verify.py --check`도 재실행해 source 경계·동결 build/package·기존 Arena stage 검사를 통과했다. 이는 새 성능 실험이 아니다.

## 먼저 구분할 결과

- **C09는 하드웨어 회귀 실험이 아니다.** 정확 compile 후 정적 27,239→27,432(+0.7085%)에서 예산상 종료했고 Arena는 없었다. 새 측정으로 weight 전송/준비와 dot의 실측 비용 관계가 달라졌다면, 같은 동결 후보의 첫 NPU 평가는 합리적인 제한 진단이다. 새 architecture로 세지 않는다.
- **C10은 O가 아니라 QKV다.** Q의 Main lookup→weight TRF가 특정 target의 `StoTrf` lowering에서 막혔다. O 코드가 바뀌지 않았고 KV도 독립 검증하지 못했다. 용량만 맞추어 O도 지원된다고 하거나, 반대로 O/TRF 전체가 불가능하다고 하면 둘 다 범위를 넘는다.
- **M01/M02는 기본 correctness PASS인 전체 경로 회귀다.** weight ingress/producer 시간이 줄었다는 정적 근거 없이 새 partial 이동·별도 reduction·broadcast·compact 비용을 붙였다. 다만 static duration만으로 NPU의 bank/stall 원인을 확정할 수 없다.
- **E47의 더 빠른 FP8 O는 일반 정확도의 대안이 아니다.** 공식 3+6 입력과 3회 측정을 통과했지만 cancellation probe에서 3,840/3,840 outputs가 실패했다. E53은 BF16 activation 유지와 scale 전 F32 보존을 함께 복구했다.
- **row interleave·열 수 변경은 여러 번 했다.** 이번 행쌍 보존 4-way + 16K + direct F32 gather는 exact mapping으로 미실행이지만 기존 행 배치 계열의 재검토다. 새 이름으로 architecture 수를 늘리면 안 된다.

## 최근 O 변경의 실제 전체 경로

| 후보 / source 출발 | producer → consumer | 없앤 일 / 새로 생긴 일 | 정적·NPU 결과 | 실패 종류·재오픈 조건 |
|---|---|---|---|---|
| E64 direct / `baseline-e55` | `s=16k+r` column-major weight DM → 같은 배치에서 16K sum | full weight 재배치를 생략하려 함 | `VRU reduce axes must be innermost in Partitioning` compile 실패; NPU 없음 | 이 표현의 reduction layout 제약. HBM mapping만 바꾸고 consumer도 그대로 두는 방법은 불가. M01이 consumer를 함께 바꾼 후속 실험 |
| E64 staged / `baseline-e55` | column-major FP8 weight DM → row-major DM → 기존 fused lookup/BF16 dot/16K sum | **15MiB DM weight copy 추가**. E55에는 없는 비용 | static 27,239→44,406; weight 13,383→13,435, reorder 17,112. job72135 O128,411 vs 동시기 E55 반복51,445 = **+149.61%**, 기본 PASS | 큰 실측 회귀. 새 ingress 이득이나 재배치 제거 없이 다시 실행할 이유 없음. M01은 후자를 이미 실행 |
| C09 / E104 `7f891f9` | H-owned 2 clusters, 16×256K, row120을 60+60으로 나눠 각 weight DM → lookup/dot/16K sum → 같은 result | 일부 weight/dot 중첩; load 6→8, LUT 준비·두 개의 작은 dot task 증가 | static 27,239→27,432; weight합13,383→13,932, dot합5,063→5,326; 첫 dot와 다음 weight 1,825 중첩. **NPU 없음** | 지원 PASS/정적 budget stop. 재오픈은 첫 NPU에서 실제 중첩을 평가하거나 준비 dependency를 바꾸는 근거가 있을 때 |
| C12 v2 / E104, 실제 본문은 C09 row staging 결합 | 60+60 Main lookup/dot → padded F32 partial DM → Sub16K sum → padded DM → 별도 Main pack → F32 gather/scale | Main dot에서 VE sum을 분리; partial write/read, Main pack, 두 stage LUT 추가. local60값의 partial은 valid240B에 32B slot을 써 **1,920B/slice/stage** | static27,239→28,404. job79329 O51,862 vs50,555/50,680 = **+2.46%**; 기본15비교 PASS, 단회·samples겹침 | v1 fullgraph 내부 SFR 오류를 v2 경계로 수정. Subreduce 자체는 primitive compilePASS. 다음 weight17,547 도착까지 기다려 Subsum12,403–13,047와 다음 dot 중첩0. 다음 operand 준비가 빨라지고 partial비용까지 줄어야 재오픈 |
| M01 DMA / actual branch `8109a27`, E64 ingress+E104 arithmetic | `s=16k+r`, local120×256 → direct lookup/BF16 dot → compact F32partial → DMA `s=16r+k` →16K sum → H480 F32 scale | E64의 15MiB repack 제거, **E104 대비 제거 이득은 없음**. 245,760B partial 및 write/copy/read 등 새 valid DM edge983,040B | static27,239→29,824; weight13,383→13,435, producer5,063 동일. job95636 O55,608 vs51,007/51,274 = **+8.74%**; 기본PASS | 최초 `VectorFinal→commit` source adapter오류를 `commit_trim`으로 수정. 최종 loss는 구현 실패가 아님. partial DMA2,028+reducer416이 producer뒤 직렬. ingress나이경계비용을 실질적으로 줄일때만 재개 |
| M01 Switch / 같은 origin·family | 위 compact partial → ring256 transpose16×16+sum | DMA partialbuffer B 제거, valid DM edge491,520B와 Switch 남음 | static31,995; weight/dot위와동일, Switch+reduce4,615. same job O60,677 = **+18.65%** | DMA variant의 단순 대체도 이미실행. Switch를 공짜로 보는 재추천 금지 |
| M02 v3 / actual branch `ecdeaca`, E104 numerical contract | c가K2048소유, sliceH%256, local15×2048 →F32scalar8Bslot → cluster별 padded fullH gather/compact → H480 crosscluster gather → F32AddF+scale | 16K sum 제거; activation leader방송, 2MiB activationreplicas(E104256KiB), paddedpartial61,440B→compact30,720B, crosscluster 이동 추가 | static37,413; weight13,512, dot5,063, broadcast4,359. job95914 O78,981 vs51,010/50,901 = **+55.00%**; 기본PASS | v1 broadcast좌표, v2 commonH factorization 문제는 소스/표현 수정으로 실제통과. 새 path가 비싼 것이 최종증거 |
| M02 v4 / 같은 family | `s=32g+r`, `h=480g+32t+r`로 H480안 interleave → direct padded crosscluster gather/compact | v3의 fullH gather뒤 extra copy제거; gather3,314+compact1,225+secondpartialVRF503 등은 남음 | static32,382로v3보다 **13.45% 짧지만**, same job O89,203으로 **12.94% 더 느림**, control대비 **+75.06%** | static이 NPU순위까지 예측하지 못한 명확한 사례. HBM연속구간이2048B라는 설명만으로 재개 불가. broadcast·partial·동기화를 포함한 완전히 다른 비용기전 필요 |

구현 근거:

- E64: `candidates/memory-layout-e55/output-column-major-staged/projection.rs:160`, weight 두 번 `to_dm`은 179–180행. `output-column-major/compile-failure.log`; 실제 package 확인은 같은 family `hardware-summary.json`.
- C09: `candidates/c09-output-pipeline/two-stage-v1/projection.rs:160`, stage helper194행, `static-screen.json`, `decision.json`.
- C12: `candidates/c12-output-main-sub/two-stage-separate-pack-v2/projection.rs:160`, Main partial194행, Sub reduce211행; `static-cost.json`/README timeline과 `docs/experiments/e111-c12-output-main-sub.json`.
- M01: `candidates/m01-partial-remap/dma-v2/projection.rs:166` producer owner, 182행 F32partial, 196행 transfer, 202행 sum. Switch는 `switch-v2/projection.rs:200`. `dma-v2/recipe.json`에 actual branch/source/reference를 분리함. `verify.py:41`은 **논리 row/partial 태그 검증**이지 emitted descriptor 전수 검사 아님.
- M02: `candidates/m02-cluster-k/khalf-v4/projection.rs:169` activation leader·Switch, 189행 weight, 207행 crosscluster gather, 219행 secondpartial VRF, 237–241행 AddF→MulF→첫BF16. `verify.py:21`은 logical 7,680 weight2048B구간·padding·두partial 합 검증이다. `analyze.py`의 compiler allocation extent와 valid payload는 서로다름.

위 modern 후보의 source·logical layout proof·graph/schedule·full package provenance·기본 NPU는 서로 다른 증거층이다. M01/M02의 기존 verifier는 모든 **emitted DMA descriptor byte**의 경계를 열거하지 않는다. 기본 NPU PASS도 전 입력 정확성 증명은 아니다. 두 후보는 단회 allocation에 전후 control과 각각 두 변형을 실행했고 추가 독립반복·extra입력은 회귀 때문에 생략했다.

## 초기 성공과 열 수 sweep을 다시 추천하면 안 되는 이유

| 계열 / 구현 출발 | 실제 바뀐 경로 | 이미 있는 증거 / 현재 의미 |
|---|---|---|
| E2 lookup+dot fusion | FP8→BF16을 DPE입력에 직접 연결하여 full BF16 weight DM materialization 제거 | O404,222→334,457(−17.26%). 현재 BF16 O도 이미 사용. predecode→DM→reload를 추가하면 이 제거 비용이 다시 생김 |
| E3 O256 / original `90f3c3f` | 한 cluster의 slice`s`가`h=256t+s`인15rows×4096을 full BF16 dot. 원래4×1024 BF16 partial/addtree 제거; scalar BF16 8Bslot+gather/compact | O334,457→195,210(−41.63%). contiguous15rows padded16의 첫 DMA factorization failure는 interleave로 수정. README의 NPU대기 문구보다 후속 e3보고서가 최신 |
| E3 direct/E1hier | full activation을 단일 큰 ring에만 의존하던 방식에서 HBM직접복제/계층방송 | E3 195,210→143,976; E1hier 145,998→116,132. 현재 O는 K256별 직접분배여서 같은 fullvector구조가 아님 |
| E8 2Hclusters | c가1920H 소유, 240validslices×8rows×4096; fullvector입력방송; cluster마다 독립출력 → canonical gather | O112,661→76,147(−32.41%). HBM축 highH이름 분리로 paddingfactorization수정. 현재 이미2Hclusters. M02의2Kclusters와 다름 |
| E9 residual8 / E26 parallel epilogue | O후단을 H480×8로 유지하며 residual을 병렬화, 이후 channel/RMS/residual 배치통일로 중간gather/왕복제거 | E9 76,147→69,651(−8.53%); E26 59,027→52,579(−10.92%). E122가 후단계열을 계승 |
| E14 BF16 O8 | Hclusters2×32rowgroups×8K, local60×512. full4096activation replica/broadcast를 K512direct로교체. **sum후BF16→scale** | O68,945→65,694(−4.72%). 현재scale전F32와numerical경계/후단이다르므로 현대O8의측정값으로대체하면안됨 |
| old `output-column-sweep` O4/O2 | 같은BF16 O8을4×1024/local30rows,2×2048/local15rows로변경. 512validslices·30,720B/slice유지 | static32,842→35,834/37,447, weight13,512모두동일, dot5,063동일. xDMA1,318→2,090→3,632; scalar gather/compact2,185추가. **NPU없음**. 15×2048타일은M02전에도이미compile됨 |
| E24 nativeFP8 hi/lo | BF16 x를amax/power2scale후두FP8항으로근사, pairedDPE→F32partial DM→8Ksum→BF16→scale | 65,662→59,027(−10.10%)이나뒤E47경계실패가근사의위험을실증. 당장복원할정확한대안아님 |
| E29 directpacked / E31 RMS | quantizerhi/lo를각component destination에직접commit하여2DMcopy제거; RMS Switchgather | E29 O52,543→49,228(−6.31%), E31 49,228→45,355(−7.87%). E29 저장최적화는유효아이디어였으나현재O에는해당quantizer가없음 |
| E37 FP8 O4 | 8×512→4×1024이나32rowgroups유지, `[Qs/1024#8]`로반만valid: 총256validslices, 61,440B/slice | static24,997→25,873. weight13,512→13,435이나DPE1,238→2,190, 물리GReduce ratio8/390동일. **NPU없음**. 논리4열=물리4wayreduction아님 |
| E47 FP8 O16 | 16×256/local120rows,512validslices. quantizer그룹256, F32partial1#8=3,840B/slice 후16Ksum | static24,997→24,621; weight13,512→13,383. jobs70664/70681/70688 O44,309/44,663/44,588, median44,588 vs45,748(−2.54%). **뒤경계FAIL, E53으로대체** |
| FP8 O32 / E47 origin | 32×128/local240rows,512validslices | static24,621→37,344. weight13,383→26,218. emittedlastpacket256→128B. **NPU없음**;128B가특정bank원인이라는결론은불가 |
| E53 BF16 O8 / clean `989461e` | 현재O16을8×512/local60rows로변경. F32dot/sum/scale→첫BF16유지 | static27,239→27,752(+1.88%), weight13,383→13,512, x933→1,318, dot5,063동일, gather470→469. **NPU없음**. actualweightpacket둘다256B; 현대O8이NPU에서느리다는증거는아님 |

각 수치는 JSON의 `comparisons`에서 report/reference/raw로그로 연결된다. 초기 report 일부는 `provenance.source_binary_relationship.verified:false`다. 원시 NPU값의 존재와 보존 source가 그 image를 생성했는지의 증거 강도는 분리해야 한다. E47은 뒤 `repeat-artifact-verification.json`에서 실제 세 upload·동결 source·image 동일성을 추가 확인했다. 초기 candidate README의 “NPU pending”을 현재 최종판정처럼 그대로 인용하면 여러 성공 실험을 놓친다.

추가 정적 실험: `output-residual-sweep`의8/16/32slice는35,557/35,527/35,457(최대−0.28%), NPU없음. 32sliceTUgather의flit packet오류를typedDMA로수정했다. `output-rms-reciprocal`은reciprocal만으로정적효과0, scalar EPS+sqrtfusion은32,842→32,560(−0.86%), NPU없음. reciprocal는CPU에서BF16불일치가관측되어bitwise동치아님. 이들도weight ingress문제의새해법으로세지않는다.

## 행 interleave의 실제 descriptor까지 확인한 내용

`output-fp8-row-interleave-two/projection.rs:174`, `output-fp8-row-interleave-four/projection.rs:174`는 **E24 8K shard**의mapping만바꾼다.

- control: `h=1920c+60g+u`, slice=`8g+k`.
- row2: `h=1920c+120g+2u+s`, slice=`8(2g+s)+k`.
- row4: `h=1920c+240g+4u+s`, slice=`8(4g+s)+k`.

둘다512slices×60rows×512columns. FP8hi/lo→paddedF32partial→8Ksum→**BF16** 뒤 `SwitchConfig::InterTranspose`로 canonical60rowgroups를 복원한다(row4 source227행, `slice1:4,slice0:8,time0:15`). staticE24 32,252→33,475/34,435; **weight13,512 동일**, 추가Switch1,223/2,183. standalone두후보는NPU미실행이다. row2를E31directpacked+parallel후단과합친E34만 job70472 O47,308 vsE31반복45,355(**+4.31%**)를측정했다. 따라서“row2/row4모두hardware실패”도,“interleave를한번도안함”도틀리다.

이번에 기존 `artifacts/output-row-interleave-schedules/o-row-{control,two,four}-graph.json`의 **graph operator17 `DmaCommand#O24`, input28→output29**를직접열거했다.

| descriptor항목 | control | row2 | row4 |
|---|---|---|---|
| src0의행loop | `H_1=60:4096` | `H_2=60:8192` | `H_4=60:16384` |
| 같은Qs분해 | `Qs_256=2:256`, `Qs_512=8:512` | 동일 | 동일 |
| PE별descriptor수 / 마지막packet | 8 /256B | 8 /256B | 8 /256B |
| source전체범위 | `[0,15,728,640)` | 동일 | 동일 |
| sourcepacket중복/누락 | 없음,61,440개 | 없음,61,440개 | 없음,61,440개 |
| destination | 512slices각`[0,30,720)` | 동일 | 동일 |
| 실제src/dstpacketmapping | 위control식전수일치 | 위row2식전수일치 | 위row4식전수일치 |

source의끝원소stride를packet폭으로분리하고, 나머지loop의Cartesian product를전수열거했다. src256Bpacket이전부정렬되어HBM전체를정확히한번덮고, dest의cluster/PE/slice/localbyte가위mapping과일치한다. graphhash·8개src/dstdimensions·exclusivebounds는JSON에보존했다. **이검사는메모리주소coverage이며, descriptor기재순서=실제HBM요청발행순서/transaction수/bankconflict라는증거는아니다.** 이미동일한총packet수와정적DMA를보이는oldrowfamily라서, rowstride만바꾼다는이유로속도개선을가정하면안된다.

## F32/scale 경계와 재사용할 수 있는 부분

`output-bf16-activation-e47/projection.rs`의현재경로는 BF16 x256 TRF, FP8weight DM→MainBF16lookup→BF16DPE/F32accumulate→16K F32sum→**2행8B F32commit**→H480F32gather→F32channelscale→첫BF16→기존RMS/gamma→BF16→residual→BF16이다.

E47boundary job70785 case7은expected BF16`0.609375`에전출력이`0.5`,abs차`0.109375`로허용`0.05609375`초과. 원본job70799와E53job70810은모든3,840행일치했다. E53은BF16activation+scale전F32를함께바꿨으므로단일변경인과실험은아니다. 또한probe8case가모든유효입력동등성을증명하지않는다. 근거는 `docs/experiments/output-boundary-comparison.md`와각원시JSON이다.

`output-compact-f32-partials`의2행F32commit뒤최종BF16 4행재transpose실패,4행F32transpose`max_in_rows=2`실패는당시표현제약이다. E53은**2행F32그대로typedgather**하여해결했다. “F32compact가불가능”이라는재사용금지는틀리지만,currentO에는이미해결된변경이다.

`output-bf16-fused-channel-scale-e53/projection.rs:283`은16Ksum바로뒤scaleVRF곱→첫BF16→BF16gather를**compile성공**시켰다(320행reduce,323행MulF,326행BF16). 별도Main401과gather24를줄이지만scaleDMA579를dot앞에둬static+154가됐다. **NPU없음**.첫`H%8#16`scale-fetch오류는actual`H%8`로수정했고SubVRF는valid480B/leader다. 재활용조건은scale준비가이미끝나는새producer경로라서579의새criticaldependency가사라질때다. “scalefusion은지원안됨/하드웨어loss”는잘못된분류다.

## 현재 shortlist에 적용할 제한

1. **행쌍보존4-way ingress + 기존16K + 직접F32H480gather:** `h=1920c+480g+8t+2r+j`, slice=`64g+16r+q`; `c<2,g<4,r<4,t<60,j<2,q<16`. exactmapping은미실행이며oldrow2/row4의단순동일소스가아니다. 인접H두개가한owner에남아2F32=8Bcommit을지킬수있는논리좌표다. 그러나새source/compile가아직없으므로API지원과emittedbounds는미확인이다. 이번Python논리검사는3840H와1920pair목적지coverage만검증했다.

   가장큰추가비용위험은**현재F32gather의연속480B×32단위가8B×1,920단위로분산되는것**이다. 총15,360B는같아도연속단위수가60배다. candidate의한leader에서8Bsourcepair는매8B로인접하지만목적지H480에서32B간격이며4rowlanes가서로사이를채운다. 실제DMAtransaction수가60배라고주장하지않고, compilerdescriptor·전체gather비용의첫gate로삼아야한다. baseline대onecandidate ingress와전체gather를함께비교하고, canonical복구Switch/fullweightrepack/새partialboundary를조용히추가하지말아야한다. 그것들을추가해야지원된다면완전비용을다시평가한다.

2. **동결C09첫NPU/중첩진단:** 기존실행되지않은측정경로이며새family는아니다. source·weightmapping·입력·control이미지고정, 더작게쪼개는sweep없음. 새profile에서확인한weight준비/dot비용관계와실제NPU전체cycles를묻는정도로범위를제한한다. 기존static+0.71%만으로NPUloss라판정할수없다. 반대로static중첩1,825를그대로speedup으로바꾸지도않는다. 두번LUT·load·작은dot·메모리lifetime증가를전부포함하고, 전후동시control로측정해야한다.

3. **weight TRF / BF16 materialize는 보류:** C10의 Q8×H3840=61,440B는 논리 64KiB 안이지만 `.to_trf()`에서 `StoTrf→lower_fetch_unit→renegade feature is enabled in non renegade-s config`로 실패했다(`direct-v1/projection.rs:23`, exact log `evidence/c10-direct-v1/compile.log`). kernel_audit가 SDK0.8.1 `engine/fetch_adapter.rs:183`의 `fetch_table_lookup`을 `Tu::Main` 전용으로 확인해 Sub lookup 직접 경로는 공개 API 근거도 없다. Main 경로에는 C10 target 제약 해소 근거가 먼저 필요하다. BF16 DM materialization은 weight 확장에 valid payload 기준 30MiB write+30MiB read를 추가하고 E2의 제거 이득을 되돌리므로, 그 비용을 포함한 완전 그래프에서 명확한 소비자 이득 없이는 후보를 열지 않는다. 이는 측정한 bus traffic이 아니며 padding·allocation extent와도 다르다. DMN/bank 수동 offset은 설치 SDK의 공개 API를 확인하지 못해 즉시 후보로 세지 않는다.

5%는기본탐색예산cutoff다. 정적0.7%증가는하드웨어회귀판정이아니며, 실측5%미만이더라도정확성/재현성조건을대신하지않는다. 이번후보의목적은새parameter숫자를늘리는것이아니라**현재15MiB ingress와후속F32gather의어느주소순회·producer/consumer계약을바꾸면완전실행경로가짧아지는지**를구분하는것이다.

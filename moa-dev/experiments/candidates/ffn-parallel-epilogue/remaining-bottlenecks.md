# E36 safe FFN의 남은 병목과 다음 판단 기준

2026-09-21. 분석 대상은 이 폴더의 고정된 [candidate.json](candidate.json)(SHA256 `97cd593f04a79e79c6be5cd9375e4657448625142abb64b827d6aa1e62de5074`), [candidate-graph.json](candidate-graph.json), [mlp.rs](mlp.rs), [ops.rs](ops.rs)다. Exact `ops::decoder_feedforward` makespan은 **162,209 cycles**다. E31 기준 167,296과의 차이 -5,087은 [기존 기록](README.md)에 있다. 이 문서는 추가 compile을 수행하지 않은 정적 분석이며, 후속 E36 공식15개·추가30개 PASS 및 E38 결과는 각 측정 기록에 근거해 아래에 연결한다. 공식 순위는 판정하지 않는다.

**상태 갱신:** [E35 full256](../../../../docs/experiments/e35-full256-native.json)는 native packet4를 유지한 채 공식15개 비교를 통과했다(FFN811,219). 따라서 native packet4 자체가 불가능하다는 결론은 틀리며, 최소 packet4 NPU 진단은 보류·동결됐다. 240→256 live slice, row layout, 실제 zero tail, 복사와 scheduling 변화가 함께 들어갔으므로 어느 하나가 유일한 해결 원인인지는 [원인 감사](../../../../docs/experiments/e35-full256-native-cause-audit.md)처럼 분리해야 한다. Direct weight commit + direct scale copy의 [E38 job70525](../../../../docs/experiments/e38-full256-directcommit.json)는 공식15개 PASS, FFN482,511cycles다. 이는 weight+scale 변형만의 결과이며 weight-only 변형은 NPU 미실행이다. Safe BF16 [E36](../../../../docs/experiments/e36-ffn-parallel-epilogue.json)은 공식15개 PASS, FFN378,021cycles이며 [추가70506](../../../../docs/experiments/e36-ffn-epilogue-extra-correctness.json)도30개 PASS다. 후속 [E39](../../../../docs/experiments/e39-native-tail-zero.json)는 전체15개 PASS, FFN415,457이지만 E37보다 느리다. [E40](../../../../docs/experiments/e40-row2-tail-zero.json)은 FFN 세 입력 모두 non-finite로 실패했다. 실제30행의 [E41](../../../../docs/experiments/e41-native-row30.json)/[E42](../../../../docs/experiments/e42-native-row30-row256.json)는 각각 전체15개 PASS, FFN426,442/456,849로 역시 느리다. 이 분기들을 성능 대체안으로 채택하지 않았다.

가장 큰 두 축은 **전체 weight 복호화의 Main 점유**와 **HBM weight/scale 및 activation 분배의 직렬 DMA 작업량**이다. Dot/scale/reduction도 남아 있지만 GeGLU 산술이나 후단 fusion만으로 큰 격차를 닫기는 어렵다. Decode를 줄이면 현재 가려진 DMA가 드러나므로 둘을 함께 줄이는 설계가 필요하다.

## 실제 시간과 입력 의존성

아래 시간은 instruction `lifetime.begin/end`이며 구간은 `[begin,end)`다. 실행 순서는 소스의 up/gate 선언 순서와 다르다. Compiler는 **gate를 먼저 읽고 계산**한다. 각 표의 stage 시간 합을 전체 절감 가능 시간으로 읽으면 안 된다.

| 단계 | 실제 schedule, node ID | 연산 시간 합 | 무엇을 기다리는가 / 무엇과 겹치는가 |
| --- | --- | ---: | --- |
| Pre-FF RMS Main | 3,675–8,349; 54/60/106/116/134 | 4,407 | input/residual DMA 뒤 실행. 첫 gate weight DMA와 겹침 |
| Gate FP4 HBM load | 4,285–30,760; 71 | 26,475 | RMS는 8,349에 끝났지만 Main은 30,772까지 다음 weight를 기다림 |
| Gate FP4→FP8 DM | 30,772–40,635; 142 | 9,863 | weight DMA 30,760, LUT Sub setup 3,134에 완료 |
| Activation 두-cluster 복제 | DMA 30,772–39,124(node145), Main ring16 40,724–44,827(node167), TRF 44,851–46,074(node189) | DMA8,352 / Main4,103 / Sub1,223 | DMA는 gate 1차 decode에 가려짐. Main broadcast는 gate 두 번째 decode와 같은 Main을 사용 |
| Gate FP8→BF16 DM | 44,851–64,314; 187 | 19,463 | 1차 decode 및 Main broadcast 뒤 실행. activation TRF와 up weight prefetch가 겹침 |
| Gate 8-row blockdot/scale/sum ×4 | 64,593–75,072; 205/220/232/245 | 8,812 | 각 dot2,203. 첫 activation/scale VRF는 46,074/44,214에 이미 준비됨. 뒤 tile의 scale VRF 적재는 dot 사이에 실행 |
| Up FP4 HBM load | 44,960–71,435; 191 | 26,475 | gate 두 번째 decode와 gate dots에 겹침. 완료되어도 Main/LUT 준비 때문에 decode는 75,087 시작 |
| Up FP4→FP8→BF16 DM | 75,087–84,950(node263), 84,974–104,437(node298) | 29,326 | FP4 load는 이미 71,435 완료. 같은 Main 자원의 gate dots 및 두 lookup stage가 직렬 |
| Up 8-row blockdot/scale/sum ×4 | 104,716–115,659; 321/343/364/384 | 8,812 | span10,943에는 gate GeGLU Main605, scale Sub, launch/staging 간격도 포함 |
| GeGLU | gate gather/global scale 106,919–110,015; Erf Sub110,407–110,750; up gather/global scale/final product124,922–125,838 | Main1,521 / Erf343 | 하나의 연속 stage가 아님. Gate branch는 up dots 사이, 마지막 up branch는 down 1차 decode 뒤에 실행 |
| Down FP4 HBM load | 84,977–114,404; 303 | 29,427 | up BF16 decode/dots에 겹침. Main이 up dot를 마치는115,659까지 기다림 |
| Down FP4→FP8 DM | 115,659–124,922; 391 | 9,263 | packed weight와 LUT가 이미 준비됨. up output gather DMA115,738–118,130과 겹침 |
| Down scales 및 activation | scale HBM118,130–127,904(node405); activation DMA127,904–137,296(node432); TRF138,896–139,639(node459) | scale9,774 / activation9,392 / TRF743 | GeGLU는125,838에 준비됐지만 DMA는 scale load 완료를 기다림. activation/TRF는 아래 BF16 decode에 가려짐 |
| Down FP8→BF16 DM | 125,859–144,122; 431 | 18,263 | GeGLU 마지막 Main 작업 뒤 실행. 첫 down dot의 activation TRF139,639보다4,483 늦게 완료 |
| Down 12-row blockdot/scale/8-column sum ×5 | 144,401–155,414; 468/484/496/508/520 | 8,925 | 각 dot1,785. span11,013에는 다음 tile scale Sub443×4와 시작 간격 포함 |
| H480×8 후단 | 155,414–162,209 | elapsed6,795 | gather469, globalScale401, square401, RMS gather289/EPS·sqrt282/scalar DMA319, gamma401, residual341, gate401, HBM648, Sub/staging/final sync 포함 |

직접 의존성을 확인한 대표 경로는 다음과 같다.

- Gate packed tensor는 schedule node71의 output59 → node142의 input59 → FP8 output101 → node187 input101로 연결된다. 두 번째 decode가 만든 BF16 matrix를 row view로 읽어 node205 등의 blockdot가 실행된다. 이 row view에는 pointer/index alias가 있으므로 단순히 input tensor의 직전 Core producer 시간만 추적해 weight가 준비되었다고 판정하면 안 된다.
- Down은 node303 output168 → node391 input168/output195 → node431 input195/output206 → row view → node468 이후 dot다. 동시에 GeGLU node425 output205 → node432 input205/output207 → TRF(node459)라는 별도 입력 가지가 합류한다.
- 따라서 현재 down activation DMA **9,392를 없앤다고 9,392만큼 빨라지는 것은 아니다**. 현 schedule에서 TRF가139,639에 준비되어도 BF16 weight가144,122까지 준비되지 않는다. 반대로 decode를 줄이면 이 activation/scale 경로가 다음 병목이 된다.

`candidate-graph.json`은 lookup table·tensor view·DPE/VE dataflow를, `candidate.json`은 compiler가 선택한 resource 시간 배치를 보여 준다. 여기서 말하는 완료 경로는 실제 정적 스케줄의 자원 제약을 포함하며, 하드웨어 카운터나 tensor DAG만의 이론적 longest path가 아니다.

## Main·DMA·Sub는 얼마나 겹치는가

각 context의 모든 nonzero interval을 합집합으로 계산했다. 같은 instruction이 Main과 VectorEngine 양쪽 context를 가질 수 있으므로 두 값을 더하지 않는다. Core의 긴 sync/reuse interval 역시 CPU 산술 busy time이 아니다.

| Context | interval union | 전체162,209 대비 |
| --- | ---: | ---: |
| MainContext | 125,274 | 77.2% |
| DmaEngine | 135,607 | 83.6% |
| SubContext | 19,378 | 11.9% |
| VectorEngine | 34,748 | 21.4% |

Main∩DMA는 **104,921** cycles다. 그 안에서 Sub까지 겹치는10,547도 이미 포함한다. DMA가 Main과 겹치지 않는 시간은30,686, Main이 DMA와 겹치지 않는 시간은20,353이다. Main/DMA/Sub 세 context 모두 비활성인 시간은3,569이며 나머지에는 Sub 단독2,680도 있다. `125274+135607+19378`을 kernel 지연으로 더하면 안 된다.

Main의125,274는 다음과 같이 정확히 나뉜다.

| Main 작업 | cycles | Main 중 비중 |
| --- | ---: | ---: |
| FP4→FP8→BF16 full matrix decode 6개 | **86,178** | **68.8%** |
| Up/gate blockdot+FP32 scale/reduction 8개 | 17,624 | 14.1% |
| Down blockdot+FP32 scale/8-slice reduction 5개 | 8,925 | 7.1% |
| 입력 ring16 broadcast | 4,103 | 3.3% |
| Pre-FF RMS | 4,407 | 3.5% |
| GeGLU Main | 1,521 | 1.2% |
| Global scale부터 residual까지 새 후단 Main | 2,516 | 2.0% |

BF16로 넓히는 두 번째 lookup만 **57,189**(19,463×2+18,263)이다. 이 값이 “native FP8이면 그대로57,189 절약”이라는 뜻은 아니다. Quantization·block result 저장·scale 적용·추가 memory lifetime이 들어가고, 기존에 겹친 DMA가 드러난다.

DMA 명령은 이 dump에서 서로 겹치지 않으며 duration 합도 union과 같은135,607이다. 그 중 packed FP4 HBM load82,377, local-scale HBM load17,352로 **99,729**가 세 projection의 weight/scale이다. 원래 layout의 논리 payload는 projection당 `3840×15360/2 + 3840×15360/16 = 33,177,600B`, 세 개99,532,800B다. 이 byte 수가 복제·padding·lookup table까지 포함한 실제 버스 전송량은 아니다. 두 activation DM 분배가 추가17,744(8,352+9,392)를 쓴다.

현재의 **직렬 DMA 명령·duration을 그대로 유지**하면서 Main decode만 공짜로 만드는 가상 모델에서도 makespan은135,607보다 작아질 수 없다. 이 모델의 최대 속도비는162,209/135,607≈**1.196배**다. 세 BF16 lookup table HBM load838씩을 함께 제거해도 DMA 작업량만133,093이 남는다. 이는 실제 RNGD의 보편적인 bandwidth roofline이 아니다. Layout·transfer 경로·동시성·추가 명령을 바꾸면 전제가 달라지지만, “decode 비중53%이니 그것만 없애면2배”라는 계산이 왜 성립하지 않는지를 보여 준다.

## 재시도 전에 읽어야 할 기존 결과

| 이미 시도한 방향 | 확인된 결과와 원인 범위 | 같은 접근을 다시 시작할 조건 |
| --- | --- | --- |
| [Tile마다 F8→BF16 lookup](../ffn-blockscale-after-dot/schedule-analysis.md) | Single-cluster E8은 연산 Main 감소에도 LUT 상수가3→33개, table DMA+25,140/Sub+38,790. 정적392,312→403,875, 실측820,226→921,693. 원인은 multiply 수만 본 데 있었다 | Lookup setup을 실제로 공유하거나 횟수를 줄이는 구체적 API/IR 근거가 있어야 함. 작은 tile sweep 반복 금지 |
| [전체 BF16 materialization 재사용](../ffn-blockscale-two-cluster-bf16-reuse/README.md) | 현재 안전 경로의 기반. 각 matrix LUT 한 번, 여러 row tile에서 재사용. 이전8-row tiled LUT는 table11개, 이 단계는5개였음(당시 down 방식이 달라 전체 schedule 직접비교 금지) | BF16 DM을 없앨 때도 table/VRF/lifetime 비용을 함께 비교 |
| [F4 LUT→F32 cast→BF16 fused decode](../ffn-blockscale-two-cluster-fused-bf16/fused-decode-investigation.md) | E13 job70162 run0 tolerance 내30.96%, 다음 launch timeout. SDK는 API chaining을 설명하므로 금지된 API라고 단정할 수 없으나 packet/table packing 변화의 원인은 미확정 | 전체 FFN 재제출 대신16개 FP4 code와 adjacent nibble/table layout의 독립 oracle이 먼저. Scalar exactness·compile PASS는 충분하지 않음 |
| [Native FP8 up/gate](../ffn-native-fp8-upgate/README.md)와 [pair-free](../ffn-native-fp8-two-dot-diagnostic/README.md) | E22/E25/E28와 E32 모두 첫 FFN launch timeout. Pair를 없애도 해결되지 않았으므로 pair-only 원인설은 약해짐. 정확한 실패 instruction은 모름 | E35가 native packet4/full256으로 통과했으므로 최소 packet4 NPU 진단은 보류. 통과한 full256에서 불필요한 복사/zero-fill을 하나씩 제거하되 원인 혼입을 유지 기록 |
| [Full256 live slice](../ffn-native-fp8-full256/README.md) | E35 공식15개 PASS, FFN811,219. Static324,369 vs E25의166,220이며 실제zero-fill/copy·30-row gather 비용이 큼. Native packet4 실행 가능성을 확인했으나 단일 원인 규명은 아님 | Direct weight commit/scale copy E38 static191,836과 tail-only memset 후속처럼 안전한 통과 경로에서 낭비를 줄임. E35 통과를 변경 후보에 자동 승계하지 않음 |
| [BF16 block partial→scale DPE](../ffn-scale-contraction-streamed/README.md) | E23 FFN만 공식3개 PASS,384,728 단일 측정. 결합 QKV 실패라 유효 전체 score 없음. Extra BF16 rounding 존재, 정적171,727은 같은 기준167,296보다 느림 | 같은 E36 base에 옮겨 완전한 correctness/extra 입력 및 반복측정으로 비교. 큰 개선의 증거로 취급하지 않음 |
| [Down scale DPE](../ffn-down-scale-contraction/README.md) | E33 job70455 FFN3개 모두 FAIL, tolerance 내 약13–16%. 새 BF16 rounding뿐 아니라 layout/copy/DPE도 달라 원인 미확정 | 단순 rounding 오차로 무시하거나 빠른 성능값으로 기록하지 않음 |
| [BF16 FFN 8-way row interleave](../ffn-bf16-row-interleave/README.md) | Static171,429→185,673(+8.31%), DMA+18,086. QKV의 실측 row-interleave 성공이 FFN에 그대로 전이되지 않음 | Weight/scale DMA 및 restore overhead가 낮아지는 새 매핑 근거 필요 |
| [Down 8→2 columns](../ffn-down-two-columns/README.md) | Weight DMA29,427→26,218, scale9,774→5,255지만 activation9,392→39,136, decode+1,800. Static171,429→190,591 | 더 연속적인 HBM 접근만 보고 채택 금지. Activation multicast까지 해결 필요 |
| [Down2 HBM relay](../ffn-down-two-columns-hbm/README.md) | Activation 비용을12,403까지 낮췄지만 기준9,392보다 큼. Static172,233 vs171,429, NPU 미검증 | HBM relay를 무조건 개선으로 일반화하지 말 것 |
| [GeGLU 2-cluster 유지](../ffn-geglu-two-cluster/README.md) | 제거된 복원 DMA4,784보다 down 분배9,392→28,648 증가가 큼. Direct/hier16 scale 모두 static 회귀 | Global-scale 복제와 down 입력 분배까지 더 싸지는 전체 경로가 필요 |
| [BF16 packet2/packed lanes](../ffn-blockscale-packed-lanes/README.md) | 5변형 compile 실패. Time→Lane 승격, VE4연속 scale operand, F8→F32 padded fetch가 장애 | 지원되는 VRF/packet fetch 배치를 독립 검증하기 전 같은 shape를 반복하지 않음 |

## 다음 우선순위와 중단 기준

1. **Decode와 DMA를 함께 줄이는 native/stream 경로가 가장 큰 설계 여지다.** E35의 full256 native 원형은 실행 가능성이 확인됐으므로 최소 packet4 진단을 중복하지 않는다. 그 통과 경로에서 실제zero tail을 보존하며 direct weight/scale commit, tail-only memset, 작은 row-interleave의 전달 비용을 먼저 분리하고 full-matrix BF16 store/refetch 제거량, quantizer·TRF 준비, local scale, F32 block buffer와 HBM/DM 작업량을 같이 본다. BF16 stream→BF16 block/scale-DPE는 별도 대안이지만 E23의 작은 단일 측정 개선과 새로운 rounding을 넘는 증거는 아직 없다. “F32 block 결과를 유지하면 정확하다”는 이유만으로 대형 padded intermediate가 싸다고 가정하지 않는다.
2. **Weight·scale DMA layout과 activation multicast를 하나의 변경으로 평가한다.** 시작부 Main idle22,423은 gate weight를 기다리고, 나중에는 decode를 줄일수록 직렬 DMA가 제한한다. 현재 세 weight/scale load99,729가 큰 고정 작업량이다. Down의 더 긴 column segment는 이미 load를 낮췄지만 입력 복제가 악화했다. 새로운 partition 또는 transfer 방식은 같은 logical weights/scales를 처리하고, 복원/activation/zero-fill까지 합친 I/O가 낮아지는 한 가지 설계만 prescreen한다. 현재8-way interleave·2-column direct·2-column HBM relay를 그대로 재시도할 근거는 없다. 새로운 mapping도 original HBM layout/공식 입력을 변경하거나 kernel 밖에서 weight를 미리 변환해서 비용을 숨기는 접근과 구분해야 한다.
3. **Blockdot 이후 scale/reduce의 lane 활용은 두 번째 단계다.** 현재 그 Main 합은26,549로 decode보다 작고 Sub tile-scale 준비 간격도 있다. Packet packing이나 두 번째 DPE를 쓰면 arithmetic뿐 아니라 padding·VRF load·extra casts·commit 경계가 같이 바뀐다. SDK가 허용하는 최소 사례가 먼저이며, down E33 failure를 해결하지 않은 채 전체 FFN으로 확대하지 않는다. 기존 packet 전체-reduce/zero-mask lane 대안은 E35 통과 이후 우선 보류하며 새 근거 없이 중복 구현하지 않는다.

Pre-FF RMS에 E31 gather 방식을 재사용하거나 새 후단의 staging을 줄이는 일은 작은 보조 후보로 분류한다. 후단 전체가6,795이고 그 안의 RMS·residual·store는 필요한 연산이다. 이를 전부 제거하는 비현실적 가정조차162,209/155,414≈1.044배다. GeGLU Main1,521/Erf343을 최우선으로 최적화할 근거도 없다.

평가의 [세 커널 기하평균](../../../../docs/research/competition-techniques.md)에서는 FFN만2배 빨라져도 전체 score는 `2^(1/3)≈1.260배`다. E36의 정적3.04% 감소를 그대로 실측으로 얻는다고 가정해도 전체 score 증가는 약1.03%에 그친다. 현재 순위 격차를 이 정적 수치로 환산하지 않는다. 실제 채택은 동일 immutable package의 공식 전체 비교·추가 입력·반복 cycle로 판단하고, QKV/O에서 더 낮은 위험으로 같은 비율 개선을 얻을 수 있다면 병행하는 것이 합리적이다.

재계산 방법: `instructions`의 `lifetime`을 context별 정렬·구간 합집합으로 집계하고, Main은 위 node 목록과 `description`의 실제 source 위치로 분류한다. Lookup/Core pointer alias는 graph와 소스를 함께 따라간다. 다음 후보의 “절감”은 개별 duration 합이 아니라 같은 compiler/base에서 나온 새 makespan과 그 뒤 실측으로 보고해야 한다.

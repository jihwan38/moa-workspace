# E40 실패 뒤 gather/unsafe rename의 physical layout 감사

2026-09-21. **E40 job70552의 FFN 세 설정 모두 출력 index0에서 non-finite로 실패했다.** QKV/O는 통과했지만 FFN432,862 cycles는 실패 결과이며 성능 후보로 인정하지 않는다. [원본 log](../../../logs/e40-row2-tail-zero-job-70552.log), [measurement](../../../../docs/experiments/e40-row2-tail-zero.json)가 근거다. Harness는 첫 non-finite에서 비교를 즉시 중단한다. 따라서 `at0`은 모든 원소가 NaN이거나 어떤 중간 단계가 먼저 실패했다는 뜻이 아니다.

조사한 [mlp.rs](mlp.rs)의 SHA256은 `28cf2313c22e5065c0e32a09e44dc28f7375991a31602d20e898a6fca25e087f`이며 실제 E40 capture의 MLP와 같다. 이 감사에서 source 변경, 추가 compile, NPU 실행은 하지 않았다. Static dump는 앞선 exact compile 산출물이며 root의 전체 binary에서 관측한 내부 메모리 dump는 아니다.

## Row30이 output에서 여전히 Row32로 padded되는가

**현재 compiler dump에서는 그렇지 않다.** Input read tile에만 `Resize(FfnRow32→30)#32`가 있다. Gather가 반환하는 DM은 `Pair(Resize(FfnRow32→30),FfnSlice%2)`이고, local axis를 감싼 padding은 없다. 60BF16=120B를 실제로 commit한 뒤 rename한다.

[physical-layout-audit.json](physical-layout-audit.json)에 실제 graph tensor·operator와 schedule/hash를 보존했다. 아래 graph와 schedule의 tensor ID는 서로 다른 단계의 번호이므로 혼용하지 않는다.

| 단계 | Graph tensor(up/gate) | Schedule tensor(up/gate) | 실제 local shape | 전체 physical allocation |
| --- | --- | --- | --- | ---: |
| Gather input view | 141 / 144 | input view를 생성하는 IndexAccess 별도 | `FfnRow=30#32` | 32,768B |
| Gather output | 142 / 145 | 181 / 212 | `(FfnRow=30,FfnSlice%2)` | **61,440B** |
| Unsafe renamed output | 146 / 147 | 182 / 213 | `L%60` | **61,440B** |

Gather output의 Cluster는Dummy2(2), physical Slice는 `[FfnSlice/2,1#2]`(128×2)다. 따라서 allocation61,440B는 `2×256×60×2`와 일치한다. 만약 local64 BF16가 남았다면65,536B가 되어야 하지만 그런 size는 기록되어 있지 않다. 유효 group만 세면2×128×60 BF16=30,720B이며, 나머지 물리 slice는 Slice축의 padding이다. 이 padding은 alias에서도 그대로 보존한다.

Alias 전후의 graph per-slice base address도 각각 같았다: up246016, gate26880. Schedule의 aggregate address도 up125960192, gate13762560으로 alias 전후 동일하다. Address를 바꾸거나64→60 stride를 암묵적으로 압축하는 reshape가 아니다.

## 실제 Switch·transpose·commit과 TDMA

두 gather의 emitted TuExec에는 다음이 기록되어 있다.

```text
fetch: FfnSlice256, FfnRow first30, packet1#16
Switch: Dim1Broadcast(dim1_volume=2, dim0_volume=1)
stream: Slice[FfnSlice/2,1#2], Time[FfnRow30,FfnSlice%2]
transpose: in_rows=4, out_rows=1, BF16, Single
transpose output: [FfnRow/2=15, (FfnRow%2)#8, FfnSlice%2]
commit_in_size: 8B (=4EA)
commit sliced: [FfnRow/2=15, FfnRow%2=2, FfnSlice%2=2]
commit output: [FfnRow=30,FfnSlice%2=2]
commit_projection: 0+[None=15:4, None=4:1]
```

Source의 `[(FfnRow%2,FfnSlice%2)#16]`은 lowering에서 `[FfnRow%2#8,FfnSlice%2]`로 표현되지만 실제 commit은 전체 앞4값만 쓰고 다음 packet으로 간다. 이 메타데이터에 따르면15개의4-value packet, 총60BF16가 연속 저장된다. 실패했던 첫 시도 `[FfnRow%2,FfnSlice%2#8]`의 내부 hole layout은 현재 source가 아니다.

뒤의 typed DMA graph op165/191은 source와 destination 모두 `None=1:120` byte extent로 복사한다. Source group step은8,388,608, destination은4,194,304로 기록되며, source에서 Slice padding 한 칸을 건너뛰고 destination의 연속 row groups로 가는 배치에 대응한다. Source는 양쪽 cluster의4PE씩 총8개 descriptor, destination은 cluster0의4PE×2DMN에 분배된다. 어느 descriptor도128B의 local64 행이라고 선언하지 않는다.

물리 좌표를 식으로 비교하면 gather output의 유효 cell은 `(cluster=c, physicalSlice=2*g, local=2*r+s)`다. Rename의 `L=7680*c+60*g+2*r+s`에서 `L/7680=c`, `L/60%128=g`, `L%60=2*r+s`이므로 **유효 cell의 물리 좌표와 Slice padding 위치를 모두 보존**한다. 이는 기존 CPU logical-row 증명에 compiler physical size/commit/TDMA 확인을 추가한 것이다.

## 확인한 것과 확인하지 못한 것

- 확인: compiler가 선언한 gather output은local60이고 unsafe alias와 byte size/address/physical Slice padding이 같다. 단순한 “Row30의 숨은 padding32 때문에 local64를60으로 rename했다”는 설명을 지지하는 dump 증거는 없다.
- 확인: 실제 zero initialization/write dependency 검사는 계속 통과하지만, 그것은 소비 이전의 주소 범위 정의 여부를 검증한 것이며 값의 하드웨어 정확성을 보장하지 않는다.
- 미확인: 하드웨어가 두-slice Switch와 복합축 transpose의 valid/padding lane을 이 dump 그대로 처리했는지. Compiler가 선언한 commit extent와 실행 결과가 같다는 증거는 아직 없다.
- 미확인: 새로운 HBM row/scale mapping, compact output, 다음 TDMA, allocator/lifetime 중 어느 단계가 처음 잘못된 값을 만들었는지. E40은 이 변경들을 함께 포함한다.
- 미확인: non-finite의 최초 발생 지점. 잘못된 입력/미정의 lane, 과대값, 후속 GeGLU·RMS 전파 등은 가능한 설명일 뿐 관측된 원인이 아니다. “단순 permutation이니 안전하다”거나 “non-finite이므로 padding 버그다”라고 확정하지 않는다.

E39의 row256+tail-only 경로는 job70551에서 전체 비교 PASS(FFN415,457)였다. E40에서 추가된 것은 row2의 **HBM view + Switch/transpose + canonical gather + 달라진 scheduling** 묶음이다. 이 차이가 원인 탐색 범위를 좁히지만 tail-only와의 상호작용까지 배제하지는 않는다.

**추가 확인:** [E41 row30+row2 job70562](../../../../docs/experiments/e41-native-row30.json)는 공식15개 비교를 모두 통과했다(FFN426,442). 따라서 row2 mapping/Switch/rename 전체가 항상 잘못되었다거나 row2만이 E40 원인이라고 할 수 없다. E41의 `FfnRow=30`은 실제30행만 처리하고 zero-tail을 없앴으며, 두 번째 DPE output을 `[FfnRow,1#4]`의 **행당8B BF16 scalar slot**에 commit한다. E40은32행 결과를 transpose로 contiguous BF16에 담아 **행당2B stride**로 읽는다. Output producer·read stride·row count·zero staging·allocation이 함께 달라졌다. E41은 최종 compact 뒤 같은120B/group typed gather를 사용하므로, 그 final rename/TDMA가 모든 상황에서 잘못됐다는 주장도 약해진다.

Root의 E42 row30+기존row256 job70576은 제출된 대조이며 이 감사에서 결과를 가정하지 않는다. 현재 증거로 E40의 첫 실패 instruction이나 정확한 원인을 확정하지 않는다. 요청에 따라 여기서 원인 감사를 동결하며 새 diagnostic·compile·NPU 제출은 추가하지 않는다.

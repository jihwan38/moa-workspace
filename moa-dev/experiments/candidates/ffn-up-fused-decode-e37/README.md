# E48: up weight만 fused decode

**[job70683](../../../../docs/experiments/e48-up-only-fused-decode.json)은 all15 PASS**다. FFN 3개 입력의 cycle은 387,277 / 381,532 / 380,694이며 median 381,532는 E37 반복 기준 379,741보다 0.47% 높다. 한 번의 screen이고 성능 개선이 관측되지 않아 채택하지 않았다. 정적 makespan 감소를 실제 속도 개선으로 해석하지 않는다.

## 소스와 실행 경로

Frozen E37과 E48 snapshot을 비교하면 변경 파일은 `src/device/shared/mlp.rs` 하나다. 변경 블록 앞·뒤는 byte-identical이며 `project_up_and_gate`의 up decode만 바뀌었다.

`ops::decoder_feedforward` → 기존 pre RMSNorm → `feedforward_sharded` → activation의 두 cluster 복제/TRF 적재 → up/gate projection → GeGLU → down projection → global scale → sharded post RMSNorm/residual/layer scalar → HBM 순서다. 이 중 gate/down, 8행 dot 및 local scale 연산, BF16 경계, 후단 함수는 그대로다.

up weight 경로는 기존 `FP4 LUT→FP8 DM→FP8 LUT→BF16 DM`에서 `FP4 LUT→FP8→fetch_cast F32→cast BF16→BF16 DM`으로 바뀐다. FP4 16개 값은 F32와 BF16에 정확히 표현 가능하다. 다만 이 수학적 사실만으로 compiler/hardware 전체 구현의 정확성이 보장되지는 않는다.

두 경로의 DM mapping은 동일하다: cluster `Dummy2`, slice `L/32%240#256`, local `[L%32,H]`, 즉 **2×240 유효 slice, 각 32×3840**이다. Fused 소스는 8개 F32를 BF16 `8#16` packet으로 cast한 뒤 유효 8개만 16B commit한다. 새로운 E48 lowered graph는 저장되지 않았으므로 이 packet 설명은 실제 소스 contract이며 register 수준 IR을 직접 확인한 결과로 확대하지 않는다.

E48 MLP SHA256: `3a7449532a74223fefddb96bca519ff2e6945722f86e38409e6cd0bc3f9a74ac`. 실제 업로드 host SHA256 `5bb1965a48e3be5fd7001235b61b94b4d5d1f547a2d3f49b7b69b243fb7e733c`는 3,667,008B다. ELF registry를 독립 파싱해 FFN image `40f60d35…`(201,216B), QKV/O image가 각각 한 번 들어 있음을 확인했다. QKV/O image는 E37 ELF에도 byte-identical하게 존재한다(`binary-audit.json`).

## 정적 schedule 변화

기준 schedule은 FFN parallel epilogue의 162,209-cycle artifact다. 같은 MLP와 FFN device image가 E37에 사용됐다. E48 raw schedule은 `../../artifacts/ffn-up-fused-decode-e37/schedule.json`에 보존했고 `analyze_schedule.py`로 아래 수치를 재현할 수 있다.

| 항목 | E37 기준 | E48 | 차이 |
|---|---:|---:|---:|
| makespan | 162,209 | 157,866 | −4,343 (−2.68%) |
| Main duration 합 | 125,274 | 113,491 | −11,783 |
| 모든 weight decode Main 합 | 86,178 | 74,395 | −11,783 |
| DmaLoad duration 합 | 110,745 | 109,907 | −838 |
| DmaStos duration 합 | 24,214 | 24,214 | 0 |
| DmaStore duration 합 | 648 | 648 | 0 |
| Sub duration 합 | 19,378 | 17,826 | −1,552 |
| 마지막 down dot 종료 | 155,414 | 151,071 | −4,343 |
| 마지막 dot 이후 후단 | 6,795 | 6,795 | 0 |

Duration 합에는 겹치는 실행이 있으므로 critical path가 아니다. Core 합의 −10,978 역시 `ExplicitSync` 대기 25,276→14,298 감소이며 순수 계산량 감소로 읽으면 안 된다.

실제 schedule은 소스 호출 순서와 달리 **gate가 먼저**다. E48 gate의 두 decode O142/O187은 각각 9,863/19,463 cycle로 그대로다. gate 4개 dot 뒤 **up fused O258은 75,014–92,557, 17,543 cycle**이다. 기존 up decode는 9,863+19,463=29,326 cycle였다. 이후 up dot O298/O320/O341/O361이 103,700까지 진행된다.

행렬 HBM DMA 자체의 duration은 변하지 않았다: up/gate 각각 26,475, down 29,427, up/gate scale 각각 3,789, down scale 9,774다. DmaLoad 합의 −838은 lookup 준비용 load 하나가 없어진 데 해당한다.

11,783-cycle decode 절감 전부가 makespan에 반영되지는 않는다. down weight DMA는 84,977–114,404에서 80,658–110,085로 앞당겨졌지만, down scale DMA는 118,130–127,904에서 129,683–139,457로 늦어졌다. down BF16 decode 종료는 5,568 cycle 앞당겨졌지만 첫 dot 시작은 4,343 cycle만 앞당겨졌다. 이 schedule은 입력/scale 준비의 겹침과 순서가 달라졌음을 보여 주며 실제 메모리 bandwidth 병목의 원인을 확정하지는 않는다.

## allocation·lifetime과 probe의 차이

Physical padding 포함 up FP4 DM은 31,457,280B, BF16 DM은 125,829,120B다. E48은 기존 FP8 intermediate **62,914,560B** allocation을 제거했다. Lookup 보조 tensor는 기존 8MiB 두 개에서 4MiB 한 개로 줄었다. 이는 제거된 allocation 크기이며 peak SRAM 감소량을 의미하지 않는다.

E48의 fused input T119는 physical pages 13/14, table T141은 page1, output T142는 pages 2/3/4/5/6/7/8/0을 사용한다. 세 operand 사이의 페이지는 겹치지 않는다. 이 사실만으로 전체 allocator 안전성을 증명하지 않는다.

T142의 표시 lifetime은 75,014–92,581에서 끝나지만 뒤의 Core tile 연산이 T159/T161/T162/T163 alias를 만들고 실제 dot은 103,700까지 이 BF16 weight를 읽는다. 부모 SSA tensor의 lifetime 끝을 backing allocation 해제 시점으로 해석하면 안 된다.

Fused O258에는 `MainContext`와 `VectorEngine` context가 표시된다. 기존 두 lookup은 `MainContext`만 가진다. 그동안 x_trf T118은 44,787–103,700 동안 살아 있고, 첫 up local-scale VRF T146은 decode 실행 중인 76,170–76,673에 적재되어 94,960까지 유지된다. 독립 probe에는 이 resident activation/scale 및 후속 8행 tile consumers가 없다. 현재 schedule에서 명백한 충돌을 확인하지 않았으며 E48은 이 배치에서 실제 정확도 검사를 통과했다.

E13은 이전 FFN에서 **up과 gate 둘 다** fused였고 실패했다. tagged fullshape probe70658은 동일한 local 크기의 decoder만 정확히 검증했다. E48은 이제 up-only fused를 사용하는 전체 FFN의 세 공식 입력도 통과했음을 보여 준다. 어느 결과도 E13 실패 원인을 특정하거나 임의 입력 전체의 정확성을 증명하지는 않는다.

이 감사는 기존 source/schedule/ELF/결과만 읽었다. 추가 compile, NPU 제출, live source 수정, Git 작업은 없다. 근거와 한계는 `evidence.json`, 개별 op/tensor 기록은 `schedule-summary.json`에 있다. 확인일: 2026-09-21.

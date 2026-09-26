# C09 frozen two-stage — 최초 NPU 진단과 E127 채택

**E127 채택 완료. E126의 초기 종료·미채택 판단은 후속 검증으로 대체됐다.**
[검증 기록](validation/README.md) · [채택 구성](../../../../baseline/e127.json).
원본 C09 source/layout/tile과 exact O image를 유지했고 QKV/FFN은 E122와 같다.
세 allocation의 paired O 변화율은−2.4960% / −3.8255% / −3.9303%, 중앙값−3.8255%다.
Candidate 표준45비교와 extra30비교가 PASS했다. 반복을 제외하면 커널별9개 표준 입력이다.
별도 pre-RMS projection probe8case/30720출력도 PASS했으며 probe timing은 성능에 포함하지 않는다.
5%는 추가 parameter tuning 예산 기준이며 채택의 최소 개선율이 아니다.
후속 [E127 공식 제출 `bc21122c`](../../../../docs/experiments/e127-official-submit-20260927.json)은
기본15비교 PASS,6.0699449×, 공개 리더보드53/69위로 완료됐다. 채택 검증과 공식 평가를 구분한다.
아래 E126 측정·trace·원본 decision은 당시 기록이며, 후속 채택 근거는 위 validation에 있다.

Parent617aaef; branch `diag/c09-first-npu-20260927`. E125 M08 종료 기록을 main/origin에
통합한 뒤 시작한다. E122 source에 **기존 C09 projection을 byte 그대로** 조합한다.
QKV/FFN은 E122 image를 요구하며, O도 기존 C09 exact image와 일치해야 한다.
새 architecture나 새 tile 변형이 아니다. 정적+0.71%만 측정했던 원 prototype의
실제 하드웨어 weight-stage overlap과 전체 O 이득 여부를 검증한다.

[원본 C09](../c09-output-pipeline/README.md), [전체 경로·추가 비용·종료 조건](frozen-two-stage-v1/recipe.json).
BF16 activation, K16 F32 합산, channel scale 후 첫 BF16, canonical32×480B gather를 유지한다.
두 개의60행 compact weight DMA 및 출력 commit 범위를 실제 graph에서 검사한다.
전체 O trace로 첫 dot 시작/두 번째 weight overlap/setup 비용을 관측하고,
표준 전체 host의 E122 before→candidate→E122 after로 판정한다. Profiler 구간은 합산하지 않는다.
최초 screen은 원본3입력으로 범위를 제한했다.
**E126 / Arena105820: 기본 정확도 PASS, 단회 O −2.496% 관측.** 당시5% 탐색 예산 기준으로
반복·확장 정확도·채택 없이 보존하고 E122를 유지했다. 이 초기 판단은 E127로 대체됐으며,
원본 recipe/decision의 종료 문구는 최초 실험 시점의 기록으로 보존한다.

## Exact 재현과 정적 비용

첫 compile PASS. O image80,640B SHA`fe90ed758c27fe40a11e78ff5132e7b98152fc8cd79208d49c22053672859495`는
원본 C09와 byte-identical이다. 정적 전체27432 vs control27239(+0.71%)도 같다.
HBM load 두 개의 source view는 각각0/60행 offset을 갖고, descriptor의 local60행
compact extent와 함께 해석해야 한다. Source code의 tile만으로 bounds를 판정하지 않는다.

| 정적 경계 | Control | C09 |
|---|---:|---:|
| 첫 dot 시작 | 16157 | 9740 |
| 마지막 dot 종료 | 21220 | 20207 |
| F32 gather 시작 | 21220 | 21413 |
| 마지막 dot→gather gap | 0 | 1206 |
| Gather duration | 470 | 470 |

C09 stage0 dot9740–12403과 stage1 weight10578–17544의 overlap은1825다.
두 번째 table DMA가9740–10578로 stage1 weight 앞에 있다. C09에는 mutable output의
`IndexAccess/IndexWrite` 연쇄가 추가된다. Gather destination의 Scratchpad→ClusterSync
패턴 자체는 control에도 있으며, 새 sync가 삽입됐다고 해석하지 않는다. 그 일정과
마지막 dot→gather gap이 달라졌다는 사실까지만 확인했으므로 NPU gap도 별도 관측한다.
IndexWrite가 그 대기의 직접 원인인지는 현재 graph만으로 확정하지 않는다.
위 compiler cycles를 NPU cycles로 환산하거나 overlap을 예상 speedup으로 더하지 않는다.

## 첫 NPU 결과

전체7stage/57비교 PASS. 표준 host의 O는 E122 before51237 → C09 49923 →
E122 after51165 cycles다. 앞뒤 중앙값51201 대비 **−2.4960%**다.
원본3입력과 한 allocation만 사용했으며 반복된57비교를 확장 정확도로 세지 않는다.
이 E126 수치 자체는 단회 screen의 관측이다. 독립 반복과 채택은 별도 E127 후속 기록이다.

| 별도 trace, 각3runs×2clusters 중앙값 | Control | C09 |
|---|---:|---:|
| Task 시작→첫 dot | 28945 | 17148 |
| Task 시작→마지막 dot 종료 | 40953 | 39542.5 |
| 첫 weight 시작→마지막 weight 종료 | 26660 | 30073 |
| 첫 weight 시작→마지막 dot 종료 | 38679 | 36802 |
| 마지막 dot→gather 시작 | **2** | **2** |
| Gather duration | 1199 | 1228 |

C09 dot0와 weight1은 **7392.5 cycles 중첩**하며6관측 모두7153–7468 범위다.
그러나 weight0 종료→weight1 시작이2154.5이고, 그 사이 table1 DMA 구간은2044다.
같은60행의 dot0는9564, dot1은6724다. 동시실행 중인 첫 stage의 marker가 더 길지만
table/DM/HBM 특정 자원 충돌로 확정할 counter는 없다. 반복 setup·분할 전송·작은 dot과
대기 비용을 포함한 전체 경로의 이득은 위2.50%다. 겹치는 interval을 더해 절감량을 만들지 않는다.

**정적 finaldot→gather1206 대기는 실제 NPU에서 재현되지 않았다.** 양측6관측 모두2다.
따라서 정적 sync gap을 이번 하드웨어의 병목 또는 제거 가능한1206cycles로 제시하지 않는다.
첫 dot은 상당히 앞당겨졌고 실제 overlap도 있지만, 최종 dot/전체 O의 개선 폭은 작다.
4/8-stage sweep보다 다음 변경이 반복 setup·공급 envelope·첫 stage 구간 증가 중
무엇을 실제로 제거하는지 설명해야 한다. 추가 parameter 탐색은 종료하고, 동결 prototype의
제한적 채택 검증은 별도 E127에서 완료했다.

[판정](decision.json), [원본7stage 결과](e126-c09-first-npu-screen-job-105820-results.json),
[trace 분석](profile-analysis-c09-first-npu-v1.json), [graph 전송 검사](descriptor-analysis-c09-first-npu-v1.json),
[독립 NPU 해석](independent-npu-review.md), [독립 package·원시 업로드 감사](independent-review.md).
표준 control은 실제 E1223-kernel host이며, diagnostic control은 O만 실행하는 같은 O image의
과거 E104 generic host다. Standard/diagnostic 후보의 세 kernel image가 같고,
진단 label 수정은 profile name영역454B뿐이다. Task code/ABI/constant는 보존했다.

재검증:

```sh
python3 moa-dev/experiments/candidates/c09-output-first-npu/verify_descriptors.py c09-first-npu-v1 --check
python3 moa-dev/experiments/candidates/c09-output-first-npu/analyze_profile.py c09-first-npu-v1 --check
python3 moa-dev/experiments/candidates/c09-output-first-npu/results.py e126-c09-first-npu-screen 105820 --check
```

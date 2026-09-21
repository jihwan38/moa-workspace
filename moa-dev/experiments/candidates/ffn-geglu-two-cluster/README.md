# GeGLU까지 up/gate의 2-cluster layout 유지

**채택하지 않는다.** 두 후보 모두 컴파일에는 성공했지만 기존 up8/down12 정적 makespan 204460보다 느렸다. Direct global-scale 복제는 205824, hier16 global-scale 복제는 206339다. NPU 미검증이다. `mlp.rs`는 마지막 hier16 실험본을 보존한 파일이며 추천 교체본이 아니다.

## 경로와 경계

기존 up/gate는 `Dummy2 cluster × L/32%240#256 slice × L%32 local`에서 계산한 뒤, 각 결과를 canonical L%7680으로 모으고 기존 1-cluster UpGateRows로 다시 분배했다. GeGLU에서는 인접 두 slice를 모아 L120개씩 연산한 뒤, 결과를 두 down cluster로 재분배했다.

후보는 up/gate 결과를 원래 2-cluster layout에서 그대로 받아 slice별 32개 원소의 GeGLU를 계산한다. Up/gate global-scale F32 곱 뒤 BF16, gate의 F32 erf 식, F32 up×GELU와 `/2`, 최종 BF16 순서는 동일하다. `validate_layout.py`는 scalar operation sequence와 BF16 cast가 동일함을 확인한다. 출력 수학을 GELU 근사나 다른 rounding 방식으로 바꾸지 않았다.

UpGateCluster의 Dummy2는 **L의 상위 행**이고 DownCluster의 Dummy2는 **H의 상위 출력 행**이다. 후보는 GeGLU 결과를 먼저 cluster별 `Slice/L%7680`으로 typed compact하고 Dummy2를 `L/7680`으로 명시적으로 rename한다. 그 다음 typed DMA가 새 DownCluster Dummy2/H row group으로 복제하고 L을 8개 column group으로 나눈다. 두 Dummy2의 의미가 같다고 가정하여 tensor를 그대로 재해석하지 않는다.

`mlp.direct-scale.rs`는 각 scale을 모든 유효 slice로 직접 DMA한 첫 비교군이다. `mlp.rs`는 `broadcast_geglu_scale`에서 cluster별 ring16 leader만 DMA하고 main switch로 broadcast한 다음 기존 TagMode::Zero VRF 읽기를 사용한다. 이 변경은 scale 값이나 수치 경계를 바꾸지 않는다.

## 실제 정적 비용

| 항목 | 기존 1-cluster GeGLU | 2-cluster direct scale | 2-cluster hier16 scale |
| --- | ---: | ---: | ---: |
| makespan | **204460** | 205824 | 206339 |
| instructions | 340 | 326 | 332 |
| DMA union | 132947 | 157205 | 146955 |
| Main union | 168570 | 167916 | 168474 |
| Vector union | 135233 | 135123 | 135123 |
| Sub union | 18180 | 18092 | 18092 |

제거한 up/gate 출력 복원 DMA는 합계 4784 cycles(각 출력 1026+1366), GeGLU 앞 paired gather는 Main 합계 588이었다. GeGLU 자체의 원소 연산은 이미 작아서 vector 합 감소는 110 cycles뿐이다.

반면 새 canonical GeGLU 결과의 down 분배 DMA는 기존 9392에서 **28648**로 증가하고, 그 앞 compact에 1026이 추가된다. Global-scale direct DMA는 합계 2656→11416으로 늘었다. Hier16은 새 global-scale DMA 합을 1166으로 줄이지만 main switch 두 개를 추가하며 전체 makespan은 515 cycles 더 늘었다. DMA 합 감소만으로 실제 의존성을 포함한 전체 schedule이 개선된다고 판단할 수 없는 사례다.

기존 분산 L120 source는 down 입력 분배에 유리하다. 이 작은 단계만 보고 GeGLU를 2클러스터로 옮기는 것보다 전체 입력 재분배 비용을 함께 측정해야 한다. 향후 검토 지점은 canonical single-slice gather를 거치지 않는 합법적인 L축 typed copy이지만, padding과 서로 다른 high-axis 역할의 정합성을 먼저 증명해야 한다. 현재 후보는 그 미구현 경로의 성능을 주장하지 않는다.

## 파일과 검증

`mlp.single-geglu.rs`는 출발점, `mlp.direct-scale.rs`는 첫 비교군, `mlp.rs`와 `change.patch`는 마지막 hier16 실험이다. `static-comparison.json`에 source/artifact hash와 구체적 source/context 비용, `logical-index-proof.json`에 행 주소와 수치 연산 불변 검사를 보존했다.

Scratch `/tmp/moa-ffn-geglu-two-cluster`, 별도 target `/tmp/moa-ffn-geglu-two-cluster-target`에서 FFN 하나만 컴파일했다. [Direct-scale schedule](../../artifacts/ffn-geglu-two-cluster/decoder_feedforward-direct-scale.json), [hier16 schedule](../../artifacts/ffn-geglu-two-cluster/decoder_feedforward.json), [컴파일 로그](../../artifacts/ffn-geglu-two-cluster/compile.log). live source, 공유 target, NPU, Git은 변경하지 않았다.

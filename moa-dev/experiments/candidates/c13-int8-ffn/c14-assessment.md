# C14 컴파일 실패의 재검토

C14-A v3의 마지막 실패는 `project_down_tile`의 첫 `x.to_dm`다.
Producer는 cluster마다 240 slices × 16 BF16 값(유효 3840개)을 물리 256 slices에
두며, consumer는 두 column × 1920개로 받는다. Compiler가 이 직접 DMA에서
3840개의 유효 범위와 4096개의 padding 포함 범위를 변환하지 못했다.

이는 이 배치의 lowering 실패다. L-stage FFN을 구현할 수 없다는 결론이나 성능
탈락으로 해석할 근거는 없다. E74 `project_up_and_gate`는 같은 종류의 padded
row 출력을 먼저 canonical slice에 compact한 뒤 다음 배치로 보낸다.
C14도 각 stage·cluster의 3840개만 compact한 뒤 ring leader로 보내는 경로를
별도로 구현·검증할 수 있다. 전체 L=15360개의 완성을 기다리는 경로로 되돌릴
필요는 없다. 다만 추가 DMA 비용과 실제 stage overlap은 측정해야 한다.

이 문서는 실제 v3 코드·로그와 E74의 구현을 대조한 후속 설계 검토다.
C14 수정본을 컴파일·NPU 실행했다는 뜻은 아니다. 이번 재개 작업은 C13을
실행 가능한 상태와 실제 정확도 검사까지 진행한다.

# E43 down activation 4-row hierarchical broadcast

E37의 `project_down_sharded` 입력 복사만 바꾸는 후보다. 기존에 두 cluster의32개 row-group×8개 column slice마다1920개 BF16 값을 보내던 복사를, row-group4개 중1개에 먼저 전송한 뒤 Switch로 나누려 했다. Dot, scale, dtype 경계, RMS, residual은 변경하지 않았다.

**최초 exact compile 네 시도는 모두 실패, NPU 미실행.** `[H/240%8,1#4,L/1920]`와 명시적 replica 축, 익명4배 반복을 사용한 Broadcast1 경로는 backend `modulo must divide the size` panic을 냈다. Named replica 축의 CustomBroadcast32는 `Switch snoop bitmap disagrees ... output slice8`로 거절됐다. 네 번째 시도 뒤 live 소스는 E37로 복구했다.

`evidence.json`에 시도별 source와 log hash, `candidate.patch`에 네 번째 소스와 E37 차이를 보존했다. 이 네 시도에는 유효 schedule이 없어 실제 DMA 시간·전체 지연·정확성을 주장하지 않는다. 논리적으로 예상한 복사량 감소만으로 hardware mapping을 통과한다고 가정할 수 없다.

**후속 다섯 번째 조합은 exact compile PASS, 정적 회귀로 NPU 보류다.** 별도 [mlp.custom-explicit.rs](mlp.custom-explicit.rs)는 anonymous4 출력을 그대로 두고 Switch만 `CustomBroadcast { ring_size: 32 }`로 바꾼다. E37의 up/gate에서 이미 사용하던 explicit anonymous replication 패턴을 적용했다. 기존 네 실패 파일은 수정하지 않았다. DMA는135,607→131,718로 줄지만 Main Switch4,103이 추가되어 makespan162,209→166,181(+2.45%)이다. [후속 분석](custom-explicit-review.md)과 [별도 evidence](custom-explicit-evidence.json)에 성공한 정확한 조합과 비용을 구분했다. 향후 decode를 제거해 Main의 실행 경로가 달라지면 DMA 감소가 유용할 가능성은 있지만, 아직 검증되지 않은 가설이다.

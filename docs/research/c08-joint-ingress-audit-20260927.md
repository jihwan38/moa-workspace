# C08 E105 joint QKV 입력 DMA 재점검

2026-09-27. **Padding이 있는 destination tile에 HBM을 직접 전송하는 의심 패턴은 확인했다. C08의 실제 DMA extent·stride·source bounds는 아직 확인하지 못했다.** 새 compile·Arena 실행·후보 수정 없이 기존 파일만 읽었다. 다른 후보에서 발견한 padding 문제를 E105의 확정 원인으로 옮기지 않는다. [기계 판독 기록](c08-joint-ingress-audit-20260927.json).

[Frozen v2 source](../../moa-dev/experiments/candidates/c08-fused-qkv/evidence/69c0337482d058362ec1316125b8c4dccf1f33bb3df9dc9b5726551e91f2cadc.rs)의 `project_qkv_fused`는 Q를 2048행씩 둘로 나눠 K/V와 함께 네 matrix slot을 가진 DM에 넣는다. 286/288/290행은 각각 `QkvMatrix=2`인 Q를 `2#{!}4` destination tile의 offset0에, `QkvMatrix=1`인 K/V를 `1#{!}4` tile의 offset2/3에 `to_dm_view`한다. 294/296/298행의 channel scale도 같은 방식이다. 읽기 쉬운 [fused.rs](../../moa-dev/experiments/candidates/c08-fused-qkv/fused.rs)의 23–35행에 동일 코드가 있다. 전체 frozen source가 실제 v2 compile 입력과 byte-identical이며 빌드 manifest SHA256와 일치함을 확인했다.

보존 schedule에서 weight DMA는 #96(Q), #120(K), #135(V), scale DMA는 #65/#74/#90이다. 이들 description이 위 여섯 source call과 대응한다. 그러나 schedule의 source snippet·tensor dependency·duration만으로 전송한 HBM 주소 범위를 확정할 수 없다. K/V load가 각각 26,475 compiler cycles라는 기존 관측 역시 전송 byte 수나 범위 초과의 증거로 바꾸지 않았다.

Host `moa-dev/experiments/artifacts/c08-fused-qkv/c08-fused-contiguous-v{1,2}`와 Docker `moa2026-dev:/tmp/moa-c02-c03/c08-fused-contiguous-v{1,2}`를 확인했다. v1은 compile 실패 보존물이고, v2에는 schedule과 exact/image binary가 남아 있다. 이 네 경로 안에서는 기존 graph·IR·vISA dump를 찾지 못했다. 이번 감사에서는 binary decoder로 descriptor를 복원하지 않았다. **Q/K/V 또는 scale을 몇 배 읽었다거나 source 범위를 넘었다는 수치는 제시하지 않는다.** [E105 job74530](../../moa-dev/experiments/candidates/c08-fused-qkv/c08-screen-fused-results.json)의 QKV 정확도 실패는 유지하며, 원인은 미확정이다.

재개할 경우 첫 단계는 frozen v2의 isolated exact compile과 descriptor 검증으로 한정한다. Q/K/V 및 scale의 HBM binding에서 emitted source offset·extent·stride·마지막 byte까지 연결해 원본 tensor bounds와 비교해야 한다. 그 결과가 잘못된 coverage를 증명할 때에만 compact ingress와 명시적 배치 복사를 수정 후보로 검토한다. 복사 비용과 마지막 weight가 준비된 뒤에야 시작하는 통합 dot의 제약도 함께 평가해야 하며, 이번 의심 패턴만으로 같은 fused 구현의 NPU 재실행을 권하지 않는다.

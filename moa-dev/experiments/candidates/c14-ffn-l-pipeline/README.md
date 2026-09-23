# C14-A — L 구간별 FFN과 producer/consumer 공동 배치

**컴파일 실패로 종료. Arena·공식 제출 없음, E104 유지.** 브랜치는
`opt/c14-ffn-l-pipeline`, 출발 소스는 E74
`9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`, 성능 비교군은 E104다.
[판정](decision.json) · [최초 설계](candidate.json) ·
[최종 소스](bf16-two-stage-v3/mlp.rs) · [E74 대비 변경](e74-to-c14-v3.patch).
하나의 구조를 구현하며 compile 3회(최초 1회, 표현 수정 2회)를 수행했다.
세 개의 독립 구조나 성능 실험으로 세지 않는다.

구현한 소스는 두 개의 L=7680 구간 각각에서 up/gate → 분산 GeGLU → down을 연결한다.
Up/gate와 down 모두 cluster가 구간 내부 L의 상위 3840개를 맡는다.
Down의 slice는 H의 30행 × 두 L column으로 바꾼다. 이전 GeGLU 실험의
H-owned down cluster와 full-vector canonical gather를 함께 제거한다.
구간 및 cluster의 F32 출력 부분합을 합친 뒤 BF16으로 변환한다.

E74의 BF16 입력/streamed BF16 weight dot/F32 block partial, 과거 2-cluster
GeGLU의 scalar 연산 순서, C04 ring32의 leader 방송 기법을 선택한다.
각 방법의 기존 전체 mlp.rs를 덮어쓰지 않고 하나의 새 호출 경로로 조합한다.
Public decoder_feedforward 인터페이스와 RMS/residual/layer scale은 유지한다.

작은 중간 activation 제거만으로 큰 이득을 기대하지 않는다. Weight HBM
payload는 동일하며 새 down weight pitch, 방송, stage/cluster F32 합산,
두 배의 큰 dot stream 수까지 전체 schedule과 NPU 측정에 포함한다.
완성된 image/schedule이 없어 overlap·실제 weight load 횟수·NPU 정확도는
미검증이다. 소스의 의도한 의존성만으로 실제 중첩이나 개선을 주장하지 않는다.

## 컴파일 결과와 범위

| 버전 | 변경 | 최초 오류 |
|---|---|---|
| v1 | 전체 FFN 구현, HBM tile 후 내부 TileL 축으로 rename | `LowLevelReshape: a tiled view is not supported` |
| v2 | 전체 HBM view를 Stage×TileL로 먼저 나눈 뒤 Stage 선택 | 첫 up weight DMA의 `cannot factorize common axis TileL` |
| v3 | E74처럼 L-high cluster를 별도 Dummy2로 표현 | GeGLU → down leader DMA의 `cannot resize 3840 to 4096 through TileL_1920=2` |

v2→v3은 기본 수정 횟수에 대한 한 번의 명시적 예외다. E74에서 쓰던 별도
cluster 축이 구체적인 해결 근거였고, [v3 recipe](bf16-two-stage-v3/recipe.json)에
총 3회로 제한한 이유를 build 전에 기록했다. 세 번째 실패 뒤 추가 수정하지 않았다.

최종 오류는 [project_down_tile](bf16-two-stage-v3/mlp.rs)의 첫 `x.to_dm`이다.
Producer는 각 cluster의 유효 240 slices×16행=3840개를 물리 256 slices에 둔다.
Down은 같은 cluster 안에서 두 column×1920개로 받아 H row group에 방송한다.
Compiler는 producer의 4096개 물리 extent와 consumer의 3840개 유효 extent를
이 직접 DMA에서 조정하지 못한다. 논리 인덱스가 맞아도 이 physical mapping은
lowering되지 않는다. NPU가 C14 계열 전체를 지원하지 않는다는 뜻은 아니다.

이전의 full-vector gather로 되돌리는 추가 수리는 하지 않았다. Padding 없는
producer와 그에 맞춘 consumer 등 다른 연결을 시험하려면, padding 변환 비용과
연산 배치가 함께 달라지는 별도 조합으로 기록해야 한다. 다음 독립 후보는 C13이며
이번 작업에서는 구현하지 않았다.

## 검증 및 재현

`verify_layout.py`로 각 버전의 L 15360개 생산/소비 좌표, H 3840행의 weight
구간 분할, packed/scale HBM pitch와 8B 정렬, GeGLU scalar 연산 및 BF16 cast
순서를 확인했다. 모두 PASS다. 이 검사는 ISA lowering·실제 DMA·NPU 정확도
검증을 대신하지 않는다. Down의 F32 합산 순서는 바뀌며 수치 오차는 미검증이다.

각 `build-*-repro.json`은 E74 source origin과 E104 비교 계획을 분리해 기록한다.
`evidence/`에 compile 입력 mlp.rs와 로그를 동결했고, 원본 입력·container 입력·
snapshot의 hash가 같음을 확인했다. 세 버전 모두 image/schedule을 만들지 못했다.

```bash
python3 moa-dev/experiments/candidates/c14-ffn-l-pipeline/verify_layout.py bf16-two-stage-v3
python3 moa-dev/experiments/structural_build.py c14-ffn-l-pipeline bf16-two-stage-v3 --plan
python3 moa-dev/experiments/structural_build.py c14-ffn-l-pipeline bf16-two-stage-v3 --label NEW-LABEL
```

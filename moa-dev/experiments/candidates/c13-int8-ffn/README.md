# C13 — FFN의 INT8 곱셈과 I32 누산

**전체 FFN 컴파일은 scale DMA의 8바이트 정렬 제약으로 종료했다. 성능 미평가,
미채택, E104 유지.** 브랜치 `opt/c13-int8-ffn`. 출발 소스는 E74
`9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`, 비교 계획은 E104다.
[판정](decision.json) · [사전 설계와 중단 기준](candidate.json) ·
[최종 소스](integer-v2/mlp.rs) · [E74 대비 diff](e74-to-c13-v2.patch).

E104의 한 helper에 정수 연산을 끼워 넣은 실험이 아니다. E74의 BF16 입력에서
출발해 `feedforward_sharded → quantize_h → project_up_and_gate → geglu →
project_down_sharded → quantize_l` 호출 경로의 up/gate/down 세 projection을
모두 INT8 weight·activation, I32 block dot, F32 복원으로 작성했다. 전체 Rust
frontend는 통과했지만 DMA lowering 실패로 실행 가능한 FFN image는 없다.

선택한 기존 방법은 E74의 F32 weight block-scale 합산과 BF16 비선형 경계,
C07의 두 lane/unzip 기법, C04의 down ring32 전송이다. E104의 global input
scale과 세 FP8 항은 사용하지 않는다. 다만 up/gate 행 소유권, 전체 GeGLU 완성
후 down을 시작하는 의존성, down의 H-owned cluster와 8-column 합산은 유지했다.
C14의 L-stage 배치를 다시 시험한 것은 아니다. 공개 `decoder_feedforward`
인터페이스, RMS/residual/layer scale, host와 fixture를 바꾸지 않았다.

## 계산과 비용

각 16개 BF16 입력 블록에서 `s = 2^floor(log2(max(amax, 2^-120))) / 64`를 구한다.
`hi = clamp(round(x/s), -127, 127)`, `lo = round((x/s-hi)*128)` 두 INT8 항으로
입력을 근사한다. 반올림은 설치된 SDK의 fixed-point 변환에 맞춰 half-away-from-zero다.
FP4 E2M1 weight는 두 배 값 `2w`를 INT8로 정확히 표현한다. Dot 뒤에
`128*dot_hi + dot_lo`를 I32에서 계산하고 F32로 변환한 뒤 `s/256`과 weight
block scale을 적용한다. Up/gate는 BF16으로 내보내 기존 GeGLU에 연결하고,
그 BF16 결과를 다시 양자화해 down에서 같은 정수 계산을 한다.

블록당 정수 결과의 절댓값 상한은 `16*12*(128*127+64) = 3,133,440`으로
I32와 F32 정확 정수 범위 안이다. 이는 activation 근사와 최종 F32 합산 오차를
없앤다는 뜻은 아니다. 작은 값과 큰 값이 같은 블록에 섞이면 작은 값이 사라질 수 있다.

두 항 × 64개 INT8 입력으로 4개 block dot을 얻는 경로를 택했다. E74의 16개
BF16 입력 packet보다 큰 packet을 쓰지만, 이 사실만으로 전체 속도를 예측하지 않는다.
FP4→FP8 lookup→F32 widening→정수 변환은 weight 전체를 지나며, 별도 scale 계산,
hi/lo 생성, GeGLU 결과의 canonical gather, scale 전송을 추가한다. Weight HBM
payload와 F32 block partial 저장·합산도 남는다. 특히 weight 변환은 정수화 전에
8개 F32 단위로 처리되어 E74의 FP8 materialization보다 stream 길이가 늘어난다.
완성된 schedule이 없으므로 이러한 비용의 중첩이나 순이익은 확인하지 못했다.

## 확인한 결과

| 검사 | 결과 | 해석 범위 |
|---|---|---|
| FP4→INT8 + 2-lane integer dot + I32 radix 결합 primitive | 첫 compile PASS | 작은 API 경로 지원, 전체 FFN/NPU 검증 아님 |
| CPU 전체 FFN 계산식, 기존 fixture 3개 | 11,520개 출력 PASS, 최대 절대오차 0.0078125 | 고정 fixture의 수학적 screen, NPU mapping/rounding/reduction tree 검증 아님 |
| 전체 FFN v1 | compile FAIL | `reduce<H/L>`가 유지하려던 block time 축까지 줄임 |
| 전체 FFN v2 | compile FAIL | scale gather의 4B stride가 DMA의 8B 정렬에 맞지 않음 |

작은 primitive의 정적 schedule은 4,630 cycles이며 전체 FFN이나 E104와 비교할
수 없다. CPU 검증은 기존 `atol=0.01 + rtol=0.01*abs(expected)`를 그대로 적용했다.
최대 오차/허용오차 비율은 0.5866 미만이다. FP4 16개 code의 정수 변환과
누산 범위도 확인했다. BF16 유한값 65,280개를 인접 bit-pattern 블록으로 묶은
검사와 zero/half-tie/outlier/최대·최소값 검사를 추가했지만 모든 블록 조합의
동치를 증명한 것은 아니다. [수치 결과](numeric-screen.json),
[실행 script와 reference/fixture provenance](numeric-provenance.json).

v1의 최대값 계산은 H/L 축 전체를 reduce했다. v2는 블록 안의 16개 원소만
`IntBlockElt` 축으로 이름 붙여 그 축을 reduce한다. 이 사전 기록된 한 번의
shape 수정 뒤, `quantize_h` 마지막 `restore.to_dm`에서 실패했다. Producer는
각 scale을 32B padded packet에 저장하고 consumer는 F32 scalar를 4B 간격으로
모으려 한다. 생성된 `strides([4,16,4,...])`가 compiler의 8B 정렬 검사에 걸린다.
[최종 로그](evidence/c13-integer-v2/compile.log).

AGENTS.md의 한 번 수정 한도에 따라 추가 packing 수정은 하지 않았다. 이 결과는
INT8 구조의 성능 회귀나 계열 전체의 실패를 뜻하지 않는다. 후속 C13 수정은 scale을
8B 이상 묶음으로 생산·전송하고 dot consumer의 block 인덱스까지 맞추는 계약을 먼저
정해야 한다. 같은 계열의 layout 수정으로 세며 새 독립 구조로 집계하지 않는다.

## 재현

각 `build-*-repro.json`은 E74 출발 source와 E104 비교 계획을 분리하고, 실제
container 입력·동결 source/log hash를 기록한다. `evidence/`의 실패 source도
보존한다. CPU reference 세 파일은 선택한 E74 archive와 일치한다. SDK/CLI는
0.8.1, toolchain/Cargo.lock/container image는 build 기록과 판정에 남겼다.

```bash
python3 moa-dev/experiments/structural_build.py c13-int8-ffn integer-v2 --plan
python3 moa-dev/experiments/structural_build.py c13-int8-ffn integer-v2 --label NEW-LABEL
python3 moa-dev/experiments/candidates/c13-int8-ffn/build_probe.py probe-v1 --label NEW-PROBE-LABEL
```

수치 script는 PyTorch와 safetensors가 설치된 환경에서 실행한다. `--reference-dir`은
E74의 `scripts/`, `--fixture`는 기존 `fixtures.safetensors`, `--out`은 결과 JSON을
지정한다. 정확한 사용 경로와 파일 hash는 `numeric-provenance.json`에 있다.
Arena·공식 제출은 수행하지 않았다. FFN image가 없어 정확도/성능 실험에 보낼 수 없다.

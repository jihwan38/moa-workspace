# C01: FFN block scale와 합산을 두 번째 DPE로 이동

2026-09-22, `opt/c01-scale-dpe`. **이번 C01 실험은 미채택이며 E55를 유지한다.**
Up/gate 단독 E65의 반복 중앙값은 2.72% 낮았지만 실행 범위가 겹친다.
Down E66과 결합 E67은 기본 정확도를 통과했으나 첫 성능 screen에서 느렸다.
이 폴더는 실행한 후보·진단·실패를 보존한다. 현재 runtime의 `src/`에는 적용하지 않았다.

기준은 [E55 구성표](../../../../baseline/e55.json), commit
`873f64ed94229c88cd18d2731bdcca7eea0d6365`다. 모든 전체 FFN 후보는 이 기준의
`src/device/shared/mlp.rs`만 교체했다. [검증 결과](summary.json)는 후보마다 46개 compile
입력, host에 포함된 image, 실제 Arena 업로드의 hash를 연결한다. QKV/O image는 E55와 같다.

## 실측과 채택 판단

각 job은 변경하지 않은 기본 harness의 3개 입력을 사용한다. 3회 행은 job 중앙값들의
중앙값이다. **음수는 cycle 감소**, 양수는 증가다. 비교 기준은 이번에 재실행한 E55다.

| 구성 | FFN cycles | E55 대비 | job 수 | 기본 정확도 / 판정 |
|---|---:|---:|---:|---|
| E55 대조 | 378,270 | — | 3 | 45개 출력 비교 PASS |
| E65: up/gate만 scale-DPE + streamed BF16 decode | 367,967 | −2.72% | 3 | 45개 PASS; 개선 미확인 |
| E66: down만 scale-DPE + tail 수정 | 413,913 | +9.42% | 1 | 15개 PASS; 회귀 screen 후 중단 |
| E67: E65 + E66 | 392,118 | +3.66% | 1 | 15개 PASS; 회귀 screen 후 중단 |

측정 원본: [E55 대조](../../../../docs/experiments/e55-c01-control-repeats.json),
[E65 반복](../../../../docs/experiments/e65-c01-upgate-scale-dpe-repeats.json),
[E66](../../../../docs/experiments/e66-c01-down-scale-dpe.json),
[E67](../../../../docs/experiments/e67-c01-combined-scale-dpe.json).
단독/조합 구성표는 [E67 조합 기록](../../../../docs/experiments/combinations/e67-scale-dpe.md)에 있다.

E65의 job72754/72760/72762는 **367,967 / 375,645 / 366,983**,
E55 job72759/72766/72767은 **375,552 / 378,270 / 379,669**였다.
중앙값 기준 범위 폭은 각각 2.35%, 1.09%이고 두 범위는 93 cycles 겹친다.
느린 E65 실행을 제외하지 않았다. 이는 통계적 유의성 검정이 아니라, 기존
[채택 정책](../../../../baseline/README.md)의 보수적인 겹침 기준에 따른 보류다.

E65는 별도 job72775에서 **추가 입력 run3..8의 30개 비교도 PASS**했다.
[추가 입력 기록](../../../../docs/experiments/e65-c01-upgate-extra.json)은 PRNG·분포·오차 기준을
바꾸지 않고 run 번호만 확장하며, 세 device image가 기본 package와 같음을 검사한다.
이 진단의 cycles는 위 표에 합치지 않았다. E66/E67의 추가 6입력·반복 성능 검사는
첫 screen의 회귀 때문에 진행하지 않았다. 이번 후보의 전체 모델 추론·새 FFN 최종 출력
상쇄 경계 검사·공식 제출은 하지 않았으므로, 기본 PASS를 일반 입력 정확도로 확대하지 않는다.

## 바뀐 실행 경로

실제 호출은 `host/runtime.rs: Decode::mlp → ops::decoder_feedforward →
shared::mlp::feedforward_sharded → project_up_and_gate / project_down_sharded`다.
`decoder_feedforward`는 residual을 읽고 pre-RMS, FFN, post-RMS, residual/layer scale을
거쳐 같은 residual HBM을 갱신한다. Sliding/full layer가 같은 FFN을 사용한다.

E55의 block 계산은 `p[r,b]=sum16(BF16(x)*BF16(weight))`를 **F32**로 유지하고
VE에서 FP8 scale을 F32로 넓혀 곱한 뒤 block 합을 구한다. C01은
`p → BF16 partial DM`, `FP8 scale → BF16 → TRF`, `BF16 partial × BF16 scale →
두 번째 DPE의 F32 합`으로 바꾼다. Scale의 widening은 정확하지만 partial에는 새 BF16
반올림이 생긴다. DPE 합산 순서도 기존 VE와 같다고 가정하지 않는다.

| 소스 | 실제 변경과 비용 |
|---|---|
| [mlp.upgate-materialized.rs](mlp.upgate-materialized.rs) | 전체 BF16 weight DM을 유지하면서 up/gate scale-DPE 적용. 정적 202,779로 +25.01%; NPU 미실행 |
| [mlp.upgate-streamed.rs](mlp.upgate-streamed.rs), E65 | FP4→FP8 DM은 유지. FP8→BF16 lookup 뒤 바로 첫 dot을 연결해 전체 BF16 weight DM 저장 제거. 32행 block partial을 한 번에 생성하고 Sub에서 scale을 FP8→F32→BF16로 변환. Down은 E55 그대로 |
| [mlp.down.rs](mlp.down.rs), E66 | Up/gate는 E55 그대로. Down은 8 column, column당 120 blocks를 유지하고 12행×5 pass를 20행×3 pass로 변경. BF16 partial과 scale을 128칸에 놓고 tail 8칸을 실제 zero로 덮은 뒤 DPE, 8-column F32 합산, BF16 출력 |
| [mlp.combined.rs](mlp.combined.rs), E67 | E65 up/gate와 E66 down을 한 source에서 결합. 단독 결과의 합으로 성능을 추정하지 않고 별도 실행 |

Down 후보는 E33 계열의 cluster 내부 compact 후 epilogue 배치로 옮기는 두 단계 gather도
포함한다. E55의 직접 gather 대비 정적 DMA 451 cycles가 추가된다. 따라서 E66은
scale 연산 한 개만 바꾼 ablation이 아니다. 새 partial staging·tail 수정·tile·gather 비용을
포함한 전체 구현 비교다. C01 안의 각 변경을 독립적인 가속 효과로 등록하지 않는다.

## D01: 과거 down 실패의 padding 문제 분리

[scale_probe.rs](scale_probe.rs)는 첫 dot 대신 알려진 F32 block partial을 받아
실제 down shape(H=3840, L=15360, 2 clusters×32 row groups×8 columns)에서
BF16 partial128, BF16 scale128, column별 F32 dot, 8-column F32 합을 모두 HBM에 출력한다.
[probe_scale.rs](probe_scale.rs)는 출력 메모리를 NaN으로 초기화하고 독립 oracle와 전수 비교한다.

세 입력은 row/block 좌표표와 정확한 2진 scale, column/16-block 경계의 단일 impulse,
인접 partial 상쇄다. 처음 두 입력에서는 partial BF16 rounding 오차가 없다.

| 진단 | Arena job | 관측 |
|---|---:|---|
| [원래 전송](scale_probe.original.rs) + [원래 host](probe_scale.original.rs) | 72741 | 전체 128칸을 먼저 zero-fill해도 compact 120칸 전송 뒤 tail에 다음 row 값 등이 나타남. 반올림 오차 없는 입력도 실패 |
| [16칸 clear](scale_probe.clear16.rs) + [112-block oracle](probe_scale.clear16.rs) | 72780 | 112..127을 지워 prefix112만 계산한 진단은 PASS. 유효 112..119를 버리므로 120-block 해결책이 아님 |
| 마지막 8칸 raw BF16 clear | 72783 | 120개 유효 block과 zero tail을 모두 보존. 3입력×4단계, 총 **23,696,640개 값이 정확히 일치** |

최종 수정은 compact copy와 scale lookup **이후**, zero로 채운 BF16 16칸 tensor에서
8칸을 Sub로 읽어 partial128/scale128의 120..127에 `commit_view`한다.
F32 zero를 BF16로 cast해서 같은 tail에 쓰는 시도는 compiler internal error가 났고,
sub-view `memset`은 SDK가 지원하지 않았다. Raw BF16 identity copy는 컴파일·실제 장치
검사를 모두 통과했다. [컴파일 실패·성공 로그 근거](compiler-log-evidence.json).

이 결과는 **이 SDK와 이 mapping의 compact→padded 경로에 존재한 문제**를 재현하고
우회한 것이다. 모든 DMA의 결함이나 E33 실패 전체의 유일한 원인을 증명한 것은 아니다.
첫 weight dot, GeGLU, 최종 normalization은 D01 범위 밖이다. 완전한 FFN 검사는
수정 후 별도 E66/E67에서 수행했다. 원래 host는 live/tail mismatch 수를 나눠 기록하지
않았으므로, 원본 mismatch 전부가 tail에만 있었다고 주장하지 않는다.

**수치 한계는 별개로 남는다.** 상쇄 입력의 F32 partial은
`1+15/256`, `−1−14/256`이 번갈아 나오고 scale은 1이다.
960 blocks의 원래 F32 합은 **1.875**, BF16 partial 뒤의 합은 **3.75**다.
패딩을 고친 NPU가 후자의 oracle와 정확히 일치한다. 이는 block partial 단계의
구체적인 반올림 반례이며, 전체 FFN 최종 출력의 tolerance 실패를 측정한 사례는 아니다.
Hi/lo partial 보정은 이번에 구현하지 않았다.

## 비용과 다음 판단

아래는 compiler 0.8.1 schedule의 interval union이다. 엔진끼리 겹치므로 합을 전체
latency로 해석하지 않는다. 실제 NPU cycles와도 다른 값이다.

| 구성 | 전체 정적 cycles | Main | Sub | DM→DM | 명령 수 |
|---|---:|---:|---:|---:|---:|
| E55 | 162,209 | 125,274 | 19,378 | 24,214 | 242 |
| Up/gate materialized | 202,779 | 125,036 | 24,088 | 24,214 | 170 |
| E65 | 167,867 | 109,140 | 24,088 | 24,214 | 168 |
| E66 | 169,681 | 127,818 | 31,335 | 42,635 | 331 |
| E67 | 166,904 | 111,684 | 36,045 | 42,635 | 257 |

E65는 Main 구간을 줄이지만 DMA interval은 E55와 같고 전체 정적 일정은 오히려 길다.
정적 예측만으로 NPU 성능을 대체할 수 없다는 결과다. E66/E67의 120→128 partial/scale
copy 6개는 각각 2,995, 합 **17,970 정적 cycles**다. 추가 gather 451을 합하면
DM→DM 증가 18,421과 일치한다. Scale lookup의 추가 HBM load도 2,514 cycles다.
실측 회귀의 정확한 instruction별 기여는 이 schedule만으로 분해하지 않는다.

따라서 다음 우선순위는 추가 BF16 partial을 만들지 않는 **C02 F32 packet scale**이다.
C01 재시도에는 직접 padded commit/scale 적재로 여섯 복사를 없애는 검증된 경로,
또는 추가 반올림 비용을 감수할 만큼 큰 이득이라는 새 근거가 필요하다.
단순 tile sweep이나 hi/lo 보정을 즉시 확대하지 않는다. C02–C07은 이번에 구현하지 않았다.

## 재현과 기록

현재 `moa2026-dev` container와 SDK 0.8.1, 저장된 E55 기본/추가 fixture를 사용한다.
[컴파일 환경·binary hash](environment.json)도 고정했다.
큰 binary·schedule·원시 로그는 Git 제외인 `moa-dev/experiments/artifacts/`와
`moa-dev/logs/`에 남고, 소스·hash·측정 결과·D01 로그 본문은 이 branch에 보관한다.
새 환경에서는 기존 fixture 생성 도구로 동일 fixture를 준비해야 하며 hash가 다르면 build가 중단된다.

저장소 root에서 실행한다. 같은 target을 사용하는 빌드는 순서대로 실행하고,
재빌드마다 새 `--label`을 지정한다. 빌드 도구 자체는 Arena에 제출하지 않는다.

```bash
python3 moa-dev/experiments/candidates/c01-scale-dpe-e55/verify.py --check
python3 moa-dev/experiment_index.py --check --check-live
python3 moa-dev/experiments/candidates/c01-scale-dpe-e55/build.py down --label c01-down-rebuild-1
python3 moa-dev/experiments/candidates/c01-scale-dpe-e55/build.py combined --label c01-combined-rebuild-1
python3 moa-dev/experiments/candidates/c01-scale-dpe-e55/build.py upgate-streamed --extra --label c01-extra-rebuild-1
python3 moa-dev/experiments/candidates/c01-scale-dpe-e55/build_probe.py fixed --label d01-rebuild-1
```

[build.py](build.py)는 고정 commit과 [tracked lock](../../../locks/gemma4-Cargo.lock)에서
새 crate를 만들고, 실제 container 입력을 해시 대조한다. Exact FFN을 먼저 컴파일한 뒤
해당 crate의 release cache만 지우고 host를 빌드해 exact image가 한 번 들어갔는지 검사한다.
QKV/O image와 fixture/runner도 기준 hash를 확인한다. 실패 시 packaging/제출로 넘어가지 않는다.
[build_probe.py](build_probe.py)의 fixed 재빌드는 실제로 실행해 job72783의 6개 compile
입력, device image, 전체 host package와 byte-identical임을 확인했다.
[재빌드 검증 기록](reproduction.json).

전체 후보 빌드 근거: [E65](build-upgate-streamed.json), [추가 host](build-upgate-extra.json),
[E66](build-down-repro.json), [E67](build-combined-repro.json).
진단 근거: [전체 job·업로드·출력](diagnostics.json), [최종 probe 빌드](build-probe-clear8.json).
새 build label은 별도 `build-<label>-repro.json`을 남겨 기존 측정 근거를 덮어쓰지 않는다.

운영 중 오류도 기록했다. Tail 수정 compile 실패 뒤 이전 binary를 job72748에 잘못 올렸고
즉시 취소했다. 업로드 hash가 원본 job72741과 같고 실행 시간은 0.002306초였다.
수정 실험이나 성능 결과에 포함하지 않는다. 이후 의존 단계는 성공한 subprocess만 통과하고
새 source/image/host/package를 대조하도록 만들었다. 실패 중간 source는 남아 있는 사본과
로그만 보존하며, 모든 중간 편집본을 복원할 수 있다고 주장하지 않는다.

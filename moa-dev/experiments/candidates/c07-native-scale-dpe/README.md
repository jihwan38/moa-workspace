# C07 — Native FP8 FFN의 결합 구조 실험

**채택: E104.** 기본19개 구성 비교 후, FP8 weight 중간 저장과 projection별 input reload를
포함한 구조가 독립3회에서 FFN **353154 → 335761 cycles (−4.93%)**를 기록했다.
기본45개/추가30개 비교와 projection207360개가 PASS다. QKV/O의 image는 동일하다.
E93의 단회 −10.13%는 미채택 관측이다. [구성표](../../../../baseline/e104.json) ·
[공식 제출](../../../../docs/experiments/e104-official-submit-20260922.json).

E79의 작은 파라미터 변경에 탐색을 제한하지 않고, 입력 표현·weight decode·DPE dot·block
scale 합산·행 배치·down 분할을 함께 바꾼다. 소스 부모는 E79
`2735b5899f41195dedeab1e076c82317e24efcb2`, SDK는 0.8.1이다. E55는 고정 참고군,
E79는 각 Arena batch 앞뒤의 동시 대조군이다. QKV/O는 같은 image를 유지한다.

## 실제 구현

공통 진입점 `decoder_feedforward → feedforward_sharded`와 공개 인자·가중치 FP4/E4M3
형식은 그대로다. [prepare.py](prepare.py)는 자체 E41의 quantizer/행 배치와 E79의
GeGLU·down·epilogue를 조합한다. 공개 FP8/scale-DPE 아이디어를 참고했지만 외부
참가자의 함수 본문을 가져오지 않았다. 생성된 디렉터리가 모두 실행된 후보는 아니다.
실제로 빌드한 소스는 `build-*-repro.json`과 [frozen-evidence.json](frozen-evidence.json)이 특정한다.

입력 흐름은 다음과 같다.

1. 기존 pre-RMS의 BF16 activation을 실제 값에서 계산한 공통 power-of-two scale과
   두 개 E4M3 또는 세 개 E5M2 term으로 표현한다. 고정 fixture의 값을 하드코딩하지 않는다.
2. FP4 packed weight를 FP8로 unpack해 곧바로 native DPE로 보낸다. 기존 FP8→BF16
   weight 변환을 없앤다. `materialized` 조합은 FP8 weight DM을 거쳐 unpack과 DPE를 분리한다.
3. 첫 DPE는 16원소 block dot을 계산한다. `pair16`은 두 lane 결과를 VE에서
   `hi + lo/16`로 합치고, `time`은 term 축을 DPE time reduction으로 합친다.
4. 두 번째 합산은 (a) BF16 block partial × BF16 scale DPE, (b) F32 block partial ×
   F32 scale VE, (c) BF16 high/residual 두 partial을 두 번째 DPE로 합치는 방식이다.
5. up/gate를 BF16 canonical row 순서로 복원한 뒤 기존 GeGLU를 호출한다. Full-native
   조합은 down도 새 FP8 경로로 바꾸며, up-only 조합은 E79 down을 그대로 호출한다.

`project_rows_with_scale_contraction`이라는 공통 helper 이름이 F32-VE 조합에도 남아 있다.
그 조합에서는 두 번째 DPE가 아니라 실제 함수의 VE reduction을 확인해야 한다.

| 축 | 구현한 구조와 계약 |
|---|---|
| FP8 표현 | E4 pair16: `lo=RN_E4(16*(x/s-hi))`; E4 time2: unamplified residual; E5 three: 세 번의 residual 표현 |
| 공통 scale | E4 `2^floor(log2(amax))/128`; E5 `/16384`. 0/작은 amax용 하한은 소스에 명시 |
| up row | row2 `L=7680*c+60*(s/2)+2*r+s%2`; contiguous `L=7680*c+30*s+r`, 30 live rows/slice |
| row 복원 | Contiguous F32 30행을 2원소/8B 단위 commit한 후 typed DMA gather, 마지막에 BF16 cast |
| term lifetime | up/gate가 같은 TRF를 공유하거나 retained DM에서 각 projection의 TRF를 별도로 reload |
| down 분할 | 4 column × width3840 × 30행; 2 column × width7680 × 15행 및 parity gather |
| down quantization | 480원소 tile로 residual VRF 크기를 제한; leader DMA와 ring broadcast 포함 |

[down.py](down.py)·[down.rs](down.rs)는 native down,
[extensions.py](extensions.py)는 time/column 분할,
[precision.py](precision.py)는 F32 partial·E5·보정 DPE·입력 reload·FP8 materialization을 구현한다.
E79를 비교군으로 유지하는 것은 이 분할/산술/배치를 유지해야 한다는 뜻이 아니다.

## 결과를 읽는 방법

[측정표](measurements.md)와 [performance-summary.json](performance-summary.json)의 값은 실제 업로드를 다시 내려받아
hash를 확인한 기본 benchmark 결과다. 원본은 `c07-screen-*-results.json`이다.
각 stage는 별도 프로세스지만 같은 batch의 stage들은 독립 반복이 아니다.
정확도에 실패한 stage의 cycles는 성능표에서 제외한다. 다른 stage 때문에 전체 job이
FAILED여도 성공 stage와 앞뒤 대조군이 완전하면 그 성공 stage는 단회 관측으로 기록한다.

E86–E104는 서로 다른 19개 기본 실행 구성에 부여했다. 별도 direct-DMA 구조도 projection 진단했으나 실패했다. 같은 구성의 compile 수정 횟수를
최적화 방법 수로 세지 않는다. 한 번 빨랐던 결과를 바로 baseline으로 채택하지 않는다.
E103/E104만 동일 batch를3회 실행했다. 최종 기본 측정·진단·채택 판정은 [decision.json](decision.json)에 기록한다.

## 정확도와 실패의 범위

[numerics.py](numerics.py) / [numerics.json](numerics.json)은 독립 CPU 수치 검사다.
공식 분포의 실제 pre-RMS 입력 9개에서 E4 pair16은 입력당 0–4개, E4 time2는
16–29개 원소가 원 BF16 값과 달라졌다. E5 three는 이 9개 입력을 모두 복원했다.
이는 **일반 BF16의 exact decomposition이나 NPU 정확도 증명은 아니다**.
큰 값과 `2^-32`를 같은 scale로 표현하는 예에서는 E5 three도 작은 값을 잃는다.

BF16 block partial을 두 번째 DPE에 넣으면 새로운 반올림이 생긴다. CPU projection
상쇄 예는 F32 partial 합 `1.875`를 BF16 partial 합 `3.75`로 바꾼다. 두 BF16
high/residual 표현은 그 예를 복구하지만, 추가 저장·계산이 실제 NPU 성능을 악화시켰다.
이 CPU 예는 전체 FFN의 공식 tolerance 위반을 직접 측정한 결과가 아니다.

[probe_stream.rs](probe_stream.rs)는 6개 full-shape projection 입력을 독립 F64 oracle과
비교한다. 전체 FP4 code, block/shard 경계, block 간 상쇄, zero, 작은 값이 섞인
두 상쇄 사례를 검사한다. 각 입력의 up/gate는 각각 15360개, down은 3840개로
총 207360개다. Oracle은 최종 BF16 값을 exact 비교하므로 공식 최종 FFN tolerance보다
강한 검사이며 입력 분포도 다르다. Probe image는 benchmark image와 별개이고,
probe의 cycles나 다른 후보의 PASS를 성능/정확도 근거로 승계하지 않는다.

초기 E5/F32 후보의 probe는 up/down을 통과했지만 gate 일부 행이 Inf/NaN 또는 다른
값을 출력했다. 원인 후보를 분리하려고 입력 reload, weight materialization, scale 복원
위치를 따로 바꿨다. 이 관찰만으로 compiler/LUT 버그라고 확정하지 않는다.
2-column down의 실행 timeout 역시 원인을 확정하지 않은 실패다.

확인한 compile 제약도 보존한다. 128/256B DPE input packet은 32/64B 제약에 걸렸고,
`DmTensorView.clone()`은 lowering되지 않아 owned DM과 `.view()`로 바꿨다. 일부
BF16 compensation tile은 VRF allocation에 실패해 3행 tile로 변경했다. 네 번째 zero
FP8 term을 넣는 배치 실험은 최종적으로 compiler internal error로 끝났다. 이들은
측정된 성능 한계나 모든 가능한 구현의 불가능성을 뜻하지 않는다.

## 재현과 수정

현재 generator는 마지막 설계 상태다. **과거 build의 재현은 현재 variant 파일이 아니라
해당 build의 동결 소스**를 사용해야 한다. [freeze.py](freeze.py)는 모든 성공/실패 시도의
MLP, diagnostic wrapper/host, compile log, Arena log를 내용 hash로 중복 제거해 보존한다.
나머지 full-build 입력은 명시한 E79 commit과 tracked Cargo lock에서 가져온다.
Binary·fixture·schedule은 ignored artifacts에 있고 fingerprint는 build 기록에 남긴다.

새 소스 실험은 저장소 루트에서 실행한다. 기존 label은 덮어쓰지 않는다.

```bash
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/prepare.py
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/build.py up-e5-three-f32-ve-late --label NEW-LABEL
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/screen.py prepare NEW-BATCH NEW-LABEL
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/screen.py submit NEW-BATCH
# Arena terminal status를 확인한 다음 수집한다.
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/screen.py collect NEW-BATCH
```

Full build target과 probe build target은 서로 독립적이다. 같은 target을 쓰는 full build끼리,
probe build끼리는 직렬 실행한다. `--extra`는 runs3..8의 별도 host/fixture를 사용하며
동일 device image 여부를 확인해야 한다. 결과를 추가한 뒤 다음 순서로 검증한다.

```bash
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/freeze.py
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/results.py
python3 moa-dev/experiments/candidates/c07-native-scale-dpe/verify.py
python3 moa-dev/experiment_index.py --check --check-live
```

`verify.py`는 46개 full-build 입력, 실패/성공 로그, exact compile image와 실제 host의
embedded image, unchanged QKV/O, 실제 Arena archive, 로그 재파싱과 집계를 확인한다.
`results.py`의 ID/group에 등록되지 않은 새 batch는 원본에만 보존되므로 새 구성을 넣을
때 ID와 catalog도 함께 갱신한다. 공식 점수는 별도 official submission 결과만 사용한다.

## 이번에 닫은 결론과 다음 탐색

E102(weight materialization)와 E104(materialization + reload)는6-case projection을 모두
통과했다. Reload만 한 E103, scale 위치를 바꾼 E101, direct-DMA 진단은 실패했다.
따라서 이번 구현에서 FP8 weight 중간 저장을 유지한 E104를 채택한다. 이것은 모든
streamed unpack의 불가능성이나 compiler 내부 원인을 증명한 결과가 아니다.
E102의 단회 수치는 E104와 비슷하지만 E104만3회 반복과 추가 입력까지 확인했다.

E104의 점수 기여 예상은 QKV/O가 같을 때 약1.7%다. 이 작은 개선을 더 세분화하지
않고 C07을 마무리한다. 다음 C08에서는 Q/K/V 세 projection을 묶는 weight/scale 배치와
공유 dot pipeline을 우선 시험한다. 기본 정확도와 대조1회로 개선 폭을 먼저 판단한다.
[정적 함수별 자원 시간](static-summary.json)은 후속 설계 참고이며, 겹치는 시간을 더해
실측 latency나 절감 가능량으로 해석하지 않는다.

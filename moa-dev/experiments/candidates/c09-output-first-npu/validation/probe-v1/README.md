# C09 채택 검증: RMS 이전 projection 경계

Arena job **105952**에서 8개 입력의 30,720개 BF16 출력 모두 독립 F64 oracle과 일치했다.
비유한 출력과 불일치는 0개다. 부호가 다른 0만 동등하게 취급하며, 나머지는 BF16 bit 단위 비교다.
24개 입력 checksum, 실제 업로드된 tar의 host/fixture/runner SHA256도 확인했다.
결과는 [`../probe-results.json`](../probe-results.json)과
[`docs/experiments/e127-c09-probe-correctness.json`](../../../../../../docs/experiments/e127-c09-probe-correctness.json)에 있다.

`probe.rs::output_before_rms`는 원래 O public entry와 같은 BF16 `[Ns,Gs,Ds]` 입력을
`[Qs]` view로 바꾼 뒤 **동결 `project_output`을 직접 호출**한다. 이 함수 안의 두 60행 stage,
16-way F32 합산, canonical gather, 원래 F32 channel-scale 곱셈, 첫 BF16 cast가 전부 실행된다.
반환 BF16 `[H]`를 HBM에 저장하고 RMS·residual을 실행하지 않으므로 최종 정규화가 오차를 가리지 않는다.

컴파일 입력 `src/device/sliding/projection.rs`의 13,250B 전체가 원 C09 snapshot과 byte-identical이다.
SHA256: `5692369dd5b8f6ddbeecdfaa9289a9683fae18798b920a71775362f306560fe6`.
원 crate 파일은 그대로 사용하고 `lib.rs`에 `pub mod probe` 한 줄과 별도 wrapper/host만 추가했다.
이 진단 image는 60,928B, SHA256
`c411f3ae30eb726362ff37338a2129a52b90042ece91199e5ad775d80a530a73`다.
원 whole-O image와 동일하다는 주장이 아니며 새 HBM 출력 경계를 가진 **정확도 전용 실행**이다.
실제 whole image의 추가 표준 입력 검증과 성능 재현은 root의 별도 E127 기록을 사용한다.

| Case | 검증 대상 |
|---|---|
| 0 | 입력 ±0, 출력 초기값 NaN sentinel이 모두 쓰였는지 확인 |
| 1 | K=255 단독 비영 입력, 전체 3,840개 행의 서로 다른 weight/scale |
| 2 | K=256 단독 비영 입력, 256열 group 경계 |
| 3 | K=4095 단독 비영 입력, 마지막 열 경계 |
| 4 | K=0/255/256/511/512/2047/2048/4095의 혼합 부호 입력 |
| 5 | 모든 32개 120행 group의 local 0/58/59/60/61/118/119만 비영 weight; 나머지 출력은 0 |
| 6 | 매 256열의 x=[+1,-1,254×2^-22], weight=[1,1,254×240]; 작은 항과 상쇄 보존 |
| 7 | dot=1+2^-8 또는 그 두 배, scale=1.5/1.25/0.5/0/2^-10/3/0.75/2; scale 이전 BF16 반올림 검출 |

모든 case에서 **모든 행**을 비교하며 local stage 59/60/119와 전역 cluster 1919/1920의
표본을 별도 로그에도 남긴다. Case5의 선택 행은 모두 합쳐 224개이며 stage 사이의 쓰기 offset,
mutable destination 잔류값, 전역 gather 경계를 겨냥한다. Case1–4는 전체 행이 비영이다.
Case7은 dot을 먼저 BF16으로 cast하는 잘못된 순서를 사용하면 1,920개 expected bit가 달라진다.

`generate.py`는 SDK 함수나 장치 quantizer를 호출하지 않는다. 저장된 BF16 x/scale을 F64로
해석하고 E4M3 bit를 수학적으로 decode한 뒤 F64 dot·scale 결과를 BF16 round-to-nearest-even으로
반올림한다. Python NumPy와 Rust host는 weight 생성 공식을 독립 구현하고 입력별 checksum을
대조한다. 모든 입력은 유한 dyadic 값이며 이 사례들의 dot/scale 결과가 F32에 정확히 표현됨도
확인한다. 이 마지막 assert만으로 임의의 F32 중간 합이 정확함을 증명하지는 않는다.
본 검증은 한정된 입력에 대한 실제 NPU 일치를 입증하며, 모든 가능한 입력이나 다른 reduction의
정밀도를 보증하지 않는다. 기존 E47 probe의 checksum/HostTensor/launch 패턴을 재사용하되,
그 probe의 최종 RMS oracle 및 느슨한 tolerance는 사용하지 않았다.

빌드 target은 `/tmp/moa-c09-validation-probe/target`로 분리했다. 첫 exact compile과 host build는
성공했고, package 코드가 `kernels` 바로 아래만 검색해 nested SDK image를 못 찾은 assertion은
[`../probe-build-attempt1.json`](../probe-build-attempt1.json)에 보존했다.
재컴파일 없이 recursive image 검색으로 package만 수리했다. 원 build 시점 driver는
[`build-driver-at-compile.py`](build-driver-at-compile.py)에 hash-identical snapshot으로 남겼다.
최종 [`../probe-build.json`](../probe-build.json)의 `build_driver_snapshot`, `packaging_repair`가
이 관계를 기록한다. runtime source·oracle·image의 수정은 없었다.

재검증(컴파일·Arena 제출 없음):

```bash
python3 moa-dev/experiments/candidates/c09-output-first-npu/validation_probe.py --check
```

이 명령은 production 원본 전체/실제 compile source, 원 source 파일 보존, wrapper/host/oracle,
exact image의 host 포함, 로그의 8개 완전한 결과와 24개 checksum, 다운로드 tar의 실제 업로드
digest, 두 결과 JSON의 일치를 다시 검사한다. 라이브 소스나 baseline 채택 상태는 수정하지 않는다.

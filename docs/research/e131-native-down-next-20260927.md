# E131 이후 native down 후보 감사 — 2026-09-27

현재 근거는 **E131 미채택, E128 유지, 다음 down 계산 표현 변경은 미실행 가설**이다.
아래 후보 소스의 축약 경로는 `moa-dev/experiments/candidates/` 기준이며, SDK 경로는
Docker `moa2026-dev` 설치본을 가리킨다. 이번 감사에서는 새 소스·compile·Arena 실행을 하지 않았다.

E131은 제거한 저장량만으로 속도를 예측할 수 없음을 다시 보였다. 동시 E128 앞뒤 대조를 포함한
전체 FFN 결과는 다음과 같다. 각 job의45PASS는 세 process의 기본 검사 합이다.

| E131 | E128 before → candidate → E128 after | 앞뒤 중앙값 대비 | 정확도 범위 |
|---|---|---:|---|
| two-row v1,109430 |328536 →378328 →328473|+15.166763%|whole45PASS; 별도109373 oracle8cases/245760출력PASS|
| four-row v2,109408 |328813 →348871 →328144|+6.208169%|whole45PASS; 별도109411 probe launch timeout, oracle완료아님|

[첫 v1 결과](../../moa-dev/experiments/candidates/ffn-joint-direct-fused/e131-joint-two-row-screen-job-109430-results.json),
[v2 결과](../../moa-dev/experiments/candidates/ffn-joint-direct-fused/e131-joint-four-row-screen-job-109408-results.json),
[v1 oracle](../../moa-dev/experiments/candidates/ffn-joint-direct-fused/validation/probe-results-v1.json),
[v2 diagnostic 실패](../../moa-dev/experiments/candidates/ffn-joint-direct-fused/validation/probe-results-v2.json).
두 후보 모두 채택하지 않고 추가 행수 조정도 중단한다. V2 timeout의 물리 원인은 미확정이며
기본 wholeFFN 정확도 실패 또는 native family 전체의 불가능성으로 확대하지 않는다.

## 1. FP4→BF16 직접 lookup: 현재 공개 API로는 불가

설치 SDK0.8.1의 `TableLookupCast<bf16> for f4e2m1` 구현이 없다. 아래 설치본은 `/tmp/moa-o-sdk-std` mirror와 SHA256까지 동일함을 Docker read-only로 재확인했다.

- `/home/moa/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/src/cast.rs:44–85`: 고정 테이블은 paired FP4→E4M3/E5M2, non-paired FP8→BF16 네 조합뿐. SHA `a5bc81ff9e06b89f4d5e25567592782af7e99ef90a2789a00c45a49012430d28`.
- `src/engine/fetch_adapter.rs:177–215`: Main Fetch Unit lookup이고 runtime table 인자가 없으며 scalar trait bound를 요구한다. Sub에는 lookup register가 없다. SHA `7e06090bccd65935e6e6793761831a9e3b1d08e6ffcfe3eff89f48344319b27e`.
- `src/engine/mod.rs:261–272`: lookup 이전 fetch position만 lookup 가능. `PositionFetchTableLookup`에 두 번째 lookup 허용 impl이 없어서 FP4→E4→BF16 두 lookup을 같은 fetch에 연결할 수도 없다. `:308–320`은 contract가 Collect에서만 가능하고 Cast 뒤에는 contract가 없음을 명시. SHA `79627b91df5d73f26843b8a1975851cca4c27260b4581a29e66434f4314791bd`.
- `src/cast.rs:352–377`: floating DPE는 BF16/BF16, E4/E4, E5/E5만 지원. FP4→E5 stream에 현재 BF16 activation TRF를 붙이는 우회도 불가.

이는 공개 primitive/type 상태 및 설치 SDK 설명의 한계다. 비공개 compiler 내부나 하드웨어의 모든 가능한 custom ISA 경로가 불가능하다는 증명은 아니다. 새 compile은 필요 없고 실행하지 않았다.

현재 경로는 `ffn-joint-direct-fused/joint-four-row-v2/mlp.rs:315–368`: compact FP4 HBM→FP4 DM→E4 DM→BF16 lookup/BF16 DPE→F32 block partial DM→F32 scale/120-block sum/4-column sum이며 BF16 weight DM은 이미 없다. 단순 direct BF16 lookup 가설을 구현 가능 후보로 올리면 안 된다.

E50 `ffn-down-fused-decode-e37/mlp.rs:398–424`는 실제 FP4lookup→E4→fetch_cast F32→BF16 **DM** 후 별도 dot이다. 구형 E37 조합의 기본 NPU PASS/단회 약0.94% 이득이지만, 현재 경로에 그대로 이식하면 56.25MiB E4 intermediate 대신112.5MiB BF16 intermediate를 재도입한다. E73 `c05-streamed-decode/direct/mlp.rs:379–420`는 E4 DM을 유지하고 반복 BF16lookup/dot/scale/reduce로 partial DM을 제거한 과거 후보이며 전체FFN +10.24%였음(`README.md` 측정표). 둘 다 요청한 no-weight-DM directBF16 경로가 아니다.

## 2. E5M2 native down: 정확 조합은 보존 소스에서 미시도

C07의 E5 이름을 native down 실행으로 읽으면 안 된다.

- `c07-native-scale-dpe/prepare.py:122–129`: `full-time4-f32-ve`는 E4 full source의 F32 partial 버전이고, `e5_three_terms`는 `up-*` source에만 적용된다.
- `up-e5-three-f32-ve/mlp.rs:343–401`: down은 BF16 activation TRF + E4 materialization + BF16 lookup DPE다. E101–E104 계열도 E5 up/gate와 BF16 down을 조합했다.
- `quantize_down`/`project_native_down_rows`가 존재하는 보존 C07 full-* 소스11개를 확인했다. 모두 E4M3이며 해당 down 구간에 E5M2는 없었다. `full-time4-f32-ve/mlp.rs:330–395`는 E4 두 항과3840원소 amax/480원소 residual tile, `:413–441`는 H-half cluster ownership·global4columns×3840, `:444–484`는 directFP4→E4 native→F32partial→scale/reduce다.
- C07 전체 native down이 모두 실패한 것도 아니다. `measurements.md`와 `docs/experiments/e93-c07-full-time4-late.json`, `e95-c07-full-f32.json`: E93/E95의 기본 wholeFFN stage는 PASS이며 E79 대비−10.13%/−7.12% 단회였다. 반복/강한 down oracle 증거는 아니다. E90/E98/E99의 E4 배치는 실행/정확도 실패로 분리한다.
- `projection-probe-full-v1-results.json`은74373 timeout/완료false이며 raw log는 case0 up/gate oracle 두 줄 후 timeout이다. down 입력 oracle/결과까지 도달하지 않았다. 이 진단을 E5 down 실패나 down 수치 오류로 인용하면 안 된다.
- C13 `c13-int8-ffn/README.md`: GeGLU 뒤 INT8 재양자화+weight 변환+integer partial 경로다. v7 기본PASS,+18.58%이고 E5 native down의 실행증거가 아니다.

따라서 현재 M03의 **L-half cluster ownership(각7680 L),4 local columns×1920,60Hrows/slice**를 유지하면서 실제 GeGLU BF16을 three-term E5로 양자화하고 FP4→E5 direct native와F32 block scale/sum을 연결하는 조합은 미시도로 분류하는 것이 맞다. Native-down 및 residual 표현 자체는 C07 가족의 재조합이며 새 독립 원리라고 부를 수 없다. 위 판단은 확인한 보존 후보 소스/결과 범위이며 모든 과거 branch의 미보존 파일 부재를 증명하는 것은 아니다.

API상 FP4→E5/BF16→F32→E5cast/E5native는 있지만 실제1920/7680 shape, scale 위치·수명·broadcast와 정확도는 미검증이다. 바뀐 GeGLU dynamic range에서 three-term 표현은 일반 BF16 exact 보장이 없고, preRMS-H quantizer의 기존8case PASS를 승계할 수 없다. per-cluster7680 amax와 per-column1920 amax는 다른 산술/통신 설계이므로 먼저 하나를 명시해야 한다. 실제 quantization 및 residual DM/TRF 준비·방송·3항 native 비용 모두 wholeFFN에 포함해야 한다.

Direct FP4→E5를 성공적으로 연결하면 down E4 DM write/read 각58,982,400B(56.25MiB)를 논리상 제거한다. E5를 먼저 DM에 materialize하면 형식만 바뀌고 그 저장은 제거되지 않는다. 기존 F32 block partial은14,745,600B(14.0625MiB)/방향이며 partial 경계를 유지하는 제안에서는 여전히 남는다. 이 수치는 payload accounting이고 bus 실측이나 cycle 절감 예측이 아니다. 위 E131 whole-NPU 회귀를 반영하면 같은 joint-copy 경로의 타일 조정보다 down 계산 표현 변경을 다음 연구 질문으로 삼을 근거가 있다. 이 감사 자체는 실행 검증이나 채택을 뜻하지 않는다.


## 3. E93/E95의 positive screen과 미완료 검증을 정확히 구분

| 후보 | Arena | E79 before → candidate → after FFN median | before 대비 | 앞뒤평균 대비 | 후보 기본 검증 |
|---|---:|---|---:|---:|---|
| E93 full-time4-late |74336|352705 →316977 →351897|−10.129712%|−10.026653%|15PASS, FFN3입력PASS|
| E95 full-f32 |74335|352176 →327093 →351992|−7.122291%|−7.098022%|15PASS, FFN3입력PASS|

출처: `docs/experiments/e93-c07-full-time4-late.json`, `e95-c07-full-f32.json`, 대응 `c07-late-e79-control.json`, `c07-precision-e79-control.json`. E95 job전체는다른stage실패로FAILED지만 E95 candidate_1은 valid=true다. 실패job상태만보고E95를실패로분류하면안된다. 이숫자는전체FFN조합이며 down단독효과가아니다.

E93 `full-time4-late-scale/mlp.rs:449–496`는 E4두항native 이후block마다BF16로반올림하고secondBF16DPE에서scale/sum한다. E95 `full-time4-f32-ve/mlp.rs:444–484`는E4두항을유지하되 F32blockpartial/VEscale/sum이므로그BF16partial반올림위험은없다. 양쪽up/gate와down모두바뀐전체조합이다.

`c07-native-scale-dpe/decision.json:44–50`의E93미채택이유는 single screen, E4/BF16-partial precision limitations, no validated final combination이다. 이후 E93/E95자체에대한강한oracle/extra6/독립repeat실패는찾지못했다. 보존 `build-projection-probe-*.json` 전체는E90(`full-time-contig-v2`)또는up-E5후보에만바인딩돼있다. E93/E95가검증을실제로실패했다는주장을뒷받침하지않는다.

`numerics.py:33–40`의E4입력복원오차는preRMS H벡터검사이며GeGLU후down입력을검사하지않았다. `:54–58`의BF16partial상쇄반례도CPU projection intermediate 예이고 wholeFFN오차측정이아님을명시한다. E5probe의gate행오류와BF16down PASS 또한 E4native-down원인규명이아니다. 따라서E93/E95는 **단회positive, 채택검증미완료**로남겨야하고 native-down가족전체실패로닫으면안된다. 현재확인한README/decision에는E93positive를보존하지만, 서술을요약할때이구분이소실될위험이있다.

## 4. 다음 큰 변경의 구체 조합안 — 아직 미구현

Source origin/control은 E128 `cc82b9c`로고정하고검증된up/gate만그대로둔다. E93의directnative down 및E95의F32blockpartial원리를현재M03 L-half소유권에맞춰조합한다. 완성경로는 다음과같다.

`E128 up/gate →기존GeGLU BF16[M03 local120] →기존 cluster별7680 BF16 compact →cluster당amax/2의지수scale 하나 →480값단위 E5 hi/lo/third 실제양자화 →3항DM조립 →각cluster의4개1920L조각을M03 ring32로전달/방송 →E5 TRF[3,1920] 및cluster-scale VRF →compactFP4 HBM[M03 H60×L1920] →Main FP4lookupE5→3항native16blockdot→F32 xscale복원→compactF32blockpartial[60,120] →기존E128 blockscale/120block F32합/4column F32합 →두cluster F32합→기존BF16/global scale/RMS/residual/epilogue`.

상기안은per-cluster7680 amax를선택한구체계획이다. 전체L15360amax나column1920amax와혼동하지않는다. Quantizer를broadcast전에실행해64Hrowgroup의같은활성화를중복양자화하지않는것이소유권변경의효과다. 480chunk는16개/cluster이며현재E128preRMSquantizer의실제세항식과VRF크기제약을재사용하되새축/cluster보존은별도검증한다. C07은H-half클러스터+global4columns3840이므로현재L-half/4localcolumns1920경로와동일하지않다. 동시joint up/gate와달리 down은한projection이므로E131packed jointassembly를추가할이유가없다.

제거: 전체down E4weight DM write/read각56.25MiB, FP8→BF16 LUT준비/stream. 유지: 원FP4HBM/FP4DM원본읽기, weight scaleHBM, F32partial14.0625MiB/방향, 후단모든계산. 추가: postGeGLU amax·scale·두잔차단계·termassembly·scale배포·E5세항TRF·native세항시간축. 모든slice에복제된activation payload는BF16 512×1920×2=1.875MiB에서E5 512×3×1920=2.8125MiB로0.9375MiB늘며 실제bus/복사횟수는graph별산정해야한다. Packedlookup은원decode의1회가directnative의1회로이동하고BF16lookup1회가사라지는구성이지 E131처럼lookup을8회로늘리는구성이아니다.

첫gate는 새fullgraph의compactHBM descriptor/term-scale정합/직접lookup F32partial좌표, 실제quantizer포함down oracle(GeGLU유래범위·cluster7680경계·1920column·16block·상쇄/zero/tiny), 이후 contemporaryE128 wholeFFN controls다. 현재 BF16 down 수치 정확도를 그대로 승계할 수 없고, E5 근사가 오류를 키우면 원인별 기록 후 판단한다. 이 완성 경로의 개선 폭이나 10위 진입은 보장하지 않는다. 추가partial fusion은E73의반복lookup/setup비용을다시만드는별도조합이므로이계획에몰래포함하지않는다. 미구현·미평가이며 여기서는 실행하지 않았다.

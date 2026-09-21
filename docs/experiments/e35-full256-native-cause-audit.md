# E35 full256 native FFN 성공의 원인 구분

2026-09-21. **E35 job70485는 native FP8 retained packet4 → two-lane Sequential → pair unzip/zip 조합을 실제로 실행하고 공식 FFN3개 입력을 모두 통과했다.** 따라서 이 조합 또는 packet4 자체가 RNGD에서 항상 실행 불가능하다는 주장은 반박된다. 반면 이전240-valid-slice 후보 timeout의 근본 원인은 아직 확정할 수 없다.

확인 자료는 [E35 공식 결과](e35-full256-native.json), [원시 로그](../../moa-dev/logs/e35-full256-native-job-70485.log), frozen E25/E35 source와 [full256 후보](../../moa-dev/experiments/candidates/ffn-native-fp8-full256/mlp.rs)다. 새 build/NPU 실행이나 live source/Git 변경은 하지 않았다.

## 실행된 소스와 검증 범위

E35 frozen `src/device/shared/mlp.rs`는 후보와 byte-identical이며 SHA256은 `e835f288722aac9b6528ff0fb3504d53bb87e4da10ba2fe625113db6c55926dc`다. exact compile의 FFN kernel SHA256 `a4082c4987cd934b48d9bb2706caf0fcce2223aabd6f0e9691f06b130f15851f`가 E35 전체 test ELF에 동일 bytes로1회(offset977,204) 포함됨을 독립 확인했다. test ELF SHA `bdc8fda7ef2a775742093538d8496e1e80ca0ebf9aa6dd830e0861f6d1758397`와 raw log SHA `a4c9653e761af9995dfa7208b48d8e7b1f29c80aa924ecc0597c1ce65e01ff42`도 내려받은 package evidence와 같다.

전체15 comparisons PASS다. FFN cycles813,658 /810,077 /811,219, median811,219이며 각 run의 max absolute error0.00586 /0.00977 /0.01562, 모든 원소가 원래 허용오차를 만족했다. Q97,301/O45,799는 같은 job의 다른 커널 결과다. FFN의 추가 BF16 block 반올림과 runtime FP8 입력 근사도 **이3개 입력에서는** 허용오차를 통과했지만 모든 입력의 동등성이나 추가 seed PASS를 증명하지는 않는다. E31 safe FFN의 약389k보다 느리므로 E35는 실행 가능성 진단 결과이지 현재 성능 최선값이 아니다.

## 실제 비교 기준은 E25

full256은 E25 `ffn-native-fp8-scale-contraction`에서 출발했다. E22와 E32도 이전 timeout 사례지만 두 번째 scale 처리와 dtype 경계가 달라 단일 변경 대조군으로 묶으면 안 된다.

E25↔E35 actual snapshot에서 `ops::decoder_feedforward` 본문, generic shared RMS `normalize`, shared residual 파일은 byte-identical이다. MLP의 `feedforward` 시작부터 파일 끝까지도 동일해서 runtime quantizer/broadcast, GeGLU, global scales, safe down이 그대로다. 두 native helper의 row axis를 `FfnRow→L%32`로 정규화하면 **helper 전체가 byte-identical**하다. 즉 packet64의4개16원소 block 유지, Sequential hi/lo, lo/16 합산, runtime scale, BF16 partial cast, 두 번째 BF16 local-scale DPE dot까지 같은 계산 구조다.

동시에 전체 source에서 Q/O projection 및 그 RMS helper는 이후 개선으로 달라졌다. 이들은 해당 FFN 호출 그래프의 새로운 산술 변경은 아니며 E35 FFN kernel이 full256 exact kernel과 일치하는 것을 확인했다. 그러나 서로 다른 job 사이의 실행 상태/스케줄까지 모두 같았던 완전한 단일 요인 실험이라고 주장하지 않는다.

## 240 view와256 live 경로의 차이

| 항목 | E25 timeout 경로 | E35 PASS 경로 |
| --- | --- | --- |
| 각 cluster의 row slice |`L/32%240#256`:240 valid+16 padding|`FfnSlice=256`:모두 valid|
| 실제 row 소유 |`L=7680*c+32*slice+row`, row0–31|`L=7680*c+256*row+slice`, row0–29|
| DPE row32 중 실제 row |32개 전부|30개 실제+2개 실제0 dummy|
| Weight decode/저장 |FP4→FP8 후 바로 소비|FP4→FP8 row30 DM→zeroed row32 DM로 typed copy|
| Local scales |row32를240valid slice로 DMA|row30 DMA→zeroed row32 DM로 typed copy|
| Output 복원 |padded slice를 제거하는 typed compact|first30 row만256-way Switch gather,4BF16/8B commit|
| Native DPE/pair helper |packet4+Sequential pair|row axis 이름 외 동일|

두 매핑을 독립 전수 검사해 각각15360개 real row를 정확히 한 번씩 포함함을 확인했다. E35의 HBM view `[Dummy2,FfnRow=30,FfnSlice,H]`의 linear index는 원래 `L*H+h`와 같다. Local scale도 마지막H/16 축만 다르고 같은 row 식이다. real weight/scale 값을 누락하거나 다른 행 scale과 결합한 변경은 없다.

E25에서도 activation hi/lo와 scalar는 먼저256개 slice 전체에 복제된다. 이후 `240#256` view로 제한한다. Weight/scale은240개 valid slice만 typed DMA로 쓰므로 padding16개 값은 보장하지 않는다. **그것만으로 E25 소스가 잘못됐다는 뜻은 아니다.** top padding은 논리 유효 데이터가 아니며, 정상 구현은 유효 결과를 계산할 때 그 값을 데이터처럼 소비하면 안 된다. 같은240-slice BF16 경로의 기존 PASS도 padding 표현 자체를 금지된 것으로 볼 수 없게 한다.

E35는 padded16 slice를 실제0으로 바꾼 단순 수정이 아니다. row 소유를 재배치해256개 모두30개 real row를 맡게 하고, **각 slice의 local row30/31**을 실제0으로 채웠다. Up/gate 각각 FP8 weight32-row와 scale32-row tensor 전체에 `memset(0)`을 수행한 뒤 valid30-row만 복사한다. SDK0.8.1 `DmTensorViewMut::memset`은 live cell만 채우므로 이처럼32개를 실제 live row로 선언한 것이 중요하다. `#` 또는 `#{0}` 표시만으로 초기화했다고 가정하지 않았다. 출력에서는 dummy2row를 제외해 원래15360개 값만 GeGLU로 넘긴다.

## 입증된 것과 남은 원인 가설

입증된 사실은 **이 full256 배치에서 기존 native packet4/pair/scale-DPE 계산이3개의 공식 입력을 완료하고 통과했다**는 것이다. E32의 pair-free timeout은 pair 제거만으로 충분하지 않았음을 보여 주고, E35는 pair를 유지해도 성공하는 경로가 있음을 보여 준다. 따라서 pair 자체 또는 retained packet4 자체를 timeout의 충분조건이라고 부를 수 없다.

여전히 단독으로 입증되지 않은 것은 “240 padding이 유일한 원인”, “padding 쓰레기 값을 읽은 것이 원인”, “zero tail 초기화가 해결책”, “256개 live slice만 쓰면 항상 해결”, “TRF stale state/firmware bug” 같은 주장이다. E35에서는 live-slice 수와 함께 real-row 배치, 실제 memset, 큰 DM copy, output gather, DM/TRF/VRF lifetime 및 compiler scheduling이 함께 바뀌었다. 이전 timeout 로그에는 실패 instruction 위치나 tensor dump가 없으므로 그중 하나로 좁힐 자료가 부족하다.

특히 실제 zero tail은 새32-row destination의 추가2row를 올바르게 정의하는 데 필요한 구성이다. 하지만 이 초기화가 이전 timeout을 없앤 원인이라는 인과 증거와는 다르다. output gather 변경 역시 후반 경로이지만 이전 hang 위치를 모르므로 원인 후보에서 무조건 제외할 수 없다.

정적324,369 cycles의 큰 비용에는 weight destination copy61,875×2, scale copy4,275×2, weight memset15,626×2, scale memset1,226×2, output gather8,457×2가 있다. 노드 합은 전체 makespan이 아니다. 현재 root/rngd_docs의 direct-LUT commit 후속 실험은256-valid/real-zero-tail/row 식/DPE helper를 유지하며 큰 weight copy를 제거하는 별도 후보로 해석해야 한다. 성공하면 그 개선된 경로의 실행 가능성을 더 확인하지만 여전히 원래240 timeout의 원인 하나를 확정하지는 않는다.

masked TRF lane8 우회 설계는 E35 성공을 반영해 구현 전에 중단했고 [미실행 설계 메모](../../moa-dev/experiments/candidates/ffn-native-fp8-mask8-design/README.md)로만 보존했다. 이후 문서에서 “native packet4는 동작하지 않는다”는 일반화 대신 “이전240-valid 구현은 timeout, E35 full256 구현은 공식3개 입력 PASS; 원인 미확정”으로 서술한다.

# 2026-10-01 E154 scalar 공급 수리 채택 · 공식 제출 준비

현재 runtime은 **E154 / baseline-e154 / `3f384063f5231e00ad286f365209806e5043fc83`**이다. E151 소스에 frozenE153의
RMS/quantizer 소유권과8복제scale 수리를 합성했다. 성능control은별도E151이다.
Compile input46중3파일변경/43불변, QKV/O이미지는E151과byte-identical이다.

| Arena | E151 before / after | E154 FFN | Paired 변화 |
|---|---:|---:|---:|
|141633|295176 / 295622|288044|-2.489853%|
|141648|294688 / 295855|288616|-2.254027%|
|141661|294564 / 294892|287581|-2.424948%|

독립3allocation의paired중앙은 **-2.424948%**다. 표준distinct9, 원rawupgate8의
245760 BF16 exact(±0동등), 원fullFFN회귀3840exact가PASS했다. Raw는실제새quantizer를
호출하지만preRMS는제외한다. 원RMS emitted산술/F32경계/3840좌표와전체FFN을별도로검증했다.
임의입력이나모델전체동치증명은아니다. 원reference/host/허용오차는유지했다.

GlobalMax와원pow2scale을8개유효quant소유자에유지하여중간scalarTDMA를제거한다.
Actualratio256/8valid복제이며최종all512scalar복제는anonymous8source의초기화된0..7을
읽는typedDMA로남는다. Latepacked23040B, seed184320B, ring5898240B와원native/E151down은
유지한다. 과거M06형태의현재조합수리이며새architecturefamily나옛timeout원인입증이아니다.
E153의37k겹친scale준비구간전체를절감했다고주장하지않는다.

[최종gate](../../moa-dev/experiments/candidates/ffn-sharded-scale-replicated/adoption/independent-final-review.json) ·
[채택판정](../../moa-dev/experiments/candidates/ffn-sharded-scale-replicated/adoption-decision.json).
공식제출은다음단계이며아직E154점수/순위없다. 마지막E151은6.8348×,
publicbestE150은6.838931882×/57위이며요청한다섯계단상승은미달성이다.
아래기록은각작성시점상태다.

# 2026-10-01 E154 replicated scale 수리 feasibility 준비

E153의 canonical scalar를8개quant shard로보내는TDMA를제거할수있는
Max8복제→원pow2scale→localSubVRF 조합을병렬준비한다. E151source/control,
frozenE1533파일에수리하나추가. M06유사시도와실패를새기법으로세지않는다.
E153내부계측과actualgraph를확인한뒤NPU선정한다. 현재runtimeE151.
[후보](../../moa-dev/experiments/candidates/ffn-sharded-scale-replicated/README.md).

# 2026-10-01 E153 첫 실측 회귀 · 한 번의 내부 비교 선정

현재 채택 runtime은 E151이다. E153은 원RMS F32 산술을 유지하면서 local BF16H480
출력을 직접 양자화하고 packed만 늦게 두cluster에 복제한다. 실제3파일/43불변,
원RMS emitted 산술/3840좌표/globalMax/23040B/copy범위/VRF 검사는 PASS했다.
그러나 첫141557 whole45PASS에서298332 vsE151294978/294929, **+1.145435% 회귀**다.
채택·추가입력NPU·parameter sweep은 진행하지 않는다. Raw8/fullFFN은 빌드만/NPU0.
한 번의 frozen E151/E153 이름만 바꾼 내부 계측으로 globalMax/latepacked와 뒤쪽 일정을
구분하여 구체적인 수리 여부를 고른다. 겹친 구간을 절감량으로 더하지 않는다.
[판정](../../moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/decision.json). 아래 기록은 각 작성 시점의 상태다.

# 2026-10-01 E153 pre-RMS shard 직접 양자화 실험 시작

Source/control E151, actualstart `2a49667318718b2f0e2fda00b7feb8a3f3727515`.
원RMS산술/F32경계보존, 최종localBF16shard→cluster0양자화→latepackedCL2복제.
기존초기canonicalgather/복제/reshard대신새globalMax/lateDMA비용까지whole검증한다.
E152 gate분할은포함하지않고M06v4timeout해결도주장하지않는다.
[후보](../../moa-dev/experiments/candidates/ffn-prerms-sharded-consumer/README.md).

# 2026-10-01 E151 공식 완료 · public best E150 유지

현재 runtime은 **E151 / `baseline-e151` / `8adbbae1776aa8080a96348a9d9a64022a0cf6d0`**이다.
Source와 성능 control은 각각 명시한 E150이며, compile input46개 중 `mlp.rs`만 바뀌었다.
QKV/O 이미지는 E150과 byte-identical이다. 공식 `2ac6e446`는 15비교 PASS, QKV92,705 / O42,977 / FFN295,105, **6.8348×**다.
10월1일19:48:33 KST public best는 E150 `ac53ad0a`, **6.838931882× / 57/80위**를 유지한다.
E151의 FFN 개선에도 동일 O 이미지의 측정 변동으로 전체 점수는 직전보다 약0.05971% 낮다.
[공식 기록](e151-official-submit-20261001.json) · [최신 제출/최고 기록 구분](../research/e151-official-rank-result-20261001.json).
큰 개선이나 요청한 다섯 계단 상승을 달성했다고 주장하지 않는다.

| Arena | E150 before / after | E151 FFN | Paired 변화 |
|---|---:|---:|---:|
|141382|297622 / 308406|295031|−2.634532%|
|141396|297217 / 297109|295185|−0.665628%|
|141411|298487 / 297199|295709|−0.716485%|

독립 세 allocation의 paired 중앙은 **−0.716484859%**이며 aggregate 중앙값 비율
−0.892416474%와 구분한다. 첫 after308406이 높았던 기록을 제외하지 않았고, 세 candidate는
각 before/after 양쪽보다 빠르다. 동일 host·fixture·runner·바깥 패키지와 실제 upload를 검증했다.
표준 distinct9, raw-down9의34560 exact BF16(±0동등), 실제 fullFFN 회귀3840 exact가
원본 reference/허용오차로 통과했다. 이는 선택된 입력의 유한 검증이다.
[성능](../../moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/adoption/performance-summary.md) ·
[최종 gate](../../moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/adoption/independent-final-review.json) ·
[채택 결정](../../moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/adoption-decision.json).

Scale HBM은 row low2 소유권의 연속480값으로 읽고 Sub cast→InterTranspose4→원래12행
5760B VRF에 공급한다. Scale HBM/DM payload3,686,400B는 줄지 않으며 F32 Switch routing
14,745,600B와 설정 비용을 포함한다. 원weighted E5·두 down native·F32 partial·five12
scale/reduce 산술·후단은 유지한다. E150의10,244cycle 준비 간격을 절감량으로 세거나,
새 E151에서 같은 경계가 사라졌다고 내부 계측 없이 단정하지 않는다.

채택은 재현된 작은 이득과 정확도 검증에 근거한다. 추가 row/ring 튜닝은 중단한다.
병렬 E152 gateweight15+15는 E150에서 별도로 출발했으며 E151 변경을 상속하지 않는다.
E152는 이후45비교PASS에도whole +10.05543% 회귀로종료했다. 합성채택하지않는다. 아래항목은각작성시점의상태다.

# 2026-10-01 E151 down scale 공급 실험 시작

`opt/ffn-down-scale-ingress-switch`, actualstart `3d4a600e068429114b7b5f49fe9a8482492c5d31`.
Source/control E150를 명시하고, row low2 scale 적재→SubSwitch4→기존VRF를 구현한다.
원weighted계산/native/partial은 고정하며 actualgraph 뒤 whole 비교한다.
[후보](../../moa-dev/experiments/candidates/ffn-down-scale-ingress-switch/README.md).

# 2026-10-01 E152 gate 분할 회귀 종료

E152141424는45비교PASS지만FFN327706 vs E150297644/297885, bracket대비
+10.05543% 회귀로미채택한다. 컴파일일정의입력전송interleave가전체이득이되지는않았다.
Raw-upgate/fullFFN은원본reference와빌드검증만완료, boundaryNPU0. 추가stage/row튜닝없다.
E151은별도검증된개선이며이후보에포함되지않는다.
[종료판정](../../moa-dev/experiments/candidates/ffn-gate-weight-stages/decision.json).

# 2026-10-01 E152 gate weight 공급 병렬 실험 시작

`opt/ffn-gate-weight-stages`, actualstart `ffce3481842b8f71a635a2dc2ba773518856ec05`.
Source/control E150이며 미채택 E151을 포함하지 않는다. 초기activation gather의 실제대기를
목표로 gate만15+15 compact/native→원partial직접쓰기를검증한다. 별도추가DMA/lookup비용포함.
두후보가모두양성이면합성은새로검증한다.
[후보](../../moa-dev/experiments/candidates/ffn-gate-weight-stages/README.md).

# 2026-10-01 E150 공식 완료 · E151 scale 공급 경로 선택

공식 `ac53ad0a`는 15비교 PASS, QKV 92,917 / O 42,550 / FFN 296,854 cycles,
score **6.838931882**. 19:11:39 KST public snapshot은 **57/80위**, 종전과 순위 동일하다.
점수 +1.223413%이며 QKV/O는 동일 이미지이므로 그 측정변동을 FFN 코드 이득으로 세지 않는다.
초기 다섯 계단 목표인 현재 52위 score7.103142846까지 추가 score3.863337%가 필요하다.
[공식 기록](e150-official-submit-20261001.json), [순위 계산](../research/e150-official-rank-result-20261001.json).

현재 E150 자체 readiness141326은 51비교 PASS. 같은 cluster 시계의
last-down-native→first-reduce 중앙10,244cycles이며 scaleDMA가 native 종료보다8,965cycles
늦게 끝난다. 각 scale VRF→reduce는 다섯 쌍 모두6관측에서+1이다. 이 시간창은 겹침을
포함하므로 절감량이 아니다. E151은 scale HBM 소유권을 row low2/연속480값으로 바꾸고
Sub cast→InterTranspose4→원래12행 VRF로 공급하는 조합을 선택한다. 원native/partial/
weighted정밀도는 유지하고 추가14.0625MiB F32 routing과 setup까지 whole FFN으로 평가한다.

# 2026-10-01 E150 FFN down 공급 분할 채택·공식 제출 pending

현재 채택 runtime은 **E150 / `baseline-e150` / `ad1ecb3d3868c7e0ca2e4d1428bbd66bf1111821`**이다.
공식 제출은 아직 pending이며, 새 공식 점수·순위는 확인되지 않았다. Source recipe와 성능 control은
별도로 선택한 E147이며 미채택 E148/E149 O 변경을 포함하지 않는다. 46개 compile input 중
`src/device/shared/mlp.rs` 하나만 바뀌고 QKV/O 이미지는 E147과 byte-identical이다.

| Arena | E147 앞뒤 중앙 | E150 FFN | Paired 변화 |
|---|---:|---:|---:|
|141242|303942|297276|−2.193182%|
|141259|303909.5|298524|−1.772074%|
|141276|304448.5|297731|−2.206449%|

독립 세 allocation의 paired 변화 중앙은 **−2.1931816%**다. Aggregate 중앙값 비율
−2.043482%와 구분한다. 표준 세 입력 반복의 전체135비교와 extra6의30비교가 PASS하여
standard distinct9다. 별도 raw-down9의34560 BF16 exact(±0동등), 실제 production FFN
회귀 사례의3840 exact 출력과 원본13입력/reference/업로드 검증도 PASS했다.
이는 선택된 입력 검증이며 arbitrary BF16·모델 전체 정확도 보장이 아니다.
[최종 gate](../../moa-dev/experiments/candidates/ffn-down-weight-stages/adoption/independent-final-review.json) ·
[채택 결정](../../moa-dev/experiments/candidates/ffn-down-weight-stages/adoption-decision.json).

Down FP4의 local60행을30+30 compact ingress/native로 나누고 기존 full60 F32 partial의
per-slice `[0,14400)`, `[14400,28800)`에 직접 쓴다. 원래 five12-row scale/reduce가
rows24..35의 half 경계를 정상 소비하며 weighted3항·quantizer·TRF·후단은 그대로다.
새 assembly는 없지만 weight DMA와 native/lookup 시작이 각각 하나 늘었다. Weight
29,491,200B 및 F32 partial write/read 각각14,745,600B는 줄지 않았다. 추가 비용을 포함한
whole 개선을 채택하며 source 순서가 DMA 우선권을 보장하거나 trace 겹침이 절감량이라는
주장은 하지 않는다. [후보와 비용](../../moa-dev/experiments/candidates/ffn-down-weight-stages/README.md).

Stage/row 수 튜닝은 중단한다. 같은 고정 이미지의 `readiness/` 내부 계측이 병렬 진행 중이며
완료 전 원인이나 후속 효과를 확정하지 않는다. 공식 제출 및 추가 구조 선택은 root가 관리한다.
아래 기록은 각 시점의 상태다.

# 2026-10-01 E150 FFN down 공급 분할 시작

`opt/ffn-down-weight-stages`, actualstart `024a0fd1d711d14c5c6355deb76f377828022d47`, source/control E147.
현재실측upgather/downweight겹침에대해원60rowFP4를compact30+30으로적재하고
두weightednative가원래full60F32partial의분리된절반에직접쓴다. five12소비자와
quant/TRF/3항계수/나머지커널은고정. 추가lookup/native/DMA비용포함whole검증예정.
[후보](../../moa-dev/experiments/candidates/ffn-down-weight-stages/README.md).

# 2026-10-01 E149 종료·E150 FFN down 공급 분할 선택

E149 O K1024ingress/MainSwitch4는45비교PASS지만57414/bracket42650(+34.616647%)
회귀로미채택. 추가ring/row튜닝·profile·boundaryNPU없다. E147 runtime유지.
다음E150은현재FFN 실측141134의upgather/downweight겹침에근거해 down의원60행을
30+30compactHBM/native로분리한다. 기존F32partialfull60root에직접쓰고5×12scale/reduce와
weighted3항/quant/TRF는유지한다. 새copy없음, 대신lookup/native/DMA하나가추가된다.
분할만으로gather우선권이생긴다고가정하지않고actualschedule/whole검증한다.
[종료판정](../../moa-dev/experiments/candidates/o-k1024-ingress-switch/decision.json).

# 2026-10-01 E149 시작: O의 마지막 weight 준비 경로 변경

`opt/o-k1024-ingress-switch`, actualstart `9b2e7795280201a017ebcd41ad5e773de5656a86`. Source/control E147,
기존60+60 weight stage와 K256 native 계산은 유지하며, 원weight를 K1024 구간/
row low2 소유권으로 적재한 뒤 MainInterTranspose4로 원소비자에 공급한다.
실제 descriptor와 native stream 검증 후 whole bracket. E148 추가 scalar/VRF 변경은 없다.
[후보](../../moa-dev/experiments/candidates/o-k1024-ingress-switch/README.md).

# 2026-10-01 E148 종료: 입력을 앞당겨도 마지막 weight가 늦어져 회귀

E147 runtime 유지. E148 직접 VRF 후보는141077/141134 whole O에서 각각
+3.288206/+4.473631%로 느려져 미채택. 첫45비교·두 번째whole/진단63비교PASS,
standard distinct3이며 raw9/fullcase7/extra6는 빌드만·NPU미실행이다.
실제입력 TRF 준비23,860.5→12,004cycles에도 마지막 native는 양쪽모두 weight1끝+2,
6 run/cluster 전부같다. 추가bitmap/달라진일정비용을입력절감과함께평가한다.
다음선택은 E147 source/control에 O60행 stage의 K1024 ingress→MainSwitch4→원K256
native60을결합하는 E149다. E148 source를다음후보에상속하지않는다.
[종료판정](../../moa-dev/experiments/candidates/o-scalar-residual-vrf/decision.json),
[현재O/FFN실측](../../moa-dev/experiments/candidates/o-scalar-residual-vrf/readiness/profile-analysis.json).

# 2026-10-01 E148 첫 whole 회귀·고정 이미지 비용 진단

`e148-direct-v1` exact/static24908과 descriptor/register gate는 통과했다. Arena141077은
원래45비교 PASS, O43945 vs E14742473/42619(앞뒤중앙42546)로 **+3.288206% 회귀**다.
채택하지 않는다. Raw9/fullcase7 패키지는 원본참조·실제source/image까지 빌드했으나 NPU는
실행하지 않았다. Extra6도 빌드만 했다. 추가 타일/항 수 반복은 없다.

원인이 direct Main→VRF의 Main/Sub/VE 동시 점유인지, scalar 방송·weight 준비/후단인지
구분하는 E147/E148 이름-only O paired profile 한 allocation을 진행한다. 같은 control
host의 unchanged E142 FFN info/trace도 포함해 현재 FFN 후속 구조 판단에 사용한다.
중첩 marker 시간창은 절감량으로 합산하지 않는다. 실제 비용 변화가 확인될 때만 수리한다.

# 2026-10-01 E148 O scalar·누적 복원 직접 VRF 실험 시작

채택 runtime은 E147이다. `opt/o-scalar-residual-vrf`, 실제 시작 `f9451626d508197901deee9bafbb03aa8edf2d34`에서
source E147fb45202와 별도 control baseline/e147.json을 선택했다. O만 scale3Main→1Main,
scalar/계수 Sub ring16→VRF, 두 F32 누적 복원 Main→VRF를 결합한다. 원래4항/2native/
60+60weight/후단은 유지한다. QKV/FFN 불변을 확인하고 exact gate→whole screen으로 진행한다.
논리 DM write/read각1MiB 감소에 Main/Sub/VE 점유와 bitmap/VRF 수명 비용이 추가된다.
아직 성능/정확도/채택 결과는 없다. 양성은 크기와 무관하게 동결·반복·경계 검증한다.

# 2026-10-01 E147 채택·공식 제출 완료

현재 runtime은 **E147 / `baseline-e147` / `fb45202`**다. 공식 제출 `aabfb8c6`은
23파일 접수·SDK0.8.1 계약 검사 및 공식15비교를 통과했다.
QKV94391/O42212/FFN305502cycles, 팀 최고 **6.756274761×(+0.120995%)**, **55/80위 그대로**다.
QKV는 직전 공식95918대비1.591985% 감소했다. O/FFN이미지는 같으며 해당 측정변동을
E147의 코드 효과로 귀속하지 않는다. 최신30위8.068829902×까지 총점+19.4272%가 필요하다.
[리더보드](leaderboard-e147-20261001.json) · [격차 계산](../research/e147-rank-gap-20261001.json).
[공식 기록](e147-official-submit-20261001.json). E144 full-H weight/native/
head/cache를 유지하고 scale 생산3 Main→1 Main, scalar·계수 Sub ring4→VRF를 채택했다.
작은 복제 TDMA24KiB가 사라지는 대신 shared bitmap16KiB와 Sub 설정 비용이 포함된다.
E144 O/FFN 이미지는 byte-identical이다.

| Arena | E144 앞뒤 중앙 | E147 QKV | 변화 |
|---|---:|---:|---:|
|137097|95594|94483|−1.162207%|
|137106|96800|94610|−2.262397%|
|137115|95999|95234|−0.796883%|

Paired 감소 중앙 **1.162207%**이며 각 후보는 앞뒤control 양쪽보다 빨랐다. 표준9입력,
raw9의73728 exact BF16(±0동등), full2의16384 원허용오차 출력/cache8380416 및
실제source46·image·업로드 연결 검증이 통과했다. CPU finiteBF16 scale65280개 동일성은
이 scalar에 한정되며 arbitrary NPU/model 보장이 아니다.
[최종 gate](../../moa-dev/experiments/candidates/qkv-scalar-sub-vrf/adoption/independent-final-review.json) ·
[채택 결정](../../moa-dev/experiments/candidates/qkv-scalar-sub-vrf/adoption-decision.json).

E146은 첫 양성 뒤 독립 반복이 부호를 바꿔 paired중앙+0.406876%로 미채택했다.
양성5%미만만으로 버린 것이 아니다. E147은 같은 작은 이득이 반복되어 채택한다.
추가파라미터튜닝은 중단한다. 공식 제출은 완료됐지만30위권에는 여전히 못 미친다. 완료 후보 branch와 tag, 원시 실패/실측 기록은 보존한다. 아래는 당시 상태다.

# 2026-10-01 E147 scalar 생산·Sub→VRF 직접 배포 검증 진행

E146 완료 기록을 main `2bdbe9137b999176dabf510cfe517267cbea7298`에 병합한 뒤
`opt/qkv-scalar-sub-vrf` branch를 시작했다. Source recipe는 E144 `3586c41`,
성능 control은 별도 `baseline/e144.json`이며 미채택 E146 weight staging은 포함하지 않는다.
`qkv-scalar-sub-vrf/scalar-v1`은 처음 exact compile을 통과했다(static48436; NPU 성능 아님).

Finite BF16 exponent map/Mul→H960 Max→floor→K4 Max로 기존 세 Main scale 계산을
하나로 합친다. Scale32B와 packed coefficient16B를 Sub ring4 방송→32B VRF로
직접 보내며 기존 소비자 DM용 TDMA를 제거한다. 원래 head용 scale DMA와 full weight
/native/Term/K4/firstBF16/head/cache/O/FFN은 유지한다. Main 두 시작 및 max 중간값을
제거하지만 입력491520개의 곱셈 및 Sub Switch/bitmap 설정을 포함한 비용을 비교한다.
CPU finite BF16 65280개 scale 식 일치와 실제 graph/register gate, 원본 전체 QKV
대조실험을 구분한다. 아직 E147 NPU 결과/채택/공식 제출은 없다. 아래 기록은 당시 상태다.

# 2026-10-01 E146 완료·미채택, scalar 공급 변경으로 진행

E146 v4는 세 independent Arena137020/137025/137030에서 각각45비교를 통과했지만,
E144 앞뒤 control 중앙 대비 QKV 변화가 −1.494967%, +0.710594%, +0.406876%로
중앙 **+0.406876% 회귀**다. 첫 양성은 재현되지 않았고 채택하지 않는다. 원래 raw9
73,728 BF16 exact(±0 동등), full2의16,384 출력 tolerance 및8,380,416 비대상 cache는
통과했다. Extra6는 빌드만 했고 성능 재현 실패 후 NPU 실행하지 않았다.
[완료 기록](../../moa-dev/experiments/candidates/qkv-qk-weight-stages/README.md).

현재 runtime/source는 E144 그대로다. E146 source와 v1–v3 compiler 실패를 보존하고
row/stage sweep은 중단한다. 다음 E147은 E144를 source/control로 따로 선택하여
scale 생성의 세 Main 경계를 한 개로 합치고 scale/계수를 Sub broadcast에서 VRF로
직접 전달하는 조합을 검증한다. 작은 scalar의 TDMA 소비자 저장/재적재를 없애되
Sub Switch/setup/VRF 비용과 각 입력 exponent 처리 비용이 추가된다. D-E145의
scale 완료~VRF 구간9,914cycles는 대기 포함 측정이며 절감 약속이 아니다.
Q weight 시작이 scale DMA 완료 뒤라는 시간상 관측도 단독 인과 증명이 아니다.
완료 E146 기록 통합 뒤 별도 branch에서 구현한다. 아래 진행 기록은 당시 상태다.

# 2026-10-01 E146 Q/K weight 단계 공급 실험 진행

현재 채택 runtime은 E144 그대로다. D-E145 job136924의81비교 및 원본 이미지/계측 연결 검증을
완료하고 진단 branch `0396ed8`을 main `6ec6269`에 병합했다. 현재 E144의 모든6개
run/cluster에서 V native는 activation TRF 완료1cycle 뒤, Q/K는 각 weight 완료2cycles 뒤
시작했다. Scalar DMA는 V weight 완료보다 먼저 시작하므로 단순한 순차 DMA 대기로 단정하지 않는다.
[원본 결과](../../moa-dev/experiments/candidates/e144-qkv-readiness/README.md).

E146 branch는 `opt/qkv-qk-weight-stages`, 실제 시작 `6ec626988820cb3fa7ba18da5dc417d42cb9b8b4`다.
[선택 근거](../research/e146-qkv-weight-stages-selection-20261001.md)에 따라 Q32→16+16,
K16→8+8의 full-H weight 공급과 native 소비를 묶으며 V/정밀도/원래 canonical gather는 유지한다.
Useful weight30MiB는 같고 DMA/native setup은3→5다. 독립 control은 `baseline/e144.json`.
v1–v3은 allocator에서 실패했으며 실측이 없다. vISA에 남는 상수조건 Branch/tensor Phi를
제거한 v4가 exact compile을 통과했다. v4는 v2의 초기화와 v3의 추가 복사를 포함하지 않는다.
정적49218cycles는 성능 판정이 아니며 실제 descriptor/register gate와 whole screen을 진행한다.

채택은 실측/검증 대기, 임의 tile tuning은 중단, 연구는 늦은 Q/K weight 완료 의존성 제거다.
2026-10-01 00:05 KST 공식공개조회는55/80·6.748109877×로 직전 제출과 같았다.
[원 응답](leaderboard-e145-start-20261001.json). 아래 기록의 순위/상태는 당시 관측이다.

# 2026-10-01 D-E145 현재 E144 병목 계측 후 구현 진행

사용자가 병목 확인 후 구현·실험·개선 채택까지 연속 진행하도록 지시했다. 현재 source46은
E144와 일치하며 기존 완료 branch가 main에 포함됨을 확인했다. D-E145는
`diag/e144-qkv-readiness`, 실제 시작 `08d9bb1597456c7fe7413229358d6d5077cdd7ba`다.
E142/E144 실제 QKV image의 이름만 바꾼 paired profile을 한 allocation으로 실행한다.
E142의 검증된 diagnostic host는 재검증하여 재사용하고 E144 host만 새로 조립한다.
O/FFN image는 이번 두 source에서 동일하다. 이전 E138-v2 계측을 현재 병목으로 대입하지 않는다.

관측: preRMS·BF16 seed/ring·global max/scale·네 항/coeff·TRF, 각 Q/K/V weight/native,
F32 gather/scale·head/cache 완료. 가장 늦은 실제 의존성에 따라 지원되는 producer/consumer
수정을 선택하고, 전체 QKV 실측이 좋아지면 동결·독립 반복·정밀 검증·채택·공식 제출한다.
정적 schedule이나 중첩 구간을 합산하여 예상 speedup으로 삼지 않는다. 계측 자체의 채택은
해당 없음; 임의 row/term sweep은 중단 유지; 연구 결정은 이번 관측으로 내린다.

# 2026-09-30 E144 QKV 채택·공식 제출 완료

현재 runtime은 **E144 / `baseline-e144` / `3586c41`**다. E142 FFN과 O 이미지는
그대로이며, QKV는 full-H E4 행 적재와 Main Switch/native K4 분배를 함께 채택했다.
Weight 총량30MiB는 같고 별도 weight transpose DM은 없다. 입력 네 항 준비, 추가 Switch,
F32 gather·scale·head/cache까지 포함한 whole QKV 성능이다.

| Arena | E142 앞뒤 중앙 cycles | E144 QKV cycles | 감소율 |
|---|---:|---:|---:|
| 136536 | 103,584 | 95,909 | 7.4094% |
| 136554 | 104,288.5 | 95,910 | 8.0340% |
| 136573 | 104,227.5 | 95,899 | 7.9907% |

Paired 감소율 중앙 **7.9907%**. 표준 고유9입력, raw9의73,728 BF16 exact(±0 동등),
원래 full2의16,384출력 허용오차와8,380,416개 비대상 cache 보존 검사가 통과했다.
원래 host/참조를 바꾸지 않았고 실제 source46·이미지·업로드 tar·로그를 연결했다.
Full2는 모든 값의 exact 일치를 뜻하지 않으며 유한 사례 검증이다.

[최종 검증](../../moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption/independent-final-review.json) ·
[채택 결정](../../moa-dev/experiments/candidates/qkv-fullh-weight-switch/adoption-decision.json) ·
[실행 기록](../../moa-dev/experiments/candidates/qkv-fullh-weight-switch/README.md).

브랜치 `diag/e142-e138v2-qkv-readiness`와 `opt/qkv-fullh-weight-switch`의 완료 기록을
main에 병합했다. 두 branch tip과 v1의 source/compile 실패를 보존했다. v2는 v1의
Switch 출력행 순서 문제를 slice low2 row ownership으로 수정한 동일 후보다.
추가 파라미터 튜닝은 종료한다. 다음 구조 선택은 현재 E144의 실제 입력·weight·gather
준비 시점을 먼저 구분하며, E138-v2의 이전 내부 cycle을 현재 병목으로 단정하지 않는다.
공식 제출 `2a66e430`은15비교 PASS, QKV95,918 / O42,945 / FFN296,581cycles로 완료했다.
팀 최고점 **6.748109877×**(+3.40094%)이며2026-09-30 23:01 KST의 순위는 **55/80**,
이번 작업 시작 snapshot은58/79였다. O/FFN이미지가 같으므로 공식값 변동을QKV효과로 세지 않는다.
현재30위8.068829902×까지 총점+19.5717%가 필요하다(고정leaderboard, 동률 기준).
[공식 결과](e144-official-submit-20260930.json) · [리더보드](leaderboard-e144-20260930.json) ·
[격차 계산](../research/e144-rank-gap-20260930.json) · [다음 판별 실험](../research/after-e144-priority-20260930.md).
아래 이전 순위와 상태는 당시 관측이다.

# 2026-09-30 D-E143 완료·E144 weight ingress/stream 분배 조합 선택

D-E143136455는81비교와 실제 이미지/업로드/76·92개 operation 연결 검증을 통과했다.
Whole QKV104551→108642→104192로 E138 v2는 E142 앞뒤 중앙 대비4.0916% 느렸다.
모든6개 run/cluster 관측에서 V native는 TRF 완료1cycle 뒤, Q/K는 각자의 weight DMA
완료2cycles 뒤 시작했다. [원본 계측](../../moa-dev/experiments/candidates/e142-e138v2-qkv-readiness/README.md).

[선택한 E144](../research/e144-qkv-fullh-switch-selection-20260930.md)는 E138 v2의 네 항
입력/TRF·native·Term/K4·head 계산을 유지하고 weight를 canonical fullH row 소유권으로
적재한 뒤 Main stream Switch로 H960 소비 위치에 보낸다. 별도 전체 weight transpose
저장은 추가하지 않으며 실제 Switch/reader/lifetime 비용과256B ingress descriptor를
검증한다. Source recipe는 E142 나머지44파일+E138 v2 QKV caller/projection의 조합이며
현재 E142 FFN을 보존한다. 성능 control은 별도 E142다. 아직 E144 실측은 없다.
첫 fullh-v1은 Switch OutTime 순서 계약 오류로 MIR에서 실패했다. 단순 row 이름 변경이
아닌 ingress 소유권을 Row%4/Row/4로 맞추는 fullh-v2를 선택했다. 행마다 H3840은
유지하지만 slice 내부 행은 stride4이며, 초기 contiguous-row 가정과 다르다.
원 v1 소스·실패 로그는 보존하고 v2의 실제 descriptor와 좌표를 새로 검증한다.

최신9월30일22:25 KST [공개 조회](leaderboard-qkv-start-20260930.json)에서 BEEF는
58/79위,6.526159×로 점수는 같고 다른 팀 개선에 따라 순위가 바뀌었다.
30위8.068830×까지 총점23.64% 상승이 필요하다. 이 후보의 예상 개선율로 주장하지 않는다.
아래는 계측 착수 당시 기록이다.

# 2026-09-30 D-E143 QKV 입력·후반 weight readiness 비교 진행

Runtime은 E142이며 live source 검사와 기존 완료 branch의 main 포함 여부를 확인했다.
새 branch는 `diag/e142-e138v2-qkv-readiness`, 실제 시작은
`6c3a3f51188a53efa68f5ada5f88b6d2bd0123ce`다. 후보 ID는
`e142-e138v2-qkv-readiness`다. [선택 근거](../research/after-e142-priority-20260927.md)의
남은 질문을 실제 동결 이미지로 판별한다. 새 커널 구현이나 타일 튜닝은 아직 선택하지 않았다.

E142 standard host와 frozen E138 native-v2 host의 QKV를 비교하며, profile-name bytes만
바꾼 각 diagnostic image의 code/constants/주소/IO 동일성을 검증한다. 두 원본 host의
FFN은 서로 다른 채택본이므로 FFN 성능은 비교하지 않는다. 각각의 세 image와 source
binding을 확인하고 QKV whole 앞뒤 대조와 info/trace를 한 allocation에 넣는다.

측정 대상은 preRMS→seed/ring→amax/terms/coefficients→TRF, Q/K/V 각각의 weight DMA와
native, reduction/head/RoPE/cache 완료다. 첫 activation과 후반 weight 중 무엇이 완료를
늦추는지로 다음 구현을 고른다. 겹친 시간창이나 static schedule을 절감량으로 합산하지 않는다.
채택은 새 커널이 없어 해당 없음, 추가 routing/row/term 튜닝은 중단 유지, 연구는 이 한정
진단으로 다음 구조를 선택한다. 아래의 공식 순위는9월27일 기록이며 새 조회와 구분한다.

# 2026-09-27 E142 채택·공식 제출 완료

공식 제출 **e1f58b9e**가15비교를 통과했고 **6.526159×,46/70위**로 완료했다.
QKV104474/O42694/FFN302799cycles이며 E140 대비 총점+0.841%, 순위는 그대로다.
QKV/O image는 동일하므로 해당 실행 변동은 FFN 변경 효과로 해석하지 않는다.
30위7.896678×까지는 총점21.00% 또는 세 커널 각각17.36% 감소가 필요한 산술 격차다.
[공식 결과](e142-official-submit-20260927.json) · [리더보드](leaderboard-e142-20260927.json) ·
[현재 격차](../research/e142-rank-gap-20260927.json).

다음 우선순위는 [QKV 후반 weight 공급과 입력 준비의 구분](../research/after-e142-priority-20260927.md)이다.
E138 v2의 남은4.10% 회귀를 양자화 비용으로 단정하지 않는다. 동결 이미지의 제한된
readiness 비교가 다음 구현을 고르는 단계이며, 아직 새 계측/후보 커널은 실행하지 않았다.
M06와 네 항 bit 구성 검토도 보존했다. 이들 검토는 새 구현의 성공 근거가 아니다.

현재 runtime은 **E142 / baseline-e142 / b3fe0ab**다. E141의 six-row helper만
실제 E103 full30 direct native→F32 block partial→6행 blockscale/H reduction으로
교체했다. 안전한 입력 복사·M03 gather/GeGLU·E134 weighted down 및 QKV/O는 유지한다.
[구성표](../../baseline/e142.json) · [채택 결정](../../moa-dev/experiments/candidates/ffn-direct-full30-partials/adoption-decision.json).

| Arena | E140 앞뒤 중앙 | E142 FFN | 감소율 |
|---|---:|---:|---:|
|111706|313534|302889|3.3952%|
|111719|314887.5|301615|4.2150%|
|111721|314166.5|302796|3.6193%|

3개 독립 allocation의 paired 감소율 중앙값 **3.6193%**다. 같은 소스·package로 반복했고
추가6입력111737까지 표준9입력, 원8사례 actual up/gate oracle111704의245760 exact,
actual whole FFN111736의3840 exact 출력을 검증했다. 46개 compile input에서
`shared/mlp.rs`만 변경됐고 QKV/O image는 E140과 같다.
[최종 검증](../../moa-dev/experiments/candidates/ffn-direct-full30-partials/validation/final-verification.json).

전체 FP8 weight112.5MiB write/read를 각각 없앴지만 F32 partial28.125MiB write/read가
각각 추가된다. E141 대비 lookup/native10→2와 별도scale/reduce10개 비용까지 포함한
완전한 경로의 이득이다. 이번 조합을 채택하고 추가 row/stage/term 튜닝은 종료한다.
이 개선만으로30위 도달을 주장하지 않으며, 공식 제출 결과는 위에 별도로 기록했다.
아래는 앞선 실험 당시 기록이다.

# 2026-09-27 D-E141 계측 완료·E142 full30 부분합 조합 선택

현재 채택/공식 runtime은 E140이다. 계측111627은57비교와 실제137/151개 operation
연결을 검증했다. E141의 첫 gate native는 E140보다15434cycles 빨랐지만 마지막 up
native 종료는10888cycles 늦었고 GeGLU는10810cycles 늦었다. 별도 whole은
313898→325748→313736으로3.8019% 회귀했다. 첫 screen3.1516%와 함께 기록한다.
[계측·선택 근거](../../moa-dev/experiments/candidates/e140-e141-lookup-profile/README.md).

Up/gate native 전체 구간은64649→91078cycles로 늘었다. Candidate up첫 native는
lookup table 준비1cycle 뒤 시작했고 이후 table 준비는 native와 겹쳤다. Table window
합집합66420.5cycles를 절감 가능량으로 읽지 않는다. DirectMain의 lookup/native/scale을
내부 service time으로 분리한 결과도 아니다.

E142는 이미 실제 compile되었던 E103 full30→F32 block partial→6행 scale/reduce
helper를 수정된 E141 주변 경로와 조합한다. lookup/native10→2, 전체Main10→12이며
28.125MiB F32 partial write/read를 각각 추가한다. Weight FP8 materialization은 계속 없다.
프로파일이 반복 준비·소비 비용을 바꿀 근거를 제공했으며 실제 whole 개선은 미검증이다.
기존 E103 오답은 보존하고 새 partial bounds/같은245760-output oracle/whole을 통과해야 한다.
소스 원점은 E141+E103 helper, 비교군은 별도 E140이며 새 named branch에서 진행한다.
아래는 완료된 앞선 기록이다.

# 2026-09-27 E141 직접 FP4 경로 실행 복구·전체 FFN 회귀

Runtime은 E140이다. E135 소스의 up-term 복사 세 곳만 고친 E141은
전체 job111571의45비교와 원래 up/gate oracle111574의245760 exact BF16 출력을
모두 통과했다. 두 기존 timeout 이미지와 같은 bitmap 주소 배치에서 전송 범위만
줄였고 정상 실행이 복구됐다. 과거 timeout은 계속 원본 그대로 보존한다.

전체 FFN은 E140314403 → E141324186 → E140314159cycles,
앞뒤 중앙314281 대비 **3.151638% 회귀**다. 이 후보는 미채택, 파라미터 튜닝은 중단한다.
[검증·원본 결과](../../moa-dev/experiments/candidates/ffn-independent-direct-copy-repair/README.md).
추가 full-FFN stress는 준비만 했으며 성능 회귀 후 실행하지 않았다.

큰 FP8 저장112.5MiB의 논리 쓰기/읽기를 각각 없앴지만 lookup 준비는2→10회로 늘었다.
다음은 고정 E140/E141 실제 이미지의 내부 계측 한 번이다. 반복 lookup·TRF 준비가
완료를 지연하는지 확인하여, 기존 full30-row native→F32 block partial→별도 scale/reduce
조합을 다시 검사할 근거가 있는지 판단한다. E103은 이 조합의 오답 기록이 있으나
그 소스에도 같은 잘못된 복사가 있고 최종 descriptor가 없어 당시 원인은 미해결이다.
이번 복구만으로 E103의 산술 문제까지 해결됐다고 주장하지 않는다.

E140 공식 제출47a9240e는15비교를 통과했고 **6.47173296×,46/69위**로 완료했다.
QKV102409/O43055/FFN314108cycles이다. 기존 QKV image의 실행 변동이 점수 상승에
크게 기여했으므로 복사 수리의 성능 개선으로 해석하지 않는다. 30위는7.89667762×다.
[공식 원본](e140-official-submit-20260927.json) · [리더보드](leaderboard-e140-20260927.json).
아래는 직전 기록이다.

# 2026-09-27 E140 복사 범위 수정 채택·E135 직접 weight 경로 복구 선택

현재 runtime은 **E140 / baseline-e140 / c56b737**이다. E137에서 up/gate 입력의
세 term 복사만 수정했다. 실제1440B를 전송하던 descriptor를 각각480B Sub stream으로
바꿔 source overread와 packed destination 초과 쓰기를 없앴다. QKV/O image와
FFN 산술은 동일하다. [채택 결정](../../moa-dev/experiments/candidates/down-native-consumer-local/adoption-decision.json).

3개 독립 Arena의 paired 변화율은 -0.2101/-0.1847/-0.0568%로 사실상 중립이다.
표준9입력과 실제 전체 FFN 정밀 사례3840출력이 통과했다. 큰 속도 개선으로 주장하지 않는다.
별도 local-down v4는 오답, 같은 복사를 수리한 v5는45 PASS지만3.0981% 회귀하여 미채택이다.
Raw-down9사례 PASS는 수정된 upstream을 지나지 않았으므로 전체 검증으로 세지 않았다.

E135의 whole/oracle timeout 이미지 둘 다 이 초과 쓰기가 통신 제어용 bitmap을 덮는
것을 확인했다. [독립 감사](../../moa-dev/experiments/candidates/ffn-independent-direct-fused/e140-bitmap-audit.md).
다음 E141은 frozen E135의 독립 FP4→6행 fused up/gate에 이 복사 세 곳만 수리한다.
112.5MiB 논리 FP8 write/read 제거의 전체 이득을 lookup 준비 증가까지 포함해 다시 측정한다.
이 수정이 timeout의 유일한 원인인지는 아직 미확인이다.

공식 제출은 아직 E137 **46/69위·6.4427554×**다. 최신 조회에서30위는7.8965272×로
상승했다. 세 커널을 같은 비율로 줄이는 산술 기준18.41%가 더 필요하며,
E140 수정만으로 순위가 달라졌다고 주장하지 않는다. 아래는 당시 작업 기록이다.

# 2026-09-27 E138 입력 전달 수정까지 미채택·E140 FFN 실측 준비

Runtime/control은 E137이다. E138 whole111168은45비교 PASS였지만
QKV104068→117376→103860으로 **12.9006% 회귀**했다.
원래 wholeQKV 경계2사례와 raw Q/K/V9사례73728개 출력도 통과했다.
Names-only계측111237에서consumer입력DMA26638cycles와TRF직후native시작을
확인하여 seed+BF16ring32 전달수정v2를실행했다. Whole111314는45PASS지만
QKV103603→108007→103895로 **4.1041% 회귀**였다.
개선된실험판과기존채택본대비이득을구분한다. 둘다미채택이며선택한routing수정은중단한다.
V2추가경계도구는준비만했고compile/NPU미실행이다.
[전체결과](../../moa-dev/experiments/candidates/qkv-native-consumer-local/README.md).

E139 job111124는현재 FFN137개 operation·36비교 검증을 완료했다.
GeGLU→downTRF36680cycles, 실제native는모든run/cluster에서TRF완료1cycle뒤
시작했다. 이시간창은wholeFFN약11.7%이며예상절감량이아니다.
Gate gather/downweight의중첩을추가절감량으로합하지않는다.
[현재 FFN 계측](../../moa-dev/experiments/candidates/e137-ffn-readiness-profile/README.md).

[선택한 E140](../research/e140-down-local-selection-20260927.md)은기존canonical
BF16gather뒤K1920소비위치에서같은cluster7680scale/K4max와weighted3항을
준비하여leader/termring을없애는경로다. 실제시작e35ea35, sourceE137334a705와
성능control baseline/e137.json을분리했다. 첫VRF할당실패와두부분view표현실패를
보존하고, local-v4에서잔차준비만4×480으로나눈flatouterchunk표현이exact통과했다.
전체실측은아직없다. 단순K2up/gate변형은C15실측회귀와중복이므로선정하지않았다.
아직새이득·공식제출·순위변화는없다. 아래는당시작업기록이다.

# 2026-09-27 E138 QKV 구현·E139 현재 FFN 계측 진행

Runtime/control은 E137이다. [실행 선택](../research/e138-e139-execution-selection-20260927.md)에
source recipe, 실제 추가/제거 비용과 각 gate를 기록했다. 실제 시작 commit은
`d6e6c62e6ae80c9b39b192c4a318a0230e79ee20`이다.

- `opt/qkv-native-consumer-local` / E138: K4×H960 소비자 로컬4항 E4 입력과
  activation TRF를 Q/K/V에서 재사용하고 원본 E4 weight를 native로 계산하는 전체 경로.
  Source와 성능 control은 각각 E137334a705/baseline/e137.json으로 명시했다.
  현재 BF16 lookup/fullH 복제와 새 양자화/K4합산/정규출력 복원 비용을 비교한다.
  원본 public interface·수치 허용오차·head/cache 의미를 유지하며 O/FFN image도 유지한다.
- `diag/e137-ffn-readiness` / E139: 현재 whole FFN의 up/gate부터 native-down/후단까지
  actual image로 한정 계측한다. 이름bytes 외 실행 code/constants/IO를 바꾸지 않는다.
  Down activation이 정말 완료를 지연하는지 먼저 판별하고 양자화 로컬화 여부를 정한다.

QKV whole positive는5%미만이어도 source동결 후3독립allocation/추가6/원래경계 검증으로
채택판단한다. Source버그는 구체원인이 있을 때 수리하며 회귀/timeout/지원불가를 구분한다.
새실측·채택·순위개선은 아직 없다. 아래는 앞선 완료 기록이다.

# 2026-09-27 E137 공식 제출·순위 반영 완료

공식 제출 **`330d5595`**이 server contract/build/evaluate와 기본15비교를 통과했다.
QKV103,960 / O43,026 / FFN313,827cycles,
**6.4427554×**, BEEF **46/69위**다.
직전 E134의6.1405524×/51위에서 점수 **+4.9214%**,
순위 **51→46위**다.
[제출·서버 원시 로그](e137-official-submit-20260927.json) · [리더보드](leaderboard-e137-20260927.json) ·
[현재 격차 계산](../research/e137-rank-gap-20260927.json).

공식 O는-13.9291% 변했다.
QKV/FFN image는 E134와 동일하므로 두 공식 실행의 해당 변동을 O최적화 효과로 세지 않는다.
이번 변경의 비교 근거는 별도 동시 E134 controls를 둔3개 Arena의 O감소율 중앙12.2978%다.

30위 eclab의7.7711479×까지는 여전히 격차가 있다. 다른 팀 점수를 고정한 산술에서
세 커널을 같은 비율로 줄이면 각각 **17.09%**,
O를 유지하고 QKV/FFN만 같은 비율로 줄이면 각각 **24.51%**가 더 필요하다.
현재 수상권/30위 도달을 주장하지 않는다. E137은 검증된 O개선을 확보했으며
[후속 연구](../research/after-e137-priority-20260927.md)는 QKV의 소비자 소유권 재설계를 우선 검토한다.
아래 준비/미제출 표시는 당시 상태로 남긴다.

# 2026-09-27 E137 native O 채택 완료, 공식 제출 준비

현재 runtime은 **E137 / `baseline-e137` / `334a705`**다.
[현재 구성](../../baseline/e137.json) · [최종 채택 결정](../../moa-dev/experiments/candidates/o-native-activation-resident/adoption-decision.json) · [실험·검증](../../moa-dev/experiments/candidates/o-native-activation-resident/README.md).
실제 branch 시작5c94b51, source recipe/control은 각각E1344f8fe7f다.

Frozen v2는 native dot이 짧아졌지만 입력 양자화·512KiB term 배포 이후에 소비를 시작해
전체 O가14.0551% 느렸다. 내부 NPU 계측110511을 근거로 v3는 BF16 입력을 실제 소비 slice로
먼저 보내고 각 rowgroup에서 같은 K4096 scale을 K16 Max로 구해4항을 직접 준비한다.
원본입력 복제16→256KiB와 증가한 local양자화, 남은 term저장/TRF적재까지 전체 실측에 포함했다.

| 독립 Arena | E134 앞/뒤 중앙 기준 | E137 O | 감소율 |
|---|---:|---:|---:|
|110558|49535|42526|14.1496%|
|110573|48980.5|42957|12.2978%|
|110584|49122|43097|12.2654%|

Paired 감소율 중앙값 **12.2978%**, job중앙값 집계49,122→42,957cycles다.
표준9입력(기본3+추가6), 실제 raw O9사례34,560exact BF16, 원래 full O case7의
RMS/residual 포함3,840출력이 통과했다. 일반BF16·임의입력·전체모델 정확성 증명은 아니다.
46개 compile input은 실측source/tag/live와 일치하며 변경은 `sliding/projection.rs` 하나다.
QKV/FFN image는 E134와 byte-identical이다. 추가 parameter tuning은 종료한다.

공식 제출 전 로컬 payload 검증을 진행한다. 공식점수·순위는 완료 후 별도 기록한다.
[다음 연구 판단](../research/after-e137-priority-20260927.md): QKV의 K4×H960 activation-resident
소비자 경로를 우선 검토하되 source/descriptor/정밀도/전체QKV비용을 새로 확인한다.
E137의 K256 성공을 fullH3840에 그대로 대입하지 않는다. FFN local-down 준비는 현재
native-down의 readiness 병목이 확인될 때만 선택하며 E135 timeout의 수리로 혼동하지 않는다.
아래는 각 실험 당시 기록으로, 이전 pending/미채택 표시는 그 시점의 상태다.

# 2026-09-27 E135/E136 검증 완료, E137 native O 선택

Runtime/control은 E134를 유지한다. [전체 경로 선택](../research/top30-paths-20260927.md),
[문헌·설치 SDK 대조](../research/top30-literature-20260927.md),
[30위 격차](../research/top30-leaderboard-20260927.json)를 기준으로 큰 비용을 바꾸는 두 경로를
실제 구현·NPU 검사했다. 이번 두 후보에서 새 성능 이득은 확보하지 못했다.

- [E135](../../moa-dev/experiments/candidates/ffn-independent-direct-fused/README.md)는 각각
  compact FP4에서6행 fused dot/scale/H-reduction을 직접 구동하여 up/gate FP8 저장과
  joint assembly를 피했다. Exact/descriptor/register 검사는 통과했으나 whole110314와
  production up/gate oracle110316 모두 첫 launch timeout(code4)이다. E134 control은
  정상 완료했다. 출력·FFN cycles가 없어 오답/성능회귀를 판정할 수 없다. 원인은 미해결이며
  packet수225대450 parity만으로 bank수리를 정당화하지 않는다. 채택보류·행수/padding튜닝중단.
- [E136](../../moa-dev/experiments/candidates/qkv-native-weight-resident/README.md)는 원본 E4
  weight Sub→TRF와4항 activation stream, canonical 출력행을 결합했다. Whole110367
  45PASS였지만 QKV116321 vs E134102925/104796, bracket대비 **+11.9973%**였다.
  실제 fullQKV 상쇄2case는16384출력과8380416 비대상cache 모두PASS다. 정밀도 수리는
  검사범위에서 유효하지만 추가 weightTRF/양자화/전파를 포함한 성능은 회귀했다. 미채택·튜닝중단.

다음은 [E137 O activation-resident 선택](../research/e137-native-o-selection-20260927.md)이다.
작은4항 activation을 TRF에 두고 원본 E4 weight를 Main으로 흘리며 C09두60행 ingress를
유지한다. Global amax/실제4항 quantization/복제/coeff를 포함하고 Term→K16 합산을 fuse한
뒤 기존 F32 channel-scale pass에서 공통scale을 복원한다. E136의 큰 weightTRF 적재와
O weight-resident의 Row15 반복읽기를 피하는 조합이다. E47 native O의 정밀도·중간저장
수리이지 독립 새family나 개선 보장이 아니다. 실제 source/control은E134를 별도로 명시한다.
양의 whole실측은5%미만이어도 source를 동결해 제한된 반복·추가입력·원래O경계 검증으로 진행한다.

# 2026-09-27 18:04 KST E134 공식 제출·순위 반영 완료

공식 제출 **`892178e8`**이 서버 contract/build/evaluate와 기본15비교를 통과했다.
QKV103091 / O49989 / FFN314619cycles, **6.1405524×**, BEEF **51/69위**다.
제출 직전 E127의6.0699449×/53위에서 점수 **+1.1632%**, 순위 **53→51위**다.
[제출·원시 로그](e134-official-submit-20260927.json) · [공식 리더보드](leaderboard-e134-20260927.json) ·
[현재 점수 격차](../research/e134-rank-gap-20260927.json).
새 submissionCode와 세 cycles가 실제 리더보드와 일치하고69팀 점수 계산도 검증했다.
QKV/O는 같은 image이므로 두 공식 실행 간의 변동을 이번 FFN 최적화 효과로 세지 않는다.

20위8.175122×/10위8.872707×다. 다른 기록을 고정하고 QKV/O가 같다면 FFN을
추가57.62%/66.85% 줄여야 각각 도달한다. 세 커널을 같은 비율로 줄이는 가정에서도
각24.89%/30.79%가 필요하다. 현재 확보한 후보로 수상권 진입을 보장하지 않는다.
현재 runtime과 작업 규칙은 아래 E134 채택 기준이며, 추가 weighted-v3 튜닝은 종료했다.
과거의 official_submission:false/pending 표시는 각 기록 생성 당시 상태로 보존한다.

# 2026-09-27 E134 weighted native down 채택 완료

현재 runtime은 **E134 / `baseline-e134` / `4f8fe7f`**다.
[채택 구성](../../baseline/e134.json) · [실험·검증](../../moa-dev/experiments/candidates/ffn-native-down-weighted-residual/README.md).
`opt/ffn-native-down-weighted-residual`, 실제 시작543a947; source는 미채택 E133v2,
성능 control은 E128이다. 같은 native-down 가족의 정밀도 수리이며 독립 아키텍처로 세지 않는다.

E5 hi/lo×16/third×256과 native F32 계수[s,s/16,s/256]·term 합산을 함께 바꿨다.
Down FP8 weight materialization의 논리적 DM 쓰기·읽기를 각각56.25MiB 제거했다.
추가 양자화·3항 방송·96B 계수·native→VE·F32 합산 비용을 포함한 전체 FFN이
독립3allocation에서 **−4.4947/−4.4518/−4.0041%**, paired중앙 **−4.4518%**였다.
중앙 candidate314266 / contemporary control328908.5cycles다. QKV/O image는 E128과 같다.

표준9입력, 별도 실제 down9case34560출력이 통과했다. 같은 공개 full-FFN image를 쓴
구성 반례109965에서는 E128/E134가3840개 모두 BF16 정답과 일치했고 E133은1920개가
원래허용오차를 벗어났다. 정답은 공식 PyTorch dense MLP/RMS로 계산했으며 입력13개 hash도
독립 대조했다. 구성 반례는 generator성분 범위 안의 상관 입력이지 관측된PRNGseed가 아니다.
전BF16 정확성·전입력 허용오차·전체 모델 검증을 주장하지 않는다.

v1/v2 SFR backend 실패와 v3 Main 계수경로 수리, 원본 source/build/image/upload/log는 보존했다.
5%미만이라는 이유로 버리지 않고 정밀도 보완 후 채택했다. 추가 파라미터 탐색은 종료한다.
채택 결정·튜닝 종료·큰 구조 연구를 별도로 기록했다. 큰 순위 개선은 QKV/O를 포함한
전체 실행경로 연구가 더 필요하며, E134 하나로20위/10위 진입을 약속하지 않는다.
[실패 회수와 다음 판단](../research/e134-recovery-and-next-20260927.md)에 남은 비용·조건부 O 입력 공급 수리를 구분했다.
공식 제출 결과는 이 문서 상단에 별도 기록했다. 아래는 각 작업 당시의 상태다.

# 2026-09-27 E131 회복 가능성: 전체 NPU 내부 계측 완료

`diag/e131-cost-recovery`, 시작 `8ee5615`. Runtime과 성능 대조군은 E128이다.
동결 E128/E131v2의 원본 image와 같은 입력을 사용해 packed/scale 복사,
입력/table 준비, native/fused 계산, gather와 down 진입의 실제 의존 순서를 비교한다.
원본 비계측 host를 별도로 대조하며 진단 name 변경은 kernel code/constants를 바꾸지 않는다.
[Arena109609의57비교가 PASS](../../moa-dev/experiments/candidates/e131-cost-recovery-profile/README.md).
표준 전체 FFN은 E128328539/329639 대비 E131349566으로+6.2223%이며 기존 회귀가 재현됐다.
진단 trace의 각 cluster Task 상대 시각 중앙값은 다음과 같다.

| 경계 | E128 | E131v2 |
|---|---:|---:|
| 입력 HBM 시작 |2478|104492.5|
| pre-RMS 완료 |16040|115391|
| 첫 native 시작 |123889.5|185874.5|
| GeGLU 완료 |231075.5|248867.5|
| down BF16 dot 시작 |260566.5|282359.5|

E131의 모든6관측에서 up-scale HBM 완료8cycles 뒤 입력 HBM이 시작했다.
반대로 E128의 up gather에는 큰 down weight 전송과 겹치는 긴 대기가 있어 E131이 일부를 회수한다.
따라서 최초입력+102k/첫dot+62k/최종store+22k를 합산하거나 각각 제거 가능량으로 취급하지 않는다.
단순 packed 복사 두 개의 비용보다 input/weight 공급과 후속 소비 순서가 중요한 연구 질문이다.
이는 이번 실행 순서의 관측이며 필수 수학적 의존성·특정 bank 원인을 증명하지 않는다.

[공개 interleave 축 이름 수리 진단](../../moa-dev/experiments/candidates/e131-interleave-name-probe/README.md)은
동일shape의 이름 변경만 exact compile했다. SDK가 lowercase Ident를 거부해 실패했으며,
기존 native contraction의 내부 interleave 축 불일치는 미해결이다. Image/NPU 결과는 없다.

E131v2의 이전 단회 screen에서는 E128 동률까지20,392.5cycles의 순절감이 필요했다.
이는 candidate 시간의5.8453%이며 회수 가능한 overhead를 측정한 값은 아니다.
행수 튜닝은 종료하되, 실제 지연과 이를 바꿀 지원되는 전체 경로가 확인되면 구조 변경을 재개한다.
O의 K4×30행/C09 두 stage16·14행/F32 직접 gather 조합은 별도
`opt/o-owned-k4` worktree에서 E132로 구현했다. 기존 H 축의 local tile 표현을
독립 OutputRow 축으로 수리한 v2가 exact/descriptor gate를 통과했다. Arena109654는45PASS이나
O가 E12848583/49216 대비55052로+12.5819% 회귀했다. 동결 profile109683도57PASS,
whole+15.7718%로 회귀했다. 첫 weight 시작은+5295.5cycles 늦지만 gather 증가는147cycles다.
control의 inputDMA→weightDMA→inputTRF 발행이 candidate에서는 inputDMA→inputTRF→weightDMA로 바뀐다.
이를 수리하려는 v3의 source 생성 순서 변경은 exact image까지 v2와 동일해 NPU를 반복하지 않았다.
해당 prototype 채택·tile tuning은 중단하고, 실제 입력/weight 발행 의존성을 바꿀 방법이 있을 때 재개한다.
이는 과거 K4의 BF16 scalar compaction 비용을 제거하고 C09 공급을 조합한 재개이며
새 독립 원리 또는 입증된 큰 개선으로 세지 않는다.
E133은 `opt/ffn-native-down-e5`, 진단 통합 뒤 실제 시작0f601fe, source/control E128로 구현했다.
기존 up/gate를 유지하고 cluster별 실제 GeGLU 출력7680을 E5 세 항으로 양자화·방송한 뒤
direct FP4→E5 native down/F32 block partial/기존 scale·합산·후단을 연결한다. Weight joint 복사는 없다.
v1은 새 term-copy DMA가480B source를1440B로 확장해 descriptor gate에서 중단했다.
명시적인480B Sub 복사로 고친 v2는 exact compile·실제 주소/범위/scale소유권/register gate를 통과했다.
전체 FFN의 세 독립 allocation109740/109750/109751에서 각각−3.8264/−3.4336/−3.8376%로 개선됐다.
후보 기본45비교와 추가6입력30비교가 통과했다. 실제 down helper의7개 mapping 사례26880출력도
BF16 reference와 일치했다. 그러나8번째 극소값 사례는3840출력 모두1.40625→0.703125로50% 손실됐다.
이는 raw down 진단이며 전체 FFN 실패와 같지 않다. 별도 CPU 구성에서는 생성기의 성분별 값 범위 안에서도
최종 FFN 허용오차 초과 가능성이 확인돼 채택은 보류한다. 관측된 PRNG seed/전체 NPU 실패를 주장하지 않는다.
성능 이득은 동결·보존하고, 작은 residual의 표현 scale과 native F32 term 복원을 함께 바꾸는 수리를 검토한다.
Down 단독 순위 약속은 없다.
[최신 점수 목표·재개 기준](../research/e131-recovery-rank-analysis-20260927.json).

# 2026-09-27 E129–E131 큰 구조 실행: 신규 채택 없음

현재 runtime은 **E128 / baseline-e128** 그대로다. 공식 재제출은 없었다.
15:53 KST 조회는 BEEF53/69위6.0699449×, 10위8.8727073×다.
[조회 원본](../research/top10-leaderboard-e128-research-20260927.json).

- [E129 projection-owned FFN](../../moa-dev/experiments/candidates/ffn-projection-clusters/README.md):
  전체 source 구현 뒤 Cluster tile의 실제 target lowering 미지원으로 compile 실패. NPU 성능 실패가 아니다.
- [E130 Main/Sub QKV](../../moa-dev/experiments/candidates/qkv-wavefront/README.md):
  whole10933945PASS,104856→119470cycles(+13.9372%). 미채택·추가 튜닝 중단.
- [E131 joint direct FFN](../../moa-dev/experiments/candidates/ffn-joint-direct-fused/README.md):
  FP8 weight 저장 제거와 fused scale/reduction·M03 consumer를 연결한 완성 조합.
  two-row10943045PASS/+15.1668%, four-row+tail10940845PASS/+6.2082%로 둘 다 미채택.
  two-row 독립 oracle245760출력 PASS. four-row 분리 진단은 case0 launch timeout;
  whole 오답으로 분류하지 않으며 검증 완료도 아니다. 추가 tile/source 수정은 종료한다.

E131의 순 logical DM write/read 감소는 각49.21875MiB지만 실측은 회귀했다.
복사·lookup·동기화 비용을 포함한 whole 결과이며 특정 stall 원인은 측정하지 않았다.
다음 표현 변경 후보는 [native down 재검토](../research/e131-native-down-next-20260927.md)에 남긴다.
E93/E95의 positive를 검증된 채택본 또는 down 정확도 실패로 잘못 분류하지 않는다.
아래 기록의 현재/예정 표시는 각 작업 당시 상태다.

# 2026-09-27 E129 projection-owned FFN: target lowering 제약 확인

[전체 source·실패 근거](../../moa-dev/experiments/candidates/ffn-projection-clusters/README.md):
cluster0 up/cluster1 gate를 한 direct producer로 묶는 전체 FFN을 구현했으나,
설치 compiler가 공개 `cluster_tile`의 Cluster IndexStep을 지원하지 않아 첫 ingress에서 실패했다.
NPU 정확도·성능 미평가이며 느린 구조로 판정하지 않는다. E128 runtime은 유지한다.
별도 E131은 E123 joint assembly 비용을 명시하고 direct native fusion·M03 consumer를 조합한다.
QKV E130은 별도 worktree에서 Main F32 dot/Sub consumer 전체 경로를 실측 준비 중이다.

# 2026-09-27 E128 native FFN dot/scale/reduce 채택 완료

현재 runtime은 **E128 / `baseline-e128` / `cc82b9c`**다.
[실험](../../moa-dev/experiments/candidates/ffn-native-fused-scale/README.md) · [구성표](../../baseline/e128.json).
`opt/ffn-native-fused-scale`, 실제 시작 `57e6a5b`, source·별도 성능 control E127.
고정6행 native dot→Mul0(input scale)→Mul1(block scale)→F32 H 합산으로
up/gate partial DM28.125MiB write/read가 실제 graph에서 제거됐다. Native 시작은2→10이다.
정적155190 vs154440(+0.486%)와 달리 NPU3allocation은 FFN−0.7879/−0.6569/−0.5564%,
paired중앙−0.6569%다. 후보기본45·추가30비교(9 distinct standard inputs),
별도8case245760 projection출력 PASS. QKV/O image는E127과동일하다.
Positive를동결해검증후채택했으며 tile sweep은없다. 이번큰저장삭제의실측효과는작았다.
추가튜닝은중단하고 FP8 weight 저장/joint packed복사를없앨ownership구조는별도연구로남긴다.
새공식제출없음;공식순위는아래E127기록이다. 검증host의checksumAPI·출력경로수리로그도보존했다.

# 2026-09-27 E127 이후 큰 중간 저장 제거 shortlist

[현재 코드·과거 실패·SDK를 대조한 후보 분석](../research/e127-large-cost-shortlist-20260927.md)을 추가했다.
첫 검증 대상은 native FFN dot→입력 scale→block scale→F32 합산의 단일 Main stream이다.
Up/gate F32 partial28.125MiB write/read를 제거하되 native dot 시작은2→10으로 늘어난다.
E73의 반복 BF16 lookup과 C11의 partial VRF handoff를 피하는 국소 데이터 흐름 교체이며 미구현·미측정이다.
더 큰 구조 연구는 projection별 cluster 소유권으로 FP8 전체 저장과 joint packed 복사를 함께 없애는 안이다.
전송 계약·direct 정확도부터 확인해야 한다. E103/direct reload와 v7b/interleave를 새 후보로 반복하지 않는다.
새 compile/Arena/제출은 없으며 runtime은 아래 E127 그대로다.

# 2026-09-27 E127 공식 제출·리더보드 반영 완료

현재 runtime E127을 공식 제출 **`bc21122c`**로 평가했다. 서버 contract/build/evaluate 완료,
기본15비교 PASS. QKV103186 / O49155 / FFN330948 cycles, **6.0699449×**다.
제출 직전 E1045.9265771× 대비 **+2.4191%**, 공개 리더보드는 **53/69위 →53/69위**다.
새 submissionCode가 실제 leaderboard에 반영됐음을 확인했다.
[제출·원시 로그](e127-official-submit-20260927.json) · [제출 전 순위](leaderboard-e127-before-20260927.json) ·
[제출 후 순위](leaderboard-e127-20260927.json).
QKV 변경은 없으므로 두 공식 실행의 QKV 차이를 새 최적화 효과로 세지 않는다.
아래 채택·실험 기록의 미제출 표시는 해당 작업 완료 당시의 상태다.

# 2026-09-27 E127 / 동결 C09 채택

현재 runtime은 **E127 / `baseline-e127`**다. E122의 QKV/FFN image를 유지하고,
O에 원본 C09의 두 stage 공급을 그대로 채택했다. [채택 구성](../../baseline/e127.json) ·
[성능·확장 정확도 검증](../../moa-dev/experiments/candidates/c09-output-first-npu/validation/README.md).

E126과 새 독립2allocation의 paired O 변화율은 **−2.4960% / −3.8255% / −3.9303%**,
중앙값은 **−3.8255%**다. 실제 stage host/fixture/runner가 동일하며,
표준 candidate45비교와 추가6입력30비교가 모두 PASS했다. 표준 입력은 반복을 제외하면
커널별9개다. 별도 pre-RMS projection probe8case/30720출력도 PASS했다.
Probe는 별도 diagnostic image이며 표준 성능 집계에는 포함하지 않는다.

**E126의 초기 종료·미채택 판단은 E127 후속 검증과 채택으로 대체됐다.**
5%는 추가 parameter tuning의 예산 기준이며, 작은 이득의 제한적 채택 검증을 금지하거나
runtime 채택의 최소 개선율을 뜻하지 않는다. Source/layout/tile을 더 조정하지 않고
기존 구현의 개선을 검증해 회수했다. 큰 구조 개선 탐색은 별개이며 새 공식 제출은 없다.
아래 기록의 runtime/미실행/종료 상태는 각 실험 당시를 가리킨다.

# 2026-09-27 E126 / frozen C09 최초 NPU 완료 — 당시 기록

[C09 최초 NPU](../../moa-dev/experiments/candidates/c09-output-first-npu/README.md)는
과거 exact image를 재현하고57비교 PASS. Arena105820 전체 O49923 vs E122
before51237/after51165로 **단회 −2.496%**를 관측했다. 추가 튜닝·반복·확장 정확도·채택 없이
5% 탐색 예산 기준으로 종료하고 소폭 positive option으로 보존한다. Live E122 유지.
Trace의 첫 dot은28945→17148, dot0/weight1 실제 overlap7392.5다. 다만 weight공급
전체 구간은26660→30073으로 늘어 최종 dot 종료는40953→39542.5의 제한적 개선이다.
정적 마지막dot→gather1206 gap은 NPU에서재현되지않았고 양측모두2cycles였다.
[이번 실행 결과와 다음 설계 조건](../research/o-weight-ingress-execution-20260927.md).
M08 기록617aaef를 main/origin에 통합한 뒤 이 별도 진단을 실행했다. 공식 제출 없음.

# 2026-09-27 E125 / M08 O row-pair4 종료

[M08](../../moa-dev/experiments/candidates/m08-output-pair4-ingress/README.md)는
actual DMA coverage·BF16/F32 경계 검사와57개 기본 비교를 통과했지만,
Arena105767 전체 O58095 vs E122 before51418/after50906으로 **+13.55% 회귀**했다.
Trace에서 weight26769→27190으로 단축이 없고 F32 gather1201→8222로 증가했다.
같은15360B gather의32×480B→1920×8B 분할 비용을 전체 경로에서 확인했다.
추가 sweep 없이 종료; live E122 유지, 공식 제출 없음. 기록을 main에 통합한 뒤
동결 C09 two-stage의 **최초 NPU 진단**으로 이어갔다. 완료된 E126 결과는 상단에 있다.
정적+0.71%만으로 종료했던 prototype의 실제 overlap/whole O를 확인했으며 새 architecture로 세지 않는다.

# 2026-09-27 O weight ingress 연구 완료

[O 전송·배치 종합 분석](../research/o-weight-ingress-20260927.md)을 추가했다.
현재 O와 동일한 image의 graph로 15 MiB의 정확한 source coverage와 DMN 교대를 확인했다.
첫 진단 후보는 행쌍 보존 4-way ingress + 기존 16K F32 합산 + 직접 H480 gather다.
현재 gather 32×480B가 예상 1920×8B로 분할되는 비용까지 전체 경로에서 비교해야 한다.
C09는 정적 +0.71% 후 NPU 미실행이므로 최초 하드웨어 진단의 가치가 남아 있다.
이번에는 새 compile/Arena/runtime 변경·공식 제출이 없다. 채택본은 아래 E122다.

# 2026-09-27 E122 채택 및 전송 계약 재검증

현재 runtime은 **E122 / `baseline-e122` / `c3553ab`**다. M03 원본을 그대로 채택했다.
[검증](../../moa-dev/experiments/candidates/m03-producer-local-ffn/validation/README.md):
동일 실행 stage의6allocation에서 모두개선, paired비율중앙값 FFN−1.4846%,
추가6표준입력과7-case241920 projection출력 PASS. QKV/O image는E104와같다.
5%는탐색예산문턱이며 이미반복검증된작은이득의채택금지가아니라는예외를기록했다.
공식제출은아직E104이며 Arena결과를공식점수로환산해주장하지않는다.

[E123 corrected direct FFN](../../moa-dev/experiments/candidates/ffn-direct-boundary-recompute/README.md)은
전체 기본60비교 PASS, 단회329395cycles로 E104 앞뒤 대비−1.9179%, 동시 M03 대비−0.7392%다.
작은 이득으로 반복·확장 정확도·채택 없이 종료했다. Direct primitive 실패를 일반
native DPE 한계로 보았던 해석은 철회하되 이 단회 결과도 일반 동치로 확대하지 않는다.

E124 corrected joint QKV도 종료했다. Arena105093의45비교가 모두통과했으나
QKV129258 vs 앞뒤 E104103899로 +24.41% 회귀해 미채택했다. [C08 exact 감사](../../moa-dev/experiments/candidates/c08-joint-ingress/README.md)에서
K/V padded destination DMA의 실제 source 범위 초과를 확인했다. 원본과 image가 같다.
이는 과거 family의 전송 계약 수리이며 새로운 독립 architecture가 아니다.
수정 뒤에는 정확하지만 느린 실행으로 판정할 근거가 생겼다. 추가 반복·튜닝은없다.

[이번 실행 전체 결과와 다음 판단 기준](../research/top10-execution-20260927.md).
다음 미실행 질문은 O weight 준비 지연의 DMA 순서/DM 배치 민감도다. 단독 전송
진단을 커널 speedup으로 계산하지 않고, 유망한 배치만 완성 O 경로에서 검증한다.

# 이전 구조 실험 기록 — 아래의 현재/미실행은 당시 상태

작업 폴더: `/home/jihwan38/moa-workspace`. Private GitHub: `jihwan38/moa-workspace`.
최신 조사: `research/rngd-dataflow-20260925` / 시작 commit `f125ca6`.
보완 조사: 2026-09-26 / `opt/m02-cluster-k`의 `a46baa4` 이후.
당시 채택 runtime: E104. 최신 채택본은 문서 상단 및 [baseline/current.json](../../baseline/current.json)의 E127이다.

## 최신 감사 — 10위권 격차와 누적 실험 재평가, 2026-09-27

[코드·실험·논문 재점검](../research/top10-audit-20260927.md)을 추가했다.
9월27일00:41 KST 공식 조회는 BEEF52/69위5.926577×, 10위8.859047×다.
FFN만 바꿔 그 점수를 맞추려면335987→100594 cycles가 필요하므로 QKV/O의
큰 비율 개선 경로도 병행해야 한다. 메모리 배치 전체가 잘못됐다고 단정하지 않는다.

M03의 첫 E117은 단회였지만, 후속 companion을 포함하면 동일 package의5개 allocation에서
앞뒤 E104 대비1.35–1.63% 감소, 감소율 중앙값1.4891%다.
[누적 원본·해시·계산](../research/m03-cumulative-audit-20260927.json).
동일3입력 반복이며 확대 정확도나 새 채택을 뜻하지 않는다. 아래 단회 설명은 첫 실험의
역사 기록이다. 5% 탐색 예산 cutoff와 제한적 채택 검증은 구분할 필요가 있다.
QKV/O 내부 계측, direct-FP4 고정 shape boundary 재계산 후보, M06 timeout 기제 진단의
조건과 중복 여부를 정리했다. 새 compile/Arena/채택/공식 제출은 없고 runtime은E104다.

## 최신 작업 — M07 weight 전송 분할, E121 개선 미확인으로 종료

M06 기록 `0c6dea5`까지 main/origin에 통합하고 `opt/m07-weight-ingress-chunks`에서
[M07](../../moa-dev/experiments/candidates/m07-weight-ingress-chunks/README.md)를 구현했다.
E74 crate + 동결 M03 전체 FFN을 source로 선택하고 세 weight의 HBM 적재만5분할한다.
M04와 달리 FP8 변환·dot은 전체 크기의3개 stream을 유지한다. Local transfer 변경이며
새 전체 FFN 아키텍처 가족으로 세지 않는다.
v1은 compile PASS지만 up/gate 각 tile이30행 전체 descriptor로 넓어져 정적 비용이 컸다.
v2는 compact6행 임시 buffer를 만들었으나 assembly DMA가30행을 읽으려 해 범위 검사 실패.
v3은 해당 assembly만 raw FP4 Sub copy로 고쳐 전체 compile·host PASS다.
v3의15개 HBM weight command는 의도한5분할 payload이며 전체 FP8 변환·dot은3개다.
입력 준비 앞의 상대 대기는 줄지만 pre-RMS 자체가 늦어져 정적 전체는 M03보다 늘었다.
Arena104739: E104335871 → v3 336059 → 원본 M03 330118 → E104335305.
기본60비교 PASS, 후보 FFN3입력/11520출력 PASS다. E104 앞뒤 중앙값 대비+0.14%,
동시 M03 대비+1.80%로 채택할 이득이 없어 추가 튜닝 없이 종료했다. 단회 screen이며
내부 NPU 시각·stall은 계측하지 않았다. [E121](e121-m07-weight-ingress-chunks.json).
v1은 중복 전송 descriptor 비용으로 정적 종료, v2는 compile 실패, v3은 실측 종료다.
세 구현 변형을 독립 아키텍처3개로 세지 않는다. Live E104 유지; 새 채택·공식 제출 없음.

## 최신 작업 — M06 실행 timeout으로 보류, E119/E120

M05 기록 `83d09dd`까지 main/origin에 통합하고 `opt/m06-sharded-input-pipeline`에서
[M06](../../moa-dev/experiments/candidates/m06-sharded-input-pipeline/README.md)를 구현했다.
예전 E53 FFN sharded RMS와 QKV/O 직접 packing의 전송 방식을 현재 native3항 입력과
M03 구조에 결합했다. pre-RMS gather/입력 재분배/scale 재분배/3항 packing DMA를
제거하는 경로다. 과거 방법을 새 독립 아키텍처로 세지 않는다.
v1/v2/v3은 SFR lowering에서 실패했다. Packed lo의 cast/add를 Sub→Main으로
옮긴 v4와 별도 term buffer·packing DMA를 복구한 v5는 전체 compile·host build PASS다.
그러나 Arena104566/104588에서 두 후보 모두 첫 FFN 출력 비교 전에 device timeout.
같은 작업의 E104 before/M03/E104 after는 각각45비교 PASS여서 접속 실패가 아니다.
Direct packing만이 원인이라는 증거도 없다. 원인 미확정으로 보류하며 추가 변형은 중단했다.
[실패 E119](e119-m06-sharded-packed-invalid.json) · [실패 E120](e120-m06-sharded-dma-invalid.json).
유효 FFN 성능·정확도·채택·공식 제출 없음. 실행 source는E104다.

## 최신 작업 — M05 입력 준비·M03 조합, E118 개선 미확인으로 종료

M03 기록 `db7cbf4`까지 main/origin에 통합한 뒤 `opt/m05-local-activation-ready`에서
[M05](../../moa-dev/experiments/candidates/m05-local-activation-ready/README.md)를 시험한다.
M03의 local GeGLU/L-owned down 배치에 M04의 producer-local 입력 준비만 결합했다.
M04의15개 row-tile decode/dot 대신3개 전체 weight stream을 유지한다.
첫 activation 복제 DMA 전에 amax를 시작할 수 있는지가 구체적 가설이며,
새 독립 아키텍처 가족이나 타일 크기 sweep으로 세지 않는다.
Arena104458: E104335775 → M05 335317 → 원본 M03 331749 → E104337758.
기본60비교 PASS. E104 앞뒤 중앙값 대비−0.43%, 동시 M03 대비+1.08%여서 종료했다.
정적 amax는 앞당겨졌지만 최종 입력 방송 완료는 M03보다 늦어졌다. 내부 NPU 시각
측정은 아니며, 전체 실측에서도 유의미한 이득이 없다. 추가 튜닝·채택·공식 제출 없음.
실행 source E104 유지. [E118](e118-m05-local-activation-ready.json).

## 최신 작업 — M03 producer-local 전체 FFN, E117 소폭 개선으로 종료

진단 기록 `6ea3664`까지 main/origin에 통합하고 `opt/m03-producer-local-ffn`에서
[M03](../../moa-dev/experiments/candidates/m03-producer-local-ffn/README.md)를 구현했다.
실제 source origin은 E74이며 C14-B의 local GeGLU/L-owned down에 E104의 검증된
materialized native up/gate를 조합했다. C14의 두 L stage는 한 stage로 바뀌었다.
전체 L gather 대신 cluster-local 120행 compact와 local down 입력 분배를 사용하며,
마지막에 두 F32 H partial을 더한다. C14 가족의 재조합이다.

첫 source compile·host build PASS, QKV/O image는 E104와 같다. 정적154440은
E104157045 대비 작은 감소지만 NPU 실측 성능으로 해석하지 않는다.
Arena104289: **E104335669 → M03 330837 → E104335107 cycles**, 앞뒤 비교군 중앙값
대비 **1.36% 감소**다. 총45비교 PASS, 후보 FFN3입력/11520출력 PASS.
단회 screen이며 확대 정확도·독립 반복은 없다. 5% 실험 예산 기준에 따라 추가 타일
튜닝 없이 종료하고 소폭 개선된 조합 선택지로 보존했다. 실행 source E104 유지,
새 채택·공식 제출 없음. [E117](e117-m03-producer-local-ffn.json).
전체 FP8 저장량과 초반 입력 준비 대기는 이 구조에서도 남는다. 이것을 메모리 배치가
전역 최적이라는 근거나 10위권 진입 방법이 확보됐다는 주장으로 해석하지 않는다.

## 최신 진단 — 직접 FP4/native 경로 보류, 정확도 미해결

`c49300d`까지 main/origin에 통합한 뒤 `diag/fp4-native-boundary`에서
[원본·진단 기록](../../moa-dev/experiments/candidates/fp4-native-boundary/README.md)을 남긴다.
현재 목표는10위권이며, 순위 상승이나 새 runtime 채택은 아직 없다.

단일 native projection의 실제 입력 준비·dot·block scale·행 합산은4입력에서
통과했다. 같은 입력 표현을 공유하는 두 projection은 최종 BF16 gather 전에
틀렸고, 한 joint producer로 바꾸면 첫 projection의16행에만 오류가 남았다.
입력 양자화를 수행한 slices0..7의 첫 행과 일치한다. 저수준 원인은 아직 미확정이다.
옛 C07 실패 image도 다시 실행해 실패를 재현했고 materialized 원본은 통과했다.
새 진단의 materialized control에는 timeout이 있어 전체 paired PASS로 세지 않는다.
후속 v8의 scale 경계 이동도 실패했다. v9c/v9d 행 padding은 lowering assertion으로
NPU 미실행이며, v10(job104243)의 첫 projection 전체 warmup도 두 실제 출력에
non-finite 값을 만들었다. 이 경로는 원인 미해결 상태로 보류하고 반복 수정을 중단했다.
새 FFN 성능 결과·채택·공식 제출은 없다. 이 진단을 최적화 가족/성능 실험으로 세지 않는다.
당시 후속이었던 M03의 완료 결과는 위에 있다: 검증된 weight materialization을 유지하면서
producer-local GeGLU와 L-owned down을 한 stage로 연결한다.

## 최신 작업 — M04 입력 준비·weight 타일 공급 결합, E116 종료

진단 기록 `db89a50`까지 main/origin에 통합한 뒤 `opt/m04-tiled-weight-supply`에서
[M04 전체 경로](../../moa-dev/experiments/candidates/m04-tiled-weight-supply/README.md)를 구현했다.
입력 amax/3항 quantization을 원래 cluster에서 먼저 수행하고 완성된 표현을 복제한다.
Up/gate는6행, down은12행씩 HBM 적재→FP8 DM→dot→F32 partial→scale/reduce로 연결했다.
한 파일의 완성 경로 조합이며 기존 row/column 소유권·GeGLU·공개 입력 계약은 유지한다.
M03의 producer-local GeGLU/down 구조를 구현한 것으로 세지 않는다.

**첫 source compile·기본 NPU 정확도 PASS, FFN은 13.67% 느려 미채택**했다.
Arena 103108: E104 334906 → M04 381507 → E104 336369 cycles. 총45비교 PASS이며
후보 FFN은3입력/11520출력을 검사했다. [E116](e116-m04-tiled-weight-supply.json).
확장 정확도·독립 반복·공식 제출은 없고, 채택 runtime은 **E104 유지**다.

정적 graph에서 pre-RMS→amax gap은23250→0이지만 pre-RMS 자체는 더 늦어졌다.
스케줄상 FP8 텐서의 최대 동시 크기는56.25→22.5MiB로 감소했고 전체 FP8 저장량은 그대로다.
DMA 명령35→74, dot3→15, 정적 전체157045→165192다. 실제 NPU 내부 stall 비용을
이 수치로 환산하지 않는다. [판정·해시·원본 로그](../../moa-dev/experiments/candidates/m04-tiled-weight-supply/decision.json)를 남기고,
회귀 prototype의 tile sweep 없이 종료한다. 두 변경 각각이 나쁘다는 판정은 아니다.
M04 종료 당시 M03은 미구현이었다. 후속 재시도는 반복 setup/전송/저장 비용을 바꾸는 구체적 근거가 필요하다.

## 이전 작업 — E104 whole-FFN 내부 NPU 진단 완료

M02와 보완 조사 기록을 main/origin의 `913f9ac`까지 통합하고 `diag/e104-weight-path`에서
[내부 NPU profile 진단](../../moa-dev/experiments/candidates/e104-weight-path/README.md)을 실행했다.
별도 stage 출력 대신 기존 image의 trace marker를 읽어 원래 FFN 흐름을 측정했다.
Arena102797·동일 package 재실행102842 성공, 총78개 출력 비교 PASS다.
Kernel 계산 task bytes는 E104와 동일하며 diagnostic host와 profile 이름만 별도 처리했다.

Weight 적재는 gate약47.6–47.8k/up약48.0–48.2k/down약65.7–66.0k cycles,
FP8 전체 저장은 각각약20–21k, native up/gate dot·부분합은각각약12k다.
초반 큰 gate weight 전송 중 작은 입력 복제 구간이 길어지고, pre-RMS 끝부터
activation amax 시작까지 다른 Main 연산이 없는 구간은 중앙값42.2k였다.
입력 방송도23.4k다. 이 구간들은 겹치고 대기를 포함하므로 더해서 비용 비중을 만들지 않는다.

다음 구조는 weight 공급·작은 입력 준비의 순서/배치와 FP8 전체 중간 저장을 함께 검토한다.
M03은 아직 미구현이며, 채택 baseline은 E104다. 진단 자체를 최적화·순위 개선으로 세지 않는다.

## 이전 조사 — 현재 리더보드 격차와 FFN weight 경로

[코드·정적 근거·관련 대회 자료](../research/e104-bottleneck-20260926.md)를 보완했다.
2026-09-26 18:25 KST의 공식 API 응답에서 BEEF는69팀 중52위, 5.926577×,
3위는9.251393×다. 해당3위 cycle까지 QKV36.01%, O29.73%, FFN41.53% 감소가 필요하다.
FFN weight 적재→FP8 전체 DM 저장→dot 경로가 우선 병목 가설이다.
실제 NPU에서 내부 비용을 분해한 결과는 아직 없으며 기존 정적 진단만 재검증했다.
이번 작업은 조사·기록이며 새 compile/Arena/채택/공식 제출은 없다.
당시 M03은 미실행이었고 최소 비용 진단을 먼저 수행하기로 했다. 실행 결과는 위 최신 작업에 있다.

## 이전 작업 — M02 cluster K-half 소유권 검증 완료

M01의 `ecdeaca`까지 main/origin에 fast-forward 통합하고 `opt/m02-cluster-k`에서
전체 O 경로를 새로 구현했다. [M02 설계·수정·실측](../../moa-dev/experiments/candidates/m02-cluster-k/README.md).
cluster는 H 반쪽 대신 K2048 반쪽을 소유하고, 각 slice가15×2048 dot을 수행한다.
Ring32 leader 입력 방송, F32 partial gather/압축, 두 cluster의 F32 합+scale을 포함한다.
15×2048 tile 자체는 이전 두열 실험에도 있었고 이 조합의 새 ownership을 검증했다.

**두 완성 경로 모두 compile·기본 NPU PASS 후 큰 회귀로 종료했다.**
Arena95914: E10451010 → v3 full-H 집결78981 → v4 H480 묶음89203 → E10450901.
앞뒤 control 중앙값 대비 각각**+55.00%, +75.06%**다.
[E114](e114-m02-full-h-gather.json), [E115](e115-m02-grouped-interleave.json).
총60비교 PASS이며 각 후보 O는3입력·11520출력 원소를 검사했다.

v1 방송 축, v2 H%256→H%480 직접 DMA factorization 오류를 구체적으로 수정했다.
v3는 full-H 집결로 실행 가능성을 확인했고, v4는 producer row를 H480 경계에 맞춰
그 중간 집결을 없앴다. 정적 전체37413→32382로 줄었지만 NPU 순서는 반대로 나타났다.
Weight 적재·dot의 정적 interval은 줄지 않았고 partial 의존성이 추가됐다.
NPU에서 어떤 bank/전송/stall이 얼마를 차지했는지는 측정하지 않았다.

source snapshot4개·exact compile6회(그중 package 재compile2회)·host2개·Arena1회.
한 architecture의 두 경로이며 독립 가족 두 개로 세지 않는다.
추가 sweep·확장 정확도·독립 반복·채택·공식 제출 없이 종료하고 **E104를 유지**한다.
Source/log/hash/실측/판정은 branch에 보존한다. 다음 미실행 사전 후보는 **M03**이다.

## 이전 작업 — M01 partial 재배치 검증 완료

사용자의 실행 순서 지정에 따라 **M01을 먼저** 진행한다. 조사 기록을 main/origin에
`8109a27`까지 fast-forward 통합한 뒤 `opt/m01-partial-remap`을 만들었다.
[M01 설계·두 전송 경로·검증 기록](../../moa-dev/experiments/candidates/m01-partial-remap/README.md).
E64의 column-major 적재 뒤 weight 전체 재배치를 제거하고, 동일 배치에서 dot 후
compact F32 partial을 DMA 또는 Switch로 옮기는 완성 O 경로를 검사한다.
단순 column 수 sweep이 아니며 두 경로를 별도 아키텍처 가족으로 세지 않는다.

**두 경로 모두 compile·기본 NPU 정확도 PASS 후 실측 회귀로 종료했다.**
Arena95636: E10451007 → DMA55608 → Switch60677 → E10451274 cycles.
앞뒤 control 중앙값 대비 DMA+8.74%, Switch+18.65%다. 각 process의15비교가 모두 PASS다.
[E112 DMA](e112-m01-partial-dma.json), [E113 Switch](e113-m01-partial-switch.json).

첫 `dma-v1`의 commit-adapter source 오류는 v2에서 수정했다. 정적 일정에서는
weight 적재13383→13435, dot stream5063→5063으로 줄지 않았고,
새 partial 이동/합산이 직렬로 추가됐다. 정적 makespan은 DMA29824, Switch31995,
E10427239다. NPU stall 분해 측정과 구분한다.
긴 튜닝·추가 반복 없이 종료, E104 실행 소스 유지, 공식 제출 없음.
후속 미실행 설계는 M02와 M03이다.

## 이전 작업 — RNGD 데이터플로·메모리 매핑 조사

논문·공식 programming guide·SDK 0.8.1 및 과거 시도를 대조해
[전체 경로 후보와 비용](../research/rngd-dataflow-candidates-20260925.md)을 정리했다.
[설계 JSON](../research/rngd-dataflow-candidates-20260925.json),
[출처·해시](../research/rngd-dataflow-evidence-20260925.json),
[좌표·논리량 검사](../research/rngd-dataflow-accounting-20260925.json).

다음 구현 우선순위는 **M02 O cluster K 소유권 → M03 producer-local FFN**이다.
M03은 C14 가족을 native up/gate·한 L stage와 조합하는 재설계이며 새 가족으로 세지 않는다.
M01은 E64의 weight repack을 partial 재배치로 바꾸는 조건부 후보로,
적재·소비 경로의 이득을 먼저 제한된 진단으로 확인해야 한다.
후보별 전체 graph, HBM/DM/TRF/VRF 소유권, 추가·제거 작업, 첫 검증/종료 조건을 기록했다.

이번 작업은 조사·설계·정수 좌표 검사까지다. 후보 compile/Arena/제출은 없고,
실행 소스·채택 runtime은 E104다. 후보의 속도 향상이나 순위 상승은 아직 입증하지 않았다.

## 이전 작업 — 세 구조 실행

진행 순서는 **C14-B → C15 → C12-B**다.
[설계 범위·비용 가설·과거 시도 대조](structural-round-20260923.md).
첫 `opt/c14-stage-compact`의 C14-B는 전체 compile·기본 NPU 정확도 PASS 후,
FFN397348 vs 동시 E104336341/339553(+17.58%)로 종료했다.
[source·측정·비용 분석](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/stage-compact.md).
C15도 up/gate column 소유권을 교체해 compile·기본 NPU PASS 후 FFN385621 vs
E104335618/335432(+14.93%)로 종료했다. [전송 비용·수정 기록](../../moa-dev/experiments/candidates/c15-upgate-columns/README.md).
C12-B도 전체 compile·기본 NPU PASS 후 O51862 vs E10450555/50680(관측+2.46%)로
종료했다. [컴파일 분리 진단·완성 경로·측정](../../moa-dev/experiments/candidates/c12-output-main-sub/README.md).
세 후보 모두 기본 NPU까지 실행했으며 채택·공식 제출은 없다. 실행 source는E104다.
세 실험 branch의 기록을 main에 fast-forward 통합했다. 각 branch와 원격 tip도 보존했다.
[통합 이력](structural-round-20260923-integration.json), [전체 재검증](structural-round-20260923-verification.json).

## 이전 작업 — main 통합과 E104 비용 진단

**main과 origin/main을 `873f64e`에서 `f9f905b`로 fast-forward해 push했다.**
`opt/c13-int8-ffn`의 `2c51fe2`까지 26개 commit과 작업 규칙·안내 정리 commit을
포함한다. 확인한 모든 기존 local/origin 실험 브랜치 tip이 이 이력에 포함되어,
별도로 병합할 기존 실험 변경은 없다. [통합 확인 기록](integration-20260923.json).

실행 소스는 채택 E104이며, C13/C14는 후보 snapshot에만 남는다. AGENTS.md와
조합 양식에 최신 비교군의 비용 근거, 제거·추가 작업, 실제 의존성/자원 경쟁,
설계 선택을 바꾸는 최소 진단과 그 한계를 기록하도록 보완했다. Root README의
현재 채택본 표시도 E79에서 E104로 수정했다.

당시 브랜치는 **`diag/e104-cost-attribution`**였다. E104의 기존 schedule 293개
instruction을 source와 대조하고 긴 node 18개·context 중첩을 재현 가능한 JSON으로
남겼다. [첫 정적 분석·다음 판별 질문](../../moa-dev/experiments/diagnostics/e104-cost-attribution/README.md).
Weight 준비와 activation 분배 중 무엇을 먼저 크게 바꿀지 결정하는 최소 NPU
진단은 아직 미구현·미실행이다. 이것은 새 최적화 가족이나 E104 구조에 탐색을 고정하는
규칙이 아니다. C14 compact 전송 수정은 아직 미구현인 후속 선택지이며, C12도 미구현이다.

## 이전 작업 — C13 재개 후 NPU 검증 완료

작업 브랜치는 `opt/c13-int8-ffn`이었다. **C13 v7은 전체 compile과 변경 없는 NPU
기본 정확도 검사를 통과했지만, 동시 E104보다 FFN이 18.58% 느려 미채택**했다.
[전체 구현·초기 실패·수정·실측](../../moa-dev/experiments/candidates/c13-int8-ffn/README.md).

초기 v2를 compile 횟수 한도로 닫은 뒤, 사용자의 지적을 받아 구체적인 구현 오류를
수정했다. 8B scale packing과 명시적 입력 복제로 v6을 compile했다. 첫 Arena
실행 E107은 non-finite FFN 출력으로 실패했다. 분리 진단에서 긴 K=3840 integer
unzip/zip 뒤에 scale 곱셈을 붙이는 조합의 block 대응 오류를 확인했다. v7은 정확한
정수 결과를 F32 DM에 저장한 뒤 별도 tag-zero stream에서 scale을 적용한다.

Arena job 78730: **E104 before 334969 → C13 397318 → E104 after 335156 cycles**.
각 process의 기본 15비교가 모두 통과했으며 FFN 3입력/11520출력도 기존 기준 그대로
PASS다. 단회 screen이며 공식 점수·독립 반복 결과가 아니다. 성능 회귀가 커서 더
튜닝하지 않고 종료했다. E104 실행 소스 유지, 공식 제출 없음.

E74를 출발로 up/gate/down 모두 INT8 두 항/I32 누산으로 구현한 같은 C13 구조다.
7개 source variant와 분리 진단을 여러 독립 구조로 세지 않는다. 0 입력의 raw
restoration scale은 CPU subnormal 대신 0으로 나오는 차이가 있어 진단 전체를
PASS로 처리하지 않았으며, 일반 BF16 입력 동치나 확대 검증은 주장하지 않는다.

AGENTS.md에는 수정 가능한 source/layout 오류를 compile 횟수만으로 포기하지
않도록 보완했다. C14의 기존 padding 전송 실패도 불가능성이나 성능 탈락의 근거는
아니다. [C14 후속 검토](../../moa-dev/experiments/candidates/c13-int8-ffn/c14-assessment.md)의
stage 내부 compact 경로는 아직 미구현이다. C12 역시 미구현이며 기존 제안 범위는
국소 Main/Sub dataflow 변경이다.

## 이전 작업 — C14-A

C14 작업 브랜치는 `opt/c14-ffn-l-pipeline`이다. **C14-A는 producer→down DMA의
layout lowering 실패로 종료**했다. [구현·세 버전·종료 근거](../../moa-dev/experiments/candidates/c14-ffn-l-pipeline/README.md).

E74 `9e2058ec0947498d7cb693fbe98e7b5f0b5c9ea7`에서 출발해 BF16 stream 계산,
분산 GeGLU, L-owned down cluster, 두 L 구간의 F32 partial 합산을 한 경로로
구현했다. E104는 비교 계획에만 사용했다. 기존 전체-stage 대기와 H-owned down
배치를 함께 변경한 whole-FFN 재설계 시도이며, 완성된 image는 생성되지 않았다.

Compile 3회: 최초 tiled-view reshape 실패 → HBM 축 분할 순서 수정 후 factorize
실패 → 별도 cluster 축 표현으로 수정 후 padded producer→down column DMA 실패.
기본 수정 횟수를 한 번 넘긴 구체적 근거와 총 3회 제한은 v3 recipe에 사전 기록했다.
논리 좌표/pitch/GeGLU 연산 순서 검사는 통과했지만 NPU 정확도·성능은 미평가다.
Arena·공식 제출 없음, E104 실행 소스 유지. C14 전체가 느리다는 판정은 아니다.
당시 다음 우선순위였던 C13의 실행 결과는 위 최신 작업에 기록했다. C12는 미구현이다.

## 이전 작업 — C09/C11/C10

사용자가 승인한 실행 순서는 **C09 → C11 → C10**이다.
[설계·과거 시도·종료 기준](../research/structural-candidates-20260923.md).

| 후보 | 브랜치 | 상태 |
|---|---|---|
| C09 | `opt/c09-output-pipeline` | [종료](../../moa-dev/experiments/candidates/c09-output-pipeline/README.md): compile PASS, 정적 +0.71%, Arena 미실행 |
| C11 | `opt/c11-ffn-partial-vrf` | [종료](../../moa-dev/experiments/candidates/c11-ffn-partial-vrf/README.md): compile PASS, 정적 +3.55%, Arena 미실행 |
| C10 | `opt/c10-direct-weight-trf` | [종료](../../moa-dev/experiments/candidates/c10-direct-weight-trf/README.md): StoTrf lowering에서 현재 target 설정 제약으로 compile 실패 |

각 후보는 E104 소스에서 별도로 출발한다. 다음 브랜치는 이전 실험의 기록을 이어받으며,
채택하지 않은 kernel 구현은 후보 snapshot에 보존한다. Main 병합·공식 제출 상태와
Arena 실험 상태를 구분한다.

이번 요청의 세 후보는 각각 **최초 compile 1회**로 종료했다. C09/C11은 E104 control도
각 1회 compile했으며 exact image hash가 채택본과 동일했다. Arena 실행·공식 제출은
없다. 성능표의 +0.71%/+3.55%는 compiler schedule이며 NPU cycle 변화가 아니다.

세 실험의 마지막 브랜치는 `opt/c10-direct-weight-trf`이며 C09 → C11 → C10 순서의 기록을
모두 포함한다. 세 후보 브랜치는 private origin에 각각 push했다. E104 실행 소스는
바꾸지 않았고 `experiment_index.py --check --check-live`로 확인한다.
당시 후속 미구현 후보는 C12(Main/Sub 분리), C13(INT8 DPE), C14(FFN 전체 구간
pipeline)였다. C14-A의 후속 실행 결과는 위 최신 작업에 기록했다.

## 사용자 지적 이후 변경 범위 재검토

실제 candidate diff와 호출 경로를 다시 읽었다. C09/C11을 C10과 묶어서 세 개의 큰
아키텍처를 충분히 시험한 것처럼 설명한 것은 범위를 과장했다. 이번 실험의 정확한
분류는 다음과 같다. 기존 소스·빌드·판정 기록은 그대로 보존한다.

| 후보 | 그대로 남긴 구조 | 실제 변경 범위 |
|---|---|---|
| C09 | 두 cluster·16-column BF16 dot, F32 합산·scale, 후단 RMS/residual | `project_output`의 행별 전송/계산 의존성을 두 구간으로 분리한 국소 dataflow 변경 |
| C11 | E5M2 세 항 입력, FP4→FP8 weight DM, up/gate→GeGLU→down 전체 단계 의존성, down 구현 | `project_rows_with_scale_contraction`의 partial DM을 VRF로 바꾼 국소 dataflow 변경 |
| C10 | Q/K/V 개별 projection, 입력 전체 복제, 기존 head 후처리 | Weight/activation의 TRF·stream 역할과 row/lane 배치를 교체한 projection 재설계 시도. Lowering 실패로 성능 미평가 |

C07 native FP8/scale 계산 조합과 C08 QKV 통합·열 분할은 더 큰 변경을 실제로
시도한 이력이 있다. 그러나 build·parameter 변형 수를 독립 아키텍처 탐색 수로
해석할 수 없다. 이력만으로 현재 구조가 최적에 가깝다거나 상위권의 구현 방식을
안다고 결론 내리지 않는다.

최근 결과가 주는 정보는 제한적이다. C09는 추가 준비/전송이 overlap을 상쇄했고,
C11은 dot 2→30개로 실행 비용이 늘었다. 이들은 그 구현의 compiler schedule 관측이다.
NPU에서 해당 비율로 느려진다고 확정하거나 전체 설계 계열을 폐기하는 근거가 아니다.

다음 큰 설계는 전체 projection 또는 FFN 실행 graph를 먼저 쓰고, 상속한 데이터
소유권·weight 중간 저장·scale 합산·stage 완료 대기 중 무엇을 없앨지 명시한다.
기존 함수 내부 한 곳의 수정으로 설계 범위를 축소하지 않는다. 작은 지원 여부 검사는
그 큰 설계의 일부를 검증하는 용도로 사용한다. 빠른 종료 규칙은 유지한다.
C12도 현재 제안 범위는 국소 자원 배치 변경이므로 자동으로 다음 대형 설계로 세지 않는다.
C14의 FFN 단계 간 의존성 재설계와 C13의 계산 표현 교체처럼 넓은 가설을 우선 검토하되,
실제 개선 여부는 아직 미검증이다. 이번 재검토에서 새 kernel을 구현·실행하지 않았다.

## 소스 선택의 E104 고정 제거

소스 선택 수정 브랜치는 `fix/recipe-driven-experiments`다. 후속 지적을 받고 실제 빌더를
확인한 결과, `structural_build.py`는 JSON의 `base_commit`을 사용하지 않고 E104 상수를
`common.BASE`에 대입하고 있었다. 변경하지 않은 image의 검사도 E104로 고정돼 있었다.
이 고정은 내가 만든 구현 선택이며, 기록만으로 조합 탐색이 수행됐다고 볼 수 없다.

빌더가 recipe의 출발 commit과 variant별 파일 선택을 실제로 사용하도록 수정했다.
성능 비교군은 `comparison_recipe`로 분리했다. 과거 build를 image 기준으로 쓸 때는
선택한 출발 source와 일치하는지 검사한다. 같은 destination의 중복 덮어쓰기는 거부한다.
[실제 입력 형식과 조립 범위](combination-template.md)를 따른다.

검증: E55/E79/E104 source 선택과 E104 비교군의 독립성, variant별 source 선택,
기준 image의 잘못된 승계 거부, 중복 destination 거부, 기존 세 recipe 호환성,
실제 Git archive→변경 파일 복사 결과까지 **6개 테스트 PASS**다. 실제 조립 검사는
첫 Docker 호출 직전에 멈추도록 만든 테스트다. 새 compiler/NPU/공식 제출 실행은 없다.
기존 실험의 frozen source/log/build JSON과 채택 E104는 변경하지 않았다.

이 수정은 선택한 조합을 빌더가 따르게 만든 것이다. 과거 method ID를 자동으로
서로 호환되는 코드로 합치는 기능은 아니다. 다음 실험은 출발 구조·과거 방법·제거할
충돌 요소의 조합부터 정하고, 그 조합을 담은 실제 코드와 JSON이 일치하는지 확인한다.
# 2026-09-27 실행 후속: E104 QKV/O 실측

[D-E104-QO](../../moa-dev/experiments/candidates/e104-qkv-output-profile/README.md)는
동결 E104의 task bytes를 유지한 내부 계측이다. Arena104914/104933의108개 비교가
통과했고76개 QKV/22개 O engine을 source와 대조했다. O dot은 weight 준비 직후,
Q dot은 weight와 table 준비 직후 시작한다. Q 입력 준비 뒤20,118 / O 입력 준비
뒤20,955 cycles 대기가 관측된다. 입력 방송 제거 시간을 전체 절감으로 계산하지
않는다. K/V의 Main 순서와 Q table 준비도 다음 설계에 포함한다.

[M06 timeout 감사](../research/m06-timeout-mechanism-20260927.md)는 generic code4,
공통 global-Max/scalar-broadcast 경계까지 특정했다. Bank starvation을 입증할
장치 증거가 없어 특정 수리 없이 성능 실험을 재개하지 않는다.

# C12-B: O projection의 Main dot / Sub 합산

`opt/c12-output-main-sub`. C09의 두 weight 구간을 가져와 Main dot producer와
Sub F32 column reducer를 분리했다. 비교 source origin은E104, 실제 projection
조합은C09 snapshot을 수정한 [v2](two-stage-separate-pack-v2/projection.rs)다.
단순 engine 이름 교체나 새 산술 모델이 아니라 producer/transfer/consumer 경로
변경이다. FFN 전체 재설계로 세지 않는다.

**전체 compile·기본 NPU 정확도 PASS, 관측 O cycles +2.46%로 미채택.**
Job79329: E10450555 → C12 51862 → E10450680 cycles(각 process 3입력 중앙값).
앞뒤 비교군 중앙값50617.5 대비다. 각 process15비교와 후보 O3입력11520출력이
기존 기준 그대로 PASS다. 단회 sample 범위는 겹치므로 통계적으로 확정된 보편적
회귀라고 주장하지 않는다. 5% 이상 개선 근거가 없어 반복·추가 튜닝 없이 닫았다.
[실측](../../../../docs/experiments/e111-c12-output-main-sub.json), [판정](decision.json).

## 전체 경로

`ops::sliding_attention_output → projection::project_output →
project_output_partial → reduce_output_partial → apply_output_channel_scale_f32
→ post RMS → add_output_residual`.

2 cluster×16 row groups×16 input columns의BF16 dot이다. 각 weight 구간은
60 output rows×256 inputs/slice. 독립 두 DM allocation에 HBM FP8 weight를 읽고
Main에서 FP8→BF16 lookup→dot→F32 scalar padded packet을 바로 DM에 저장한다.
Producer에는 vector reduction·cast·transpose가 없다. 각 partial은60×1#8 F32,
1920 physical bytes/slice(유효240B)다. Sub는16 column의 F32 sum을 실행하고,
v2는 결과를 padded DM에 저장한 뒤 짧은 Main transpose로 compact하게 묶는다.
Canonical output DMA→F32 channel scale→첫 BF16 변환과 기존 epilogue를 유지했다.

## 컴파일 오류를 분리한 과정

[v1](two-stage-padded-v1/projection.rs)은 Sub 합산→transpose를 같은 명령으로
연결했지만 전체 graph의 SFR lowering에서`no entry found for key`로 실패했다.
예전 C05와 같은 오류 위치만으로 Sub 경로 불가능을 단정하지 않았다.

[분리 진단](support-diagnostic.json)은 같은60행·16-column·padded F32 배치를 썼다.
Main 합산, Sub 합산, Sub transpose, Sub 합산+transpose 네 경로 **모두 compile PASS**다.
따라서 오류는 해당 연산 자체의 일반적 미지원으로 좁혀지지 않았다. 전체 graph의
혼합 context/할당 조합에 관련된 것으로 추정하며 정확한 내부 원인은 미확정이다.
[v2 recipe](two-stage-separate-pack-v2/recipe.json)의 DM 경계와 Main packing으로
전체 compile 및 NPU 실행을 성공시켰다. 수정 전 snapshot·실패 로그도 보존했다.
Primitive 검사는 실행 지원의 compile 증거일 뿐 NPU 정확도나 성능 검사가 아니다.

SDK0.8.1의 generic-Tu API와 공식 [execution context](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/index.html),
[commit adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/commit-adapter.html),
[transpose](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/transpose-engine.html),
[scheduling](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html)를 대조했다.
Frontend signature만으로 실제 mixed-context NPU lowering 지원을 확정하지 않는다.

## 왜 중첩 이득이 생기지 않았나

[정적 일정](static-cost.json): E10427239 → C12 28404(+4.28%).

| 구간 | 시작 | 끝 |
|---|---:|---:|
| 첫 Main dot |9740|12403|
| 두 번째 weight DMA |10581|17547|
| 첫 Sub column sum |12403|13047|
| 첫 Main packing |13047|13372|
| 두 번째 Main dot |17547|20210|

첫 dot과 두 번째 DMA는1822 modeled cycles 겹친다. 그러나 첫 Sub 합산 종료 후에도
다음 weight가4500 cycles 더 늦게 도착한다. 그래서 목표였던 **Sub 합산과 다음 Main
계산의 중첩은0**이다. Producer의Main-only context 분리는 실제 일정에 나타났지만,
더 이른 operand 준비 없이 합산만 분리한 이 조합에는 이득이 없었다. 두 lookup setup,
padded partial write/read, 합산 후 추가 DM 경계와 packing 비용도 포함한 결과다.
Interval 길이를 더해 NPU stall이나 성능 기여율로 환산하지 않는다.

기본 screen 이후 미채택·공식 제출 없음. QKV/FFN image는E104와 같다. 원본 입력이나
평가 허용 오차는 수정하지 않았다. 다음 재시도는 다음 구간 weight 준비를 앞당기거나
partial 경계 비용을 줄이는 구체적인 조합이 있을 때만 한다.

독립 읽기 전용 감사에서 새로운 signature·좌표 매핑 오류를 찾지 못했다. 다만 **이 후보 전체를 적용하면 E53의 BF16 O 수정이 사라진다.** [projection.rs](projection.rs)의 `project_output`부터 파일 끝까지는 경계 probe에 실패한 E47 native O와 byte-identical이다. 향후 통합 시 QKV prefix와 [ops.rs](ops.rs)의 QKV 입력/projection body만 E53에 옮기고 O image가 유지되는지 다시 검사해야 한다.

검토한 동결 파일은 projection SHA `8b68ffa018ce4bf9a06b1bee1772ccb837f9df9ef0883d11285fb6d7d9151662`, ops SHA `ffdab428fd4c8126b77e145b75073c907732bf0fa9ca830a123fe828e7b08888`이다. [manifest](manifest.json)의 네 파일 해시를 실제 파일과 대조했다. 보고된 exact compile 52,282와 image `ae9b778d…`는 확인했지만 이 감사에서 새 compile·NPU 실행은 하지 않았다.

- `project_query`부터 `project_key_value` 끝까지는 동결 E18 helper와 byte-identical이다. 입력만 E47 `normalize_sharded_inter_slice`에서 받아 E18의 ring32 BF16 broadcast에 연결한다. Ops 공개 함수 10개 signature는 E18/E47과 같고, 수정한 QKV 입력/projection body 밖의 ops 텍스트도 E47과 같다. QKV head RMS·RoPE·cache write suffix는 그대로다.
- 입력의 480값×8 shard는 각 cluster에 전체 H3840을 나타낸다. Typed DMA가 각 32-slice ring의 leader에 전체 H를 모으고 `CustomBroadcast { ring_size:32 }`가 실제 BF16 데이터를 32개 slice로 복제한다. 그 뒤 `[Dummy256/32,32]→Replicated` reshape는 채워진 256개 slice의 이름만 바꾼다. Q/KV view에도 동일한 전체 H가 있다.
- Q row는 `q=2048*c+64*g+8*r+s`이고 slice=`8*g+s`, local=`r`이다. Switch 후 slice=`8*g+r`, local=`s`로 복원된다. Q4096개 전부에 대해 역변환과 `(Ns,Gs,Ds)` head mapping을 검사했다. KV는 `p=1024*c+32*g+8*r+s`, `r<4`다. Switch의 `4#8` padding은 끝까지 유효행4개로 제한되며, head reshape에서도 `Ds/8%4#8`을 유지한다. KV2048개 전부의 좌표와 typed head gather를 확인했다. Padding을 유효값으로 둔갑시키는 reshape는 발견하지 못했다.
- 각 scale은 Switch 뒤의 동일 row mapping으로 불러온다. V도 K와 동일 BF16 helper를 사용하지만 서로 다른 weight/scale 인자를 받는다. 이후 V RMSNorm은 유지되며 V에는 RoPE가 없다. K/V scatter는 원래 `kv_offset`을 그대로 사용한다. 원래 harness는 zero cache의 target slot만 비교하므로 이 새 candidate의 non-target cache 보존을 동적으로 검사했다고 주장하지 않는다.

정밀도 한계는 남아 있다. BF16 activation 자체를 TRF에 보존하고 FP8 weight는 BF16 lookup으로 확장하므로 hi/lo 입력 양자화 손실은 제거된다. 그러나 projection 36/101행의 raw-dot BF16 cast가 channel scale보다 앞에 있다. 공식 `scaled_projection`은 F32 dot에 scale을 곱한 뒤 BF16으로 변환한다. 따라서 기존 E18의 이른 반올림·F32 누적 순서 차이는 그대로이며, head normalization이 이를 항상 지운다는 보장도 없다. Pre/head RMS의 sqrt/div와 CPU rsqrt 차이, overflow/underflow·FTZ도 전 입력 동등성의 남은 제약이다.

E18의 과거 실측 PASS는 helper 재사용의 근거다. 이 새 E47 sharded-RMS 조합 자체의 NPU PASS나 경계값 안전성을 대신하지 않는다. 공식 QKV 입력은 `signs/input_rms_weight`로 구성되어 일반적인 activation 동적 범위 검증에도 한계가 있다. 채택한다면 공식·extra 입력과 독립 QKV 경계 입력 검증이 필요하다. 이번 작업은 source·live target·Git·NPU를 변경하지 않았다.

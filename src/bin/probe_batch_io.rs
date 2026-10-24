//! Diagnostic-only exact comparison of B16 embedding against 16 scalar launches.
use std::time::Instant;

use furiosa_opt_gemma4::axes::{B, H, W};
use furiosa_opt_gemma4::{Chip, ops, ops_batch_io};
use furiosa_opt_std::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokens: [usize; 16] = [0, 1, 7, 255, 256, 511, 512, 4095,
                              4096, 8191, 8192, 16383, 16384, W::SIZE - 2, W::SIZE - 1, 7];
    assert_eq!(B::SIZE, tokens.len());
    let mut device = Device::new(ops_batch_io::embed_tokens.topology())?;
    let mut bytes = vec![0u8; W::SIZE * H::SIZE * 2];
    for &token in &tokens {
        for h in 0..H::SIZE {
            let value = (((token % 127) * 17 + h * 13) % 1009) as f32 / 256.0 - 2.0;
            let offset = (token * H::SIZE + h) * 2;
            bytes[offset..offset + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    let table: HbmTensor<bf16, Chip, m![W, H]> = HostTensor::<bf16, m![W, H]>::from_buf(bytes).to_hbm(&mut device.pdma).await?;
    let mut batch: HbmTensor<bf16, Chip, m![B, H]> = HostTensor::<bf16, m![B, H]>::zero().to_hbm(&mut device.pdma).await?;
    let mut single: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    for permutation in 0..2 {
        let ordered: Vec<_> = if permutation == 0 { tokens.to_vec() } else { tokens.into_iter().rev().collect() };
        let offsets: HbmTensor<i32, Chip, m![B]> = HostTensor::<i32, m![B]>::from_vec(
            ordered.iter().map(|&token| (token * H::SIZE * 2) as i32)
        ).to_hbm(&mut device.pdma).await?;
        let started = Instant::now();
        launch(ops_batch_io::embed_tokens, (&mut device, &table, &offsets, &mut batch)).await?;
        let batch_us = started.elapsed().as_micros();
        let got = batch.to_host::<m![B, H]>(&mut device.pdma).await?.into_vec();
        let mut mismatches = 0usize;
        let mut scalar_us = 0u128;
        for (row, &token) in ordered.iter().enumerate() {
            let offset: HbmTensor<i32, Chip, m![1]> = HostTensor::<i32, m![1]>::from_vec([(token * H::SIZE * 2) as i32])
                .to_hbm(&mut device.pdma).await?;
            let started = Instant::now();
            launch(ops::embed_token, (&mut device, &table, &offset, &mut single)).await?;
            scalar_us += started.elapsed().as_micros();
            let expected = single.to_host::<m![H]>(&mut device.pdma).await?.into_vec();
            for h in 0..H::SIZE {
                if got[row * H::SIZE + h].to_bits() != expected[h].to_bits() {
                    if mismatches < 4 {
                        eprintln!("mismatch permutation={permutation} row={row} token={token} h={h} got={} expected={}",
                                  got[row * H::SIZE + h].to_f32(), expected[h].to_f32());
                    }
                    mismatches += 1;
                }
            }
        }
        println!("BATCH_EMBED_RESULT {}", serde_json::json!({
            "permutation": permutation, "values": B::SIZE * H::SIZE,
            "mismatches": mismatches, "batch_launch_us": batch_us,
            "scalar_launches_us": scalar_us,
            "timing_scope": "host launch/wait only; diagnostic, not E2E score"
        }));
        if mismatches != 0 { return Err("batched embedding differs from scalar reference".into()); }
    }
    Ok(())
}

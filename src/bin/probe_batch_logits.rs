//! Compare every B16 vocabulary logit with the original scalar head.
//! Unit-magnitude hidden values and unit RMS weights round back to the same
//! BF16 hidden values after normalization, so both heads receive identical x.
use std::time::Instant;

use furiosa_opt_gemma4::{axes::{B, H, W}, Chip, ops, ops_batch_io};
use furiosa_opt_std::prelude::*;

fn hidden(b: usize, h: usize) -> bf16 {
    bf16::from_f32(if (b * 7 + h * 13) % 31 < 15 { -1.0 } else { 1.0 })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::new(ops_batch_io::logits_batch.topology())?;
    let mut patterns = vec![vec![0u8; H::SIZE * 2]; 17];
    for (n, row) in patterns.iter_mut().enumerate() {
        for h in 0..H::SIZE {
            let w = bf16::from_f32(((n * 5 + h * 7) % 17) as f32 / 16.0 - 0.5);
            row[h*2..h*2+2].copy_from_slice(&w.to_bits().to_le_bytes());
        }
    }
    let mut bytes = vec![0u8; W::SIZE * H::SIZE * 2];
    for (n, row) in bytes.chunks_exact_mut(H::SIZE * 2).enumerate() {
        row.copy_from_slice(&patterns[n % 17]);
    }
    let weight: HbmTensor<bf16, Chip, m![W,H]> = HostTensor::<bf16,m![W,H]>::from_buf(bytes)
        .to_hbm(&mut device.pdma).await?;
    let input: HbmTensor<bf16, Chip, m![B,H]> = HostTensor::<bf16,m![B,H]>::from_vec(
        (0..B::SIZE).flat_map(|b| (0..H::SIZE).map(move |h| hidden(b,h)))
    ).to_hbm(&mut device.pdma).await?;
    let rms: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16,m![H]>::from_vec(
        (0..H::SIZE).map(|_| bf16::from_f32(1.0))
    ).to_hbm(&mut device.pdma).await?;
    let mut batch: HbmTensor<bf16, Chip, m![B,W]> = HostTensor::<bf16,m![B,W]>::zero()
        .to_hbm(&mut device.pdma).await?;
    let mut scalar: HbmTensor<bf16, Chip, m![W]> = HostTensor::<bf16,m![W]>::zero()
        .to_hbm(&mut device.pdma).await?;
    let start = Instant::now();
    launch(ops_batch_io::logits_batch, (&mut device, &input, &weight, &mut batch)).await?;
    let cold_us = start.elapsed().as_micros();
    let start = Instant::now();
    launch(ops_batch_io::logits_batch, (&mut device, &input, &weight, &mut batch)).await?;
    let warm_us = start.elapsed().as_micros();
    let got = batch.to_host::<m![B,W]>(&mut device.pdma).await?.into_vec();
    let mut mismatches = 0usize;
    let mut scalar_us = 0u128;
    for b in 0..B::SIZE {
        let offset: HbmTensor<i32, Chip, m![1]> = HostTensor::<i32,m![1]>::from_vec(
            [(b * W::SIZE * 2) as i32]
        ).to_hbm(&mut device.pdma).await?;
        launch(ops_batch_io::extract_logits_row, (&mut device, &batch, &offset, &mut scalar)).await?;
        let extracted = scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
        for n in 0..W::SIZE {
            if extracted[n].to_bits() != got[b*W::SIZE+n].to_bits() {
                return Err(format!("logit row extraction mismatch at {b},{n}").into());
            }
        }
        let x: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16,m![H]>::from_vec(
            (0..H::SIZE).map(|h| hidden(b,h))
        ).to_hbm(&mut device.pdma).await?;
        let start = Instant::now();
        launch(ops::final_norm_and_logits, (&mut device, &x, &rms, &weight, &mut scalar)).await?;
        scalar_us += start.elapsed().as_micros();
        let expected = scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
        for n in 0..W::SIZE {
            let actual = got[b*W::SIZE+n];
            if actual.to_bits() != expected[n].to_bits() {
                if mismatches < 8 { eprintln!("row={b} col={n} got={} expected={}", actual.to_f32(), expected[n].to_f32()); }
                mismatches += 1;
            }
        }
    }
    println!("BATCH_LOGITS_RESULT {}", serde_json::json!({
        "values": B::SIZE*W::SIZE, "mismatches": mismatches,
        "batch_cold_us": cold_us, "batch_warm_us": warm_us,
        "scalar16_launch_us": scalar_us,
        "timing_scope": "host launch/wait only; scalar includes normalization; not E2E"
    }));
    if mismatches != 0 { return Err("B16 logits differ from scalar head".into()); }
    Ok(())
}

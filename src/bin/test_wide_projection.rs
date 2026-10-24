//! Independent arithmetic oracle for the B16 projection primitive.
//! Dyadic inputs make the F32 dot exact; the check includes both BF16 boundaries.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, P128, Qs},
    ops_batch_wide,
};
use furiosa_opt_std::prelude::*;

fn input(b: usize, k: usize) -> f32 {
    ((b * 17 + k * 13) % 31) as f32 / 16.0 - 15.0 / 16.0
}
fn weight(n: usize, k: usize) -> f32 {
    ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0
}
fn scale(n: usize) -> f32 {
    2f32.powi((n % 7) as i32 - 3)
}

#[tokio::main]
async fn main() {
    let mut device = Device::new(ops_batch_wide::project_query_batch.topology()).unwrap();
    let x: HbmTensor<bf16, Chip, m![P128, H]> = HostTensor::<bf16, m![P128, H]>::from_vec(
        (0..P128::SIZE).flat_map(|b| (0..H::SIZE).map(move |k| bf16::from_f32(input(b, k)))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let w: HbmTensor<f8e4m3, Chip, m![Qs, H]> = HostTensor::<f8e4m3, m![Qs, H]>::from_vec(
        (0..Qs::SIZE).flat_map(|n| (0..H::SIZE).map(move |k| f8e4m3::from_f32(weight(n, k)))),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let s: HbmTensor<bf16, Chip, m![Qs]> =
        HostTensor::<bf16, m![Qs]>::from_vec((0..Qs::SIZE).map(|n| bf16::from_f32(scale(n))))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let mut out: HbmTensor<bf16, Chip, m![P128, Qs]> = HostTensor::<bf16, m![P128, Qs]>::from_vec(
        (0..Qs::SIZE * P128::SIZE).map(|_| bf16::from_f32(f32::NAN)),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let start = std::time::Instant::now();
    launch(
        ops_batch_wide::project_query_batch,
        (&mut device, &x, &w, &s, &mut out),
    )
    .await
    .unwrap();
    let host: HostTensor<bf16, m![P128, Qs]> = out.to_host(&mut device.pdma).await.unwrap();
    let elapsed = start.elapsed();
    let actual = host.into_vec();
    let mut dot = vec![0.0f32; P128::SIZE * 17];
    for b in 0..P128::SIZE {
        for n in 0..17 {
            dot[b * 17 + n] =
                bf16::from_f32((0..H::SIZE).map(|k| input(b, k) * weight(n, k)).sum()).to_f32();
        }
    }
    let mut failures = 0;
    for b in 0..P128::SIZE {
        for n in 0..Qs::SIZE {
            let expected = bf16::from_f32(dot[b * 17 + n % 17] * scale(n));
            let got = actual[b * Qs::SIZE + n];
            if got.to_bits() != expected.to_bits() {
                if failures < 10 {
                    eprintln!(
                        "row={b} col={n} got={} expected={}",
                        got.to_f32(),
                        expected.to_f32()
                    );
                }
                failures += 1;
            }
        }
    }
    println!(
        "projection P128={} H={} Qs={} checked={} mismatches={} launch_and_download_ms={:.3}",
        P128::SIZE,
        H::SIZE,
        Qs::SIZE,
        actual.len(),
        failures,
        elapsed.as_secs_f64() * 1000.0
    );
    let start = std::time::Instant::now();
    for _ in 0..5 {
        launch(
            ops_batch_wide::project_query_batch,
            (&mut device, &x, &w, &s, &mut out),
        )
        .await
        .unwrap();
    }
    let _: HostTensor<bf16, m![P128, Qs]> = out.to_host(&mut device.pdma).await.unwrap();
    println!(
        "P128 query warm5launches1read_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0 / 5.0
    );
    assert_eq!(failures, 0);
}

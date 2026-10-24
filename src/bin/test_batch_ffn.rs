//! Hardware differential test: actual B16 FFN versus the unmodified serial core.
use furiosa_opt_gemma4::{
    Chip,
    axes::{B, H, L},
    ops_batch,
};
use furiosa_opt_std::prelude::*;

fn word(i: usize, seed: u64) -> u64 {
    let mut x = (i as u64).wrapping_add(seed);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
async fn upload<D: MaterializableScalar, E: M>(
    device: &mut Device,
    bytes: Vec<u8>,
) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(bytes)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}
async fn packed<E: M>(device: &mut Device, seed: u64) -> HbmTensor<f4e2m1, Chip, E> {
    upload(
        device,
        (0..E::SIZE / 2).map(|i| word(i, seed) as u8).collect(),
    )
    .await
}
async fn scales<E: M>(device: &mut Device, seed: u64) -> HbmTensor<f8e4m3, Chip, E> {
    HostTensor::<f8e4m3, E>::from_vec(
        (0..E::SIZE).map(|i| f8e4m3::from_f32(((word(i, seed) % 8) + 1) as f32 / 128.0)),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap()
}
async fn global(device: &mut Device, v: f32) -> HbmTensor<f32, Chip, m![1]> {
    HostTensor::<f32, m![1]>::from_vec([v])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}
#[tokio::main]
async fn main() {
    let mut device = Device::new(ops_batch::batch_ffn.topology()).unwrap();
    let values: Vec<bf16> = (0..B::SIZE * H::SIZE)
        .map(|i| bf16::from_f32((word(i, 11) % 257) as f32 / 256.0 - 0.5))
        .collect();
    let input: HbmTensor<bf16, Chip, m![B, H]> =
        HostTensor::<bf16, m![B, H]>::from_vec(values.iter().copied())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
    let up: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 13).await;
    let gate: HbmTensor<f4e2m1, Chip, m![L, H]> = packed(&mut device, 17).await;
    let down: HbmTensor<f4e2m1, Chip, m![H, L]> = packed(&mut device, 19).await;
    let us: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 23).await;
    let gs: HbmTensor<f8e4m3, Chip, m![L, H / 16]> = scales(&mut device, 29).await;
    let ds: HbmTensor<f8e4m3, Chip, m![H, L / 16]> = scales(&mut device, 31).await;
    let ug = global(&mut device, 0.125).await;
    let gg = global(&mut device, 0.125).await;
    let dg = global(&mut device, 0.125).await;
    let mut output: HbmTensor<bf16, Chip, m![B, H]> =
        upload(&mut device, vec![0; B::SIZE * H::SIZE * 2]).await;
    let start = std::time::Instant::now();
    launch(
        ops_batch::batch_ffn,
        (
            &mut device,
            &input,
            &up,
            &gate,
            &down,
            &us,
            &gs,
            &ds,
            &ug,
            &gg,
            &dg,
            &mut output,
        ),
    )
    .await
    .unwrap();
    let actual: HostTensor<bf16, m![B, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let actual = actual.into_vec();
    let mut serial_inputs = Vec::new();
    for row in 0..B::SIZE {
        let x: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::from_vec(
            values[row * H::SIZE..(row + 1) * H::SIZE].iter().copied(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        serial_inputs.push(x);
    }
    let mut serial_output: HbmTensor<bf16, Chip, m![H]> =
        upload(&mut device, vec![0; H::SIZE * 2]).await;
    let mut failures = 0;
    let mut bit_differences = 0;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let start = std::time::Instant::now();
    for (row, x) in serial_inputs.iter().enumerate() {
        launch(
            ops_batch::serial_ffn_control,
            (
                &mut device,
                x,
                &up,
                &gate,
                &down,
                &us,
                &gs,
                &ds,
                &ug,
                &gg,
                &dg,
                &mut serial_output,
            ),
        )
        .await
        .unwrap();
        let reference: HostTensor<bf16, m![H]> =
            serial_output.to_host(&mut device.pdma).await.unwrap();
        for (col, expected) in reference.into_vec().iter().enumerate() {
            let got = actual[row * H::SIZE + col];
            if got.to_bits() != expected.to_bits() {
                bit_differences += 1;
            }
            let expected = expected.to_f32();
            let got = got.to_f32();
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            max_rel = max_rel.max(error / expected.abs().max(0.01));
            if !got.is_finite() || error > 0.02 + 0.01 * expected.abs() {
                if failures < 10 {
                    eprintln!("row={row} col={col} got={got} expected={expected}");
                }
                failures += 1;
            }
        }
    }
    let serial_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = std::time::Instant::now();
    for _ in 0..5 {
        launch(
            ops_batch::batch_ffn,
            (
                &mut device,
                &input,
                &up,
                &gate,
                &down,
                &us,
                &gs,
                &ds,
                &ug,
                &gg,
                &dg,
                &mut output,
            ),
        )
        .await
        .unwrap();
    }
    let _: HostTensor<bf16, m![B, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let warm_ms = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
    println!(
        "FFN B={} H={} L={} checked={} failures={} bit_differences={} max_abs={} max_rel={} batch_cold_ms={:.3} batch_warm_launch_ms={:.3} serial16_launch_and_read_ms={:.3}",
        B::SIZE,
        H::SIZE,
        L::SIZE,
        actual.len(),
        failures,
        bit_differences,
        max_abs,
        max_rel,
        cold_ms,
        warm_ms,
        serial_ms
    );
    assert_eq!(failures, 0);
}

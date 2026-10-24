//! H960 FFN differential test; F32 reassociation may change BF16 rounding.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P128},
    ops_batch_kpart, ops_batch_wide,
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
    let mut device = Device::new(ops_batch_wide::batch_ffn.topology()).unwrap();
    let values: Vec<bf16> = (0..P128::SIZE * H::SIZE)
        .map(|i| {
            bf16::from_f32(
                (((word(i, 11) >> 40) as f32 / 16777216.0) * 2.0 - 1.0)
                    * 4.0
                    * 2f32.powi(-((word(i, 13) % 9) as i32)),
            )
        })
        .collect();
    let input: HbmTensor<bf16, Chip, m![P128, H]> =
        HostTensor::<bf16, m![P128, H]>::from_vec(values.iter().copied())
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
    let mut output: HbmTensor<bf16, Chip, m![P128, H]> =
        upload(&mut device, vec![0; P128::SIZE * H::SIZE * 2]).await;
    let start = std::time::Instant::now();
    launch(
        ops_batch_wide::batch_ffn,
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
    let actual: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    let actual = actual.into_vec();
    launch(
        ops_batch_kpart::batch_ffn,
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
    let changed: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let mut bitdiff = 0;
    let mut failures = 0;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (i, (expected, got)) in actual.iter().zip(changed.into_vec()).enumerate() {
        if expected.to_bits() != got.to_bits() {
            bitdiff += 1;
        }
        let e = expected.to_f32();
        let a = got.to_f32();
        let error = (e - a).abs();
        max_abs = max_abs.max(error);
        max_rel = max_rel.max(error / e.abs().max(0.01));
        if !a.is_finite() || error > 0.02 + 0.01 * e.abs() {
            if failures < 8 {
                eprintln!("i={i} got={a} expected={e}");
            }
            failures += 1;
        }
    }
    println!(
        "H960FFN_v_control checked={} bitdiff={bitdiff} failures={failures} max_abs={max_abs} max_rel={max_rel} cold_control_ms={cold_ms}",
        actual.len()
    );
    let mut hidden: HbmTensor<bf16, Chip, m![L, P128]> =
        upload(&mut device, vec![0; L::SIZE * P128::SIZE * 2]).await;
    launch(
        ops_batch_kpart::probe_upgate,
        (
            &mut device,
            &input,
            &up,
            &gate,
            &us,
            &gs,
            &ug,
            &gg,
            &mut hidden,
        ),
    )
    .await
    .unwrap();
    launch(
        ops_batch_kpart::probe_down,
        (&mut device, &hidden, &down, &ds, &dg, &mut output),
    )
    .await
    .unwrap();
    let split: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
    launch(
        ops_batch_kpart::batch_ffn,
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
    let fused: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
    let split_diff = split
        .into_vec()
        .iter()
        .zip(fused.into_vec())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "split_stage_vs_fused checked={} bit_differences={split_diff}",
        P128::SIZE * H::SIZE
    );
    assert_eq!(split_diff, 0);
    let mut times = [Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new()];
    for variant in [0, 1, 2, 2, 1, 0] {
        let start = std::time::Instant::now();
        for _ in 0..5 {
            if variant == 0 {
                launch(
                    ops_batch_kpart::batch_ffn,
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
            if variant == 1 {
                launch(
                    ops_batch_kpart::probe_upgate,
                    (
                        &mut device,
                        &input,
                        &up,
                        &gate,
                        &us,
                        &gs,
                        &ug,
                        &gg,
                        &mut hidden,
                    ),
                )
                .await
                .unwrap();
            }
            if variant == 2 {
                launch(
                    ops_batch_kpart::probe_down,
                    (&mut device, &hidden, &down, &ds, &dg, &mut output),
                )
                .await
                .unwrap();
            }
        }
        if variant == 1 {
            let _: HostTensor<bf16, m![L, P128]> = hidden.to_host(&mut device.pdma).await.unwrap();
        } else {
            let _: HostTensor<bf16, m![P128, H]> = output.to_host(&mut device.pdma).await.unwrap();
        }
        times[variant].push(start.elapsed().as_secs_f64() * 1000.0 / 5.0);
    }
    println!(
        "P128H960FFN stages fused_ms={:?} upgate_geglu_ms={:?} down_ms={:?} scope=5launches+stage_output_read_divided_by5 split_stages_include_extra_intermediate_HBM_IO",
        times[0], times[1], times[2]
    );
    assert_eq!(failures, 0);
}

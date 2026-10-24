use furiosa_opt_gemma4::{Chip, axes::*, ops_prefill as op};
use furiosa_opt_std::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::new(op::rmsnorm_b16.topology())?;
    let values: Vec<_> = (0..B::SIZE * H::SIZE)
        .map(|i| bf16::from_f32(((i * 17 % 129) as f32 - 64.0) / 32.0))
        .collect();
    let x: HbmTensor<bf16, Chip, m![B, H]> = HostTensor::<bf16, m![B, H]>::from_vec(values.clone())
        .to_hbm(&mut device.pdma)
        .await?;
    let norm: HbmTensor<bf16, Chip, m![H]> =
        HostTensor::<bf16, m![H]>::from_vec(vec![bf16::from_f32(1.0); H::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
    let mut out: HbmTensor<bf16, Chip, m![B, H]> = HostTensor::<bf16, m![B, H]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    launch(op::rmsnorm_b16, (&mut device, &x, &norm, &mut out)).await?;
    let got = out.to_host::<m![B, H]>(&mut device.pdma).await?.into_vec();
    let mut max_error = 0.0f32;
    for r in 0..B::SIZE {
        let row = &values[r * H::SIZE..(r + 1) * H::SIZE];
        let rms = (row
            .iter()
            .map(|v| {
                let f = v.to_f32() as f64;
                f * f
            })
            .sum::<f64>()
            / H::SIZE as f64
            + 1e-6)
            .sqrt();
        for c in 0..H::SIZE {
            let expected = bf16::from_f32((row[c].to_f32() as f64 / rms) as f32).to_f32();
            max_error = max_error.max((got[r * H::SIZE + c].to_f32() - expected).abs());
        }
    }
    println!("B16 RMS versus independent f64 oracle: max_abs_error={max_error}");
    assert!(max_error <= 0.015625);
    launch(op::post_attention_b16, (&mut device, &x, &norm, &mut out)).await?;
    let gate: HbmTensor<bf16, Chip, m![1 # 8]> =
        HostTensor::<bf16, m![1 # 8]>::from_vec(vec![bf16::from_f32(1.0); 8])
            .to_hbm(&mut device.pdma)
            .await?;
    launch(
        op::post_feedforward_b16,
        (&mut device, &x, &norm, &gate, &mut out),
    )
    .await?;
    let offsets: HbmTensor<i32, Chip, m![B]> = HostTensor::<i32, m![B]>::from_vec(vec![0; B::SIZE])
        .to_hbm(&mut device.pdma)
        .await?;
    let qf: HbmTensor<bf16, Chip, m![B, Qf]> = HostTensor::<bf16, m![B, Qf]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let kf: HbmTensor<bf16, Chip, m![B, Pf]> = HostTensor::<bf16, m![B, Pf]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let nf: HbmTensor<bf16, Chip, m![Df]> = HostTensor::<bf16, m![Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let cf: HbmTensor<bf16, Chip, m![E, Df]> = HostTensor::<bf16, m![E, Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut qof: HbmTensor<bf16, Chip, m![B, Gf, Df]> = HostTensor::<bf16, m![B, Gf, Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut kof: HbmTensor<bf16, Chip, m![B, Df]> = HostTensor::<bf16, m![B, Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut vof: HbmTensor<bf16, Chip, m![B, Df]> = HostTensor::<bf16, m![B, Df]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    launch(
        op::full_heads_b16,
        (
            &mut device,
            &qf,
            &kf,
            &nf,
            &nf,
            &offsets,
            &cf,
            &cf,
            &mut qof,
            &mut kof,
            &mut vof,
        ),
    )
    .await?;
    let qs: HbmTensor<bf16, Chip, m![B, Qs]> = HostTensor::<bf16, m![B, Qs]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let ks: HbmTensor<bf16, Chip, m![B, Ps]> = HostTensor::<bf16, m![B, Ps]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let ns: HbmTensor<bf16, Chip, m![Ds]> = HostTensor::<bf16, m![Ds]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let cs: HbmTensor<bf16, Chip, m![E, Ds]> = HostTensor::<bf16, m![E, Ds]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut qos: HbmTensor<bf16, Chip, m![B, Ns, Gs, Ds]> =
        HostTensor::<bf16, m![B, Ns, Gs, Ds]>::zero()
            .to_hbm(&mut device.pdma)
            .await?;
    let mut kos: HbmTensor<bf16, Chip, m![B, Ns, Ds]> = HostTensor::<bf16, m![B, Ns, Ds]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    let mut vos: HbmTensor<bf16, Chip, m![B, Ns, Ds]> = HostTensor::<bf16, m![B, Ns, Ds]>::zero()
        .to_hbm(&mut device.pdma)
        .await?;
    launch(
        op::sliding_heads_b16,
        (
            &mut device,
            &qs,
            &ks,
            &ks,
            &ns,
            &ns,
            &offsets,
            &cs,
            &cs,
            &mut qos,
            &mut kos,
            &mut vos,
        ),
    )
    .await?;
    println!("Head preparation kernels launched (zero-input smoke only)");
    let mut cache_k: HbmTensor<bf16, Chip, m![Tf, Df]> =
        HostTensor::<bf16, m![Tf, Df]>::from_vec(vec![bf16::from_f32(1.0); Tf::SIZE * Df::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
    let mut cache_v: HbmTensor<bf16, Chip, m![Tf, Df]> =
        HostTensor::<bf16, m![Tf, Df]>::from_vec(vec![bf16::from_f32(1.0); Tf::SIZE * Df::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
    let offsets: HbmTensor<i32, Chip, m![B = 3]> = HostTensor::<i32, m![B = 3]>::from_vec(
        (0..3)
            .map(|i| (i * Df::SIZE * 2) as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await?;
    launch(
        op::commit_full_kv_3,
        (
            &mut device,
            &kof,
            &vof,
            &offsets,
            &mut cache_k,
            &mut cache_v,
        ),
    )
    .await?;
    let actual = cache_k
        .to_host::<m![Tf, Df]>(&mut device.pdma)
        .await?
        .into_vec();
    assert!(
        actual
            .iter()
            .enumerate()
            .all(|(i, v)| v.to_f32() == if i < 3 * Df::SIZE { 0.0 } else { 1.0 })
    );
    let mut cache_k: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> =
        HostTensor::<bf16, m![Ts, Ns, Ds]>::from_vec(vec![
            bf16::from_f32(1.0);
            Ts::SIZE * Ns::SIZE * Ds::SIZE
        ])
        .to_hbm(&mut device.pdma)
        .await?;
    let mut cache_v: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> =
        HostTensor::<bf16, m![Ts, Ns, Ds]>::from_vec(vec![
            bf16::from_f32(1.0);
            Ts::SIZE * Ns::SIZE * Ds::SIZE
        ])
        .to_hbm(&mut device.pdma)
        .await?;
    let offsets: HbmTensor<i32, Chip, m![B = 3]> = HostTensor::<i32, m![B = 3]>::from_vec(
        (0..3)
            .map(|i| (((Ts::SIZE - 2 + i) % Ts::SIZE) * Ns::SIZE * Ds::SIZE * 2) as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await?;
    launch(
        op::commit_sliding_kv_3,
        (
            &mut device,
            &kos,
            &vos,
            &offsets,
            &mut cache_k,
            &mut cache_v,
        ),
    )
    .await?;
    let actual = cache_k
        .to_host::<m![Ts, Ns, Ds]>(&mut device.pdma)
        .await?
        .into_vec();
    assert!(actual.iter().enumerate().all(|(i, v)| {
        let row = i / (Ns::SIZE * Ds::SIZE);
        v.to_f32()
            == if row == 0 || row >= Ts::SIZE - 2 {
                0.0
            } else {
                1.0
            }
    }));
    println!("Live-prefix KV commits preserve padding and wrap the sliding ring correctly");
    Ok(())
}

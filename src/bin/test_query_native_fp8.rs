//! Native Qs arithmetic/layout oracle; BF16 activation differences are reported separately.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_qkv_p128_trf32 as control, ops_query_native_fp8 as op,
};
use furiosa_opt_std::prelude::*;
fn hash(i: usize, seed: usize) -> usize {
    let mut v = (i as u64).wrapping_add(seed as u64);
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    (v ^ (v >> 31)) as usize
}
fn activation(b: usize, k: usize, seed: usize) -> bf16 {
    let unit = if seed == 0 {
        ((b * 17 + k * 13) % 31) as f32 / 16.0 - 15.0 / 16.0
    } else {
        ((hash(b * H::SIZE + k, seed) % 255) as f32 - 127.0) / 128.0
    };
    let value = match b {
        0 => 0.0,
        1 => unit * 2f32.powi(-110),
        2 => {
            if k % 2 == 0 {
                0.25
            } else {
                -0.25
            }
        }
        _ => unit * 2f32.powi((b % 11) as i32 - 5),
    };
    bf16::from_f32(value)
}
fn weight(n: usize, k: usize, seed: usize) -> f8e4m3 {
    let value = if seed == 0 {
        ((n * 5 + k * 7) % 17) as f32 / 8.0 - 1.0
    } else {
        ((hash(n * H::SIZE + k / 2, seed + 7) % 16) as f32 - 8.0) / 16.0
    };
    f8e4m3::from_f32(value)
}
fn channel(n: usize) -> bf16 {
    bf16::from_f32([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, 2.0][n % 7])
}
fn quantize(x: &[bf16]) -> (Vec<f8e4m3>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(P128::SIZE);
    for row in x.chunks_exact(H::SIZE) {
        let max = row
            .iter()
            .map(|x| x.to_f32().abs())
            .fold(1.925929944387236e-34f32, f32::max);
        let scale = f32::from_bits(max.to_bits() & 0x7f80_0000) * 0.0078125;
        scales.push(scale);
        q.extend(row.iter().map(|x| f8e4m3::from_f32(x.to_f32() / scale)));
    }
    (q, scales)
}
fn cpu_dot(q: &[f8e4m3], w: &[f8e4m3], b: usize, n: usize, scale: f32, channel: bf16) -> bf16 {
    // Bounded dyadic domains keep each H960 partial and its four-part sum exact F32.
    let dot = (0..H::SIZE)
        .map(|k| q[b * H::SIZE + k].to_f32() as f64 * w[n * H::SIZE + k].to_f32() as f64)
        .sum::<f64>();
    bf16::from_f32(bf16::from_f32(dot as f32 * scale).to_f32() * channel.to_f32())
}
async fn read<E: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, E>) -> Vec<bf16> {
    let h: HostTensor<bf16, E> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec()
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::project_query_batch.topology()).unwrap();
    for seed in [0, 31, 79] {
        let x: Vec<_> = (0..P128::SIZE)
            .flat_map(|b| (0..H::SIZE).map(move |k| activation(b, k, seed)))
            .collect();
        let w: Vec<_> = (0..Qs::SIZE)
            .flat_map(|n| (0..H::SIZE).map(move |k| weight(n, k, seed)))
            .collect();
        let sc: Vec<_> = (0..Qs::SIZE).map(channel).collect();
        let (cpu_q, cpu_scale) = quantize(&x);
        let input = HostTensor::<bf16, m![P128, H]>::from_vec(x.clone())
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let weights = HostTensor::<f8e4m3, m![Qs, H]>::from_vec(w.clone())
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let scales = HostTensor::<bf16, m![Qs]>::from_vec(sc.clone())
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let mut q = HostTensor::<f8e4m3, m![P128, H]>::zero()
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let mut xs = HostTensor::<f32, m![P128,1#8]>::zero()
            .to_hbm(&mut d.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, m![P128, Qs]>::from_vec(vec![
            bf16::from_f32(f32::NAN);
            P128::SIZE * Qs::SIZE
        ])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
        launch(op::quantize_input, (&mut d, &input, &mut q, &mut xs))
            .await
            .unwrap();
        let hostq: HostTensor<f8e4m3, m![P128, H]> = q.to_host(&mut d.pdma).await.unwrap();
        let hostq = hostq.into_vec();
        let hostxs: HostTensor<f32, m![P128,1#8]> = xs.to_host(&mut d.pdma).await.unwrap();
        let hostxs = hostxs.into_vec();
        assert_eq!(hostxs.len(), P128::SIZE * 8); // Actual padded NPU HostTensor retains physical slots.
        let qdiff = hostq
            .iter()
            .zip(&cpu_q)
            .filter(|(a, b)| a.to_f32().to_bits() != b.to_f32().to_bits())
            .count();
        let sdiff = (0..P128::SIZE)
            .filter(|&b| hostxs[b * 8].to_bits() != cpu_scale[b].to_bits())
            .count();
        println!(
            "seed={seed} quantized_values={} q_raw_differences={qdiff} scale_raw_differences={sdiff}",
            hostq.len()
        );
        assert_eq!(qdiff, 0);
        assert_eq!(sdiff, 0);
        launch(
            op::project_query_batch,
            (&mut d, &input, &weights, &scales, &mut output),
        )
        .await
        .unwrap();
        let actual = read(&mut d, &output).await;
        launch(
            op::project_query_quantized,
            (&mut d, &q, &xs, &weights, &scales, &mut output),
        )
        .await
        .unwrap();
        let staged = read(&mut d, &output).await;
        let diff = actual
            .iter()
            .zip(&staged)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "seed={seed} fused_vs_staged checked={} raw_differences={diff}",
            actual.len()
        );
        assert_eq!(diff, 0);
        let columns: Vec<_> = if seed == 0 {
            (0..Qs::SIZE).collect()
        } else {
            vec![
                0, 1, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 2047, 2048, 4094, 4095,
            ]
        };
        let mut cpu_diff = 0;
        for b in 0..P128::SIZE {
            let dots: Vec<_> = (0..if seed == 0 { 17 } else { columns.len() })
                .map(|j| {
                    let n = if seed == 0 { j } else { columns[j] };
                    cpu_dot(&cpu_q, &w, b, n, cpu_scale[b], bf16::from_f32(1.0))
                })
                .collect();
            for (j, &n) in columns.iter().enumerate() {
                let expected = bf16::from_f32(
                    dots[if seed == 0 { n % 17 } else { j }].to_f32() * sc[n].to_f32(),
                );
                if actual[b * Qs::SIZE + n].to_bits() != expected.to_bits() {
                    if cpu_diff < 5 {
                        println!(
                            "cpu_difference row={b} channel={n} actual={} expected={}",
                            actual[b * Qs::SIZE + n].to_f32(),
                            expected.to_f32()
                        );
                    }
                    cpu_diff += 1;
                }
            }
        }
        println!(
            "seed={seed} quantized_input_f64_oracle checked={} raw_differences={cpu_diff}",
            P128::SIZE * columns.len()
        );
        assert_eq!(cpu_diff, 0);
        launch(
            control::project_query_batch,
            (&mut d, &input, &weights, &scales, &mut output),
        )
        .await
        .unwrap();
        let reference = read(&mut d, &output).await;
        let mut max_abs = 0.0f32;
        let mut err2 = 0.0f64;
        let mut ref2 = 0.0f64;
        let mut differences = 0;
        for (a, b) in actual.iter().zip(&reference) {
            assert!(a.to_f32().is_finite());
            let delta = a.to_f32() - b.to_f32();
            max_abs = max_abs.max(delta.abs());
            err2 += (delta as f64).powi(2);
            ref2 += (b.to_f32() as f64).powi(2);
            differences += usize::from(a.to_bits() != b.to_bits());
        }
        println!(
            "seed={seed} BF16_activation_comparison checked={} raw_differences={differences} max_abs={max_abs} relative_l2={}",
            actual.len(),
            (err2 / ref2.max(1e-300)).sqrt()
        );
        if seed == 31 {
            for variant in [0, 1, 1, 0] {
                let now = std::time::Instant::now();
                for _ in 0..20 {
                    if variant == 0 {
                        launch(
                            control::project_query_batch,
                            (&mut d, &input, &weights, &scales, &mut output),
                        )
                        .await
                        .unwrap();
                    } else {
                        launch(
                            op::project_query_batch,
                            (&mut d, &input, &weights, &scales, &mut output),
                        )
                        .await
                        .unwrap();
                    }
                }
                println!(
                    "ABBA variant={variant} warm_us={:.3}",
                    now.elapsed().as_secs_f64() * 1e6 / 20.0
                );
            }
        }
    }
}

//! Batched-quarter post-FFN exact oracle against the current fused and original paths.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use furiosa_opt_gemma4::{
    Chip, EPS, axes::*, ops_native_fp8_p512_downblocked as unpack, ops_norm_fused as control,
    ops_norm_quarter_post as fused, ops_norm_wide as norm,
};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Hidden = HbmTensor<bf16, Chip, m![P128, H]>;
fn bf(x: f32) -> f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
}
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            bf((((x >> 40) as f32) / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
async fn upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    HostTensor::<bf16, Mp>::from_buf(
        x.iter()
            .flat_map(|x| ((bf(*x).to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut d.pdma)
    .await
    .unwrap()
}
async fn zero<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(vec![0u8; Mp::SIZE * D::BITS / 8])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn bytes<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
    x: &HbmTensor<D, Chip, Mp>,
) -> Vec<u8> {
    let h: HostTensor<D, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_inner().into_buf()
}
fn exact(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    println!("{label} bytes={} mismatches={n}", a.len());
    assert_eq!(n, 0, "{label}");
}
struct State {
    residual: [Hidden; 4],
    projected: [Hidden; 4],
}
impl State {
    async fn new(d: &mut Device, rv: &[Vec<f32>; 4]) -> Self {
        Self {
            residual: [
                upload(d, &rv[0]).await,
                upload(d, &rv[1]).await,
                upload(d, &rv[2]).await,
                upload(d, &rv[3]).await,
            ],
            projected: [zero(d).await, zero(d).await, zero(d).await, zero(d).await],
        }
    }
    async fn reset(&mut self, d: &mut Device, rv: &[Vec<f32>; 4]) {
        self.residual = [
            upload(d, &rv[0]).await,
            upload(d, &rv[1]).await,
            upload(d, &rv[2]).await,
            upload(d, &rv[3]).await,
        ];
    }
}
async fn run(
    d: &mut Device,
    variant: u8,
    input: &HbmTensor<bf16, Chip, unpack::Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    gate: &HbmTensor<bf16, Chip, m![1#8]>,
    s: &mut State,
) {
    if variant == 1 {
        let [a, b, c, e] = &mut s.residual;
        launch(
            fused::post_feedforward_four,
            (d, input, weight, gate, a, b, c, e),
        )
        .await
        .unwrap();
    } else if variant == 0 {
        let [a, b, c, e] = &mut s.residual;
        launch(
            control::post_feedforward_four,
            (d, input, weight, gate, a, b, c, e),
        )
        .await
        .unwrap();
    } else {
        let [a, b, c, e] = &mut s.projected;
        launch(unpack::unpack_hidden_group, (d, input, a, b, c, e))
            .await
            .unwrap();
        for i in 0..4 {
            launch(
                norm::post_feedforward_p128,
                (d, &s.projected[i], weight, gate, &mut s.residual[i]),
            )
            .await
            .unwrap();
        }
    }
}
async fn test(d: &mut Device, seed: u64, order: [usize; 4], gate_value: f32) {
    let pv: [Vec<f32>; 4] = std::array::from_fn(|g| {
        let mut v = values(128 * 3840, seed + order[g] as u64 * 111, 3.0);
        for row in 0..128 {
            if row % 23 == 0 {
                v[row * 3840..(row + 1) * 3840].fill(0.0);
            }
            if row % 23 == 1 {
                v[row * 3840..(row + 1) * 3840].fill(bf(1e-5));
            }
        }
        v
    });
    let rv: [Vec<f32>; 4] =
        std::array::from_fn(|g| values(128 * 3840, seed + order[g] as u64 * 137 + 17, 0.75));
    // Independent CPU blocked layout, with every logical token/channel present.
    let mut blocked = vec![0u8; 512 * 3840 * 2];
    for row in 0..512 {
        for h in 0..3840 {
            let i = (((row / 256 * 16 + h / 240) * 256 + row % 256) * 240 + h % 240) * 2;
            blocked[i..i + 2].copy_from_slice(
                &((pv[row / 128][row % 128 * 3840 + h].to_bits() >> 16) as u16).to_le_bytes(),
            );
        }
    }
    let input = HostTensor::<bf16, unpack::Output>::from_buf(blocked.clone())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap();
    let weight_values = values(3840, seed + 1500, 1.75);
    let weight = upload(d, &weight_values).await;
    let gate = HostTensor::<bf16, m![1#8]>::from_buf(
        (0..8)
            .flat_map(|_| ((bf(gate_value).to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut d.pdma)
    .await
    .unwrap();
    let mut control = State::new(d, &rv).await;
    let mut candidate = State::new(d, &rv).await;
    let mut original = State::new(d, &rv).await;
    run(d, 1, &input, &weight, &gate, &mut candidate).await;
    run(d, 0, &input, &weight, &gate, &mut control).await;
    run(d, 2, &input, &weight, &gate, &mut original).await;
    exact(
        "immutable blocked producer output",
        &blocked,
        &bytes(d, &input).await,
    );
    let mut error = 0f64;
    let mut reference = 0f64;
    for g in 0..4 {
        let a = bytes(d, &control.residual[g]).await;
        let b = bytes(d, &candidate.residual[g]).await;
        exact(&format!("current fused vs quarters group{g}"), &a, &b);
        exact(
            &format!("original unpack+P128 vs quarters group{g}"),
            &bytes(d, &original.residual[g]).await,
            &b,
        );
        for row in [0, 1, 2, 31, 63, 64, 127] {
            let rms = (pv[g][row * 3840..(row + 1) * 3840]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                / 3840.0
                + EPS as f64)
                .sqrt();
            for col in 0..3840 {
                let ix = row * 3840 + col;
                let normalized = bf((pv[g][ix] as f64 / rms * weight_values[col] as f64) as f32);
                let expected = bf(bf(normalized + rv[g][ix]) * bf(gate_value));
                let got =
                    f32::from_bits((u16::from_le_bytes([b[ix * 2], b[ix * 2 + 1]]) as u32) << 16);
                assert!(got.is_finite());
                error += (got as f64 - expected as f64).powi(2);
                reference += (expected as f64).powi(2);
            }
        }
    }
    let nrmse = (error / reference.max(1e-30)).sqrt();
    println!("CPU F64 postFFN seed={seed} gate={gate_value} nrmse={nrmse}");
    assert!(nrmse < 0.003);
    let mut timing = [Vec::new(), Vec::new()];
    for variant in [0u8, 1, 1, 0, 0, 1, 1, 0] {
        let state = if variant == 1 {
            &mut candidate
        } else {
            &mut control
        };
        state.reset(d, &rv).await;
        let now = Instant::now();
        run(d, variant, &input, &weight, &gate, state).await;
        timing[variant as usize].push(now.elapsed().as_secs_f64() * 1e6);
    }
    if seed == 713 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
        for (label, variant) in [
            ("current_sequential_quarters", 0u8),
            ("batched_quarters", 1u8),
        ] {
            let state = if variant == 1 {
                &mut candidate
            } else {
                &mut control
            };
            state.reset(d, &rv).await;
            println!("POST_QUARTER_PROFILE_BEGIN variant={label}");
            PROFILE.with(|p| p.enable(true));
            run(d, variant, &input, &weight, &gate, state).await;
            PROFILE.with(|p| p.enable(false));
        }
    }
    println!(
        "TIMING seed={seed} order={order:?} gate={gate_value} current_sequential_quarters_us={:?} batched_quarters_us={:?}",
        timing[0], timing[1]
    );
}
thread_local! { static PROFILE:npu_profile::Collector=npu_profile::Collector::default(); }
#[tokio::main]
async fn main() {
    PROFILE.with(|p| tracing::subscriber::set_global_default(p.clone()).unwrap());
    let mut d = Device::new(fused::post_feedforward_four.topology()).unwrap();
    test(&mut d, 713, [0, 1, 2, 3], 1.25).await;
    test(&mut d, 1771, [3, 1, 0, 2], -0.75).await;
    test(&mut d, 5001, [2, 0, 3, 1], 0.0).await;
    test(&mut d, 9123, [1, 3, 2, 0], 1.0).await;
    println!(
        "PASS batched-quarter postFFN all512rows, independent CPUproducer mapping, originalBF16 residual/gate, zero/tiny/signed and quarter permutations"
    );
}

//! All exact consumers of persistent grouped residuals, with alternating layer types.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_batch_layouts::P512H960, ops_native_fp8_p512_downblocked::Output,
    ops_norm_blocked_chain as chain, ops_norm_blocked_residual as blocked,
    ops_norm_fused as original, ops_norm_quarter_post as control, ops_norm_quarters as quarters,
    ops_output_p128_h64_blocked::Output as AttentionOutput,
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
            bf(((x >> 40) as f32 / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
fn raw(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn residual_index(p: usize, h: usize) -> usize {
    (((((p / 64 % 2) * 32 + p / 2 % 32) * 8 + h / 480) * 4 + p / 128) * 2 + p % 2) * 480 + h % 480
}
fn pack_residual(values: &[Vec<f32>; 4]) -> Vec<u8> {
    let mut out = vec![0xa5; 512 * 3840 * 2];
    let mut seen = vec![false; 512 * 3840];
    for p in 0..512 {
        for h in 0..3840 {
            let i = residual_index(p, h);
            assert!(!seen[i]);
            seen[i] = true;
            out[2 * i..2 * i + 2].copy_from_slice(
                &((values[p / 128][p % 128 * 3840 + h].to_bits() >> 16) as u16).to_le_bytes(),
            );
        }
    }
    assert!(seen.iter().all(|x| *x));
    out
}
fn unpack_cpu(bytes: &[u8], quarter: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(128 * 3840 * 2);
    for p in quarter * 128..(quarter + 1) * 128 {
        for h in 0..3840 {
            let i = residual_index(p, h);
            out.extend_from_slice(&bytes[2 * i..2 * i + 2]);
        }
    }
    out
}
fn pack_output(values: &[Vec<f32>; 4]) -> Vec<u8> {
    let mut out = vec![0xa5; 512 * 3840 * 2];
    let mut seen = vec![false; 512 * 3840];
    for p in 0..512 {
        for h in 0..3840 {
            let i = (((p / 256 * 16 + h / 240) * 256 + p % 256) * 240) + h % 240;
            assert!(!seen[i]);
            seen[i] = true;
            out[2 * i..2 * i + 2].copy_from_slice(
                &((values[p / 128][p % 128 * 3840 + h].to_bits() >> 16) as u16).to_le_bytes(),
            );
        }
    }
    assert!(seen.iter().all(|x| *x));
    out
}
async fn upload<Mp: M>(d: &mut Device, v: Vec<u8>) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(v.len(), Mp::SIZE * 2);
    HostTensor::<bf16, Mp>::from_buf(v)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn bytes<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<u8> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_inner().into_buf()
}
fn exact(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    println!("{label} bytes={} mismatches={n}", a.len());
    assert_eq!(n, 0, "{label}");
}
async fn canonical(d: &mut Device, v: &[Vec<f32>; 4]) -> [Hidden; 4] {
    [
        upload(d, raw(&v[0])).await,
        upload(d, raw(&v[1])).await,
        upload(d, raw(&v[2])).await,
        upload(d, raw(&v[3])).await,
    ]
}
async fn pack(
    d: &mut Device,
    rows: &[Hidden; 4],
    out: &mut HbmTensor<bf16, Chip, blocked::Residual>,
) {
    launch(
        blocked::pack_four,
        (d, &rows[0], &rows[1], &rows[2], &rows[3], out),
    )
    .await
    .unwrap();
}
async fn unpack(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, blocked::Residual>,
    rows: &mut [Hidden; 4],
) {
    let [a, b, c, e] = rows;
    launch(blocked::unpack_four, (d, input, a, b, c, e))
        .await
        .unwrap();
}
async fn old(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    gate: &HbmTensor<bf16, Chip, m![1#8]>,
    rows: &mut [Hidden; 4],
) {
    let [a, b, c, e] = rows;
    launch(
        control::post_feedforward_four,
        (d, input, weight, gate, a, b, c, e),
    )
    .await
    .unwrap();
}
async fn new(
    d: &mut Device,
    input: &HbmTensor<bf16, Chip, Output>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    gate: &HbmTensor<bf16, Chip, m![1#8]>,
    residual: &mut HbmTensor<bf16, Chip, blocked::Residual>,
) {
    launch(
        blocked::post_feedforward,
        (d, input, weight, gate, residual),
    )
    .await
    .unwrap();
}
async fn empty<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(vec![0x7f; Mp::SIZE * D::BITS / 8])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn raw_tensor<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
    x: &HbmTensor<D, Chip, Mp>,
) -> Vec<u8> {
    let h: HostTensor<D, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_inner().into_buf()
}
struct Ingress {
    q: [HbmTensor<f8e4m3, Chip, m![P128, H]>; 4],
    s: [HbmTensor<f32, Chip, m![P128,1#8]>; 4],
    ffn: HbmTensor<f8e4m3, Chip, P512H960>,
    ffn_scale: HbmTensor<f32, Chip, m![P512]>,
}
impl Ingress {
    async fn new(d: &mut Device) -> Self {
        Self {
            q: [
                empty(d).await,
                empty(d).await,
                empty(d).await,
                empty(d).await,
            ],
            s: [
                empty(d).await,
                empty(d).await,
                empty(d).await,
                empty(d).await,
            ],
            ffn: empty(d).await,
            ffn_scale: empty(d).await,
        }
    }
}
async fn qkv(
    d: &mut Device,
    candidate: bool,
    rows: &[Hidden; 4],
    residual: &HbmTensor<bf16, Chip, blocked::Residual>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
    out: &mut Ingress,
) {
    let [a, b, c, e] = &mut out.q;
    let [sa, sb, sc, se] = &mut out.s;
    if candidate {
        launch(
            chain::normalize_qkv_four,
            (d, residual, weight, a, sa, b, sb, c, sc, e, se),
        )
        .await
        .unwrap();
    } else {
        launch(
            original::normalize_qkv_four,
            (
                d, &rows[0], &rows[1], &rows[2], &rows[3], weight, a, sa, b, sb, c, sc, e, se,
            ),
        )
        .await
        .unwrap();
    }
}
async fn attention(
    d: &mut Device,
    candidate: bool,
    sliding: bool,
    projected: &[Hidden; 4],
    ob: &[HbmTensor<bf16, Chip, AttentionOutput>; 4],
    post: &HbmTensor<bf16, Chip, m![H]>,
    pre: &HbmTensor<bf16, Chip, m![H]>,
    rows: &mut [Hidden; 4],
    residual: &mut HbmTensor<bf16, Chip, blocked::Residual>,
    out: &mut Ingress,
) {
    let [a, b, c, e] = rows;
    if candidate {
        if sliding {
            launch(
                chain::post_attention_sliding,
                (
                    d,
                    &ob[0],
                    &ob[1],
                    &ob[2],
                    &ob[3],
                    post,
                    pre,
                    residual,
                    &mut out.ffn,
                    &mut out.ffn_scale,
                ),
            )
            .await
            .unwrap();
        } else {
            launch(
                chain::post_attention_full,
                (
                    d,
                    &projected[0],
                    &projected[1],
                    &projected[2],
                    &projected[3],
                    post,
                    pre,
                    residual,
                    &mut out.ffn,
                    &mut out.ffn_scale,
                ),
            )
            .await
            .unwrap();
        }
    } else if sliding {
        launch(
            quarters::post_attention_ffn_input_four_blocked,
            (
                d,
                &ob[0],
                &ob[1],
                &ob[2],
                &ob[3],
                post,
                pre,
                a,
                b,
                c,
                e,
                &mut out.ffn,
                &mut out.ffn_scale,
            ),
        )
        .await
        .unwrap();
    } else {
        launch(
            original::post_attention_ffn_input_four,
            (
                d,
                &projected[0],
                &projected[1],
                &projected[2],
                &projected[3],
                post,
                pre,
                a,
                b,
                c,
                e,
                &mut out.ffn,
                &mut out.ffn_scale,
            ),
        )
        .await
        .unwrap();
    }
}
async fn state_exact(
    d: &mut Device,
    label: &str,
    rows: &[Hidden; 4],
    residual: &HbmTensor<bf16, Chip, blocked::Residual>,
) {
    let packed = bytes(d, residual).await;
    for g in 0..4 {
        exact(
            &format!("{label} quarter{g}"),
            &bytes(d, &rows[g]).await,
            &unpack_cpu(&packed, g),
        );
    }
}
async fn test_chain(d: &mut Device, seed: u64, order: [usize; 4], gate_value: f32) {
    let pv: [Vec<f32>; 4] = std::array::from_fn(|g| {
        let mut x = values(128 * 3840, seed + order[g] as u64 * 31, 2.0);
        for p in 0..128 {
            if p % 23 == 0 {
                x[p * 3840..(p + 1) * 3840].fill(0.0);
            }
            if p % 23 == 1 {
                x[p * 3840..(p + 1) * 3840].fill(bf(1e-5));
            }
        }
        x
    });
    let rv: [Vec<f32>; 4] =
        std::array::from_fn(|g| values(128 * 3840, seed + order[g] as u64 * 91 + 11, 0.75));
    let projected = canonical(d, &pv).await;
    let obv: [Vec<u8>; 4] = std::array::from_fn(|g| {
        let mut data = vec![0u8; 128 * 3840 * 2];
        for p in 0..128 {
            for h in 0..3840 {
                let i = (((p / 64 * 60 + h / 64) * 64 + p % 64) * 64 + h % 64) * 2;
                data[i..i + 2]
                    .copy_from_slice(&((pv[g][p * 3840 + h].to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        data
    });
    let ob: [HbmTensor<bf16, Chip, AttentionOutput>; 4] = [
        upload(d, obv[0].clone()).await,
        upload(d, obv[1].clone()).await,
        upload(d, obv[2].clone()).await,
        upload(d, obv[3].clone()).await,
    ];
    let ffn = upload(d, pack_output(&pv)).await;
    let input_weight = upload(d, raw(&values(3840, seed + 1000, 1.5))).await;
    let post = upload(d, raw(&values(3840, seed + 2000, 1.75))).await;
    let pre = upload(d, raw(&values(3840, seed + 3000, 1.25))).await;
    let gate = upload(d, raw(&[bf(gate_value); 8])).await;
    let mut rows = canonical(d, &rv).await;
    let mut candidate_rows = canonical(d, &rv).await;
    let mut residual = upload::<blocked::Residual>(d, vec![0x7f; 512 * 3840 * 2]).await;
    pack(d, &rows, &mut residual).await;
    exact(
        "initial pack CPU layout",
        &pack_residual(&rv),
        &bytes(d, &residual).await,
    );
    let mut old_out = Ingress::new(d).await;
    let mut new_out = Ingress::new(d).await;
    for step in 0..4 {
        let sliding = step % 2 == 0;
        qkv(
            d,
            true,
            &candidate_rows,
            &residual,
            &input_weight,
            &mut new_out,
        )
        .await;
        qkv(d, false, &rows, &residual, &input_weight, &mut old_out).await;
        for g in 0..4 {
            exact(
                &format!("QKV bytes seed{seed} step{step} quarter{g}"),
                &raw_tensor(d, &old_out.q[g]).await,
                &raw_tensor(d, &new_out.q[g]).await,
            );
            exact(
                &format!("QKV scales seed{seed} step{step} quarter{g}"),
                &raw_tensor(d, &old_out.s[g]).await,
                &raw_tensor(d, &new_out.s[g]).await,
            );
        }
        attention(
            d,
            true,
            sliding,
            &projected,
            &ob,
            &post,
            &pre,
            &mut candidate_rows,
            &mut residual,
            &mut new_out,
        )
        .await;
        attention(
            d,
            false,
            sliding,
            &projected,
            &ob,
            &post,
            &pre,
            &mut rows,
            &mut residual,
            &mut old_out,
        )
        .await;
        exact(
            &format!("FFN ingress seed{seed} step{step}"),
            &raw_tensor(d, &old_out.ffn).await,
            &raw_tensor(d, &new_out.ffn).await,
        );
        exact(
            &format!("FFN scales seed{seed} step{step}"),
            &raw_tensor(d, &old_out.ffn_scale).await,
            &raw_tensor(d, &new_out.ffn_scale).await,
        );
        state_exact(
            d,
            &format!("postattention seed{seed} step{step}"),
            &rows,
            &residual,
        )
        .await;
        new(d, &ffn, &post, &gate, &mut residual).await;
        old(d, &ffn, &post, &gate, &mut rows).await;
        state_exact(
            d,
            &format!("postFFN seed{seed} step{step}"),
            &rows,
            &residual,
        )
        .await;
    }
    unpack(d, &residual, &mut candidate_rows).await;
    for g in 0..4 {
        exact(
            &format!("final owner unpack quarter{g}"),
            &bytes(d, &rows[g]).await,
            &bytes(d, &candidate_rows[g]).await,
        );
    }
    for sliding in [true, false] {
        let mut timing = [Vec::new(), Vec::new()];
        for variant in [0usize, 1, 1, 0, 0, 1, 1, 0] {
            rows = canonical(d, &rv).await;
            residual = upload(d, pack_residual(&rv)).await;
            let out = if variant == 0 {
                &mut old_out
            } else {
                &mut new_out
            };
            let now = Instant::now();
            qkv(d, variant == 1, &rows, &residual, &input_weight, out).await;
            attention(
                d,
                variant == 1,
                sliding,
                &projected,
                &ob,
                &post,
                &pre,
                &mut rows,
                &mut residual,
                out,
            )
            .await;
            if variant == 0 {
                old(d, &ffn, &post, &gate, &mut rows).await;
            } else {
                new(d, &ffn, &post, &gate, &mut residual).await;
            }
            timing[variant].push(now.elapsed().as_secs_f64() * 1e6);
        }
        println!(
            "CONSUMER_CHAIN_TIMING seed={seed} sliding={sliding} canonical_us={:?} blocked_us={:?}",
            timing[0], timing[1]
        );
    }
    if seed == 713 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
        for variant in [0usize, 1] {
            rows = canonical(d, &rv).await;
            residual = upload(d, pack_residual(&rv)).await;
            let out = if variant == 0 {
                &mut old_out
            } else {
                &mut new_out
            };
            println!("CONSUMER_PROFILE_BEGIN variant={variant} phase=qkv");
            PROFILE.with(|p| p.enable(true));
            qkv(d, variant == 1, &rows, &residual, &input_weight, out).await;
            PROFILE.with(|p| p.enable(false));
            for sliding in [true, false] {
                println!(
                    "CONSUMER_PROFILE_BEGIN variant={variant} phase=postattention sliding={sliding}"
                );
                PROFILE.with(|p| p.enable(true));
                attention(
                    d,
                    variant == 1,
                    sliding,
                    &projected,
                    &ob,
                    &post,
                    &pre,
                    &mut rows,
                    &mut residual,
                    out,
                )
                .await;
                PROFILE.with(|p| p.enable(false));
            }
            println!("CONSUMER_PROFILE_BEGIN variant={variant} phase=postffn");
            PROFILE.with(|p| p.enable(true));
            if variant == 0 {
                old(d, &ffn, &post, &gate, &mut rows).await;
            } else {
                new(d, &ffn, &post, &gate, &mut residual).await;
            }
            PROFILE.with(|p| p.enable(false));
        }
    }
    println!("PASS consumer seed={seed} quarter_order={order:?} gate={gate_value}");
}
thread_local! {static PROFILE:npu_profile::Collector=npu_profile::Collector::default();}
#[tokio::main]
async fn main() {
    PROFILE.with(|p| tracing::subscriber::set_global_default(p.clone()).unwrap());
    let mut d = Device::new(chain::normalize_qkv_four.topology()).unwrap();
    test_chain(&mut d, 713, [0, 1, 2, 3], 1.25).await;
    test_chain(&mut d, 1771, [3, 1, 0, 2], -0.75).await;
    test_chain(&mut d, 5001, [2, 0, 3, 1], 0.0).await;
    println!(
        "PASS persistent residual all-consumer chain QKV/sliding/full/FFN/finalowner all bits with repeated feedback; no host adoption"
    );
}

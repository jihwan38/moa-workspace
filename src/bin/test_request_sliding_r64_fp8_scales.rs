//! Isolated quantized-cache proof: byte oracle, dequantized exact control, F64 math,
//! BF16 approximation report, finite future poison, sparse rows, and ABBA timings.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use fp8::Scale8;
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_pool_views::Rc64, ops_request_sliding_r64 as exact,
    ops_request_sliding_r64_fp8_scales as fp8,
};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
const RING: usize = 1024 * 8 * 256;
const POS: [usize; 16] = [
    0, 1, 31, 255, 511, 512, 1023, 1024, 1535, 15, 510, 700, 2048, 256, 800, 8191,
];
fn position(r: usize) -> usize {
    POS[r % 16]
}
fn slot(r: usize) -> usize {
    (r * 5 + 3) % 64
}
fn bf(x: f32) -> f32 {
    f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
}
fn word(i: usize, seed: u64) -> u64 {
    let mut x = (i as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| bf((((word(i, seed) >> 40) as f32) / 16777216.0 * 2. - 1.) * scale))
        .collect()
}
fn ring(r: usize, is_v: bool) -> Vec<f32> {
    let mut x = values(
        RING,
        200 + r as u64 + if is_v { 1000 } else { 0 },
        if is_v {
            1.75
        } else if r < 32 {
            0.2
        } else {
            1.5
        },
    );
    // Different scale per token/head; tests that quantization never mixes rows/heads.
    for (t, h) in (0..1024).flat_map(|t| (0..8).map(move |h| (t, h))) {
        let factor = 2f32.powi(((word(t * 8 + h, 29 + r as u64) % 5) as i32) - 2);
        for v in &mut x[(t * 8 + h) * 256..(t * 8 + h + 1) * 256] {
            *v *= factor;
        }
    }
    if r == 0 {
        x[..256].fill(0.);
        for (i, v) in x[256..512].iter_mut().enumerate() {
            *v = f32::from_bits(if i % 2 == 0 { 0x00400000 } else { 0x80400000 });
        }
        for (i, v) in x[512..768].iter_mut().enumerate() {
            *v = if i % 2 == 0 {
                2f32.powi(-112)
            } else {
                -2f32.powi(-112)
            };
        }
    }
    x
}
fn encode_e4(x: f32) -> u8 {
    let mut out = [0];
    f8e4m3::store(&mut out, 0, f8e4m3::from_f32(x));
    out[0]
}
fn decode_e4(code: u8) -> f32 {
    let e = (code >> 3) & 15;
    let m = code & 7;
    let a = if e == 0 {
        m as f32 / 512.
    } else if e == 15 && m == 7 {
        f32::NAN
    } else {
        (1. + m as f32 / 8.) * 2f32.powi(e as i32 - 7)
    };
    if code & 128 != 0 { -a } else { a }
}
fn nearest_e4(x: f32) -> u8 {
    let a = x.abs();
    let mut best = 0;
    let mut error = f32::INFINITY;
    for code in 0u8..127 {
        let e = (decode_e4(code) - a).abs();
        if e < error || (e == error && code % 2 == 0) {
            best = code;
            error = e;
        }
    }
    best | ((x.to_bits() >> 24) as u8 & 128)
}
fn check_cpu_quantizer() {
    let mut count = 0;
    for bits in 0..65536u32 {
        let x = f32::from_bits(bits << 16);
        if x.is_finite() && x.abs() <= 256. {
            assert_eq!(encode_e4(x), nearest_e4(x), "CPU E4 RNE bits={bits:04x}");
            count += 1;
        }
    }
    println!("CPU E4 nearest-even independent lookup PASS {count} normalized BF16 patterns");
}
struct Packed {
    codes: Vec<u8>,
    scales: Vec<f32>,
    decoded: Vec<f32>,
}
fn quantize(x: &[f32]) -> Packed {
    let mut codes = vec![0; x.len()];
    let mut scales = Vec::with_capacity(x.len() / 256 * 8);
    let mut decoded = vec![0.; x.len()];
    for (h, row) in x.chunks_exact(256).enumerate() {
        let amax = row
            .iter()
            .map(|x| x.abs())
            .fold(0f32, f32::max)
            .max(2f32.powi(-112));
        let scale = f32::from_bits(amax.to_bits() & 0x7f800000) / 128.;
        scales.extend([scale; 8]);
        for (j, &x) in row.iter().enumerate() {
            let x = if x.abs() < f32::MIN_POSITIVE {
                f32::from_bits(x.to_bits() & 0x80000000)
            } else {
                x
            };
            let code = encode_e4(x / scale);
            codes[h * 256 + j] = code;
            decoded[h * 256 + j] = bf(decode_e4(code) * scale);
            assert!(decoded[h * 256 + j].is_finite());
        }
    }
    Packed {
        codes,
        scales,
        decoded,
    }
}
async fn upload<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
    x: Vec<u8>,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(x)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn zero<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    upload(d, vec![0; Mp::SIZE * D::BITS / 8]).await
}
async fn ub<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    upload(
        d,
        x.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
    )
    .await
}
async fn uf<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<f32, Chip, Mp> {
    upload(d, x.iter().flat_map(|x| x.to_le_bytes()).collect()).await
}
async fn ui<Mp: M>(d: &mut Device, x: Vec<i32>) -> HbmTensor<i32, Chip, Mp> {
    upload(d, x.iter().flat_map(|x| x.to_le_bytes()).collect()).await
}
async fn raw<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
    x: &HbmTensor<D, Chip, Mp>,
) -> Vec<u8> {
    x.to_host::<Mp>(&mut d.pdma)
        .await
        .unwrap()
        .into_inner()
        .into_buf()
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<f32> {
    raw(d, x)
        .await
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect()
}
struct Pools {
    k: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    v: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    dk: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    dv: HbmTensor<bf16, Chip, m![Rc64, Ts, Ns, Ds]>,
    qk: HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    qv: HbmTensor<f8e4m3, Chip, m![Rc64, Ts, Ns, Ds]>,
    ks: HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
    vs: HbmTensor<f32, Chip, m![Rc64, Ns, Ts, Scale8]>,
}
async fn check_device_quantizer(
    d: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    expected: &Packed,
    label: &str,
) {
    let mut q = zero::<f8e4m3, m![Ts, Ns, Ds]>(d).await;
    let mut s = zero::<f32, m![Ts, Ns, Scale8]>(d).await;
    launch(fp8::quantize_ring, (d, x, &mut q, &mut s))
        .await
        .unwrap();
    let got = raw(d, &q).await;
    let n = got
        .iter()
        .zip(&expected.codes)
        .filter(|(a, b)| a != b)
        .count();
    for (i, (a, b)) in got
        .iter()
        .zip(&expected.codes)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(5)
    {
        println!("quant mismatch {label} i={i} expected={b:02x} actual={a:02x}");
    }
    assert_eq!(n, 0, "device E4 codes {label}");
    let got = raw(d, &s).await;
    let expected: Vec<u8> = expected
        .scales
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(
        got, expected,
        "device scales including live repeat packets {label}"
    );
    println!(
        "QUANT {label} PASS {} E4 values + {} replicated F32 scales",
        RING,
        1024 * 8 * 8
    );
}
async fn copy(d: &mut Device, p: &mut Pools, r: usize, k: &[f32], v: &[f32], quant_gate: bool) {
    let qk = quantize(k);
    let qv = quantize(v);
    let kh = ub::<m![Ts, Ns, Ds]>(d, k).await;
    let vh = ub::<m![Ts, Ns, Ds]>(d, v).await;
    if quant_gate {
        check_device_quantizer(d, &kh, &qk, &format!("row{r}-K")).await;
        check_device_quantizer(d, &vh, &qv, &format!("row{r}-V")).await;
    }
    let off = ui::<m![1]>(d, vec![(slot(r) * RING * 2) as i32]).await;
    launch(
        exact::copy_sliding_cache_to_pool_rc64,
        (d, &kh, &vh, &off, &mut p.k, &mut p.v),
    )
    .await
    .unwrap();
    let dkh = ub::<m![Ts, Ns, Ds]>(d, &qk.decoded).await;
    let dvh = ub::<m![Ts, Ns, Ds]>(d, &qv.decoded).await;
    launch(
        exact::copy_sliding_cache_to_pool_rc64,
        (d, &dkh, &dvh, &off, &mut p.dk, &mut p.dv),
    )
    .await
    .unwrap();
    let qkh = upload::<f8e4m3, m![Ts, Ns, Ds]>(d, qk.codes).await;
    let qvh = upload::<f8e4m3, m![Ts, Ns, Ds]>(d, qv.codes).await;
    let ksh = uf::<m![Ts, Ns, Scale8]>(d, &qk.scales).await;
    let vsh = uf::<m![Ts, Ns, Scale8]>(d, &qv.scales).await;
    let qo = ui::<m![1]>(d, vec![(slot(r) * RING) as i32]).await;
    let so = ui::<m![1]>(d, vec![(slot(r) * 1024 * 8 * 8 * 4) as i32]).await;
    launch(
        fp8::copy_quantized_ring_to_pool,
        (
            d, &qkh, &qvh, &ksh, &vsh, &qo, &so, &mut p.qk, &mut p.qv, &mut p.ks, &mut p.vs,
        ),
    )
    .await
    .unwrap();
}
fn report(label: &str, reference: &[f32], got: &[f32]) -> f64 {
    let (mut error, mut norm, mut max, mut diff) = (0f64, 0f64, 0f32, 0usize);
    for (&a, &b) in reference.iter().zip(got) {
        assert!(a.is_finite() && b.is_finite());
        error += (a as f64 - b as f64).powi(2);
        norm += (a as f64).powi(2);
        max = max.max((a - b).abs());
        diff += usize::from(a.to_bits() != b.to_bits());
    }
    let nrmse = (error / norm.max(1e-60)).sqrt();
    println!(
        "{label} n={} bitdiff={diff} nrmse={nrmse:.9} maxabs={max:.9}",
        reference.len()
    );
    nrmse
}
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], pos: usize, h: usize, g: usize) -> Vec<f32> {
    let n = (pos + 1).min(1024);
    let q = &q[(h * 2 + g) * 256..(h * 2 + g + 1) * 256];
    let scores: Vec<f64> = (0..n)
        .map(|t| {
            q.iter()
                .zip(&k[(t * 8 + h) * 256..(t * 8 + h + 1) * 256])
                .map(|(&q, &k)| q as f64 * k as f64)
                .sum()
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    (0..256)
        .map(|j| {
            (exp.iter()
                .enumerate()
                .map(|(t, &w)| w * v[(t * 8 + h) * 256 + j] as f64)
                .sum::<f64>()
                / sum) as f32
        })
        .collect()
}
// Independent F64 accumulation with the implementation's explicit BF16 storage
// boundaries. The F64 unrounded reference above remains an error measurement;
// it cannot be an equivalence gate for the original BF16 attention itself.
fn cpu_bf16_boundaries(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: usize,
    h: usize,
    g: usize,
) -> Vec<f32> {
    let n = (pos + 1).min(1024);
    let q = &q[(h * 2 + g) * 256..(h * 2 + g + 1) * 256];
    let score: Vec<f32> = (0..n)
        .map(|t| {
            bf(q.iter()
                .zip(&k[(t * 8 + h) * 256..(t * 8 + h + 1) * 256])
                .map(|(&q, &k)| q as f64 * k as f64)
                .sum::<f64>() as f32)
        })
        .collect();
    let max = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = score.iter().map(|s| (s - max).exp()).collect();
    let sum = exp.iter().map(|&x| x as f64).sum::<f64>() as f32 + furiosa_opt_gemma4::EPS;
    let weights: Vec<f32> = exp.iter().map(|&x| bf(x)).collect();
    (0..256)
        .map(|j| {
            let combined = (0..32)
                .map(|part| {
                    bf((part * 32..((part + 1) * 32).min(n))
                        .map(|t| weights[t] as f64 * v[(t * 8 + h) * 256 + j] as f64)
                        .sum::<f64>() as f32) as f64
                })
                .sum::<f64>();
            bf(bf(combined as f32) / sum)
        })
        .collect()
}
fn cpu_boundary_diagnostic() {
    let r = 47;
    let q = values(4096, 100 + r as u64, 1.);
    let k = quantize(&ring(r, false)).decoded;
    let v = quantize(&ring(r, true)).decoded;
    let ideal = cpu_attention(&q, &k, &v, position(r), 7, 0);
    let bounded = cpu_bf16_boundaries(&q, &k, &v, position(r), 7, 0);
    report(
        "CPU-only row47 h7 g0 BF16 boundaries versus unrounded F64",
        &ideal,
        &bounded,
    );
}
async fn run(
    d: &mut Device,
    p: &Pools,
    q: &HbmTensor<bf16, Chip, m![R64, Ns, Gs, Ds]>,
    count: usize,
    label: &str,
    profile: Option<&npu_profile::Collector>,
) -> Vec<f32> {
    let idx = ui::<m![R64]>(d, (0..64).map(|r| (slot(r) * RING * 2) as i32).collect()).await;
    let qidx = ui::<m![R64]>(d, (0..64).map(|r| (slot(r) * RING) as i32).collect()).await;
    let sidx = ui::<m![R64]>(
        d,
        (0..64)
            .map(|r| (slot(r) * 1024 * 8 * 8 * 4) as i32)
            .collect(),
    )
    .await;
    let mask = uf::<m![R64, Ts]>(
        d,
        &(0..64)
            .flat_map(|r| {
                (0..1024).map(move |t| {
                    if r < count && (position(r) >= 1023 || t <= position(r)) {
                        1.
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>(),
    )
    .await;
    let mut orig = zero::<bf16, m![R64, Ns, Gs, Ds]>(d).await;
    let mut reconstructed = zero::<bf16, m![R64, Ns, Gs, Ds]>(d).await;
    let mut out = zero::<bf16, m![R64, Ns, Gs, Ds]>(d).await;
    launch(
        exact::sliding_attention_r64,
        (d, q, &p.k, &p.v, &idx, &mask, &mut orig),
    )
    .await
    .unwrap();
    launch(
        exact::sliding_attention_r64,
        (d, q, &p.dk, &p.dv, &idx, &mask, &mut reconstructed),
    )
    .await
    .unwrap();
    launch(
        fp8::sliding_attention_r64,
        (
            d, q, &p.qk, &p.ks, &p.qv, &p.vs, &qidx, &sidx, &mask, &mut out,
        ),
    )
    .await
    .unwrap();
    let result = read(d, &out).await;
    let reference = read(d, &reconstructed).await;
    report(
        &format!("{label} exact dequantized BF16"),
        &reference,
        &result,
    );
    for (i, (&a, &b)) in reference
        .iter()
        .zip(&result)
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .take(8)
    {
        println!("ATTN MISMATCH i={i} expected={a} got={b}");
    }
    assert!(
        reference
            .iter()
            .zip(&result)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "dequantized control exact"
    );
    assert!(
        result[count * 4096..].iter().all(|&x| x == 0.),
        "inactive finite zero"
    );
    let original = read(d, &orig).await;
    report(
        &format!("{label} versus original BF16 (approximation, NOT a quality gate)"),
        &original,
        &result,
    );
    if label == "initial" {
        for r in 0..count {
            report(
                &format!("BF16_DELTA row={r} pos={}", position(r)),
                &original[r * 4096..(r + 1) * 4096],
                &result[r * 4096..(r + 1) * 4096],
            );
        }
        for r in [0, 1, 3, 4, 5, 6, 7, 31, 32, 47, 63] {
            let k = quantize(&ring(r, false)).decoded;
            let v = quantize(&ring(r, true)).decoded;
            let qv = values(4096, 100 + r as u64, if r < 32 { 0.2 } else { 1. });
            for (h, g) in [(0, 0), (3, 1), (7, 0)] {
                let expected = cpu_attention(&qv, &k, &v, position(r), h, g);
                let got = &result[r * 4096 + (h * 2 + g) * 256..r * 4096 + (h * 2 + g + 1) * 256];
                let baseline =
                    &reference[r * 4096 + (h * 2 + g) * 256..r * 4096 + (h * 2 + g + 1) * 256];
                report(
                    &format!("CPU_F64_UNROUNDED candidate row={r} h={h} g={g}"),
                    &expected,
                    got,
                );
                report(
                    &format!("CPU_F64_UNROUNDED BF16control row={r} h={h} g={g}"),
                    &expected,
                    baseline,
                );
                let bounded = cpu_bf16_boundaries(&qv, &k, &v, position(r), h, g);
                let e = report(
                    &format!("CPU_BF16_BOUNDARIES row={r} h={h} g={g}"),
                    &bounded,
                    got,
                );
                assert!(
                    e < 0.01 || bounded.iter().all(|x| x.abs() < 1e-30),
                    "independent F64 with original BF16 score, weight, 32-key partial, combined and output boundaries"
                );
            }
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for which in [0, 1, 1, 0, 0, 1, 1, 0] {
            let t = Instant::now();
            if which == 0 {
                launch(
                    exact::sliding_attention_r64,
                    (d, q, &p.k, &p.v, &idx, &mask, &mut orig),
                )
                .await
                .unwrap();
                a.push(t.elapsed().as_secs_f64() * 1e6);
            } else {
                launch(
                    fp8::sliding_attention_r64,
                    (
                        d, q, &p.qk, &p.ks, &p.qv, &p.vs, &qidx, &sidx, &mask, &mut out,
                    ),
                )
                .await
                .unwrap();
                b.push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
        println!(
            "ABBA count={count} BF16_us={a:?} E4_us={b:?}; quantization/host metadata excluded"
        );
        if let Some(c) = profile {
            c.enable(true);
            println!("PROFILE BF16");
            launch(
                exact::sliding_attention_r64,
                (d, q, &p.k, &p.v, &idx, &mask, &mut orig),
            )
            .await
            .unwrap();
            println!("PROFILE E4");
            launch(
                fp8::sliding_attention_r64,
                (
                    d, q, &p.qk, &p.ks, &p.qv, &p.vs, &qidx, &sidx, &mask, &mut out,
                ),
            )
            .await
            .unwrap();
            c.enable(false);
        }
    }
    result
}
#[tokio::main]
async fn main() {
    check_cpu_quantizer();
    cpu_boundary_diagnostic();
    if std::env::args().any(|x| x == "--cpu-only") {
        return;
    }
    let start = Instant::now();
    let collector = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone()).unwrap();
    let mut d = Device::new(fp8::sliding_attention_r64.topology()).unwrap();
    let mut p = Pools {
        k: zero(&mut d).await,
        v: zero(&mut d).await,
        dk: zero(&mut d).await,
        dv: zero(&mut d).await,
        qk: zero(&mut d).await,
        qv: zero(&mut d).await,
        ks: zero(&mut d).await,
        vs: zero(&mut d).await,
    };
    let qv: Vec<f32> = (0..64)
        .flat_map(|r| values(4096, 100 + r, if r < 32 { 0.2 } else { 1. }))
        .collect();
    let q = ub::<m![R64, Ns, Gs, Ds]>(&mut d, &qv).await;
    for r in 0..64 {
        copy(
            &mut d,
            &mut p,
            r,
            &ring(r, false),
            &ring(r, true),
            r == 0 || r == 63,
        )
        .await;
    }
    println!(
        "SETUP seconds={:.3}; K/V BF16=512MiB E4=256MiB scales=32MiB",
        start.elapsed().as_secs_f64()
    );
    let before = run(
        &mut d,
        &p,
        &q,
        64,
        "initial",
        if std::env::args().any(|x| x == "--profile") {
            Some(&collector)
        } else {
            None
        },
    )
    .await;
    let (mut k, mut v) = (ring(0, false), ring(0, true));
    for t in 1..1024 {
        for h in 0..8 {
            for j in 0..256 {
                k[(t * 8 + h) * 256 + j] = bf(qv[(h * 2) * 256 + j] * 128.);
                v[(t * 8 + h) * 256 + j] = bf(1000. + j as f32);
            }
        }
    }
    copy(&mut d, &mut p, 0, &k, &v, false).await;
    let poison = run(&mut d, &p, &q, 64, "future-poison", None).await;
    assert_eq!(
        before, poison,
        "per-token scaling and masking preserve future invariance"
    );
    for r in 0..7 {
        let (mut k, mut v) = (ring(r, false), ring(r, true));
        let t = position(r) % 1024;
        k[t * 2048..(t + 1) * 2048].copy_from_slice(&values(2048, 4000 + r as u64, 0.2));
        v[t * 2048..(t + 1) * 2048].copy_from_slice(&values(2048, 5000 + r as u64, 1.));
        copy(&mut d, &mut p, r, &k, &v, false).await;
    }
    run(&mut d, &p, &q, 7, "sparse7", None).await;
    run(&mut d, &p, &q, 64, "regrow64", None).await;
    println!(
        "PASS isolated E4 KV quantization, dequantized attention bit parity, independent F64, finite future invariance and 64->7->64; elapsed={:.3}s. BF16/model quality remains separate.",
        start.elapsed().as_secs_f64()
    );
}

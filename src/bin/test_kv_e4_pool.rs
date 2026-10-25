//! E4 append overhead and all-width BF16 round-trip codec byte oracles.
#[path = "support/npu_profile.rs"]
mod npu_profile;
use fp8::Scale8;
use furiosa_opt_gemma4::{
    Chip, axes::*, ops, ops_kv_e4_pool as pool, ops_pool_views::Rc64,
    ops_request_sliding_b16_e4 as attention, ops_request_sliding_r64 as exact,
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

fn bytes_bf(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}
fn bytecheck(label: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len());
    let mut n = 0;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        if a != b {
            if n < 5 {
                println!("BYTE_DIFF {label} index={i} expected={a:02x} got={b:02x}");
            }
            n += 1;
        }
    }
    println!("BYTE_ORACLE {label} bytes={} differences={n}", a.len());
    assert_eq!(n, 0);
}
fn source(n: usize, seed: u64) -> Vec<f32> {
    let mut x = values(n, seed, 3.0);
    for (h, row) in x.chunks_mut(256).enumerate() {
        let scale = 2f32.powi((h % 10) as i32 - 6);
        for x in row {
            *x *= scale;
        }
    }
    x[..256].fill(0.);
    for (i, x) in x[256..512].iter_mut().enumerate() {
        *x = f32::from_bits(if i % 2 == 0 { 0x00400000 } else { 0x80400000 });
    }
    x
}

fn expected_write(data: &mut [u8], scales: &mut [u8], x: &Packed, locations: &[(usize, usize)]) {
    for (r, &(slot, pos)) in locations.iter().enumerate() {
        let t = pos % 1024;
        let at = (slot * 1024 + t) * 2048;
        data[at..at + 2048].copy_from_slice(&x.codes[r * 2048..(r + 1) * 2048]);
        for h in 0..8 {
            let at = ((slot * 8 + h) * 1024 + t) * 32;
            for j in 0..8 {
                scales[at + j * 4..at + j * 4 + 4]
                    .copy_from_slice(&x.scales[(r * 8 + h) * 8 + j].to_le_bytes());
            }
        }
    }
}
fn decoded_ring(data: &[u8], scales: &[u8], slot: usize) -> Vec<f32> {
    (0..RING)
        .map(|i| {
            let t = i / 2048;
            let h = i / 256 % 8;
            let at = ((slot * 8 + h) * 1024 + t) * 32;
            let scale = f32::from_le_bytes(scales[at..at + 4].try_into().unwrap());
            bf(decode_e4(data[slot * RING + i]) * scale)
        })
        .collect()
}
#[tokio::main]
async fn main() {
    let start = Instant::now();
    let mut d = Device::new(pool::capture_p128.topology()).unwrap();
    let mut pk = launch(pool::zero_data_pool, &mut d).await.unwrap();
    let mut pv = launch(pool::zero_data_pool, &mut d).await.unwrap();
    let mut pks = launch(pool::zero_scale_pool, &mut d).await.unwrap();
    let mut pvs = launch(pool::zero_scale_pool, &mut d).await.unwrap();
    let (mut ek, mut ev) = (vec![0u8; 64 * RING], vec![0u8; 64 * RING]);
    let (mut eks, mut evs) = (vec![0u8; 64 * 8 * 1024 * 32], vec![0u8; 64 * 8 * 1024 * 32]);
    let (mut k, v) = (source(128 * 2048, 901), source(128 * 2048, 907));
    k[0] = 1.9921875;
    k[1] = 2f32.powi(-16);
    let (ck, cv) = (quantize(&k), quantize(&v));
    assert_ne!(
        ck.decoded[1],
        quantize(&ck.decoded).decoded[1],
        "test must expose unsafe requantization"
    );
    let mut kh = ub::<m![P128, Ns, Ds]>(&mut d, &k).await;
    let mut vh = ub::<m![P128, Ns, Ds]>(&mut d, &v).await;
    let mut qk = zero::<f8e4m3, m![P128, Ns, Ds]>(&mut d).await;
    let mut qv = zero::<f8e4m3, m![P128, Ns, Ds]>(&mut d).await;
    let mut ks = zero::<f32, m![P128, Ns, Scale8]>(&mut d).await;
    let mut vs = zero::<f32, m![P128, Ns, Scale8]>(&mut d).await;
    launch(
        pool::capture_p128,
        (&mut d, &mut kh, &mut vh, &mut qk, &mut qv, &mut ks, &mut vs),
    )
    .await
    .unwrap();
    bytecheck(
        "capture128 K",
        &bytes_bf(&ck.decoded),
        &raw(&mut d, &kh).await,
    );
    bytecheck(
        "capture128 V",
        &bytes_bf(&cv.decoded),
        &raw(&mut d, &vh).await,
    );
    bytecheck("capture128 E4 K", &ck.codes, &raw(&mut d, &qk).await);
    bytecheck("capture128 E4 V", &cv.codes, &raw(&mut d, &qv).await);
    bytecheck(
        "capture128 scaleK",
        &ck.scales
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        &raw(&mut d, &ks).await,
    );
    bytecheck(
        "capture128 scaleV",
        &cv.scales
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        &raw(&mut d, &vs).await,
    );
    let locations: Vec<_> = (0..128).map(|r| (63, 960 + r)).collect();
    let oi = ui::<m![P128]>(
        &mut d,
        locations
            .iter()
            .map(|&(s, p)| ((s * 1024 + p % 1024) * 2048) as i32)
            .collect(),
    )
    .await;
    let si = ui::<m![P128, Ns]>(
        &mut d,
        locations
            .iter()
            .flat_map(|&(s, p)| (0..8).map(move |h| (((s * 8 + h) * 1024 + p % 1024) * 32) as i32))
            .collect(),
    )
    .await;
    launch(
        pool::commit_p128,
        (
            &mut d, &qk, &qv, &ks, &vs, &oi, &si, &mut pk, &mut pv, &mut pks, &mut pvs,
        ),
    )
    .await
    .unwrap();
    expected_write(&mut ek, &mut eks, &ck, &locations);
    expected_write(&mut ev, &mut evs, &cv, &locations);
    let mut kh = ub::<m![B, Ns, Ds]>(&mut d, &k[..16 * 2048]).await;
    let mut vh = ub::<m![B, Ns, Ds]>(&mut d, &v[..16 * 2048]).await;
    let mut qk = zero::<f8e4m3, m![B, Ns, Ds]>(&mut d).await;
    let mut qv = zero::<f8e4m3, m![B, Ns, Ds]>(&mut d).await;
    let mut ks = zero::<f32, m![B, Ns, Scale8]>(&mut d).await;
    let mut vs = zero::<f32, m![B, Ns, Scale8]>(&mut d).await;
    launch(
        pool::capture_b16,
        (&mut d, &mut kh, &mut vh, &mut qk, &mut qv, &mut ks, &mut vs),
    )
    .await
    .unwrap();
    bytecheck(
        "capture16 K",
        &bytes_bf(&ck.decoded[..16 * 2048]),
        &raw(&mut d, &kh).await,
    );
    bytecheck(
        "capture16 V",
        &bytes_bf(&cv.decoded[..16 * 2048]),
        &raw(&mut d, &vh).await,
    );
    bytecheck(
        "capture16 E4K",
        &ck.codes[..16 * 2048],
        &raw(&mut d, &qk).await,
    );
    macro_rules! commit {
        ($n:literal,$kernel:path) => {{
            let locations: Vec<_> = (0..$n).map(|r| ($n, 1018 + r)).collect();
            let oi = ui::<m![B = $n]>(
                &mut d,
                locations
                    .iter()
                    .map(|&(s, p)| ((s * 1024 + p % 1024) * 2048) as i32)
                    .collect(),
            )
            .await;
            let si = ui::<m![B = $n, Ns]>(
                &mut d,
                locations
                    .iter()
                    .flat_map(|&(s, p)| {
                        (0..8).map(move |h| (((s * 8 + h) * 1024 + p % 1024) * 32) as i32)
                    })
                    .collect(),
            )
            .await;
            launch(
                $kernel,
                (
                    &mut d, &qk, &qv, &ks, &vs, &oi, &si, &mut pk, &mut pv, &mut pks, &mut pvs,
                ),
            )
            .await
            .unwrap();
            expected_write(&mut ek, &mut eks, &ck, &locations);
            expected_write(&mut ev, &mut evs, &cv, &locations);
        }};
    }
    commit!(1, pool::commit_b16_1);
    commit!(2, pool::commit_b16_2);
    commit!(3, pool::commit_b16_3);
    commit!(4, pool::commit_b16_4);
    commit!(5, pool::commit_b16_5);
    commit!(6, pool::commit_b16_6);
    commit!(7, pool::commit_b16_7);
    commit!(8, pool::commit_b16_8);
    commit!(9, pool::commit_b16_9);
    commit!(10, pool::commit_b16_10);
    commit!(11, pool::commit_b16_11);
    commit!(12, pool::commit_b16_12);
    commit!(13, pool::commit_b16_13);
    commit!(14, pool::commit_b16_14);
    commit!(15, pool::commit_b16_15);
    commit!(16, pool::commit_b16_16);

    bytecheck(
        "dataK entire pool and neighbors",
        &ek,
        &raw(&mut d, &pk).await,
    );
    bytecheck(
        "dataV entire pool and neighbors",
        &ev,
        &raw(&mut d, &pv).await,
    );
    bytecheck(
        "scaleK entire pool and neighbors",
        &eks,
        &raw(&mut d, &pks).await,
    );
    bytecheck(
        "scaleV entire pool and neighbors",
        &evs,
        &raw(&mut d, &pvs).await,
    );
    let mut rk = zero::<bf16, m![Ts, Ns, Ds]>(&mut d).await;
    let mut rv = zero::<bf16, m![Ts, Ns, Ds]>(&mut d).await;
    let query = values(16 * 4096, 991, 0.3);
    let qh = ub::<m![B, Ns, Gs, Ds]>(&mut d, &query).await;
    let mut attn = zero::<bf16, m![B, Ns, Gs, Ds]>(&mut d).await;
    let indices = ui::<m![B]>(&mut d, (1..17).map(|s| (s * RING) as i32).collect()).await;
    let scale_indices = ui::<m![B]>(
        &mut d,
        (1..17).map(|s| (s * 8 * 1024 * 32) as i32).collect(),
    )
    .await;
    let masks = uf::<m![B, Ts]>(&mut d, &vec![1.; 16 * 1024]).await;
    launch(
        attention::sliding_attention_b16,
        (
            &mut d,
            &qh,
            &pk,
            &pks,
            &pv,
            &pvs,
            &indices,
            &scale_indices,
            &masks,
            &mut attn,
        ),
    )
    .await
    .unwrap();
    let actual = raw(&mut d, &attn).await;
    let mask = uf::<m![Ts]>(&mut d, &vec![1.; 1024]).await;
    for slot in (0..17).chain([31, 63]) {
        let data = ui::<m![1]>(&mut d, vec![(slot * RING) as i32]).await;
        let scales = ui::<m![1]>(&mut d, vec![(slot * 8 * 1024 * 32) as i32]).await;
        launch(
            pool::dequant_ring,
            (
                &mut d, &pk, &pks, &pv, &pvs, &data, &scales, &mut rk, &mut rv,
            ),
        )
        .await
        .unwrap();
        bytecheck(
            &format!("dequant ring{slot} K"),
            &bytes_bf(&decoded_ring(&ek, &eks, slot)),
            &raw(&mut d, &rk).await,
        );
        bytecheck(
            &format!("dequant ring{slot} V"),
            &bytes_bf(&decoded_ring(&ev, &evs, slot)),
            &raw(&mut d, &rv).await,
        );
        if (1..17).contains(&slot) {
            let qr = ub::<m![Ns, Gs, Ds]>(&mut d, &query[(slot - 1) * 4096..slot * 4096]).await;
            let mut expected = zero::<bf16, m![Ns, Gs, Ds]>(&mut d).await;
            launch(
                ops::sliding_attention,
                (&mut d, &qr, &rk, &rv, &mask, &mut expected),
            )
            .await
            .unwrap();
            bytecheck(
                &format!("B16 E4 attention row{}", slot - 1),
                &raw(&mut d, &expected).await,
                &actual[(slot - 1) * 8192..slot * 8192],
            );
        }
    }
    println!(
        "PASS retained P128/B16 codec, non-idempotent quantization witness, all live1..16 commits, wrap, full-pool neighbors, scratch reconstruction and B16 reader; elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
}

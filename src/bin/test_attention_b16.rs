//! Deterministic NPU/CPU attention comparisons, page/ring boundaries, and future-key invariance.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_attention as full, ops_sliding_batch as sliding};
use furiosa_opt_std::prelude::*;
use sliding::Bk;

fn bf(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff0000)
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
async fn bf_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    let bytes: Vec<u8> = x
        .iter()
        .flat_map(|x| ((bf(*x).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    HostTensor::<bf16, Mp>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn f_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<f32, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    HostTensor::<f32, Mp>::from_buf(x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
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
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<f32> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec().into_iter().map(bf16::to_f32).collect()
}
fn compare(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let mut err = 0f64;
    let mut den = 0f64;
    let mut max = 0f32;
    for (&a, &b) in got.iter().zip(want) {
        assert!(a.is_finite(), "{label}: nonfinite");
        err += (a as f64 - b as f64).powi(2);
        den += (b as f64).powi(2);
        max = max.max((a - b).abs());
    }
    let nrms = (err / den.max(1e-30)).sqrt();
    println!("{label}: n={} nrms={nrms:.7} max_abs={max:.7}", got.len());
    assert!(nrms < 0.025 && max < 0.035, "{label}: numerical mismatch");
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    bf(a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>())
}

fn full_reference(q: &[f32], k: &[f32], v: &[f32], start: usize, valid: usize) -> Vec<f32> {
    let mut out = vec![0.; valid * 16 * 512];
    for row in 0..valid {
        for h in 0..16 {
            let qr = &q[(row * 16 + h) * 512..(row * 16 + h + 1) * 512];
            let mut num = vec![0.; 512];
            let mut max = -1e30f32;
            let mut sum = 0f32;
            for page in 0..=(start + row) / 512 {
                let begin = page * 512;
                let visible = (start + row + 1 - begin).min(512);
                let mut scores = vec![-1e30; 512];
                for t in 0..visible {
                    scores[t] = dot(qr, &k[(begin + t) * 512..(begin + t + 1) * 512]);
                }
                let m = scores.iter().copied().fold(max, f32::max);
                let rescale = (max - m).exp();
                let weights: Vec<f32> = scores.iter().map(|x| (*x - m).exp()).collect();
                let new_sum = weights.iter().sum::<f32>() + 1e-6;
                let weights: Vec<f32> = weights.into_iter().map(bf).collect();
                for d in 0..512 {
                    let mut total = 0.;
                    for block in 0..16 {
                        let mut partial = 0.;
                        for t in block * 32..block * 32 + 32 {
                            partial += weights[t] * v[(begin + t) * 512 + d];
                        }
                        total += bf(partial);
                    }
                    num[d] = bf(bf(num[d] * rescale) + bf(total));
                }
                sum = sum * rescale + new_sum;
                max = m;
            }
            for d in 0..512 {
                out[(row * 16 + h) * 512 + d] = bf(num[d] / sum);
            }
        }
    }
    out
}
async fn full_run(
    d: &mut Device,
    qv: &[f32],
    kv: &[f32],
    vv: &[f32],
    start: usize,
    valid: usize,
) -> Vec<f32> {
    let q: HbmTensor<bf16, Chip, m![B, Gf, Df]> = bf_upload(d, qv).await;
    let mut max: HbmTensor<f32, Chip, m![B,Gf,1 # 8]> = zero(d).await;
    let mut sum: HbmTensor<f32, Chip, m![B,Gf,1 # 8]> = zero(d).await;
    let mut out: HbmTensor<bf16, Chip, m![B, Gf, Df]> = zero(d).await;
    for page in 0..=(start + valid - 1) / 512 {
        let a = page * 512 * 512;
        let k = bf_upload::<m![Tf, Df]>(d, &kv[a..a + 512 * 512]).await;
        let v = bf_upload::<m![Tf, Df]>(d, &vv[a..a + 512 * 512]).await;
        let mask: Vec<f32> = (0..16)
            .flat_map(|r| {
                (0..512).map(move |t| {
                    if r < valid && page * 512 + t <= start + r {
                        0.
                    } else {
                        -1e30
                    }
                })
            })
            .collect();
        let mask = f_upload::<m![B, Tf]>(d, &mask).await;
        if page == 0 {
            launch(
                full::full_attention_first_page_b16,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        } else {
            launch(
                full::full_attention_next_page_b16,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        }
    }
    let mut divided = zero::<bf16, m![B, Gf, Df]>(d).await;
    launch(
        full::full_attention_divide_b16,
        (d, &out, &sum, &mut divided),
    )
    .await
    .unwrap();
    read(d, &divided).await
}
async fn full_serial(d: &mut Device, qv: &[f32], kv: &[f32], vv: &[f32], pos: usize) -> Vec<f32> {
    let q = bf_upload::<m![Gf, Df]>(d, qv).await;
    let mut max = zero::<f32, m![Gf]>(d).await;
    let mut sum = zero::<f32, m![Gf]>(d).await;
    let mut out = zero::<bf16, m![Gf, Df]>(d).await;
    for page in 0..=pos / 512 {
        let a = page * 512 * 512;
        let k = bf_upload::<m![Tf, Df]>(d, &kv[a..a + 512 * 512]).await;
        let v = bf_upload::<m![Tf, Df]>(d, &vv[a..a + 512 * 512]).await;
        let mask: Vec<f32> = (0..512)
            .map(|t| {
                if page * 512 + t <= pos {
                    f32::INFINITY
                } else {
                    0.
                }
            })
            .collect();
        let mask = f_upload::<m![Tf]>(d, &mask).await;
        if page == 0 {
            launch(
                ops::full_attention_first_page,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        } else {
            launch(
                ops::full_attention_page,
                (d, &q, &k, &v, &mask, &mut max, &mut sum, &mut out),
            )
            .await
            .unwrap();
        }
    }
    let sum: HostTensor<f32, m![Gf]> = sum.to_host(&mut d.pdma).await.unwrap();
    let sum = sum.into_vec();
    read(d, &out)
        .await
        .into_iter()
        .enumerate()
        .map(|(i, x)| bf(x / sum[i / 512]))
        .collect()
}
async fn test_full(d: &mut Device, start: usize, valid: usize) {
    assert!(start % 512 + valid <= 512);
    let size = ((start + valid + 511) / 512) * 512 * 512;
    let q = values(16 * 16 * 512, 1000 + start as u64, 0.2);
    let k = values(size, 2000, 0.2);
    let v = values(size, 3000, 1.);
    let want = full_reference(&q, &k, &v, start, valid);
    let got = full_run(d, &q, &k, &v, start, valid).await;
    compare(
        &format!("full start={start} valid={valid} CPU"),
        &got[..want.len()],
        &want,
    );
    let row = valid - 1;
    let serial = full_serial(d, &q[row * 8192..(row + 1) * 8192], &k, &v, start + row).await;
    compare(
        &format!("full start={start} serial-last"),
        &got[row * 8192..(row + 1) * 8192],
        &serial,
    );
    if valid > 1 {
        let mut future = k.clone();
        for x in &mut future[(start + valid - 1) * 512..(start + valid) * 512] {
            *x = bf(*x * 64. + 4.);
        }
        let changed = full_run(d, &q, &future, &v, start, valid).await;
        assert_eq!(
            &got[..8192],
            &changed[..8192],
            "full future key changed earlier row"
        );
        println!("full start={start} future-key invariance exact");
    }
}

fn old_position(start: usize, slot: usize) -> Option<usize> {
    if start == 0 {
        return None;
    }
    let last = start - 1;
    let p = last as isize - ((last + 1024 - slot) % 1024) as isize;
    (p >= 0).then_some(p as usize)
}
fn sliding_masks(start: usize, valid: usize) -> (Vec<f32>, Vec<f32>) {
    let old = (0..16)
        .flat_map(|r| {
            (0..1024).map(move |t| {
                if r < valid && old_position(start, t).is_some_and(|p| p + 1024 > start + r) {
                    0.
                } else {
                    -1e30
                }
            })
        })
        .collect();
    let new = (0..16)
        .flat_map(|r| {
            (0..16).map(move |t| {
                if r < valid && t <= r && t < valid {
                    0.
                } else {
                    -1e30
                }
            })
        })
        .collect();
    (old, new)
}
fn sliding_reference(
    q: &[f32],
    oldk: &[f32],
    oldv: &[f32],
    newk: &[f32],
    newv: &[f32],
    start: usize,
    valid: usize,
) -> Vec<f32> {
    let (om, nm) = sliding_masks(start, valid);
    let mut out = vec![0.; valid * 4096];
    for r in 0..valid {
        for h in 0..8 {
            for g in 0..2 {
                let qi = (r * 16 + h * 2 + g) * 256;
                let qr = &q[qi..qi + 256];
                let mut os = vec![-1e30; 1024];
                let mut ns = vec![-1e30; 16];
                for t in 0..1024 {
                    if om[r * 1024 + t] == 0. {
                        let a = (t * 8 + h) * 256;
                        os[t] = dot(qr, &oldk[a..a + 256]);
                    }
                }
                for t in 0..16 {
                    if nm[r * 16 + t] == 0. {
                        let a = (t * 8 + h) * 256;
                        ns[t] = dot(qr, &newk[a..a + 256]);
                    }
                }
                let max = os.iter().chain(&ns).copied().fold(-1e30f32, f32::max);
                let ow: Vec<f32> = os.iter().map(|x| (*x - max).exp()).collect();
                let nw: Vec<f32> = ns.iter().map(|x| (*x - max).exp()).collect();
                let sum = ow.iter().sum::<f32>() + nw.iter().sum::<f32>() + 1e-6;
                let ow: Vec<f32> = ow.into_iter().map(bf).collect();
                let nw: Vec<f32> = nw.into_iter().map(bf).collect();
                for d in 0..256 {
                    let mut total = 0.;
                    for block in 0..32 {
                        let mut partial = 0.;
                        for t in block * 32..block * 32 + 32 {
                            partial += ow[t] * oldv[(t * 8 + h) * 256 + d];
                        }
                        total += bf(partial);
                    }
                    let mut new = 0.;
                    for t in 0..16 {
                        new += nw[t] * newv[(t * 8 + h) * 256 + d];
                    }
                    out[qi + d] = bf(bf(bf(total) + bf(new)) / sum);
                }
            }
        }
    }
    out
}
async fn sliding_run(
    d: &mut Device,
    q: &[f32],
    ok: &[f32],
    ov: &[f32],
    nk: &[f32],
    nv: &[f32],
    start: usize,
    valid: usize,
) -> Vec<f32> {
    let q = bf_upload::<m![B, Ns, Gs, Ds]>(d, q).await;
    let ok = bf_upload::<m![Ts, Ns, Ds]>(d, ok).await;
    let ov = bf_upload::<m![Ts, Ns, Ds]>(d, ov).await;
    let nk = bf_upload::<m![B, Ns, Ds]>(d, nk).await;
    let nv = bf_upload::<m![B, Ns, Ds]>(d, nv).await;
    let (om, nm) = sliding_masks(start, valid);
    let om = f_upload::<m![B, Ts]>(d, &om).await;
    let nm = f_upload::<m![B, Bk]>(d, &nm).await;
    let mut out = zero::<bf16, m![B, Ns, Gs, Ds]>(d).await;
    launch(
        sliding::sliding_attention_b16,
        (d, &q, &ok, &ov, &nk, &nv, &om, &nm, &mut out),
    )
    .await
    .unwrap();
    read(d, &out).await
}
async fn test_sliding(d: &mut Device, start: usize, valid: usize) {
    let q = values(16 * 4096, 4000 + start as u64, 0.2);
    let ok = values(1024 * 2048, 5000, 0.2);
    let ov = values(1024 * 2048, 6000, 1.);
    let nk = values(16 * 2048, 7000 + start as u64, 0.2);
    let nv = values(16 * 2048, 8000 + start as u64, 1.);
    let want = sliding_reference(&q, &ok, &ov, &nk, &nv, start, valid);
    let got = sliding_run(d, &q, &ok, &ov, &nk, &nv, start, valid).await;
    compare(
        &format!("sliding start={start} valid={valid} CPU"),
        &got[..want.len()],
        &want,
    );
    let r = valid - 1;
    let pos = start + r;
    let mut sk = ok.clone();
    let mut sv = ov.clone();
    for row in 0..valid {
        let slot = (start + row) % 1024;
        sk[slot * 2048..(slot + 1) * 2048].copy_from_slice(&nk[row * 2048..(row + 1) * 2048]);
        sv[slot * 2048..(slot + 1) * 2048].copy_from_slice(&nv[row * 2048..(row + 1) * 2048]);
    }
    let sq = bf_upload::<m![Ns, Gs, Ds]>(d, &q[r * 4096..(r + 1) * 4096]).await;
    let sk = bf_upload::<m![Ts, Ns, Ds]>(d, &sk).await;
    let sv = bf_upload::<m![Ts, Ns, Ds]>(d, &sv).await;
    let sm: Vec<f32> = (0..1024)
        .map(|t| {
            if old_position(pos + 1, t).is_some_and(|p| p + 1024 > pos) {
                f32::INFINITY
            } else {
                0.
            }
        })
        .collect();
    let sm = f_upload::<m![Ts]>(d, &sm).await;
    let mut so = zero::<bf16, m![Ns, Gs, Ds]>(d).await;
    launch(ops::sliding_attention, (d, &sq, &sk, &sv, &sm, &mut so))
        .await
        .unwrap();
    let serial = read(d, &so).await;
    compare(
        &format!("sliding start={start} serial-last"),
        &got[r * 4096..(r + 1) * 4096],
        &serial,
    );
    if valid > 1 {
        let mut future = nk.clone();
        for x in &mut future[(valid - 1) * 2048..valid * 2048] {
            *x = bf(*x * 64. + 4.);
        }
        let changed = sliding_run(d, &q, &ok, &ov, &future, &nv, start, valid).await;
        assert_eq!(
            &got[..4096],
            &changed[..4096],
            "sliding future key changed earlier row"
        );
        println!("sliding start={start} future-key invariance exact");
    }
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(full::full_attention_first_page_b16.topology()).unwrap();
    for (s, n) in [
        (0, 1),
        (0, 16),
        (496, 16),
        (512, 16),
        (1008, 16),
        (1024, 16),
    ] {
        test_full(&mut d, s, n).await;
    }
    for (s, n) in [
        (0, 1),
        (0, 16),
        (496, 16),
        (512, 16),
        (1008, 16),
        (1020, 4),
        (1024, 16),
        (1032, 7),
    ] {
        test_sliding(&mut d, s, n).await;
    }
    println!("PASS: full/sliding B16 causal boundaries, CPU and original-row references");
}

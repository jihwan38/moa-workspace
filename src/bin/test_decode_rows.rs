//! Independent request row adapters: cache isolation and original attention parity.
use furiosa_opt_gemma4::{Chip, axes::*, ops, ops_decode as rows};
use furiosa_opt_std::prelude::*;
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

async fn index(d: &mut Device, n: i32) -> HbmTensor<i32, Chip, m![1]> {
    HostTensor::<i32, m![1]>::from_vec(vec![n])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn sliding_rows(d: &mut Device) {
    let qv = values(16 * 4096, 101, 0.2);
    let kv = values(16 * 2048, 102, 0.2);
    let vv = values(16 * 2048, 103, 1.);
    let q = bf_upload::<m![B, Ns, Gs, Ds]>(d, &qv).await;
    let k = bf_upload::<m![B, Ns, Ds]>(d, &kv).await;
    let v = bf_upload::<m![B, Ns, Ds]>(d, &vv).await;
    let mut expected = vec![bf(-3.25); 16 * 4096];
    let mut output = bf_upload::<m![B, Ns, Gs, Ds]>(d, &expected).await;
    for (row, pos) in [(15, 0), (0, 511), (7, 512), (3, 1023), (11, 1024)] {
        let mut ck = values(1024 * 2048, 200 + row as u64, 0.2);
        let mut cv = values(1024 * 2048, 300 + row as u64, 1.);
        let mut kh = bf_upload::<m![Ts, Ns, Ds]>(d, &ck).await;
        let mut vh = bf_upload::<m![Ts, Ns, Ds]>(d, &cv).await;
        let ri = index(d, row as i32).await;
        let off = index(d, ((pos % 1024) * 2048 * 2) as i32).await;
        let mask: Vec<f32> = (0..1024)
            .map(|t| {
                if pos >= 1023 || t <= pos {
                    f32::INFINITY
                } else {
                    0.
                }
            })
            .collect();
        let mask = f_upload::<m![Ts]>(d, &mask).await;
        launch(
            rows::sliding_attention_row,
            (
                d,
                &q,
                &k,
                &v,
                &ri,
                &off,
                &mut kh,
                &mut vh,
                &mask,
                &mut output,
            ),
        )
        .await
        .unwrap();
        let slot = pos % 1024;
        ck[slot * 2048..(slot + 1) * 2048].copy_from_slice(&kv[row * 2048..(row + 1) * 2048]);
        cv[slot * 2048..(slot + 1) * 2048].copy_from_slice(&vv[row * 2048..(row + 1) * 2048]);
        assert_eq!(read(d, &kh).await, ck, "sliding K cache commit row={row}");
        assert_eq!(read(d, &vh).await, cv, "sliding V cache commit row={row}");
        let qr = bf_upload::<m![Ns, Gs, Ds]>(d, &qv[row * 4096..(row + 1) * 4096]).await;
        let mut reference = zero::<bf16, m![Ns, Gs, Ds]>(d).await;
        launch(
            ops::sliding_attention,
            (d, &qr, &kh, &vh, &mask, &mut reference),
        )
        .await
        .unwrap();
        let reference = read(d, &reference).await;
        expected[row * 4096..(row + 1) * 4096].copy_from_slice(&reference);
        let got = read(d, &output).await;
        assert_eq!(got, expected, "sliding row parity/inactive rows row={row}");
        println!(
            "sliding row={row} pos={pos}: exact original attention, exact full-cache commit and inactive output preservation"
        );
    }
}
async fn full_rows(d: &mut Device) {
    let qv = values(16 * 8192, 401, 0.2);
    let kv = values(16 * 512, 402, 0.2);
    let vv = values(16 * 512, 403, 1.);
    let q = bf_upload::<m![B, Gf, Df]>(d, &qv).await;
    let k = bf_upload::<m![B, Df]>(d, &kv).await;
    let v = bf_upload::<m![B, Df]>(d, &vv).await;
    let mut expected = vec![bf(-3.25); 16 * 8192];
    let mut output = bf_upload::<m![B, Gf, Df]>(d, &expected).await;
    for (row, pos) in [(15, 0), (0, 511), (7, 512), (3, 1023), (11, 1024)] {
        let mut caches = Vec::new();
        for page in 0..=pos / 512 {
            let ck = values(512 * 512, 500 + row as u64 * 10 + page as u64, 0.2);
            let cv = values(512 * 512, 600 + row as u64 * 10 + page as u64, 1.);
            let kh = bf_upload::<m![Tf, Df]>(d, &ck).await;
            let vh = bf_upload::<m![Tf, Df]>(d, &cv).await;
            caches.push((kh, vh, ck, cv));
        }
        let ri = index(d, row as i32).await;
        let off = index(d, ((pos % 512) * 512 * 2) as i32).await;
        let mut qr = zero::<bf16, m![Gf, Df]>(d).await;
        let (kh, vh, ck, cv) = &mut caches[pos / 512];
        launch(
            rows::full_commit_row,
            (d, &q, &k, &v, &ri, &off, kh, vh, &mut qr),
        )
        .await
        .unwrap();
        let slot = pos % 512;
        ck[slot * 512..(slot + 1) * 512].copy_from_slice(&kv[row * 512..(row + 1) * 512]);
        cv[slot * 512..(slot + 1) * 512].copy_from_slice(&vv[row * 512..(row + 1) * 512]);
        assert_eq!(read(d, kh).await, *ck, "full K commit row={row}");
        assert_eq!(read(d, vh).await, *cv, "full V commit row={row}");
        assert_eq!(
            read(d, &qr).await,
            &qv[row * 8192..(row + 1) * 8192],
            "full Q row gather"
        );
        let mut max = zero::<f32, m![Gf]>(d).await;
        let mut sum = zero::<f32, m![Gf]>(d).await;
        let mut num = zero::<bf16, m![Gf, Df]>(d).await;
        for (page, (kh, vh, _, _)) in caches.iter().enumerate() {
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
                    (d, &qr, kh, vh, &mask, &mut max, &mut sum, &mut num),
                )
                .await
                .unwrap();
            } else {
                launch(
                    ops::full_attention_page,
                    (d, &qr, kh, vh, &mask, &mut max, &mut sum, &mut num),
                )
                .await
                .unwrap();
            }
        }
        launch(
            rows::full_divide_pack_row,
            (d, &num, &sum, &ri, &mut output),
        )
        .await
        .unwrap();
        let sums: HostTensor<f32, m![Gf]> = sum.to_host(&mut d.pdma).await.unwrap();
        let sums = sums.into_vec();
        let reference: Vec<f32> = read(d, &num)
            .await
            .into_iter()
            .enumerate()
            .map(|(i, x)| bf(x / sums[i / 512]))
            .collect();
        let got = read(d, &output).await;
        compare(
            &format!("full row={row} pos={pos} divide"),
            &got[row * 8192..(row + 1) * 8192],
            &reference,
        );
        for r in 0..16 {
            if r != row {
                assert_eq!(
                    &got[r * 8192..(r + 1) * 8192],
                    &expected[r * 8192..(r + 1) * 8192],
                    "full untouched row={r}"
                );
            }
        }
        expected[row * 8192..(row + 1) * 8192].copy_from_slice(&got[row * 8192..(row + 1) * 8192]);
        println!(
            "full row={row} pos={pos}: exact Q/cache gather/commit and inactive output preservation"
        );
    }
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(rows::sliding_attention_row.topology()).unwrap();
    sliding_rows(&mut d).await;
    full_rows(&mut d).await;
    println!("PASS: independent-request decode attention row adapters");
}

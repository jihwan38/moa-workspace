use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::{H, L}, stream_probe};

// Independent FP4 value specification, including both signed zero codes.
const FP4: [f64; 16] = [0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.];
fn activation(case: usize, k: usize) -> f64 {
    match case {
        2 => match (k / 16 % 2, k % 16) {
            (0, 0) => 1., (0, 1) => 15. / 256.,
            (1, 0) => -1., (1, 1) => -14. / 256., _ => 0.,
        },
        3 => 0.,
        _ => ((k * 7 % 17) as f64 - 8.) / 32.,
    }
}
fn code(case: usize, r: usize, k: usize, width: usize, seed: usize) -> u8 {
    match case {
        1 => {
            let edges = [0, 1, 118, 119, 120, width / 16 - 2, width / 16 - 1];
            if k / 16 == edges[(r + seed) % edges.len()] {2} else {0}
        },
        2 => [1, 2, 4, 6][(r + seed) % 4],
        _ => {
            let identity = r + seed * 15360;
            let pair = k / 2;
            let block = k / 512;
            if k % 2 == 0 {((pair % 16 + ((identity >> ((block % 4) * 4)) & 15)) % 16) as u8}
            else {((pair / 16 + 13 * identity + 7 * block) % 16) as u8}
        },
    }
}
fn scale(case: usize, r: usize, b: usize, seed: usize) -> f64 {
    if case == 2 {1.} else {[0.25, 0.5, 1., 2.][(r * 3 + b + seed) % 4]}
}
fn inputs(case: usize, rows: usize, width: usize, seed: usize) -> (Vec<u8>, Vec<f8e4m3>, Vec<bf16>) {
    let mut packed = vec![0u8; rows * width / 2];
    let mut scales = Vec::with_capacity(rows * width / 16);
    let mut expected = Vec::with_capacity(rows);
    let mut mask = 0u32;
    for r in 0..rows {
        let mut exact = 0f64;
        for b in 0..width / 16 {
            let factor = scale(case, r, b, seed);
            scales.push(f8e4m3::from_f32(factor as f32));
            let mut dot = 0f64;
            for j in 0..16 {
                let k = b * 16 + j;
                let c = code(case, r, k, width, seed);
                packed[(r * width + k) / 2] |= c << (4 * (k % 2));
                mask |= 1 << c;
                dot += FP4[c as usize] * activation(case, k);
            }
            // Chosen binary fractions make each block exact in F32.
            assert_eq!(dot, (dot as f32) as f64);
            exact += dot * factor;
        }
        assert_eq!(exact, (exact as f32) as f64);
        if case == 2 {
            assert_eq!(exact, (width / 32) as f64 / 256. * FP4[code(case, r, 0, width, seed) as usize]);
        }
        expected.push(bf16::from_f32(exact as f32));
    }
    if case == 0 || case == 3 {assert_eq!(mask, 0xffff);}
    println!("C05_ORACLE case={case} rows={rows} width={width} seed={seed} code_mask={mask:x}");
    (packed, scales, expected)
}
fn check(case: usize, name: &str, actual: &[bf16], expected: &[bf16]) -> usize {
    assert_eq!(actual.len(), expected.len());
    let mut failures = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if a.to_f32() != e.to_f32() {
            if failures < 8 {println!("C05_MISMATCH case={case} projection={name} row={i} actual={} expected={}", a.to_f32(), e.to_f32());}
            failures += 1;
        }
    }
    println!("C05_PROBE case={case} projection={name} checked={} mismatches={failures}", actual.len());
    failures
}
#[tokio::main]
async fn main() {
    let oracle_only = std::env::args().any(|a| a == "--oracle-only");
    let mut device = if oracle_only {None} else {Some(Device::new(stream_probe::upgate.topology()).unwrap())};
    let mut failures = 0;
    for case in 0..4 {
        let (wp, ws, expected_up) = inputs(case, L::SIZE, H::SIZE, 0);
        let (gp, gs, expected_gate) = inputs(case, L::SIZE, H::SIZE, 1);
        if let Some(d) = device.as_mut() {
            let x: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::from_vec((0..H::SIZE).map(|k| bf16::from_f32(activation(case, k) as f32))).to_hbm(&mut d.pdma).await.unwrap();
            let w: HbmTensor<f4e2m1, Chip, m![L,H]> = HostTensor::<f4e2m1, m![L,H]>::from_buf(wp).to_hbm(&mut d.pdma).await.unwrap();
            let g: HbmTensor<f4e2m1, Chip, m![L,H]> = HostTensor::<f4e2m1, m![L,H]>::from_buf(gp).to_hbm(&mut d.pdma).await.unwrap();
            let s: HbmTensor<f8e4m3, Chip, m![L,H/16]> = HostTensor::<f8e4m3, m![L,H/16]>::from_vec(ws).to_hbm(&mut d.pdma).await.unwrap();
            let t: HbmTensor<f8e4m3, Chip, m![L,H/16]> = HostTensor::<f8e4m3, m![L,H/16]>::from_vec(gs).to_hbm(&mut d.pdma).await.unwrap();
            let mut y: HbmTensor<bf16, Chip, m![L]> = HostTensor::<bf16,m![L]>::from_vec(vec![bf16::from_bits(0x7fc1); L::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
            let mut z: HbmTensor<bf16, Chip, m![L]> = HostTensor::<bf16,m![L]>::from_vec(vec![bf16::from_bits(0x7fc1); L::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
            launch(stream_probe::upgate, (&mut *d, &x, &w, &g, &s, &t, &mut y, &mut z)).await.unwrap();
            let y: HostTensor<bf16,m![L]> = y.to_host(&mut d.pdma).await.unwrap();
            let z: HostTensor<bf16,m![L]> = z.to_host(&mut d.pdma).await.unwrap();
            failures += check(case,"up", &y.into_vec(), &expected_up);
            failures += check(case,"gate", &z.into_vec(), &expected_gate);
        }
        let (wp, ws, expected) = inputs(case, H::SIZE, L::SIZE, 2);
        if let Some(d) = device.as_mut() {
            let x: HbmTensor<bf16, Chip, m![L]> = HostTensor::<bf16, m![L]>::from_vec((0..L::SIZE).map(|k| bf16::from_f32(activation(case, k) as f32))).to_hbm(&mut d.pdma).await.unwrap();
            let w: HbmTensor<f4e2m1, Chip, m![H,L]> = HostTensor::<f4e2m1, m![H,L]>::from_buf(wp).to_hbm(&mut d.pdma).await.unwrap();
            let s: HbmTensor<f8e4m3, Chip, m![H,L/16]> = HostTensor::<f8e4m3, m![H,L/16]>::from_vec(ws).to_hbm(&mut d.pdma).await.unwrap();
            let mut y: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16,m![H]>::from_vec(vec![bf16::from_bits(0x7fc1); H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
            launch(stream_probe::down, (&mut *d, &x, &w, &s, &mut y)).await.unwrap();
            let y: HostTensor<bf16,m![H]> = y.to_host(&mut d.pdma).await.unwrap();
            failures += check(case,"down", &y.into_vec(), &expected);
        }
    }
    println!("C05_PROBE_COMPLETE oracle_only={oracle_only} cases=4 checked={} mismatches={failures}", if oracle_only {0} else {4 * (L::SIZE * 2 + H::SIZE)});
    if failures != 0 {std::process::exit(1);}
}

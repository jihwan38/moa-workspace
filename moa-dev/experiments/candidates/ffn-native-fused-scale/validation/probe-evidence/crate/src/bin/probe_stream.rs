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
        // Sparse input touches both sides of block16/native64 and final-H boundaries.
        6 => match k {0 => 1., 15 => 15. / 256., 16 => -1., 31 => -14. / 256.,
            32 => 0.5, 63 => 1. / 16., 64 => -0.25, 3838 => 1. / 128., 3839 => -1. / 256., _ => 0.},
        // Amax128 changes restoration scale to2^-7; smaller terms remain exactly representable.
        7 => match k {0 => 128., 1 => -128., _ => match k % 16 {15 => 1. / 16., 0 => -1. / 32., _ => 0.}},
        4 => match k {0 => 1., 1 => -1., _ => 1. / 4194304.},
        5 => match k {0 => 0.53125, 1 => -0.53125, _ => 1. / 8388608.},
        _ => ((k * 7 % 17) as f64 - 8.) / 32.,
    }
}
fn code(case: usize, r: usize, k: usize, width: usize, seed: usize) -> u8 {
    match case {
        1 => {
            let edges: Vec<usize> = [0, 1, 118, 119, 120, 238, 239, 240, 358, 359, 360, 478, 479, 480, 598, 599, 600, 718, 719, 720, 838, 839, 840, width / 16 - 2, width / 16 - 1].into_iter().filter(|b| *b < width / 16).collect();
            if k / 16 == edges[(r + seed) % edges.len()] {2} else {0}
        },
        2 | 4 | 5 => [1, 2, 4, 6][(r + seed) % 4],
        6 => {
            const EDGES: [usize; 9] = [0,15,16,31,32,63,64,3838,3839];
            if EDGES.contains(&k) { [1,2,3,4,5,6,7,9,10,11,12,13,14,15][(r + k + 5 * seed) % 14] } else {0}
        },
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
    if matches!(case, 2 | 4 | 5) {1.}
    else if case >= 6 { [0.25, 0.5, 1., 2.][(r / 6 + r % 6 + 2 * (r / 30) + r / 7680 + b + seed) % 4] }
    else {[0.25, 0.5, 1., 2.][(r * 3 + b + seed) % 4]}
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
    let packed_hash = fingerprint(packed.iter().copied());
    let scale_hash = fingerprint(scales.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    let expected_hash = fingerprint(expected.iter().flat_map(|v| v.to_bits().to_le_bytes()));
    println!("E128_ORACLE case={case} rows={rows} width={width} seed={seed} code_mask={mask:x} packed={packed_hash:016x} scales={scale_hash:016x} expected={expected_hash:016x}");
    (packed, scales, expected)
}
fn fingerprint(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}
fn check(case: usize, name: &str, actual: &[bf16], expected: &[bf16]) -> usize {
    assert_eq!(actual.len(), expected.len());
    let mut failures = 0;
    let mut bad_groups = std::collections::BTreeMap::<usize, usize>::new();
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        if !a.to_f32().is_finite() || (a.to_bits() != e.to_bits() && !(a.to_f32() == 0. && e.to_f32() == 0.)) {
            if failures < 8 {println!("E128_MISMATCH case={case} projection={name} row={i} actual={} expected={}", a.to_f32(), e.to_f32());}
            failures += 1;
            *bad_groups.entry(i / 6).or_default() += 1;
        }
    }
    println!("E128_PROBE case={case} projection={name} checked={} mismatches={failures}", actual.len());
    if failures != 0 {println!("E128_BAD_ROW_GROUPS case={case} projection={name} groups={bad_groups:?}");}
    failures
}
#[tokio::main]
async fn main() {
    let oracle_only = std::env::args().any(|a| a == "--oracle-only");
    let mut device = if oracle_only {None} else {Some(Device::new(stream_probe::upgate.topology()).unwrap())};
    let mut failures = 0;
    for case in 0..8 {
        let xv: Vec<bf16> = (0..H::SIZE).map(|k| bf16::from_f32(activation(case, k) as f32)).collect();
        for (k, x) in xv.iter().enumerate() { assert_eq!(x.to_f32() as f64, activation(case,k)); }
        let x_hash = fingerprint(xv.iter().flat_map(|v| v.to_bits().to_le_bytes()));
        println!("E128_INPUT case={case} x={x_hash:016x} count={}", xv.len());
        let (wp, ws, expected_up) = inputs(case, L::SIZE, H::SIZE, 0);
        let (gp, gs, expected_gate) = inputs(case, L::SIZE, H::SIZE, 1);
        if let Some(d) = device.as_mut() {
            let x: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::from_vec(xv).to_hbm(&mut d.pdma).await.unwrap();
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
    }
    println!("E128_PROBE_COMPLETE oracle_only={oracle_only} cases=8 checked={} mismatches={failures}", if oracle_only {0} else {8 * L::SIZE * 2});
    if failures != 0 {std::process::exit(1);}
}

//! CPU-only fixed local rotation. No model/device mapping or calibration data.
#[path = "rational_a4_oracle.rs"]
pub mod rational;

pub const SEED: u64 = 0x243f_6a88_85a3_08d3;
pub const BLOCK: usize = 64;
pub const GROUP: usize = 960;
pub const ARM_NAMES: [&str; 9] = [
    "provided_bf16",
    "ideal_f64_rotation",
    "f32_bf16_rotation_only",
    "original_A4W4",
    "rotated_A4W4",
    "original_A4_only",
    "original_W4_only",
    "rotated_A4_only",
    "rotated_W4_only",
];

pub fn sign(channel: usize) -> f32 {
    let mut z = ((channel as u64) ^ SEED).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    if z >> 63 == 0 {
        1.0
    } else {
        -1.0
    }
}
pub fn bf32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
pub fn bf16_rne(x: f32) -> u16 {
    assert!(x.is_finite());
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn normal32(x: f32) -> Result<(), String> {
    if x == 0.0 || x.is_normal() {
        Ok(())
    } else {
        Err(format!(
            "outside normal-or-zero F32 scope: {:08x}",
            x.to_bits()
        ))
    }
}
fn normal64(x: f64) -> Result<(), String> {
    if x == 0.0 || x.is_normal() {
        Ok(())
    } else {
        Err(format!(
            "outside normal-or-zero F64 scope: {:016x}",
            x.to_bits()
        ))
    }
}
fn normal_bf(bits: u16) -> Result<(), String> {
    let a = bits & 0x7fff;
    if a == 0 || (a >> 7 > 0 && a >> 7 < 255) {
        Ok(())
    } else {
        Err(format!("outside normal-or-zero BF16 scope: {bits:04x}"))
    }
}

/// Six RN32 stages followed by exact normal /8. Sign comes BEFORE H.
pub fn forward32(input: &[f32; BLOCK], base: usize) -> Result<[f32; BLOCK], String> {
    assert_eq!(base % BLOCK, 0);
    let mut a = *input;
    for (k, x) in a.iter_mut().enumerate() {
        normal32(*x)?;
        *x *= sign(base + k);
    }
    for stride in [1, 2, 4, 8, 16, 32] {
        for start in (0..BLOCK).step_by(2 * stride) {
            for j in 0..stride {
                let (x, y) = (a[start + j], a[start + j + stride]);
                a[start + j] = x + y;
                a[start + j + stride] = x - y;
                normal32(a[start + j])?;
                normal32(a[start + j + stride])?;
            }
        }
    }
    for x in &mut a {
        let old = *x;
        *x *= 0.125;
        normal32(*x)?;
        if (*x as f64) * 8.0 != old as f64 {
            return Err("nonexact F32 /8".into());
        }
    }
    Ok(a)
}

/// F64 evaluation of the ideal transform; NOT an exact-real oracle.
pub fn forward64(input: &[f64; BLOCK], base: usize) -> Result<[f64; BLOCK], String> {
    let mut a = *input;
    for (k, x) in a.iter_mut().enumerate() {
        normal64(*x)?;
        *x *= sign(base + k) as f64;
    }
    hadamard64(&mut a)?;
    Ok(a)
}
fn hadamard64(a: &mut [f64; BLOCK]) -> Result<(), String> {
    for stride in [1, 2, 4, 8, 16, 32] {
        for start in (0..BLOCK).step_by(2 * stride) {
            for j in 0..stride {
                let (x, y) = (a[start + j], a[start + j + stride]);
                a[start + j] = x + y;
                a[start + j + stride] = x - y;
                normal64(a[start + j])?;
                normal64(a[start + j + stride])?;
            }
        }
    }
    for x in a {
        let old = *x;
        *x *= 0.125;
        normal64(*x)?;
        assert_eq!(*x * 8.0, old);
    }
    Ok(())
}
pub fn inverse64(input: &[f64; BLOCK], base: usize) -> Result<[f64; BLOCK], String> {
    let mut a = *input;
    hadamard64(&mut a)?;
    for (k, x) in a.iter_mut().enumerate() {
        *x *= sign(base + k) as f64;
    }
    Ok(a)
}

pub struct Quantized {
    pub codes: Vec<i8>,
    pub scales: Vec<f32>,
}
fn quantize(bits: &[u16]) -> Quantized {
    assert_eq!(bits.len() % GROUP, 0);
    let mut q = Quantized {
        codes: Vec::with_capacity(bits.len()),
        scales: Vec::new(),
    };
    for group in bits.chunks_exact(GROUP) {
        let (codes, scale) = rational::reference(group);
        assert!(scale.is_normal() && scale > 0.0);
        q.codes
            .extend(codes.into_iter().map(|x| i8::try_from(x).unwrap()));
        q.scales.push(scale);
    }
    q
}
impl Quantized {
    fn value(&self, k: usize) -> f64 {
        // Exact in F64 (a <=3-bit integer times a stored 24-bit F32 value).
        self.codes[k] as f64 * self.scales[k / GROUP] as f64
    }
    pub fn zeros(&self) -> usize {
        self.codes.iter().filter(|&&x| x == 0).count()
    }
}
pub struct Row {
    pub original: Vec<u16>,
    pub ideal: Vec<f64>,
    pub rounded: Vec<u16>,
    pub original_q: Quantized,
    pub rotated_q: Quantized,
}
impl Row {
    pub fn new(bits: &[u16]) -> Result<Self, String> {
        assert!(!bits.is_empty() && bits.len() % GROUP == 0);
        let mut ideal = Vec::with_capacity(bits.len());
        let mut rounded = Vec::with_capacity(bits.len());
        let gamma32 = 6.0 / (2f64.powi(24) - 6.0);
        let gamma64 = 6.0 / (2f64.powi(53) - 6.0);
        for (block, b) in bits.chunks_exact(BLOCK).enumerate() {
            for &x in b {
                normal_bf(x)?;
            }
            let x32: [f32; BLOCK] = std::array::from_fn(|i| bf32(b[i]));
            let x64 = x32.map(f64::from);
            let a = forward32(&x32, block * BLOCK)?;
            let exact_approx = forward64(&x64, block * BLOCK)?;
            // Both transforms can round. This check excludes final BF16 error.
            let bound = (gamma32 + gamma64) * x64.iter().map(|x| x.abs()).sum::<f64>() / 8.0;
            for (actual, reference) in a.into_iter().zip(exact_approx) {
                if (actual as f64 - reference).abs() > bound * (1.0 + 64.0 * f64::EPSILON) {
                    return Err("F32 butterfly exceeds six-stage bound".into());
                }
                let value = bf16_rne(actual);
                normal_bf(value)?;
                rounded.push(value);
                ideal.push(reference);
            }
        }
        Ok(Self {
            original: bits.to_vec(),
            ideal,
            original_q: quantize(bits),
            rotated_q: quantize(&rounded),
            rounded,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub dots: [f64; 9],
    pub identity_error: f64,
    pub identity_bound: f64,
}
impl Sample {
    pub fn bits(&self) -> [u64; 11] {
        std::array::from_fn(|i| {
            if i < 9 {
                self.dots[i].to_bits()
            } else if i == 9 {
                self.identity_error.to_bits()
            } else {
                self.identity_bound.to_bits()
            }
        })
    }
}
pub fn compare(x: &Row, w: &Row) -> Result<Sample, String> {
    assert_eq!(x.original.len(), w.original.len());
    let width = x.original.len();
    let mut dots = [0f64; 9];
    let (mut nx, mut nw) = (0.0, 0.0);
    for k in 0..width {
        let (a, b) = (bf32(x.original[k]) as f64, bf32(w.original[k]) as f64);
        let (ar, br) = (bf32(x.rounded[k]) as f64, bf32(w.rounded[k]) as f64);
        let (aq, bq, arq, brq) = (
            x.original_q.value(k),
            w.original_q.value(k),
            x.rotated_q.value(k),
            w.rotated_q.value(k),
        );
        let terms = [
            a * b,
            x.ideal[k] * w.ideal[k],
            ar * br,
            aq * bq,
            arq * brq,
            aq * b,
            a * bq,
            arq * br,
            ar * brq,
        ];
        for (sum, term) in dots.iter_mut().zip(terms) {
            normal64(term)?;
            *sum += term;
            normal64(*sum)?;
        }
        nx += a * a;
        nw += b * b;
    }
    // Original BF16 products are exact F64. Ideal-rotated F64 products, both
    // dot accumulations and F64 butterflies can round. EPS=2u conservatively
    // covers each rotated multiply+add pair in gamma_width.
    // eta=8*gamma6 bounds each block transform's relative2-norm error.
    // Dot summation contributes gamma_width for each original/rotated dot.
    // Use eps (twice unit roundoff) and inflate computed norm product for its sums.
    let eps = f64::EPSILON;
    let gamma = width as f64 * eps / (1.0 - width as f64 * eps);
    let eta = 8.0 * 6.0 * eps / (1.0 - 6.0 * eps);
    let norm = (nx * nw).sqrt() / (1.0 - gamma);
    let identity_bound =
        (2.0 * eta + eta * eta + gamma * ((1.0 + eta).powi(2) + 1.0)) * norm * (1.0 + 64.0 * eps);
    let identity_error = (dots[1] - dots[0]).abs();
    if !identity_bound.is_finite() || identity_error > identity_bound {
        return Err(format!(
            "F64 rotation identity error {identity_error} exceeds {identity_bound}"
        ));
    }
    Ok(Sample {
        dots,
        identity_error,
        identity_bound,
    })
}

#[derive(Clone, Default, Debug)]
pub struct Error {
    pub samples: usize,
    pub sse: f64,
    pub reference_ss: f64,
    pub maxabs: f64,
}
impl Error {
    pub fn add(&mut self, value: f64, reference: f64) {
        assert!(value.is_finite() && reference.is_finite());
        self.samples += 1;
        self.sse += (value - reference).powi(2);
        self.reference_ss += reference * reference;
        self.maxabs = self.maxabs.max((value - reference).abs());
    }
    pub fn relative_l2(&self) -> f64 {
        if self.reference_ss == 0.0 {
            if self.sse == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (self.sse / self.reference_ss).sqrt()
        }
    }
}
pub fn decision(errors: &[Error; 9]) -> bool {
    errors[2].relative_l2() <= 0.01 && errors[4].relative_l2() <= 0.75 * errors[3].relative_l2()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn direct(x: &[f64; 64], base: usize) -> [f64; 64] {
        std::array::from_fn(|j| {
            (0..64)
                .map(|k| {
                    x[k] * sign(base + k) as f64
                        * if (k & j).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                })
                .sum::<f64>()
                / 8.0
        })
    }
    #[test]
    fn dense_matrix_and_inverse_are_independent() {
        for base in [0, 64, 896, 960, 3776, 3840, 15296] {
            let x: [f64; 64] = std::array::from_fn(|k| ((k * 73 + 19) % 509) as f64 - 254.0);
            let y = forward64(&x, base).unwrap();
            assert_eq!(y, direct(&x, base));
            assert_eq!(inverse64(&y, base).unwrap(), x);
            assert_ne!(forward64(&y, base).unwrap(), x, "forward is not inverse");
        }
    }
    #[test]
    fn range_edges_fail_instead_of_assuming_ftz_or_no_overflow() {
        assert!(Row::new(&vec![1; 960]).is_err());
        assert!(Row::new(&vec![0x7f7f; 960]).is_err());
        assert!(Row::new(&vec![0x7fc0; 960]).is_err());
        let zero = Row::new(&vec![0x8000; 960]).unwrap();
        assert_eq!(compare(&zero, &zero).unwrap().dots, [0.0; 9]);
    }
    #[test]
    fn full_width_quantizer_and_cancelled_dot_identity() {
        for width in [3840, 15360] {
            let x: Vec<u16> = (0..width)
                .map(|k| bf16_rne((((73 * k + 127) % 511) as f32 - 255.0) / 64.0))
                .collect();
            let w: Vec<u16> = (0..width)
                .map(|k| bf16_rne(if k % 2 == 0 { 2.0 } else { -2.0 }))
                .collect();
            let a = Row::new(&x).unwrap();
            let b = Row::new(&w).unwrap();
            let s = compare(&a, &b).unwrap();
            assert!(s.identity_error <= s.identity_bound);
            assert_eq!(a.original_q.scales.len(), width / 960);
            assert!(a.original_q.codes.iter().all(|q| (-7..=7).contains(q)));
        }
    }
    #[test]
    fn gate_requires_both_fixed_conditions() {
        let mut e: [Error; 9] = std::array::from_fn(|_| Error {
            samples: 512,
            reference_ss: 1.0,
            ..Error::default()
        });
        e[3].sse = 0.16;
        e[4].sse = 0.09;
        e[2].sse = 0.0001;
        assert!(decision(&e));
        e[2].sse = 0.00010001;
        assert!(!decision(&e));
        e[2].sse = 0.0;
        e[4].sse = 0.091;
        assert!(!decision(&e));
    }
}

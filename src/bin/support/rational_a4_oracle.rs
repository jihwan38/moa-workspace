//! Independent quotient/remainder oracle. It never counts candidate thresholds.
pub const WIDTH: usize = 3840;
pub const ROWS: usize = 16;
pub const FLOOR_BITS: u16 = 15 << 7; // BF16 2^-112.

fn finite_abs_ftz(bits: u16) -> u16 {
    let a = bits & 0x7fff;
    assert!(a >> 7 != 255, "nonfinite BF16 is outside the contract");
    if a >> 7 == 0 { 0 } else { a }
}

pub fn reference(row: &[u16]) -> (Vec<i32>, f32) {
    let a = row.iter().map(|&b| finite_abs_ftz(b)).max().unwrap_or(0).max(FLOOR_BITS);
    let ae = (a >> 7) as u32;
    let am = u128::from(128 + (a & 127));
    let codes = row.iter().map(|&bits| {
        let x = finite_abs_ftz(bits);
        if x == 0 { return 0; }
        let delta = ae - u32::from(x >> 7);
        // Large denominators are handled independently of the candidate's
        // delta5 shortcut. 7*Mx<=1785 cannot round up at delta>=120.
        if delta >= 120 { return 0; }
        let numerator = 7 * u128::from(128 + (x & 127));
        let denominator = am << delta;
        let whole = numerator / denominator;
        let remainder = numerator % denominator;
        let magnitude = (whole + u128::from(remainder * 2 >= denominator)) as i32;
        assert!(magnitude <= 7);
        if bits & 0x8000 == 0 { magnitude } else { -magnitude }
    }).collect();
    let scale = f32::from_bits(u32::from(a) << 16) * f32::from_bits(0x3e12_4925);
    (codes, scale)
}

pub const CASES: usize = 13;
pub fn fixture(case: usize) -> Vec<u16> {
    assert!(case < CASES);
    if case == 12 {
        let source = fixture(11);
        return (0..ROWS * WIDTH).map(|i| {
            source[((5 * (i / WIDTH) + 7) % ROWS) * WIDTH + (73 * (i % WIDTH) + 127) % WIDTH]
        }).collect();
    }
    let mut all = Vec::with_capacity(ROWS * WIDTH);
    for row in 0..ROWS {
        let mut values = Vec::new();
        if case < 8 {
            // Every valid normalized mantissa pair for delta0..4, both signs.
            let ma = 128 + case * ROWS + row;
            let a = (128u16 << 7) | (ma as u16 - 128);
            values.extend([a, a | 0x8000]);
            for delta in 0..5 {
                for mx in 128..256 {
                    if delta == 0 && mx > ma { continue; }
                    let x = (((128 - delta) << 7) | (mx - 128)) as u16;
                    values.extend([x, x | 0x8000]);
                }
            }
        } else if case == 8 {
            let a = match row {
                0 => 0, 1 => 0x8000, 2 => 0x007f, 3 => 0x807f,
                4 => 0x0080, 5 => FLOOR_BITS - 1, 6 => FLOOR_BITS,
                _ => FLOOR_BITS + (row as u16 - 6),
            };
            values.extend([a, a ^ 0x8000, 0, 0x8000, 1, 0x8001]);
        } else {
            let exponents = [1u16, 14, 15, 16, 17, 31, 63, 100, 126, 127, 128, 200, 250, 252, 253, 254];
            let ae = exponents[row];
            let ma = 128 + ((case * 29 + row * 17) % 128) as u16;
            let a = (ae << 7) | (ma - 128);
            values.extend([a, a | 0x8000, 0, 0x8000, 1, 0x807f]);
            for delta in [0u16, 1, 2, 3, 4, 5, 6, 7, 15, 31, 63, 119, 120, 200, 253] {
                if delta >= ae { continue; }
                for mx in 128..256 {
                    if delta == 0 && mx > ma { continue; }
                    let x = ((ae - delta) << 7) | (mx - 128);
                    values.extend([x, x | 0x8000]);
                }
            }
        }
        let filled: Vec<u16> = (0..WIDTH).map(|i| values[i % values.len()]).collect();
        all.extend(filled);
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_halves_round_away_and_scale_is_separate() {
        // A=14, x=1,3,5,...,13 => exact half-integer 7*x/A.
        let a = (14f32.to_bits() >> 16) as u16;
        let mut row = vec![a];
        for x in (1..14).step_by(2) {
            let b = ((x as f32).to_bits() >> 16) as u16;
            row.extend([b, b | 0x8000]);
        }
        let (q,s) = reference(&row);
        assert_eq!(q[0],7);
        for k in 0..7 { assert_eq!(&q[1+2*k..3+2*k], &[(k+1) as i32,-((k+1) as i32)]); }
        assert_eq!(s.to_bits(), (14f32 * f32::from_bits(0x3e12_4925)).to_bits());
    }
    #[test]
    fn complete_fixture_and_permutation_contract() {
        for case in 0..CASES {
            let bits = fixture(case);
            assert_eq!(bits.len(), ROWS * WIDTH);
            for row in bits.chunks_exact(WIDTH) {
                let(q,s) = reference(row);
                assert!(q.iter().all(|x| (-7..=7).contains(x)));
                assert!(s.is_finite() && s>0.0);
            }
        }
        let original = fixture(11);
        let permuted = fixture(12);
        for row in 0..ROWS {
            let (a,sa)=reference(&original[((5*row+7)%ROWS)*WIDTH..((5*row+7)%ROWS+1)*WIDTH]);
            let (b,sb)=reference(&permuted[row*WIDTH..(row+1)*WIDTH]);
            assert_eq!(sa.to_bits(),sb.to_bits());
            for col in 0..WIDTH { assert_eq!(b[col],a[(73*col+127)%WIDTH]); }
        }
    }
    #[test]
    fn floor_and_ftz_ignore_signed_tiny_values() {
        let(q,s)=reference(&[0,0x8000,1,0x807f,0x0080,0x8080]);
        assert_eq!(q,vec![0;6]);
        assert_eq!(s.to_bits(),(f32::from_bits(u32::from(FLOOR_BITS)<<16)*f32::from_bits(0x3e12_4925)).to_bits());
    }
}

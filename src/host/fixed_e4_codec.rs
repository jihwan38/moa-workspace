//! Independent finite BF16 -> fixed E4 oracle. No SDK conversion is used.
//! E4 magnitudes are exact integer multiples of 2^-9; ties choose the even code.
pub fn e4_units(code: u8) -> u32 {
    let c = code & 0x7f;
    assert!(c <= 126);
    let e = c >> 3;
    let m = (c & 7) as u32;
    if e == 0 { m } else { (8 + m) << (e - 1) }
}
pub fn encode(bits: u16) -> u8 {
    let sign = ((bits >> 8) & 0x80) as u8;
    let mag = bits & 0x7fff;
    if mag > 0x7f80 {
        return sign | 0x7f;
    }
    let x = f32::from_bits((mag as u32) << 16) as f64 * 512.0;
    if x >= e4_units(126) as f64 {
        return sign | 126;
    }
    let mut lo = 0u8;
    let mut hi = 126u8;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if e4_units(mid) as f64 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dl = x - e4_units(lo) as f64;
    let dh = e4_units(hi) as f64 - x;
    sign | if dl < dh || (dl == dh && lo & 1 == 0) {
        lo
    } else {
        hi
    }
}
pub fn decode(code: u8) -> u16 {
    let sign = ((code as u16) & 0x80) << 8;
    if code & 0x7f == 0x7f {
        return sign | 0x7fc0;
    }
    let x = e4_units(code) as f32 / 512.0;
    assert_eq!(x.to_bits() & 0xffff, 0);
    sign | (x.to_bits() >> 16) as u16
}
pub fn table() -> Vec<u8> {
    (0..=u16::MAX).map(encode).collect()
}
pub fn encode_bytes(input: &[u8], lut: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(2)
        .map(|x| lut[u16::from_le_bytes([x[0], x[1]]) as usize])
        .collect()
}
pub fn decode_bytes(input: &[u8]) -> Vec<u8> {
    input
        .iter()
        .flat_map(|&x| decode(x).to_le_bytes())
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_finite_codes_and_exact_midpoints() {
        for c in 0..127u8 {
            assert_eq!(encode(decode(c)), c);
            assert_eq!(encode(decode(c | 128)), c | 128);
        }
        for c in 0..126u8 {
            let x = (e4_units(c) + e4_units(c + 1)) as f32 / 1024.0;
            assert_eq!(x.to_bits() & 65535, 0);
            let b = (x.to_bits() >> 16) as u16;
            let even = if c & 1 == 0 { c } else { c + 1 };
            assert_eq!(encode(b), even);
            assert_eq!(encode(b | 0x8000), even | 128);
            if b > 0 {
                assert_eq!(encode(b - 1), c);
            }
            assert_eq!(encode(b + 1), c + 1);
        }
        assert_eq!(encode(0x7f7f), 126);
        assert_eq!(encode(0xff7f), 254);
        assert_eq!(encode(1), 0);
        assert_eq!(encode(0x8001), 128);
    }
    #[test]
    fn exhaustive_finite_matches_linear_nearest_search() {
        for b in 0..=u16::MAX {
            if b & 0x7f80 == 0x7f80 {
                continue;
            }
            let x = f32::from_bits(((b & 0x7fff) as u32) << 16) as f64;
            let mut best = 0;
            let mut err = f64::INFINITY;
            for c in 0..127u8 {
                let d = (x - e4_units(c) as f64 / 512.0).abs();
                if d < err || (d == err && c & 1 == 0 && best & 1 == 1) {
                    best = c;
                    err = d;
                }
            }
            // For huge values f64 distance loses small-code differences. Saturation
            // is a distinct range rule, rather than relying on that subtraction.
            if x >= 448.0 {
                best = 126;
            }
            assert_eq!(encode(b), best | (((b >> 8) & 128) as u8), "{b:04x}");
        }
    }
}

//! CPU-only compression of current blocked E4 weights into fixed E2M1 codes.
//!
//! The source is `[rows/240, cols/960, rows%240, cols%960]`, as used by
//! `ops_native_fp8_blocked::{UpWeight, DownWeight}`. Each contiguous 960-byte
//! group gets the smallest power-of-two scale `s` with `6*s >= max(abs(w))`.
//! All-zero groups use `s=1`. Nearest-code ties select the even E2M1 magnitude
//! code; input signs, including negative zero, are retained.
//!
//! This is a new lossy weight quantizer, not a lossless codec for provided
//! NVFP4 weights. The existing per-output-row dequantization and global scales
//! remain separate and unchanged. Consumers multiply each local F32 K960 dot
//! by its group scale before the existing cross-slice sum. In particular,
//! `dense_e4_codes` does NOT contain that scale: folding it into E4 can overflow
//! or introduce another rounding boundary. No activation quantizer is changed.

use std::fmt;

pub const GROUP_SIZE: usize = 960;
pub const ROW_BLOCK: usize = 240;
const HALF_LEVELS: [u32; 8] = [0, 1, 2, 3, 4, 6, 8, 12];
const E4_LEVELS: [u8; 8] = [0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c];

#[derive(Debug, PartialEq, Eq)]
pub enum QuantizeError {
    InvalidShape { rows: usize, cols: usize },
    SizeOverflow,
    Length { expected: usize, actual: usize },
    NonFinite { index: usize, bits: u8 },
    ScaleExponent(i32),
}

impl fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { rows, cols } => write!(
                f,
                "expected positive rows%240=0 and cols%960=0, got {rows}x{cols}"
            ),
            Self::SizeOverflow => write!(f, "weight shape overflows usize"),
            Self::Length { expected, actual } => {
                write!(f, "expected {expected} E4 bytes, got {actual}")
            }
            Self::NonFinite { index, bits } => {
                write!(f, "nonfinite E4 byte 0x{bits:02x} at index {index}")
            }
            Self::ScaleExponent(e) => {
                write!(f, "E4-derived group scale exponent {e} is outside -11..=7")
            }
        }
    }
}
impl std::error::Error for QuantizeError {}

pub struct PackedF4Groups {
    /// Same blocked order as the input. Even element uses the low nibble.
    pub packed_f4: Vec<u8>,
    /// `[rows/240, cols/960, rows%240]`, one F32 value per 960 input bytes.
    pub group_scales: Vec<f32>,
    /// Fixed E2M1 values expanded to exact E4, in the original blocked order.
    /// For a matching dense-dot oracle; group scales are deliberately separate.
    pub dense_e4_codes: Vec<u8>,
}

/// Startup-only output without the full-size E4 oracle copy.
pub struct PackedF4Weights {
    pub packed_f4: Vec<u8>,
    pub group_scales: Vec<f32>,
}

// All finite E4M3FN magnitudes are integer multiples of 2^-9.
fn magnitude_units(bits: u8) -> u32 {
    let bits = bits & 0x7f;
    let exponent = u32::from(bits >> 3);
    let fraction = u32::from(bits & 7);
    if exponent == 0 {
        fraction
    } else {
        (8 + fraction) << (exponent - 1)
    }
}

fn finite(bits: u8) -> bool {
    bits & 0x7f != 0x7f
}

/// Exact finite E4M3FN decode; signed zeros are retained.
pub fn decode_e4(bits: u8) -> Result<f32, QuantizeError> {
    if !finite(bits) {
        return Err(QuantizeError::NonFinite { index: 0, bits });
    }
    let magnitude = magnitude_units(bits) as f32 / 512.0;
    Ok(if bits & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// Exact fixed E2M1 decode of the low nibble, including negative zero.
pub fn decode_f4(code: u8) -> f32 {
    let magnitude = HALF_LEVELS[usize::from(code & 7)] as f32 * 0.5;
    if code & 8 == 0 { magnitude } else { -magnitude }
}

/// Smallest admissible power-of-two scale for a finite E4 magnitude.
/// A zero maximum selects the documented all-zero-group scale 1.
pub fn scale_exponent_for_max(bits: u8) -> Result<i32, QuantizeError> {
    if !finite(bits) {
        return Err(QuantizeError::NonFinite { index: 0, bits });
    }
    let units = magnitude_units(bits);
    if units == 0 {
        return Ok(0);
    }
    for exponent in -11..=7 {
        let shift = exponent + 9;
        let fits = if shift >= 0 {
            units <= 6u32 << shift
        } else {
            units << -shift <= 6
        };
        if fits {
            return Ok(exponent);
        }
    }
    unreachable!("finite E4 is bounded by 448")
}

/// Exact nearest fixed-E2M1 encoding at a specified E4-derived scale.
/// Saturates to magnitude 6 when called outside a valid group's range.
pub fn encode_e4_at_scale(bits: u8, exponent: i32) -> Result<u8, QuantizeError> {
    if !finite(bits) {
        return Err(QuantizeError::NonFinite { index: 0, bits });
    }
    if !(-11..=7).contains(&exponent) {
        return Err(QuantizeError::ScaleExponent(exponent));
    }
    Ok(encode_finite(bits, exponent))
}

fn encode_finite(bits: u8, exponent: i32) -> u8 {
    // A code's value in E4 units is HALF_LEVELS[code] * 2^(exponent+8).
    // Scale both operands to integers, so even halfway comparisons are exact.
    let shift = (-exponent - 8).max(0);
    let target = magnitude_units(bits) << shift;
    let level_shift = exponent + 8 + shift;
    let mut best = 0;
    let mut best_distance = u32::MAX;
    for (code, level) in HALF_LEVELS.into_iter().enumerate() {
        let distance = target.abs_diff(level << level_shift);
        if distance < best_distance || (distance == best_distance && code % 2 == 0) {
            best = code as u8;
            best_distance = distance;
        }
    }
    best | ((bits >> 4) & 8)
}

pub fn quantize_blocked_e4(
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<PackedF4Groups, QuantizeError> {
    if rows == 0 || cols == 0 || rows % ROW_BLOCK != 0 || cols % GROUP_SIZE != 0 {
        return Err(QuantizeError::InvalidShape { rows, cols });
    }
    let expected = rows.checked_mul(cols).ok_or(QuantizeError::SizeOverflow)?;
    if raw.len() != expected {
        return Err(QuantizeError::Length {
            expected,
            actual: raw.len(),
        });
    }
    if let Some((index, &bits)) = raw.iter().enumerate().find(|(_, b)| !finite(**b)) {
        return Err(QuantizeError::NonFinite { index, bits });
    }
    // These small exact lookup tables avoid repeated host floating conversion
    // across the model. They do not alter the fixed hardware E2M1 lookup table.
    let lookup: [[u8; 256]; 19] =
        std::array::from_fn(|e| std::array::from_fn(|b| encode_finite(b as u8, e as i32 - 11)));
    let mut packed_f4 = vec![0u8; raw.len() / 2];
    let mut dense_e4_codes = vec![0u8; raw.len()];
    let mut group_scales = Vec::with_capacity(raw.len() / GROUP_SIZE);
    for (group_index, group) in raw.chunks_exact(GROUP_SIZE).enumerate() {
        // Finite E4 magnitude codes are ordered, including subnormals.
        let max_bits = group.iter().map(|b| b & 0x7f).max().unwrap();
        let exponent = scale_exponent_for_max(max_bits)?;
        group_scales.push(f32::from_bits(((exponent + 127) as u32) << 23));
        let table = &lookup[(exponent + 11) as usize];
        let base = group_index * GROUP_SIZE;
        for (local, &bits) in group.iter().enumerate() {
            let code = table[usize::from(bits)];
            let index = base + local;
            packed_f4[index / 2] |= code << (4 * (index % 2));
            dense_e4_codes[index] = E4_LEVELS[usize::from(code & 7)] | ((code & 8) << 4);
        }
    }
    Ok(PackedF4Groups {
        packed_f4,
        group_scales,
        dense_e4_codes,
    })
}

/// Two source passes and paired-nibble writes, with identical quantization to
/// `quantize_blocked_e4`. This omits only the dense E4 diagnostic reference.
/// Validation is completed before allocating the large packed-weight output;
/// only per-group scale metadata is allocated during the first pass.
pub fn quantize_blocked_e4_packed(
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<PackedF4Weights, QuantizeError> {
    if rows == 0 || cols == 0 || rows % ROW_BLOCK != 0 || cols % GROUP_SIZE != 0 {
        return Err(QuantizeError::InvalidShape { rows, cols });
    }
    let expected = rows.checked_mul(cols).ok_or(QuantizeError::SizeOverflow)?;
    if raw.len() != expected {
        return Err(QuantizeError::Length {
            expected,
            actual: raw.len(),
        });
    }
    let mut exponents = Vec::with_capacity(raw.len() / GROUP_SIZE);
    let mut group_scales = Vec::with_capacity(raw.len() / GROUP_SIZE);
    for (group_index, group) in raw.chunks_exact(GROUP_SIZE).enumerate() {
        let max_bits = group.iter().map(|b| b & 0x7f).max().unwrap();
        if max_bits == 0x7f {
            let local = group.iter().position(|&b| !finite(b)).unwrap();
            return Err(QuantizeError::NonFinite {
                index: group_index * GROUP_SIZE + local,
                bits: group[local],
            });
        }
        let exponent = scale_exponent_for_max(max_bits)?;
        exponents.push((exponent + 11) as u8);
        group_scales.push(f32::from_bits(((exponent + 127) as u32) << 23));
    }
    let lookup: [[u8; 256]; 19] =
        std::array::from_fn(|e| std::array::from_fn(|b| encode_finite(b as u8, e as i32 - 11)));
    let mut packed_f4 = vec![0u8; raw.len() / 2];
    for ((source, destination), &exponent) in raw
        .chunks_exact(GROUP_SIZE)
        .zip(packed_f4.chunks_exact_mut(GROUP_SIZE / 2))
        .zip(&exponents)
    {
        let table = &lookup[usize::from(exponent)];
        for (pair, byte) in source.chunks_exact(2).zip(destination) {
            *byte = table[usize::from(pair[0])] | (table[usize::from(pair[1])] << 4);
        }
    }
    Ok(PackedF4Weights {
        packed_f4,
        group_scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_and_signs() {
        for (value, want) in [
            (0.25, 0),
            (0.75, 2),
            (1.25, 2),
            (1.75, 4),
            (2.5, 4),
            (3.5, 6),
            (5.0, 6),
        ] {
            let bits = (0..127u8)
                .find(|&b| decode_e4(b).unwrap() == value)
                .unwrap();
            assert_eq!(encode_e4_at_scale(bits, 0).unwrap(), want);
            assert_eq!(encode_e4_at_scale(bits | 128, 0).unwrap(), want | 8);
        }
    }

    #[test]
    fn zeros_extremes_and_errors() {
        assert_eq!(scale_exponent_for_max(1).unwrap(), -11);
        assert_eq!(scale_exponent_for_max(0x7e).unwrap(), 7);
        assert_eq!(decode_e4(0x80).unwrap().to_bits(), 0x8000_0000);
        assert_eq!(decode_f4(8).to_bits(), 0x8000_0000);
        let mut raw = vec![0x80; 240 * 960];
        let q = quantize_blocked_e4(&raw, 240, 960).unwrap();
        assert!(q.packed_f4.iter().all(|&b| b == 0x88));
        assert!(q.dense_e4_codes.iter().all(|&b| b == 0x80));
        assert!(q.group_scales.iter().all(|&s| s == 1.0));
        raw[1919] = 0xff;
        assert!(matches!(
            quantize_blocked_e4(&raw, 240, 960),
            Err(QuantizeError::NonFinite {
                index: 1919,
                bits: 0xff
            })
        ));
        assert!(matches!(
            quantize_blocked_e4(&[], 239, 960),
            Err(QuantizeError::InvalidShape { .. })
        ));
        assert!(matches!(
            quantize_blocked_e4(&[], 240, 960),
            Err(QuantizeError::Length { .. })
        ));
        assert!(encode_e4_at_scale(1, -12).is_err());
    }
}

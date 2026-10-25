//! Deterministic histogram-selected dyadic scales for the existing F4 ABI.
//! Candidate0 is the original ceil-max power2 scale; all others are lower.
//! Exact integer SSE selection cannot worsen error against current E4 within
//! a group. It does not establish original-weight or task-quality accuracy.
use super::f4_group960::{
    GROUP_SIZE, PackedF4Weights, QuantizeError, ROW_BLOCK, scale_exponent_for_max,
};
use std::sync::OnceLock;

pub const CANDIDATES: usize = 25;
const HALF_LEVELS: [u64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

pub struct GroupError {
    pub baseline_squared_error: f64,
    pub fitted_squared_error: f64,
    pub source_squared_norm: f64,
}

struct Tables {
    scales: [f32; CANDIDATES],
    codes: [[u8; 256]; CANDIDATES],
    squared: [[u64; 128]; CANDIDATES],
}
static TABLES: OnceLock<Vec<Tables>> = OnceLock::new();

fn choice(candidate: usize) -> (i32, u64) {
    if candidate == 0 {
        (0, 8)
    } else {
        (
            1 + (candidate - 1) as i32 / 8,
            8 + ((candidate - 1) % 8) as u64,
        )
    }
}

fn build_tables(base: i32) -> Tables {
    let mut table = Tables {
        scales: [0.; CANDIDATES],
        codes: [[255; 256]; CANDIDATES],
        squared: [[0; 128]; CANDIDATES],
    };
    let common = (-9).min(base - 7);
    for c in 0..CANDIDATES {
        let (depth, mantissa) = choice(c);
        table.scales[c] =
            f32::from_bits(((base - depth + 127) as u32) << 23 | ((mantissa as u32 - 8) << 20));
        for bits in 0..127usize {
            let exponent = bits >> 3;
            let units = if exponent == 0 {
                bits & 7
            } else {
                (8 + (bits & 7)) << (exponent - 1)
            } as u64;
            let source = units << (-9 - common);
            let mut best_code = 0;
            let mut best_distance = u64::MAX;
            for (code, &half_level) in HALF_LEVELS.iter().enumerate() {
                let level = (half_level * mantissa) << (base - depth - 4 - common);
                let distance = source.abs_diff(level);
                if distance < best_distance || (distance == best_distance && code % 2 == 0) {
                    best_distance = distance;
                    best_code = code as u8;
                }
            }
            table.codes[c][bits] = best_code;
            table.codes[c][bits | 128] = best_code | 8;
            table.squared[c][bits] = best_distance * best_distance;
        }
    }
    table
}

fn tables() -> &'static [Tables] {
    TABLES.get_or_init(|| (-11..=7).map(build_tables).collect())
}

/// Diagnostic API for the finite candidate grid. Invalid parameters return None.
pub fn candidate_scale(base: i32, candidate: usize) -> Option<f32> {
    if !(-11..=7).contains(&base) || candidate >= CANDIDATES {
        return None;
    }
    Some(tables()[(base + 11) as usize].scales[candidate])
}

/// Exact nearest-even fixed E2M1 code, with explicit saturation and signed zero.
pub fn candidate_code(bits: u8, base: i32, candidate: usize) -> Option<u8> {
    if bits & 127 == 127 || !(-11..=7).contains(&base) || candidate >= CANDIDATES {
        return None;
    }
    Some(tables()[(base + 11) as usize].codes[candidate][usize::from(bits)])
}

pub fn quantize_blocked_e4_mse(
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<PackedF4Weights, QuantizeError> {
    Ok(quantize_impl::<false>(raw, rows, cols)?.0)
}

/// Same output plus per-group error sums against current E4, obtained from the
/// existing histogram. The optional report avoids a full per-weight F64 audit.
pub fn quantize_blocked_e4_mse_report(
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(PackedF4Weights, Vec<GroupError>), QuantizeError> {
    quantize_impl::<true>(raw, rows, cols)
}

fn quantize_impl<const REPORT: bool>(
    raw: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(PackedF4Weights, Vec<GroupError>), QuantizeError> {
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
    let tables = tables();
    let groups = raw.len() / GROUP_SIZE;
    let mut selected = Vec::with_capacity(groups);
    let mut group_scales = Vec::with_capacity(groups);
    let mut errors = Vec::with_capacity(if REPORT { groups } else { 0 });
    for (g, group) in raw.chunks_exact(GROUP_SIZE).enumerate() {
        let mut histogram = [0u16; 128];
        for &byte in group {
            histogram[usize::from(byte & 127)] += 1;
        }
        if histogram[127] != 0 {
            let local = group.iter().position(|&b| b & 127 == 127).unwrap();
            return Err(QuantizeError::NonFinite {
                index: g * GROUP_SIZE + local,
                bits: group[local],
            });
        }
        let maximum = histogram.iter().rposition(|&n| n != 0).unwrap() as u8;
        let base = scale_exponent_for_max(maximum)?;
        let base_index = (base + 11) as usize;
        let table = &tables[base_index];
        let mut best = 0;
        let mut best_sse = u64::MAX;
        let mut baseline_sse = 0;
        for c in 0..CANDIDATES {
            // sum(histogram)=960 and 960*(448*2^18)^2 < 2^64.
            let sse: u64 = histogram
                .iter()
                .zip(&table.squared[c])
                .map(|(&count, &error)| u64::from(count) * error)
                .sum();
            if c == 0 {
                baseline_sse = sse;
            }
            if sse < best_sse {
                best = c;
                best_sse = sse;
            }
        }
        selected.push((base_index as u8, best as u8));
        group_scales.push(table.scales[best]);
        if REPORT {
            let norm: u64 = histogram
                .iter()
                .take(127)
                .enumerate()
                .map(|(bits, &count)| {
                    let exponent = bits >> 3;
                    let units = if exponent == 0 {
                        bits & 7
                    } else {
                        (8 + (bits & 7)) << (exponent - 1)
                    } as u64;
                    u64::from(count) * units * units
                })
                .sum();
            let common = (-9).min(base - 7);
            let unit_squared = 2f64.powi(2 * common);
            errors.push(GroupError {
                baseline_squared_error: baseline_sse as f64 * unit_squared,
                fitted_squared_error: best_sse as f64 * unit_squared,
                source_squared_norm: norm as f64 / 262144.0,
            });
        }
    }
    let mut packed_f4 = vec![0u8; raw.len() / 2];
    for ((source, destination), &(base, candidate)) in raw
        .chunks_exact(GROUP_SIZE)
        .zip(packed_f4.chunks_exact_mut(GROUP_SIZE / 2))
        .zip(&selected)
    {
        let code = &tables[usize::from(base)].codes[usize::from(candidate)];
        for (pair, byte) in source.chunks_exact(2).zip(destination) {
            *byte = code[usize::from(pair[0])] | (code[usize::from(pair[1])] << 4);
        }
    }
    Ok((
        PackedF4Weights {
            packed_f4,
            group_scales,
        },
        errors,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baseline_grid_and_zero() {
        for base in -11..=7 {
            assert_eq!(candidate_scale(base, 0).unwrap(), 2f32.powi(base));
            for bits in 0..=255u8 {
                assert_eq!(
                    candidate_code(bits, base, 0),
                    super::super::f4_group960::encode_e4_at_scale(bits, base).ok()
                );
            }
        }
        let q = quantize_blocked_e4_mse(&vec![128; 240 * 960], 240, 960).unwrap();
        assert!(q.group_scales.iter().all(|s| *s == 1.0));
        assert!(q.packed_f4.iter().all(|b| *b == 0x88));
    }
}

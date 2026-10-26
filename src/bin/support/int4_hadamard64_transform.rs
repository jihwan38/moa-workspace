//! Two independently indexed CPU implementations of the frozen H64 policy.
#[path = "int4_hadamard64.rs"]
pub mod fixed;

fn check_bf(x: u16) -> Result<(), String> {
    let a = x & 0x7fff;
    if a == 0 || (a >> 7 > 0 && a >> 7 < 255) {
        Ok(())
    } else {
        Err(format!("BF16 range: {x:04x}"))
    }
}
fn check32(x: f32) -> Result<(), String> {
    if x == 0.0 || x.is_normal() {
        Ok(())
    } else {
        Err(format!("F32 range: {:08x}", x.to_bits()))
    }
}
pub fn rotate_row(bits: &[u16]) -> Result<Vec<u16>, String> {
    assert!(!bits.is_empty() && bits.len() % 960 == 0);
    let mut out = Vec::with_capacity(bits.len());
    for (block, b) in bits.chunks_exact(64).enumerate() {
        for &x in b {
            check_bf(x)?;
        }
        let x = std::array::from_fn(|j| fixed::bf32(b[j]));
        for v in fixed::forward32(&x, 64 * block)? {
            let q = fixed::bf16_rne(v);
            check_bf(q)?;
            out.push(q);
        }
    }
    Ok(out)
}
/// Independent out-of-place coordinate form. Each output reads the preceding
/// stage; upper outputs are lower-input minus upper-input, never the reverse.
pub fn reference_row(bits: &[u16]) -> Result<Vec<u16>, String> {
    assert!(!bits.is_empty() && bits.len() % 960 == 0);
    let mut out = Vec::with_capacity(bits.len());
    for (block, b) in bits.chunks_exact(64).enumerate() {
        let mut a = [0f32; 64];
        for j in 0..64 {
            check_bf(b[j])?;
            let mut z =
                ((64 * block + j) as u64 ^ 0x243f6a8885a308d3).wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            a[j] = fixed::bf32(b[j]) * if z & (1u64 << 63) == 0 { 1.0 } else { -1.0 };
        }
        for bit in 0..6 {
            let old = a;
            let stride = 1usize << bit;
            for j in 0..64 {
                a[j] = if j & stride == 0 {
                    old[j] + old[j ^ stride]
                } else {
                    old[j ^ stride] - old[j]
                };
                check32(a[j])?;
            }
        }
        for v in a {
            let x = v / 8.0;
            check32(x)?;
            assert_eq!(x as f64 * 8.0, v as f64);
            // RN-even via truncated-word/remainder, independently of add-bias.
            let raw = x.to_bits();
            let mut q = (raw >> 16) as u16;
            let tail = raw & 65535;
            if tail > 32768 || (tail == 32768 && q & 1 != 0) {
                q = q.wrapping_add(1);
            }
            check_bf(q)?;
            out.push(q);
        }
    }
    Ok(out)
}
pub fn rotate_bytes(bytes: &[u8], width: usize, reference: bool) -> Result<Vec<u8>, String> {
    assert_eq!(bytes.len(), 512 * width * 2);
    let mut result = Vec::with_capacity(bytes.len());
    for row in bytes.chunks_exact(width * 2) {
        let bits: Vec<u16> = row
            .chunks_exact(2)
            .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let rotated = if reference {
            reference_row(&bits)?
        } else {
            rotate_row(&bits)?
        };
        result.extend(rotated.iter().flat_map(|x| x.to_le_bytes()));
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_coordinates_all_groups_and_signed_zeros() {
        for width in [3840, 15360] {
            let x: Vec<u16> = (0..width)
                .map(|k| {
                    if k % 101 == 0 {
                        0x8000
                    } else {
                        fixed::bf16_rne((((k * 73 + 127) % 2047) as f32 - 1023.0) / 64.0)
                    }
                })
                .collect();
            assert_eq!(rotate_row(&x).unwrap(), reference_row(&x).unwrap());
        }
    }
    #[test]
    fn full_row_permutation_preserves_absolute_channel_signs() {
        let width = 3840;
        let bits: Vec<u8> = (0..512 * width)
            .flat_map(|i| {
                fixed::bf16_rne((((i / width) * 23 + (i % width) * 17) % 511) as f32 / 128.0)
                    .to_le_bytes()
            })
            .collect();
        let order: Vec<u8> = (0..512)
            .flat_map(|p| {
                bits[((73 * p + 127) % 512) * width * 2..((73 * p + 127) % 512 + 1) * width * 2]
                    .iter()
                    .copied()
            })
            .collect();
        let a = rotate_bytes(&bits, width, false).unwrap();
        let b = rotate_bytes(&order, width, true).unwrap();
        for p in 0..512 {
            assert_eq!(
                &b[p * width * 2..(p + 1) * width * 2],
                &a[((73 * p + 127) % 512) * width * 2..((73 * p + 127) % 512 + 1) * width * 2]
            );
        }
    }
    #[test]
    fn both_implementations_reject_unsupported_ranges() {
        for bits in [1u16, 0x7f7f, 0x7fc0] {
            assert!(rotate_row(&vec![bits; 960]).is_err());
            assert!(reference_row(&vec![bits; 960]).is_err());
        }
    }
}

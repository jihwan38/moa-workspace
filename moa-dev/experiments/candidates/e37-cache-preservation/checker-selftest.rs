use std::sync::atomic::{AtomicU64, Ordering};

struct Ts; impl Ts { const SIZE: usize = 1024; }

struct Ns; impl Ns { const SIZE: usize = 8; }

struct Ds; impl Ds { const SIZE: usize = 256; }

mod prng {
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

    pub fn name_hash(name: &str) -> u64 {
        let mut hash = FNV_OFFSET;
        for byte in name.as_bytes() {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    }

    pub fn word(seed: u64, index: usize) -> u64 {
        let mut x = (seed ^ index as u64).wrapping_add(GOLDEN);
        x = (x ^ (x >> 30)).wrapping_mul(MIX_A);
        x = (x ^ (x >> 27)).wrapping_mul(MIX_B);
        x ^ (x >> 31)
    }

    pub fn u01(w: u64) -> f32 {
        ((w >> 40) as u32) as f32 / 16_777_216.0
    }

    pub fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounding = 0x7FFF + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    }

    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn bf16_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        bf16_uniform_range(name, 0, count, lo, hi)
    }

    pub fn f32_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<f32> {
        let seed = name_hash(name);
        (0..count)
            .map(|index| lo + (hi - lo) * u01(word(seed, index)))
            .collect()
    }

    pub fn bf16_uniform_range(name: &str, offset: usize, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count * 2);
        for index in offset..offset + count {
            push_u16(&mut out, f32_to_bf16_bits(lo + (hi - lo) * u01(word(seed, index))));
        }
        out
    }

    pub fn bf16_signs(name: &str, count: usize, scale: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let magnitude = f32_to_bf16_bits(scale.abs());
        let mut out = Vec::with_capacity(count * 2);
        for index in 0..count {
            let negative = word(seed, index) >> 63 == 1;
            push_u16(&mut out, if negative { magnitude | 0x8000 } else { magnitude });
        }
        out
    }

    pub fn f8_banded(name: &str, count: usize, exp_min: u8, exp_max: u8, signed: bool) -> Vec<u8> {
        assert!(
            exp_min <= exp_max && exp_max <= 14,
            "{name}: exponent band out of range"
        );
        let seed = name_hash(name);
        let span = u64::from(exp_max - exp_min + 1);
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let w = word(seed, index);
            let exponent = exp_min + ((w >> 32) % span) as u8;
            let mantissa = ((w >> 56) & 0x7) as u8;
            let sign = if signed { ((w >> 63) as u8) << 7 } else { 0 };
            out.push(sign | (exponent << 3) | mantissa);
        }
        out
    }

    pub fn f4_nibbles(name: &str, count: usize) -> Vec<u8> {
        assert!(count % 2 == 0, "{name}: f4 element count must be even");
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count / 2);
        for pair in 0..count / 2 {
            let low = ((word(seed, pair * 2) >> 60) & 0xF) as u8;
            let high = ((word(seed, pair * 2 + 1) >> 60) & 0xF) as u8;
            out.push(low | (high << 4));
        }
        out
    }

    pub fn checksum(bytes: &[u8], word_offset: usize) -> u32 {
        let mut total: u32 = 0;
        for (block, chunk) in bytes.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            let index = (word_offset + block) as u32;
            total = total.wrapping_add(u32::from_le_bytes(word).wrapping_mul(index.wrapping_mul(2).wrapping_add(1)));
        }
        total
    }
}

static CACHE_PRESERVATION_MISMATCHES: AtomicU64 = AtomicU64::new(0);

fn cache_sentinel_bits(run: usize, cache_name: &str) -> Vec<u16> {
    let seed = prng::name_hash(&format!("cache-preservation/{cache_name}/run{run}"));
    (0..Ts::SIZE * Ns::SIZE * Ds::SIZE)
        .map(|index| {
            let word = prng::word(seed, index);
            // Magnitudes 0x3e00..0x3fff are finite, normal and nonzero.
            let magnitude = 0x3e00u16 + (word as u16 & 0x01ff);
            magnitude | (((word >> 63) as u16) << 15)
        })
        .collect()
}

fn check_cache_sentinel(
    stage: &str,
    cache_name: &str,
    run: usize,
    slot: usize,
    expected: &[u16],
    actual: &[u16],
    exclude_target: bool,
) {
    let width = Ns::SIZE * Ds::SIZE;
    let start = slot * width;
    let end = start + width;
    assert_eq!(expected.len(), Ts::SIZE * width);
    assert_eq!(actual.len(), expected.len());
    assert!(slot < Ts::SIZE && start < end && end <= actual.len());
    let mut checked = 0usize;
    let mut mismatches = 0usize;
    for index in 0..actual.len() {
        if exclude_target && (start..end).contains(&index) {
            continue;
        }
        checked += 1;
        if actual[index] != expected[index] {
            if mismatches < 8 {
                eprintln!(
                    "CACHE_MISMATCH stage={stage} cache={cache_name} run={run} target_slot={slot} index={index} slot={} head={} dim={} byte_offset={} expected=0x{:04x} actual=0x{:04x}",
                    index / width, index % width / Ds::SIZE, index % Ds::SIZE,
                    index * 2, expected[index], actual[index],
                );
            }
            mismatches += 1;
        }
    }
    CACHE_PRESERVATION_MISMATCHES.fetch_add(mismatches as u64, Ordering::Relaxed);
    println!(
        "CACHE_CHECK stage={stage} cache={cache_name} run={run} target_slot={slot} target_elements=[{start},{end}) target_bytes=[{},{}) checked={checked} mismatches={mismatches}",
        start * 2, end * 2,
    );
}

fn main() {
    let width = Ns::SIZE * Ds::SIZE;
    let size = Ts::SIZE * width;
    for run in 0..3 {
        for name in ["k", "v"] {
            let expected = cache_sentinel_bits(run, name);
            assert_eq!(expected.len(), size);
            assert!(expected.iter().all(|b| { let v = prng::bf16_bits_to_f32(*b); v.is_finite() && v != 0.0 }));
            let seed = prng::name_hash(&format!("run{run}.global.pos"));
            let pos = 1 + (prng::word(seed, 0) % 2047) as usize;
            let slot = pos % Ts::SIZE;
            check_cache_sentinel("cpu-before", name, run, slot, &expected, &expected, false);
            check_cache_sentinel("cpu-after", name, run, slot, &expected, &expected, true);
        }
    }
    assert_eq!(CACHE_PRESERVATION_MISMATCHES.load(Ordering::Relaxed), 0);
    let expected = cache_sentinel_bits(0, "boundary");
    let mut expected_failures = 0u64;
    for slot in [0, 1, 511, 1023] {
        let start = slot * width;
        let end = start + width;
        // Target first/last writes must be ignored only in the after check.
        for index in [start, end-1] {
            let mut actual = expected.clone(); actual[index] ^= 1;
            check_cache_sentinel("cpu-target-after", "boundary", 0, slot, &expected, &actual, true);
            assert_eq!(CACHE_PRESERVATION_MISMATCHES.load(Ordering::Relaxed), expected_failures);
            check_cache_sentinel("cpu-target-before", "boundary", 0, slot, &expected, &actual, false);
            expected_failures += 1;
            assert_eq!(CACHE_PRESERVATION_MISMATCHES.load(Ordering::Relaxed), expected_failures);
        }
        let mut protected = vec![];
        if start > 0 { protected.extend([0, start-1]); }
        if end < size { protected.extend([end, size-1]); }
        protected.sort(); protected.dedup();
        for index in protected {
            let mut actual = expected.clone(); actual[index] ^= 1;
            check_cache_sentinel("cpu-outside-after", "boundary", 0, slot, &expected, &actual, true);
            expected_failures += 1;
            assert_eq!(CACHE_PRESERVATION_MISMATCHES.load(Ordering::Relaxed), expected_failures);
        }
    }
    println!("CPU_CHECKER_SELFTEST PASS: 3 runs x K/V finite patterns; slot0/1/511/1023 target first/last exclusion; exact outside-boundary detection; expected injected mismatch count={expected_failures}");
}


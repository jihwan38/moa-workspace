#![allow(dead_code)]
#[path = "../../src/host/f4_group960.rs"]
mod f4_group960;
#[path = "../../src/host/f4_group960_mse.rs"]
mod f4_group960_mse;
use f4_group960::{PackedF4Weights, QuantizeError};
use f4_group960_mse::{quantize_blocked_e4_mse as serial, quantize_blocked_e4_mse_parallel as parallel};

fn fixture(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut raw = vec![0; rows * cols];
    for (g, group) in raw.chunks_exact_mut(960).enumerate() {
        for (i, value) in group.iter_mut().enumerate() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = match g % 11 {
                0 => 0,
                1 => if i % 2 == 0 { 0 } else { 128 },
                // Every finite E4 code, including exact midpoint fixtures.
                2 => ((i % 127) as u8) | (((i / 127) % 2) as u8 * 128),
                // Vary group maxima throughout all finite E4 exponents.
                3 => ((i % ((g / 11) % 126 + 1)) as u8) | ((i % 2) as u8 * 128),
                4 => [0x00, 0x80, 0x28, 0x34, 0x3a, 0x3e, 0x44, 0x4a][i % 8],
                _ => ((state & 127) as u8 % 127) | ((state >> 8) as u8 & 128),
            };
        }
    }
    raw
}

fn parity(a: &PackedF4Weights, b: &PackedF4Weights) {
    assert_eq!(a.packed_f4, b.packed_f4);
    assert_eq!(a.group_scales.len(), b.group_scales.len());
    for (i, (a, b)) in a.group_scales.iter().zip(&b.group_scales).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "scale group {i}");
    }
}

fn encode(mode: &str, raw: &[u8], rows: usize, cols: usize) -> PackedF4Weights {
    match mode {
        "serial" => serial(raw, rows, cols),
        "parallel" => parallel(raw, rows, cols),
        _ => panic!("unknown mode"),
    }.unwrap()
}

fn checksum(p: &PackedF4Weights) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in &p.packed_f4 {
        h = (h ^ u64::from(*b)).wrapping_mul(0x100000001b3);
    }
    for s in &p.group_scales {
        for b in s.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn proc_value(key: &str) -> String {
    std::fs::read_to_string("/proc/self/status").unwrap().lines()
        .find(|l| l.starts_with(key)).unwrap().to_owned()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map_or("abba", String::as_str);
    let (rows, cols) = (15360, 3840);
    let raw = fixture(rows, cols, 0x12345678a5);
    println!("ENV available_parallelism={} cpu_max={:?} affinity={:?}",
        std::thread::available_parallelism().unwrap().get(),
        std::fs::read_to_string("/sys/fs/cgroup/cpu.max").unwrap_or_default().trim(),
        proc_value("Cpus_allowed_list:"));
    let modes: Vec<&str> = if mode == "abba" {
        vec!["serial", "parallel", "parallel", "serial", "parallel", "serial", "serial", "parallel"]
    } else { vec![mode; 5] };
    let mut expected = None;
    for (iteration, mode) in modes.into_iter().enumerate() {
        let start = std::time::Instant::now();
        let output = encode(mode, &raw, rows, cols);
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let hash = checksum(&output);
        if let Some(expected) = expected { assert_eq!(hash, expected); } else { expected = Some(hash); }
        println!("ENCODE mode={mode} iteration={iteration} ms={ms:.6} packed={} scales={} fnv64={hash:016x}",
            output.packed_f4.len(), output.group_scales.len());
        std::hint::black_box(&output);
        drop(output);
    }
    println!("MEMORY {} {}", proc_value("VmHWM:"), proc_value("VmRSS:"));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_projection_shapes_and_group_permutation() {
        for (rows, cols, seed) in [(15360, 3840, 71), (3840, 15360, 93)] {
            let raw = fixture(rows, cols, seed);
            let candidate = parallel(&raw, rows, cols).unwrap();
            let control = serial(&raw, rows, cols).unwrap();
            parity(&candidate, &control);
            let groups = raw.len() / 960;
            let mut permuted = vec![0; raw.len()];
            // 73 is coprime to 61,440; use a genuine whole-group permutation.
            for g in 0..groups {
                let source = (g * 73 + 127) % groups;
                permuted[g*960..(g+1)*960].copy_from_slice(&raw[source*960..(source+1)*960]);
            }
            let shuffled = parallel(&permuted, rows, cols).unwrap();
            for g in 0..groups {
                let source = (g * 73 + 127) % groups;
                assert_eq!(&shuffled.packed_f4[g*480..(g+1)*480],
                    &control.packed_f4[source*480..(source+1)*480]);
                assert_eq!(shuffled.group_scales[g].to_bits(), control.group_scales[source].to_bits());
            }
        }
    }

    #[test]
    fn small_and_uneven_row_blocks() {
        for blocks in [1, 2, 3, 5, 7, 9] {
            let rows = blocks * 240;
            let raw = fixture(rows, 960, blocks as u64 + 11);
            parity(&parallel(&raw, rows, 960).unwrap(), &serial(&raw, rows, 960).unwrap());
        }
    }

    #[test]
    fn validation_and_earliest_nonfinite_preserved() {
        for (raw, rows, cols) in [(vec![], 0, 960), (vec![], 241, 960), (vec![0; 7], 240, 960)] {
            assert_eq!(parallel(&raw, rows, cols).err(), serial(&raw, rows, cols).err());
        }
        let rows = 9 * 240;
        let mut raw = fixture(rows, 960, 19);
        // Multiple failing workers must return the first source index, not first completion.
        for index in [480 * 960 + 959, 1440 * 960, raw.len() - 1] { raw[index] = 0xff; }
        let expected = QuantizeError::NonFinite { index: 480 * 960 + 959, bits: 0xff };
        assert_eq!(parallel(&raw, rows, 960).err(), Some(expected));
        assert_eq!(parallel(&raw, rows, 960).err(), serial(&raw, rows, 960).err());
        let huge = usize::MAX / 240 * 240;
        assert_eq!(parallel(&[], huge, 960).err(), Some(QuantizeError::SizeOverflow));
    }
}

#[path = "../../src/host/f4_group960.rs"]
mod f4_group960;
#[path = "../../src/host/f4_group960_mse.rs"]
mod f4_group960_mse;
use std::{hint::black_box, time::Instant};

fn memory() -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn main() {
    let repeats: usize = std::env::args()
        .nth(1)
        .unwrap_or("5".into())
        .parse()
        .unwrap();
    let variant = std::env::args().nth(2).unwrap_or("current".into());
    let (rows, cols) = (15360, 3840);
    let mut raw = vec![0u8; rows * cols];
    for (g, group) in raw.chunks_exact_mut(960).enumerate() {
        let cap = (g * 73 + 31) % 127;
        for (k, b) in group.iter_mut().enumerate() {
            *b = ((k * 53 + g * 17) % (cap + 1)) as u8 | ((((k + g) % 2) as u8) << 7);
        }
        group[0] = cap as u8;
    }
    println!(
        "BENCH variant={variant} rows={rows} cols={cols} bytes={} repeats={repeats} source_memory={}",
        raw.len(),
        memory()
    );
    for repeat in 0..repeats {
        let start = Instant::now();
        let (packed, dense, scales) = match variant.as_str() {
            "current" => {
                let result = f4_group960::quantize_blocked_e4(black_box(&raw), rows, cols).unwrap();
                (result.packed_f4, result.dense_e4_codes, result.group_scales)
            }
            "packed" => {
                let result =
                    f4_group960::quantize_blocked_e4_packed(black_box(&raw), rows, cols).unwrap();
                (result.packed_f4, Vec::new(), result.group_scales)
            }
            "mse" => {
                let result =
                    f4_group960_mse::quantize_blocked_e4_mse(black_box(&raw), rows, cols).unwrap();
                (result.packed_f4, Vec::new(), result.group_scales)
            }
            _ => panic!("unknown variant"),
        };
        let elapsed = start.elapsed().as_secs_f64();
        let checksum = packed
            .iter()
            .fold(0u64, |s, b| s.wrapping_add(u64::from(*b)));
        black_box(checksum);
        println!(
            "SAMPLE repeat={repeat} seconds={elapsed:.9} source_MB_per_sec={:.3} packed_bytes={} dense_reference_bytes={} scale_bytes={} checksum={checksum} memory={}",
            raw.len() as f64 / elapsed / 1e6,
            packed.len(),
            dense.len(),
            scales.len() * 4,
            memory()
        );
    }
}

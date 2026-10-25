#[path = "../../src/host/f4_group960.rs"]
mod f4_group960;
use f4_group960::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("tables") => {
            let prefix = &args[2];
            let mut codes = Vec::new();
            for exponent in -11..=7 {
                for bits in 0..=255u8 {
                    codes.push(encode_e4_at_scale(bits, exponent).unwrap_or(255));
                }
            }
            let mut scales = Vec::new();
            let mut decoded = Vec::new();
            for bits in 0..=255u8 {
                scales
                    .extend_from_slice(&scale_exponent_for_max(bits).unwrap_or(999).to_le_bytes());
                decoded.extend_from_slice(
                    &decode_e4(bits)
                        .map(f32::to_bits)
                        .unwrap_or(0xffff_ffff)
                        .to_le_bytes(),
                );
            }
            std::fs::write(format!("{prefix}.codes"), codes)?;
            std::fs::write(format!("{prefix}.exponents"), scales)?;
            std::fs::write(format!("{prefix}.decoded"), decoded)?;
            let f4: Vec<u8> = (0..16)
                .flat_map(|b| decode_f4(b).to_bits().to_le_bytes())
                .collect();
            std::fs::write(format!("{prefix}.f4"), f4)?;
        }
        Some("quantize") => {
            let rows = args[2].parse()?;
            let cols = args[3].parse()?;
            let raw = std::fs::read(&args[4])?;
            let result = quantize_blocked_e4(&raw, rows, cols)?;
            let prefix = &args[5];
            std::fs::write(format!("{prefix}.packed"), result.packed_f4)?;
            std::fs::write(format!("{prefix}.dense"), result.dense_e4_codes)?;
            let scales: Vec<u8> = result
                .group_scales
                .iter()
                .flat_map(|s| s.to_bits().to_le_bytes())
                .collect();
            std::fs::write(format!("{prefix}.scales"), scales)?;
        }
        _ => return Err("expected tables PREFIX or quantize ROWS COLS INPUT PREFIX".into()),
    }
    Ok(())
}

#[path = "../../src/host/f4_group960.rs"]
mod f4_group960;
#[path = "../../src/host/f4_group960_mse.rs"]
mod f4_group960_mse;
use f4_group960::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("mse-tables") => {
            let prefix = &args[2];
            let mut codes = Vec::new();
            let mut scales = Vec::new();
            for base in -11..=7 {
                for candidate in 0..f4_group960_mse::CANDIDATES {
                    scales.extend_from_slice(
                        &f4_group960_mse::candidate_scale(base, candidate)
                            .unwrap()
                            .to_bits()
                            .to_le_bytes(),
                    );
                    for bits in 0..=255u8 {
                        codes.push(
                            f4_group960_mse::candidate_code(bits, base, candidate).unwrap_or(255),
                        );
                    }
                }
            }
            std::fs::write(format!("{prefix}.codes"), codes)?;
            std::fs::write(format!("{prefix}.scales"), scales)?;
        }
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
        Some("quantize") | Some("quantize-packed") | Some("quantize-mse") => {
            let rows = args[2].parse()?;
            let cols = args[3].parse()?;
            let raw = std::fs::read(&args[4])?;
            let prefix = &args[5];
            let scales = if args[1] == "quantize" {
                let result = quantize_blocked_e4(&raw, rows, cols)?;
                std::fs::write(format!("{prefix}.packed"), result.packed_f4)?;
                std::fs::write(format!("{prefix}.dense"), result.dense_e4_codes)?;
                result.group_scales
            } else {
                let result = if args[1] == "quantize-mse" {
                    let (result, report) =
                        f4_group960_mse::quantize_blocked_e4_mse_report(&raw, rows, cols)?;
                    let control = f4_group960_mse::quantize_blocked_e4_mse(&raw, rows, cols)?;
                    assert_eq!(result.packed_f4, control.packed_f4);
                    assert_eq!(result.group_scales, control.group_scales);
                    let errors: Vec<u8> = report
                        .iter()
                        .flat_map(|r| {
                            [
                                r.baseline_squared_error,
                                r.fitted_squared_error,
                                r.source_squared_norm,
                            ]
                        })
                        .flat_map(f64::to_le_bytes)
                        .collect();
                    std::fs::write(format!("{prefix}.errors"), errors)?;
                    result
                } else {
                    quantize_blocked_e4_packed(&raw, rows, cols)?
                };
                std::fs::write(format!("{prefix}.packed"), result.packed_f4)?;
                result.group_scales
            };
            let scales: Vec<u8> = scales
                .iter()
                .flat_map(|s| s.to_bits().to_le_bytes())
                .collect();
            std::fs::write(format!("{prefix}.scales"), scales)?;
        }
        _ => {
            return Err(
                "expected tables PREFIX or quantize[-packed] ROWS COLS INPUT PREFIX".into(),
            );
        }
    }
    Ok(())
}

//! Host-only provided-weight diagnostic. This executable creates no Device.
//! Code selection uses integer division, independently of the device threshold implementation.
use furiosa_opt_gemma4::host::load;
use furiosa_opt_std::prelude::bf16;
use serde_json::{Value, json};
use std::{error::Error, fs::File, time::Instant};

const FLOOR_A_BITS: u16 = 15 << 7; // BF16 2^-112.
const RECIPROCAL_SEVEN_BITS: u32 = 0x3e12_4925; // RN32(1/7).
const SAMPLES: usize = 64;

// Deliberately the exact r1194 selection, including any repeated sampled rows.
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn selected_rows(rows: usize, layer: usize) -> Vec<usize> {
    (0..SAMPLES)
        .map(|pick| {
            if pick < 4 {
                [0, rows / 2 - 1, rows / 2, rows - 1][pick]
            } else {
                mix(pick as u64 + layer as u64 * 313) as usize % rows
            }
        })
        .collect()
}

fn decode_f4(code: u8) -> f32 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 == 0 { magnitude } else { -magnitude }
}

fn decode_e4(code: u8) -> Result<f32, Box<dyn Error>> {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 15 && mantissa == 7 {
        return Err(format!("nonfinite provided E4 blockscale 0x{code:02x}").into());
    }
    let magnitude = if exponent == 0 {
        mantissa as f32 * 2f32.powi(-9)
    } else {
        (8 + mantissa) as f32 * 2f32.powi(exponent as i32 - 10)
    };
    Ok(if code & 128 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

fn ftz_value(bits: u16) -> f64 {
    if bits & 0x7f80 == 0 {
        0.0
    } else {
        bf16::from_bits(bits).to_f32() as f64
    }
}

fn amax_bits(values: &[u16]) -> u16 {
    values.iter().fold(FLOOR_A_BITS, |maximum, &bits| {
        assert_ne!(bits & 0x7f80, 0x7f80, "finite-BF16 contract");
        // Positive finite BF16 bit patterns have the same order as magnitude.
        // Any subnormal is already smaller than the normal floor.
        maximum.max(bits & 0x7fff)
    })
}

fn rational_code(bits: u16, a_bits: u16) -> i32 {
    let absolute = bits & 0x7fff;
    assert!(absolute < 0x7f80 && (FLOOR_A_BITS..0x7f80).contains(&a_bits));
    if absolute & 0x7f80 == 0 {
        return 0;
    } // FTZ includes both signed zeros.
    assert!(absolute <= a_bits);
    let delta = (a_bits >> 7) - (absolute >> 7);
    if delta >= 5 {
        return 0;
    } // 7*255/(128*32) < 1/2.
    let numerator = 7i64 * (128 + (absolute & 127)) as i64;
    let denominator = ((128 + (a_bits & 127)) as i64) << delta;
    // floor(n/d + 1/2), using an independent integer quotient (not seven comparisons).
    let magnitude = ((2 * numerator + denominator) / (2 * denominator)).min(7) as i32;
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn rational_scale(a_bits: u16) -> f32 {
    let a = bf16::from_bits(a_bits).to_f32();
    // One F32 multiplication, after the separately rounded reciprocal constant.
    a * f32::from_bits(RECIPROCAL_SEVEN_BITS)
}

fn coarse_scale(a_bits: u16) -> f32 {
    // This is the original r1194 independent reference, unchanged mathematically.
    let a = bf16::from_bits(a_bits).to_f32() as f64;
    let mut scale = 2f64.powi(-114);
    while 7.0 * scale < a {
        scale *= 2.0;
    }
    scale as f32
}

#[derive(Clone, Copy)]
enum Method {
    CoarsePower2,
    Rational,
}

struct Metrics {
    groups: u64,
    values: u64,
    codes: [u64; 15],
    squared_error: f64,
    squared_reference: f64,
    max_absolute_error: f64,
    scale_min: f32,
    scale_max: f32,
    amax_min: f32,
    amax_max: f32,
}

impl Metrics {
    fn new() -> Self {
        Self {
            groups: 0,
            values: 0,
            codes: [0; 15],
            squared_error: 0.0,
            squared_reference: 0.0,
            max_absolute_error: 0.0,
            scale_min: f32::INFINITY,
            scale_max: 0.0,
            amax_min: f32::INFINITY,
            amax_max: 0.0,
        }
    }

    fn add_group(&mut self, values: &[u16], method: Method) {
        let a_bits = amax_bits(values);
        let a = bf16::from_bits(a_bits).to_f32();
        let scale = match method {
            Method::CoarsePower2 => coarse_scale(a_bits),
            Method::Rational => rational_scale(a_bits),
        };
        assert!(scale.is_normal() && scale > 0.0);
        self.groups += 1;
        self.scale_min = self.scale_min.min(scale);
        self.scale_max = self.scale_max.max(scale);
        self.amax_min = self.amax_min.min(a);
        self.amax_max = self.amax_max.max(a);
        for &bits in values {
            let code = match method {
                Method::CoarsePower2 => {
                    (ftz_value(bits) / scale as f64).round().clamp(-7.0, 7.0) as i32
                }
                Method::Rational => rational_code(bits, a_bits),
            };
            assert!((-7..=7).contains(&code));
            self.codes[(code + 7) as usize] += 1;
            self.values += 1;
            // Match r1194: error is against provided BF16 weights, before FTZ/global.
            let original = bf16::from_bits(bits).to_f32() as f64;
            let reconstructed = code as f64 * scale as f64;
            let error = reconstructed - original;
            self.squared_error += error * error;
            self.squared_reference += original * original;
            self.max_absolute_error = self.max_absolute_error.max(error.abs());
        }
    }

    fn report(&self) -> Value {
        assert_eq!(self.codes.iter().sum::<u64>(), self.values);
        let relative_l2 = if self.squared_reference == 0.0 {
            0.0
        } else {
            (self.squared_error / self.squared_reference).sqrt()
        };
        json!({"groups":self.groups,"values":self.values,
            "code_min":self.codes.iter().position(|&n| n > 0).map(|i|i as i32-7),
            "code_max":self.codes.iter().rposition(|&n| n > 0).map(|i|i as i32-7),
            "code_histogram_minus7_through7":self.codes,
            "relative_l2":relative_l2,"squared_error":self.squared_error,
            "squared_reference":self.squared_reference,"max_absolute_error":self.max_absolute_error,
            "scale_min":self.scale_min,"scale_max":self.scale_max,
            "clamped_amax_min":self.amax_min,"clamped_amax_max":self.amax_max})
    }
}

// A deterministic input fingerprint, explicitly not a cryptographic file hash.
struct Fingerprint(u64);
impl Fingerprint {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

fn oracle_boundary_checks() {
    assert_eq!((1.0f32 / 7.0f32).to_bits(), RECIPROCAL_SEVEN_BITS);
    for (bits, a_bits, expected) in [
        (0x0000, FLOOR_A_BITS, 0),
        (0x8000, FLOOR_A_BITS, 0),
        (0x007f, FLOOR_A_BITS, 0),
        (0x807f, FLOOR_A_BITS, 0),
        (0x0080, FLOOR_A_BITS, 0),
        (0x8080, FLOOR_A_BITS, 0),
        (FLOOR_A_BITS, FLOOR_A_BITS, 7),
        (FLOOR_A_BITS | 0x8000, FLOOR_A_BITS, -7),
        (0x3f00, 0x3f80, 4),
        (0xbf00, 0x3f80, -4), // exact +/-3.5 tie.
        (0x3c80, 0x3f80, 0),
        (0x7f7f, 0x7f7f, 7),
        (0xff7f, 0x7f7f, -7),
    ] {
        assert_eq!(rational_code(bits, a_bits), expected);
    }
    for a_bits in [FLOOR_A_BITS, 0x3f80, 0x7f7f] {
        assert!(rational_scale(a_bits).is_normal());
        assert!(coarse_scale(a_bits).is_normal());
    }
    // Every finite BF16 bit pattern is legal under the max-finite A bound.
    let mut checked = 0;
    for bits in 0u16..=u16::MAX {
        if bits & 0x7f80 == 0x7f80 {
            continue;
        }
        assert!((-7..=7).contains(&rational_code(bits, 0x7f7f)));
        checked += 1;
    }
    assert_eq!(checked, 65_280);
    println!(
        "A4_RATIONAL_ORACLE {}",
        json!({"finite_patterns_checked":checked,
        "explicit_boundary_cases":13,"reciprocal_seven_f32_bits":format!("{RECIPROCAL_SEVEN_BITS:08x}"),
        "scope":"host independent integer quotient; no device code or timing proof"})
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args_os().len() != 1 {
        return Err("this fixed diagnostic takes no arguments".into());
    }
    oracle_boundary_checks();
    let started = Instant::now();
    let path = load::model_dir()?.join("model.safetensors");
    let file = File::open(&path)?;
    let file_bytes = file.metadata()?.len();
    // Read-only mmap follows the repository's existing provided-weight diagnostics.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    println!(
        "A4_RATIONAL_SOURCE {}",
        json!({"path":path,"file_bytes":file_bytes,
        "tensor_count":tensors.len(),"layers":[0,6,47],"sampled_rows_per_projection":SAMPLES,
        "reference":"BF16(provided E2M1 nibble * provided E4M3 per16 blockscale), before global",
        "rational_code":"sign(x)*floor(7*abs(FTZ(x))/max(2^-112,amax)+1/2)",
        "reconstruction_scale":"RN32(A*RN32(1/7)), separate from exact rational code selection",
        "baseline":"r1194 coarse power2 row/block960 reference and identical row selection",
        "excludes":"r742 synthetic 0.1319; no actual activations, model accuracy, NPU or packing cost"})
    );
    let mut total_values = 0u64;
    let mut total_source = Fingerprint::new();
    for layer in [0usize, 6, 47] {
        for projection in ["up_proj", "gate_proj", "down_proj"] {
            let projection_start = Instant::now();
            let prefix = format!("model.language_model.layers.{layer}.mlp.{projection}");
            let packed_name = format!("{prefix}.weight_packed");
            let scale_name = format!("{prefix}.weight_scale");
            let packed = tensors.tensor(&packed_name)?;
            let blockscales = tensors.tensor(&scale_name)?;
            let (rows, width) = if projection == "down_proj" {
                (3840, 15360)
            } else {
                (15360, 3840)
            };
            if packed.dtype() != safetensors::Dtype::U8
                || packed.shape() != [rows, width / 2]
                || packed.data().len() != rows * width / 2
                || blockscales.dtype() != safetensors::Dtype::F8_E4M3
                || blockscales.shape() != [rows, width / 16]
                || blockscales.data().len() != rows * width / 16
            {
                return Err(format!(
                    "unexpected {prefix} headers: packed {:?} {:?}, scales {:?} {:?}",
                    packed.dtype(),
                    packed.shape(),
                    blockscales.dtype(),
                    blockscales.shape()
                )
                .into());
            }
            let selected = selected_rows(rows, layer);
            let mut source_fingerprint = Fingerprint::new();
            let mut bf16_fingerprint = Fingerprint::new();
            let mut metrics: [Metrics; 4] = std::array::from_fn(|_| Metrics::new());
            let (mut minimum, mut maximum) = (f32::INFINITY, f32::NEG_INFINITY);
            let (mut negative_zeros, mut positive_zeros, mut subnormals) = (0u64, 0u64, 0u64);
            for &row in &selected {
                let packed_row = &packed.data()[row * width / 2..(row + 1) * width / 2];
                let scale_row = &blockscales.data()[row * width / 16..(row + 1) * width / 16];
                source_fingerprint.add(&(row as u64).to_le_bytes());
                source_fingerprint.add(packed_row);
                source_fingerprint.add(scale_row);
                bf16_fingerprint.add(&(row as u64).to_le_bytes());
                let mut values = Vec::with_capacity(width);
                for k in 0..width {
                    let code = (packed_row[k / 2] >> (4 * (k % 2))) & 15;
                    let value = bf16::from_f32(decode_f4(code) * decode_e4(scale_row[k / 16])?);
                    let f = value.to_f32();
                    if !f.is_finite() {
                        return Err(
                            format!("nonfinite reconstruction {prefix} row{row} col{k}").into()
                        );
                    }
                    let bits = value.to_bits();
                    minimum = minimum.min(f);
                    maximum = maximum.max(f);
                    negative_zeros += u64::from(bits == 0x8000);
                    positive_zeros += u64::from(bits == 0);
                    subnormals += u64::from(bits & 0x7f80 == 0 && bits & 127 != 0);
                    bf16_fingerprint.add(&bits.to_le_bytes());
                    values.push(bits);
                }
                metrics[0].add_group(&values, Method::CoarsePower2);
                metrics[1].add_group(&values, Method::Rational);
                for block in values.chunks_exact(960) {
                    metrics[2].add_group(block, Method::CoarsePower2);
                    metrics[3].add_group(block, Method::Rational);
                }
            }
            for metric in &metrics {
                assert_eq!(metric.values, (SAMPLES * width) as u64);
            }
            total_values += (SAMPLES * width) as u64;
            total_source.add(prefix.as_bytes());
            total_source.add(&source_fingerprint.0.to_le_bytes());
            println!(
                "A4_RATIONAL_WEIGHT {}",
                json!({"layer":layer,"projection":projection,
                "packed_tensor":{"name":packed_name,"dtype":format!("{:?}",packed.dtype()),"shape":packed.shape(),"bytes":packed.data().len()},
                "scale_tensor":{"name":scale_name,"dtype":format!("{:?}",blockscales.dtype()),"shape":blockscales.shape(),"bytes":blockscales.data().len()},
                "sampled_rows":selected,"values":SAMPLES*width,"finite_min":minimum,"finite_max":maximum,
                "negative_zero_count":negative_zeros,"positive_zero_count":positive_zeros,"ftz_subnormal_count":subnormals,
                "sampled_source_fnv1a64":source_fingerprint.hex(),"sampled_bf16_fnv1a64":bf16_fingerprint.hex(),
                "fingerprint_scope":"noncryptographic; each selected row index LE64 then raw packed row+scale row; BF16 stream uses row index then LE16 decoded values",
                "coarse_power2_row":metrics[0].report(),"rational_row":metrics[1].report(),
                "coarse_power2_block960":metrics[2].report(),"rational_block960":metrics[3].report(),
                "host_elapsed_ms_including_decode_and_all_four_audits":projection_start.elapsed().as_secs_f64()*1000.0})
            );
        }
    }
    assert_eq!(total_values, 4_423_680);
    println!(
        "A4_RATIONAL_DONE {}",
        json!({"status":"PASS provided-weight diagnostic completed",
        "projections":9,"sampled_rows":576,"source_values":total_values,"quantized_code_checks":4*total_values,
        "sampled_source_chain_fnv1a64":total_source.hex(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"weight reconstruction only; no model/activation quality or complete A4W4 latency claim"})
    );
    Ok(())
}

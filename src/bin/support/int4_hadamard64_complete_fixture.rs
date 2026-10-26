//! All-row provided-weight rotation/encoding and matched complete-chain metrics.
use super::{cpu, transform};
use cpu::{Result, SampleWeight, WeightCpu, EXPANDED, HIDDEN, SAMPLES};
use furiosa_opt_std::prelude::bf16;
use std::collections::BTreeMap;

pub fn weights(
    t: &safetensors::SafeTensors<'_>,
    name: &str,
    base: &WeightCpu,
) -> Result<WeightCpu> {
    base.verify_source(t)?;
    let (rows, cols) = if name == "down_proj" {
        (HIDDEN, EXPANDED)
    } else {
        (EXPANDED, HIDDEN)
    };
    assert!(["up_proj", "gate_proj", "down_proj"].contains(&name));
    assert_eq!((base.rows, base.cols), (rows, cols));
    let stem = format!("model.language_model.layers.0.mlp.{name}");
    let packed = t.tensor(&format!("{stem}.weight_packed"))?;
    let scales = t.tensor(&format!("{stem}.weight_scale"))?;
    assert_eq!(packed.shape(), [rows, cols / 2]);
    assert_eq!(packed.dtype(), safetensors::Dtype::U8);
    assert_eq!(scales.shape(), [rows, cols / 16]);
    assert_eq!(scales.dtype(), safetensors::Dtype::F8_E4M3);
    let mut result = WeightCpu {
        rows,
        cols,
        packed: vec![0; rows * cols / 2],
        dense: vec![0; rows * cols],
        scales: vec![f32::NAN; rows * cols / 960],
        samples: BTreeMap::new(),
        global: base.global,
        source: base.source.clone(),
    };
    let selected: std::collections::BTreeSet<_> =
        (0..SAMPLES).map(|i| cpu::sample_row(i, rows)).collect();
    assert_eq!(selected.len(), SAMPLES);
    let mut seen = vec![false; result.scales.len()];
    const E4: [u8; 8] = [0, 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e];
    for row in 0..rows {
        let input: Vec<u16> = (0..cols)
            .map(|k| {
                let at = row * cols + k;
                let nib = (packed.data()[at / 2] >> (4 * (at % 2))) & 15;
                let f4 = [0., 0.5, 1., 1.5, 2., 3., 4., 6.][(nib & 7) as usize]
                    * if nib & 8 == 0 { 1.0 } else { -1.0 };
                let b = scales.data()[at / 16];
                let e = (b >> 3) & 15;
                let m = b & 7;
                assert!(!(e == 15 && m == 7));
                let a = if e == 0 {
                    m as f32 / 512.0
                } else {
                    (8 + m) as f32 * 2f32.powi(e as i32 - 10)
                };
                bf16::from_f32(f4 * a * if b & 128 == 0 { 1.0 } else { -1.0 }).to_bits()
            })
            .collect();
        // Scratch is one row only: no retained all-weight F64/rotatedBF16 array.
        let rotated = transform::rotate_row(&input)?;
        assert_eq!(
            rotated,
            transform::reference_row(&input)?,
            "independent all-weight rotation {name} row{row}"
        );
        let mut sample = selected.contains(&row).then(|| SampleWeight {
            codes: Vec::new(),
            scales: Vec::new(),
        });
        for (part, group) in rotated.chunks_exact(960).enumerate() {
            let (q, s) = transform::fixed::rational::reference(group);
            let g = cpu::weight_group(row, part, cols);
            assert!(!seen[g]);
            seen[g] = true;
            assert_eq!((g / (cols / 960) / 240) * 240 + g % 240, row);
            assert_eq!(g / 240 % (cols / 960), part);
            assert!(s.is_normal() && s > 0.0);
            result.scales[g] = s;
            for (k, pair) in q.chunks_exact(2).enumerate() {
                result.packed[g * 480 + k] = (pair[0] as u8 & 15) | ((pair[1] as u8 & 15) << 4);
            }
            for (k, &v) in q.iter().enumerate() {
                assert!((-7..=7).contains(&v));
                result.dense[g * 960 + k] =
                    E4[v.unsigned_abs() as usize] | if v < 0 { 128 } else { 0 };
                let n = (result.packed[(g * 960 + k) / 2] >> (4 * (k % 2))) & 15;
                assert_eq!(
                    if n >= 8 { n as i32 - 16 } else { n as i32 },
                    v,
                    "independent physical-word inverse"
                );
            }
            if let Some(v) = sample.as_mut() {
                v.codes.extend(q.iter().map(|&x| x as i8));
                v.scales.push(s);
            }
        }
        if let Some(v) = sample {
            result.samples.insert(row, v);
        }
    }
    assert!(seen.iter().all(|&x| x));
    assert_eq!(result.samples.len(), SAMPLES);
    base.verify_source(t)?;
    println!(
        "I4_H64_COMPLETE_WEIGHTS {}",
        serde_json::json!({"projection":name,"values":rows*cols,"groups":result.scales.len(),
        "all_row_independent_rotation":true,"global_unchanged":true,"source":result.source,"no_native_row_dequant":true,
        "provided_BF16_then_F32_rotation_then_BF16_then_rational":true,"per_row_scratch_only":true})
    );
    Ok(result)
}
#[derive(Clone)]
pub struct Error {
    pub relative_l2: f64,
    pub sse: f64,
    pub reference_ss: f64,
    pub changed: usize,
    pub maxabs: f64,
    pub values: usize,
}
pub fn error(got: &[u8], reference: &[u8]) -> Error {
    assert_eq!(got.len(), reference.len());
    assert_eq!(got.len() % 2, 0);
    let (mut sse, mut den, mut max, mut changed) = (0f64, 0f64, 0f64, 0usize);
    for (a, b) in got.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let a = u16::from_le_bytes(a.try_into().unwrap());
        let b = u16::from_le_bytes(b.try_into().unwrap());
        let x = bf16::from_bits(a).to_f32() as f64;
        let y = bf16::from_bits(b).to_f32() as f64;
        assert!(x.is_finite() && y.is_finite());
        sse += (x - y) * (x - y);
        den += y * y;
        max = max.max((x - y).abs());
        changed += usize::from(a != b);
    }
    assert!(
        den > 0.0 && den.is_finite() && sse.is_finite(),
        "positive finite reference required"
    );
    let relative_l2 = (sse / den).sqrt();
    assert!(relative_l2.is_finite());
    Error {
        relative_l2,
        sse,
        reference_ss: den,
        changed,
        maxabs: max,
        values: got.len() / 2,
    }
}
pub fn report(e: &Error, comparison: &str, round: usize, step: usize) {
    println!(
        "I4_H64_COMPLETE_DRIFT {}",
        serde_json::json!({"comparison":comparison,"round":round,"step":step,"bf16_values":e.values,
        "changed":e.changed,"relative_l2":e.relative_l2,"sse":e.sse,"reference_ss":e.reference_ss,"maxabs":e.maxabs,
        "matched_incoming_inputs":true,"common_feedback":"original q020 residual","no_task_quality_or_speed_claim":true})
    );
}

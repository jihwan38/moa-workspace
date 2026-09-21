//! Standalone O-only diagnostics. All official device sources remain unchanged.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops};
use std::io::Write;

const NAMES: [&str; 8] = [
    "signed_zeros", "column_row_boundaries", "amax_below_powers_two",
    "amax_at_powers_two", "amax_above_powers_two", "small_component_thresholds",
    "low_amax_floor", "correlated_cancellation_in_distribution_support",
];

fn checksum(bytes: &[u8]) -> u32 {
    bytes.chunks(4).enumerate().fold(0u32, |acc, (i, chunk)| {
        let mut word = [0u8; 4]; word[..chunk.len()].copy_from_slice(chunk);
        acc.wrapping_add(u32::from_le_bytes(word).wrapping_mul((2*i+1) as u32))
    })
}

fn weight_codes(case: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(H::SIZE * Qs::SIZE);
    for row in 0..H::SIZE {
        for q in 0..Qs::SIZE {
            let code = if case == 7 {
                if q % 256 < 2 { 0x38 } else { 0x77 }
            } else {
                let exponent = 7 + ((row / 120 + q / 256 + case) % 8);
                let mantissa = (3*row + q + q/16) % 8;
                let sign = ((row/8 + q/32 + q%3 + case) % 2) << 7;
                (sign | (exponent << 3) | mantissa) as u8
            };
            result.push(code);
        }
    }
    result
}

fn raw(f: &safetensors::SafeTensors<'_>, name: &str) -> Vec<u8> {
    let t = f.tensor(name).unwrap();
    assert_eq!(t.dtype(), safetensors::Dtype::U8);
    t.data().to_vec()
}
fn check_input(f: &safetensors::SafeTensors<'_>, case: usize, name: &str, bytes: &[u8]) {
    let t = f.tensor(&format!("case{case}.checksum.{name}")).unwrap();
    assert_eq!(t.dtype(), safetensors::Dtype::I64);
    let want = i64::from_le_bytes(t.data().try_into().unwrap()) as u32;
    assert_eq!(checksum(bytes), want, "case {case} {name} generator/payload checksum");
    println!("INPUT_CHECK case={case} tensor={name} bytes={} checksum={want:08x} status=PASS", bytes.len());
}
fn expected(f: &safetensors::SafeTensors<'_>, case: usize) -> Vec<f32> {
    let t = f.tensor(&format!("case{case}.expected")).unwrap();
    assert_eq!(t.dtype(), safetensors::Dtype::F32);
    let v: Vec<_> = t.data().chunks_exact(4).map(|p| f32::from_le_bytes(p.try_into().unwrap())).collect();
    assert_eq!(v.len(), H::SIZE);
    assert!(v.iter().all(|v| v.is_finite()));
    v
}
fn compare(case: usize, got: &[bf16], want: &[f32]) -> usize {
    assert_eq!(got.len(), H::SIZE);
    let mut failures = 0usize; let mut nonfinite = 0usize; let mut exact = 0usize;
    let mut max_error = 0.0f32; let mut sum = 0.0f64;
    let mut groups = [0usize; 32];
    for (row, (&value, &w)) in got.iter().zip(want).enumerate() {
        let g = value.to_f32(); let err = (g-w).abs(); let limit = 0.05f32 + 0.01f32*w.abs();
        if g.to_bits() == w.to_bits() { exact += 1; }
        if g.is_finite() { max_error = max_error.max(err); sum += f64::from(err); }
        else { nonfinite += 1; }
        if !g.is_finite() || err > limit {
            if failures < 32 {
                println!("O_MISMATCH case={case} row={row} got={g:.9e} want={w:.9e} got_bf16={:04x} want_bf16={:04x} error={err:.9e} limit={limit:.9e}", value.to_bits(), bf16::from_f32(w).to_bits());
            }
            failures += 1; groups[row/120] += 1;
        }
    }
    for row in [0,119,120,239,240,1919,1920,3839] {
        println!("O_BOUNDARY_SAMPLE case={case} row={row} got_bf16={:04x} want_bf16={:04x}", got[row].to_bits(), bf16::from_f32(want[row]).to_bits());
    }
    println!("O_GROUP_FAILURES case={case} rows_per_group=120 counts={groups:?}");
    println!("O_RESULT case={case} name={} checked={} failures={failures} nonfinite={nonfinite} exact_f32={exact} max_abs_finite={max_error:.9e} mean_abs_finite={:.9e} status={}", NAMES[case], got.len(), sum/(got.len()-nonfinite).max(1) as f64, if failures == 0 {"PASS"} else {"FAIL"});
    failures
}

#[tokio::main]
async fn main() {
    let args: Vec<_> = std::env::args().collect();
    let host_only = args.iter().any(|a| a == "--host-check-only");
    let dump = args.windows(2).find(|a| a[0] == "--dump-weights").map(|a| a[1].clone());
    let path = args.windows(2).find(|a| a[0] == "--fixture").map(|a| a[1].as_str()).unwrap_or("boundary.safetensors");
    let bytes = std::fs::read(path).expect("read immutable CPU oracle payload");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    if let Some(dir) = &dump { assert!(host_only); std::fs::create_dir_all(dir).unwrap(); }
    let mut device = if host_only {None} else {Some(Device::new(ops::sliding_attention_output.topology()).expect("create Device"))};
    println!("O_SUITE_BEGIN cases=8 atol=0.05 rtol=0.01 diagnostic=true official_performance=false host_only={host_only}");
    let mut total = 0usize; let mut completed = 0usize;
    for case in 0..NAMES.len() {
        let x = raw(&fixture, &format!("case{case}.input.x"));
        let gamma = raw(&fixture, &format!("case{case}.input.post_attn_rms_weight"));
        let scale = raw(&fixture, &format!("case{case}.input.o_weight_scale"));
        let residual = raw(&fixture, &format!("case{case}.input.residual"));
        let weights = weight_codes(case);
        for (name, data) in [("x", &x), ("post_attn_rms_weight", &gamma), ("o_weight_scale", &scale), ("residual", &residual), ("o_weight", &weights)] { check_input(&fixture, case, name, data); }
        assert_eq!(x.len(), 2*Qs::SIZE);
        for data in [&gamma, &scale, &residual] { assert_eq!(data.len(), 2*H::SIZE); }
        let want = expected(&fixture, case);
        if let Some(dir) = &dump { std::fs::write(format!("{dir}/case{case}.weights.bin"), &weights).unwrap(); }
        if host_only { completed += 1; continue; }
        let device = device.as_mut().unwrap();
        let x: HbmTensor<bf16,Chip,m![Ns,Gs,Ds]> = HostTensor::<bf16,m![Ns,Gs,Ds]>::from_buf(x).to_hbm(&mut device.pdma).await.unwrap();
        let gamma: HbmTensor<bf16,Chip,m![H]> = HostTensor::<bf16,m![H]>::from_buf(gamma).to_hbm(&mut device.pdma).await.unwrap();
        let scale: HbmTensor<bf16,Chip,m![H]> = HostTensor::<bf16,m![H]>::from_buf(scale).to_hbm(&mut device.pdma).await.unwrap();
        let weights: HbmTensor<f8e4m3,Chip,m![H,Qs]> = HostTensor::<f8e4m3,m![H,Qs]>::from_buf(weights).to_hbm(&mut device.pdma).await.unwrap();
        let mut residual: HbmTensor<bf16,Chip,m![H]> = HostTensor::<bf16,m![H]>::from_buf(residual).to_hbm(&mut device.pdma).await.unwrap();
        println!("O_LAUNCH_BEGIN case={case} name={}", NAMES[case]); std::io::stdout().flush().unwrap();
        launch(ops::sliding_attention_output,(&mut *device,&x,&gamma,&weights,&scale,&mut residual)).await.expect("O boundary diagnostic launch");
        let got: HostTensor<bf16,m![H]> = residual.to_host(&mut device.pdma).await.unwrap();
        total += compare(case, &got.into_vec(), &want); completed += 1;
    }
    println!("O_SUITE_RESULT completed={completed} planned=8 total_failed_values={total} status={}", if total == 0 {"PASS"} else {"FAIL"});
    if total != 0 { std::process::exit(1); }
}

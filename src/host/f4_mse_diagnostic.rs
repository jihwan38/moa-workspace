//! Supplied-weight-only scale-fitting diagnostic. No F4 consumer or model route.
use crate::{Chip, axes::{H,L}};
use furiosa_opt_std::prelude::*;
use super::{load::{self, Layer}, f4_group960::{PackedF4Weights,quantize_blocked_e4_packed},
    f4_group960_mse::quantize_blocked_e4_mse_report};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;

fn e4(b:u8)->f64 {
    let (e,m)=((b>>3)&15,b&7);assert!(e!=15||m!=7);
    let x=if e==0 {m as f64/512.} else {(1.+m as f64/8.)*2f64.powi(e as i32-7)};
    if b&128==0 {x} else {-x}
}
fn reconstructed(q:&PackedF4Weights,i:usize)->f64 {
    let code=(q.packed_f4[i/2]>>(4*(i%2)))&15;
    let x=[0.,0.5,1.,1.5,2.,3.,4.,6.][(code&7)as usize];
    (if code&8==0{x}else{-x})*q.group_scales[i/960]as f64
}

async fn projection<E:M,S:M>(d:&mut Device,weight:&HbmTensor<f8e4m3,Chip,E>,
    row_scales:&HbmTensor<f32,Chip,S>,rows:usize,cols:usize,layer:usize,label:&str)->Result<()> {
    let start=std::time::Instant::now();
    let raw=weight.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf();
    let read_ms=start.elapsed().as_secs_f64()*1000.;
    assert_eq!(raw.len(),rows*cols);
    let scale_bytes=row_scales.to_host::<S>(&mut d.pdma).await?.into_inner().into_buf();
    let row_scales:Vec<f32>=scale_bytes.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
    assert_eq!(row_scales.len(),rows);assert!(row_scales.iter().all(|v|v.is_finite()));
    let start=std::time::Instant::now();
    let baseline=quantize_blocked_e4_packed(&raw,rows,cols)?;
    let baseline_ms=start.elapsed().as_secs_f64()*1000.;
    let start=std::time::Instant::now();
    let (fitted,errors)=quantize_blocked_e4_mse_report(&raw,rows,cols)?;
    let fitting_report_ms=start.elapsed().as_secs_f64()*1000.;
    assert_eq!(baseline.packed_f4.len(),raw.len()/2);assert_eq!(fitted.packed_f4.len(),raw.len()/2);
    assert_eq!(errors.len(),rows*cols/960);
    let (mut old_sse,mut new_sse,mut norm,mut old_weighted,mut new_weighted,mut norm_weighted)=(0.,0.,0.,0.,0.,0.);
    let(mut changed,mut improved)=(0usize,0usize);
    for (g,error) in errors.iter().enumerate() {
        assert!(error.fitted_squared_error<=error.baseline_squared_error,"group SSE worsened");
        assert!(error.source_squared_norm.is_finite());
        old_sse+=error.baseline_squared_error;new_sse+=error.fitted_squared_error;norm+=error.source_squared_norm;
        let row=(g/(240*(cols/960)))*240+g%240;let w=(row_scales[row]as f64).powi(2);
        old_weighted+=error.baseline_squared_error*w;new_weighted+=error.fitted_squared_error*w;norm_weighted+=error.source_squared_norm*w;
        changed+=usize::from(baseline.group_scales[g].to_bits()!=fitted.group_scales[g].to_bits());
        improved+=usize::from(error.fitted_squared_error<error.baseline_squared_error);
    }
    // Independent per-value F64 decode/squared errors for complete sampled
    // groups, including both blocked row and K-part boundaries.
    for sample in 0..64 {
        let g=if sample<8 {[0,1,239,240,errors.len()/2-1,errors.len()/2,errors.len()-2,errors.len()-1][sample]}
            else {(sample*977+layer*31)%errors.len()};
        let(mut a,mut b,mut n)=(0.,0.,0.);
        for k in 0..960 {let i=g*960+k;let x=e4(raw[i]);a+=(x-reconstructed(&baseline,i)).powi(2);b+=(x-reconstructed(&fitted,i)).powi(2);n+=x*x;}
        assert_eq!(a,errors[g].baseline_squared_error,"baseline group F64 mismatch");
        assert_eq!(b,errors[g].fitted_squared_error,"fitted group F64 mismatch");
        assert_eq!(n,errors[g].source_squared_norm,"group norm F64 mismatch");
    }
    let memory=std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines()
        .filter(|s|s.starts_with("VmHWM:")||s.starts_with("VmRSS:")).collect::<Vec<_>>().join(";");
    println!("F4_MSE_WEIGHT layer={layer} projection={label} values={} groups={} changed_scales={changed} strictly_improved_groups={improved} baseline_relative_l2={} fitted_relative_l2={} baseline_row_dequant_l2={} fitted_row_dequant_l2={} D2H_ms={read_ms} baseline_encode_ms={baseline_ms} fitted_encode_and_histogram_report_ms={fitting_report_ms} independent_F64_values={} host_memory={memory} source=current_native_E4_not_original_NVFP4 no_upload_or_device_F4_consumer=true",
        raw.len(),errors.len(),(old_sse/norm.max(1e-300)).sqrt(),(new_sse/norm.max(1e-300)).sqrt(),
        (old_weighted/norm_weighted.max(1e-300)).sqrt(),(new_weighted/norm_weighted.max(1e-300)).sqrt(),64*960);
    Ok(())
}

pub async fn run()->Result<()> {
    let first=std::env::args().nth(1).unwrap_or("0".into()).parse::<usize>()?;
    let last=std::env::args().nth(2).unwrap_or("4".into()).parse::<usize>()?;
    assert!(first<last&&last<=4);
    let mut d=Device::new(crate::ops_prepack_alloc::prepack_up.topology())?;
    let start=std::time::Instant::now();let weights=load::load_weights(&mut d).await?;
    println!("F4_MSE_LOAD current_model_load_seconds={}",start.elapsed().as_secs_f64());
    for &layer in &[0usize,5,6,47][first..last] {
        let w=match &weights.layers[layer] {Layer::Sliding(l)=>&l.mlp,Layer::Full(l)=>&l.mlp};
        let n=&w.native_fp8;
        projection(&mut d,&n.up,&n.up_dequant,L::SIZE,H::SIZE,layer,"up").await?;
        projection(&mut d,&n.gate,&n.gate_dequant,L::SIZE,H::SIZE,layer,"gate").await?;
        projection(&mut d,&n.down,&n.down_dequant,H::SIZE,L::SIZE,layer,"down").await?;
    }
    println!("PASS F4_MSE_WEIGHT_ONLY projections={} all_groups_SSE_not_worse=true independent_F64_values={} no_F4_model_or_quality_claim=true",3*(last-first),3*(last-first)*64*960);
    Ok(())
}

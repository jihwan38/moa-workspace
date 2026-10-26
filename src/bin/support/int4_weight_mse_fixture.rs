//! Host-only provided-layer0 fitting/metrics. No activation or device changes.
use super::cpu;
use cpu::{HIDDEN,EXPANDED,TOKENS,SAMPLES,WeightCpu,SampleWeight,Result};
use furiosa_opt_std::prelude::bf16;
use std::collections::BTreeMap;
#[path="int4_weight_mse.rs"] pub mod fitter;

fn f4(b:u8)->f32{let x=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8==0{x}else{-x}}
fn e4(b:u8)->f32{let e=(b>>3)&15;let m=b&7;assert!(!(e==15&&m==7));let x=if e==0{m as f32/512.0}else{(8+m)as f32*2f32.powi(e as i32-10)};if b&128==0{x}else{-x}}
fn nibble(p:&[u8],i:usize)->i8{let x=(p[i/2]>>(4*(i%2)))&15;if x>=8{x as i8-16}else{x as i8}}
fn bf(bytes:&[u8],i:usize)->u16{u16::from_le_bytes(bytes[2*i..2*i+2].try_into().unwrap())}
fn value(b:u16)->f64{bf16::from_bits(b).to_f32()as f64}

#[derive(Default)]
struct Error{values:usize,norm:f64,sse:f64,max:f64,zeros:usize}
impl Error{
    fn add(&mut self,x:f64,reconstruction:f32){let y=reconstruction as f64;assert!(x.is_finite()&&y.is_finite());
        self.values+=1;self.norm+=x*x;self.sse+=(x-y)*(x-y);self.max=self.max.max((x-y).abs());self.zeros+=usize::from(y==0.0);}
    fn json(&self)->serde_json::Value{serde_json::json!({"values":self.values,"relative_l2":(self.sse/self.norm.max(1e-300)).sqrt(),
        "sse":self.sse,"reference_ss":self.norm,"maxabs":self.max,"reconstructed_zeros":self.zeros})}
}

pub fn weights(t:&safetensors::SafeTensors<'_>,projection:&str,base:&WeightCpu)->Result<WeightCpu>{
    let (rows,cols)=if projection=="down_proj"{(HIDDEN,EXPANDED)}else{(EXPANDED,HIDDEN)};
    assert!(["up_proj","gate_proj","down_proj"].contains(&projection));assert_eq!((base.rows,base.cols),(rows,cols));
    base.verify_source(t)?;
    let prefix=format!("model.language_model.layers.0.mlp.{projection}");
    let packed=t.tensor(&format!("{prefix}.weight_packed"))?;let scale=t.tensor(&format!("{prefix}.weight_scale"))?;
    assert_eq!(packed.dtype(),safetensors::Dtype::U8);assert_eq!(packed.shape(),[rows,cols/2]);
    assert_eq!(scale.dtype(),safetensors::Dtype::F8_E4M3);assert_eq!(scale.shape(),[rows,cols/16]);
    let mut result=WeightCpu{rows,cols,packed:vec![0;base.packed.len()],dense:vec![0;base.dense.len()],
        scales:vec![f32::NAN;base.scales.len()],samples:BTreeMap::new(),global:base.global,source:base.source.clone()};
    let selected:BTreeMap<usize,usize>=(0..SAMPLES).map(|i|(cpu::sample_row(i,rows),i)).collect();assert_eq!(selected.len(),SAMPLES);
    let mut choices=[0usize;26];let(mut better,mut zero_groups)=(0,0);
    let(mut baseline_sse,mut fitted_sse,mut source_ss)=(0i128,0i128,0i128);
    let(mut original_error,mut fitted_error)=(Error::default(),Error::default());
    let mut seen=vec![false;base.scales.len()];
    const E4:[u8;8]=[0,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e];
    for row in 0..rows{
        let mut sample=selected.contains_key(&row).then(||SampleWeight{codes:Vec::new(),scales:Vec::new()});
        for part in 0..cols/960{
            let g=cpu::weight_group(row,part,cols);assert!(!seen[g]);seen[g]=true;
            let values:Vec<u16>=(0..960).map(|k|{let at=row*cols+part*960+k;
                let code=(packed.data()[at/2]>>(4*(at%2)))&15;
                bf16::from_f32(f4(code)*e4(scale.data()[at/16])).to_bits()}).collect();
            let old_codes:Vec<i8>=(0..960).map(|k|nibble(&base.packed,g*960+k)).collect();
            let fit=fitter::fit(&values,&old_codes,base.scales[g])?;
            assert!(fit.selected_sse<=fit.baseline_sse);choices[fit.selected]+=1;
            better+=usize::from(fit.selected_sse<fit.baseline_sse);zero_groups+=usize::from(fit.source_ss==0);
            baseline_sse=baseline_sse.checked_add(fit.baseline_sse).ok_or("aggregate exact SSE overflow")?;
            fitted_sse=fitted_sse.checked_add(fit.selected_sse).ok_or("aggregate exact SSE overflow")?;
            source_ss=source_ss.checked_add(fit.source_ss).ok_or("aggregate exact norm overflow")?;
            result.scales[g]=fit.scale;
            let bytes=fitter::pack(&fit.codes)?;result.packed[g*480..(g+1)*480].copy_from_slice(&bytes);
            for k in 0..960{
                let q=fit.codes[k];result.dense[g*960+k]=E4[q.unsigned_abs()as usize]|if q<0{128}else{0};
                assert_eq!(nibble(&result.packed,g*960+k),q,"independent packed signed-code inverse");
                original_error.add(value(values[k]),old_codes[k]as f32*base.scales[g]);
                fitted_error.add(value(values[k]),q as f32*fit.scale);
            }
            if fit.selected==0{assert_eq!(&result.packed[g*480..(g+1)*480],&base.packed[g*480..(g+1)*480]);assert_eq!(fit.scale.to_bits(),base.scales[g].to_bits());}
            if let Some(v)=sample.as_mut(){v.codes.extend_from_slice(&fit.codes);v.scales.push(fit.scale);}
        }
        if let Some(v)=sample{result.samples.insert(row,v);}
    }
    assert!(seen.into_iter().all(|x|x));assert_eq!(choices.iter().sum::<usize>(),base.scales.len());
    assert_eq!(result.samples.len(),SAMPLES);assert!(result.scales.iter().all(|s|s.is_normal()&&*s>0.0));
    assert!(fitted_sse<=baseline_sse);base.verify_source(t)?;
    println!("I4_WEIGHT_MSE_ENCODING {}",serde_json::json!({"projection":projection,"provided_bf16_values":rows*cols,"groups":base.scales.len(),
        "candidate_choice_counts":choices.to_vec(),"strictly_improved_groups":better,"allzero_original_groups":zero_groups,
        "exact_real_baseline_sse_grid_units":baseline_sse.to_string(),"exact_real_selected_sse_grid_units":fitted_sse.to_string(),
        "exact_source_ss_grid_units":source_ss.to_string(),"squared_grid_exponent":-76,
        "exact_real_baseline_relative_l2":((baseline_sse as f64)/(source_ss as f64).max(1.)).sqrt(),
        "exact_real_selected_relative_l2":((fitted_sse as f64)/(source_ss as f64).max(1.)).sqrt(),
        "scalar_RN32_baseline":original_error.json(),"scalar_RN32_selected":fitted_error.json(),
        "source":base.source,"global_unchanged":true,"original_pair_included":true,
        "scope":"exactreal q*stored_scale SSE guarantee only; scalar RN32 metrics are separate, neither is an exact output-error decomposition"}));
    Ok(result)
}

/// Both steps of this diagnostic share the incoming original-production state.
pub fn drift(got:&[u8],reference:&[u8],comparison:&str,round:usize,step:usize){
    assert_eq!(got.len(),reference.len());assert_eq!(got.len()%2,0);
    let(mut e,mut den,mut max,mut changed)=(0f64,0f64,0f64,0usize);
    for i in 0..got.len()/2{let a=bf(got,i);let b=bf(reference,i);let x=value(a);let y=value(b);
        assert!(x.is_finite()&&y.is_finite());e+=(x-y)*(x-y);den+=y*y;max=max.max((x-y).abs());changed+=usize::from(a!=b);}
    println!("I4_WEIGHT_MSE_DRIFT {}",serde_json::json!({"comparison":comparison,"round":round,"step":step,"bf16_values":got.len()/2,
        "changed":changed,"relative_l2":(e/den.max(1e-300)).sqrt(),"maxabs":max,"matched_incoming_inputs":true,
        "common_feedback":"original production residual","no_model_quality_claim":true}));
}

pub fn same_input(got:&[u8],other:&[u8],label:&str)->usize{assert_eq!(got,other,"same original BF16/activation input {label}");got.len()}
pub const EXPECTED_PREFIX_VALUES:usize=4*TOKENS*HIDDEN;

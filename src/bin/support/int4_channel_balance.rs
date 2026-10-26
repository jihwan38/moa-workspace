//! Parameter-free provided-gamma compensation; host diagnostic only.
use super::cpu;
use cpu::{Result,WeightCpu,SampleWeight,HIDDEN,EXPANDED,TOKENS,SAMPLES};
use furiosa_opt_std::prelude::bf16;
use std::collections::BTreeMap;
#[path="rational_a4_oracle.rs"] mod rational;

fn b16(x:f64)->u16 {bf16::from_f32(x as f32).to_bits()}
fn value(b:u16)->f64 {bf16::from_bits(b).to_f32()as f64}
fn normal_or_zero(x:f32)->bool {x==0.0||x.is_normal()}
fn bf(bytes:&[u8],i:usize)->u16 {u16::from_le_bytes(bytes[2*i..2*i+2].try_into().unwrap())}
fn f4(b:u8)->f32 {let v=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8==0{v}else{-v}}
fn e4(b:u8)->f32 {let e=(b>>3)&15;let m=b&7;assert!(!(e==15&&m==7));let v=if e==0{m as f32/512.0}else{(8+m)as f32*2f32.powi(e as i32-10)};if b&128==0{v}else{-v}}
fn code(p:&[u8],i:usize)->i8 {let b=((p[i/2]>>(4*(i%2)))&15)as i8;if b>=8{b-16}else{b}}
fn put(p:&mut[u8],d:&mut[u8],start:usize,q:&[i32]) {
    const E:[u8;8]=[0,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e];
    for(i,&v)in q.iter().enumerate(){assert!((-7..=7).contains(&v));let at=start+i;
        if at%2==0{p[at/2]=(v as u8)&15;}else{p[at/2]|=((v as u8)&15)<<4;}
        d[at]=E[v.unsigned_abs()as usize]|if v<0{128}else{0};assert_eq!(code(p,at),v as i8);}
}
#[derive(Default)]
struct Error {n:usize,ss:f64,ee:f64,max:f64,zero:usize}
impl Error {
    fn add(&mut self,x:f64,y:f64){assert!(x.is_finite()&&y.is_finite());self.n+=1;self.ss+=x*x;self.ee+=(y-x)*(y-x);self.max=self.max.max((x-y).abs());self.zero+=usize::from(y==0.0);}
    fn json(&self)->serde_json::Value {serde_json::json!({"values":self.n,"relative_l2":if self.ss==0.0{0.0}else{(self.ee/self.ss).sqrt()},"sse":self.ee,"reference_ss":self.ss,"maxabs":self.max,"reconstructed_zeros":self.zero})}
}
pub struct Balance {pub original:Vec<u8>,pub prime:Vec<u8>,pub scale:Vec<f64>,pub exponents:Vec<i32>}
impl Balance {
    pub fn load(t:&safetensors::SafeTensors<'_>)->Result<Self>{
        let name="model.language_model.layers.0.pre_feedforward_layernorm.weight";
        let v=t.tensor(name)?;assert_eq!(v.dtype(),safetensors::Dtype::BF16);assert_eq!(v.shape(),[HIDDEN]);
        let original=v.data().to_vec();let mut prime=Vec::with_capacity(original.len());let mut scale=Vec::new();let mut exponents=Vec::new();let(mut identity,mut negative)=(0,0);
        for k in 0..HIDDEN {let b=bf(&original,k);let e=(b>>7)&255;assert!(e!=255,"nonfinite gamma channel{k}");
            let exponent=if e==0{identity+=1;0}else{e as i32-127};let s=2f64.powi(exponent);let exact=value(b)/s;let bp=b16(exact);
            assert_eq!(value(bp),exact,"exact gammaPrime channel{k}");assert_eq!(b16(value(bp)*s),b,"gamma inverse channel{k}");
            assert!(e==0||normal_or_zero(bf16::from_bits(bp).to_f32()));negative+=usize::from(b&0x8000!=0);
            prime.extend(bp.to_le_bytes());scale.push(s);exponents.push(exponent);}
        println!("I4_BALANCE_GAMMA {}",serde_json::json!({"source":name,"dtype":"BF16","shape":[HIDDEN],"source_fnv":format!("{:016x}",cpu::fingerprint(&original)),"prime_fnv":format!("{:016x}",cpu::fingerprint(&prime)),"channels":HIDDEN,"identity_zero_subnormal":identity,"negative":negative,"min_exponent":exponents.iter().min(),"max_exponent":exponents.iter().max(),"rule":"normal nonzero:2^floor(log2(abs(gamma))); zero/subnormal:1; nonfinite:reject","gamma_exact_inverse":true}));
        Ok(Self{original,prime,scale,exponents})
    }
    pub fn source_immutable(&self,t:&safetensors::SafeTensors<'_>)->Result<()> {assert_eq!(t.tensor("model.language_model.layers.0.pre_feedforward_layernorm.weight")?.data(),self.original);Ok(())}
    pub fn prefix(&self,old:&[u8],new:&[u8],round:usize,step:usize)->Result<usize>{
        assert_eq!(old.len(),TOKENS*HIDDEN*2);assert_eq!(new.len(),old.len());
        let(mut failures,mut subnormal)=(0,0);
        for i in 0..TOKENS*HIDDEN{let a=bf(old,i);let b=bf(new,i);let x=value(a);let y=value(b);let s=self.scale[i%HIDDEN];
            let safe=normal_or_zero(x as f32)&&normal_or_zero(y as f32);subnormal+=usize::from(!safe);
            let ok=safe&&y==x/s&&x==y*s&&b16(y*s)==a&&b16(x/s)==b;
            if !ok{if failures<8{println!("I4_BALANCE_PREFIX_DIFF round={round} step={step} index={i} old={a:04x} new={b:04x} exponent={}",self.exponents[i%HIDDEN]);}failures+=1;}
        }
        println!("I4_BALANCE_PREFIX_PROOF round={round} step={step} values={} failures={failures} nonnormal_nonzero={subnormal} exact_F64_inverse_and_BF16_bits={}",TOKENS*HIDDEN,failures==0);
        if failures!=0{return Err("prefix compensation prerequisite failed; quantized drift must not be interpreted".into());}Ok(TOKENS*HIDDEN)
    }
    pub fn activation_metrics(&self,old:&[u8],new:&[u8],qold:&cpu::InputCpu,qnew:&cpu::InputCpu,round:usize,step:usize){
        let(mut a,mut b,mut restored)=(Error::default(),Error::default(),Error::default());
        for p in 0..TOKENS{for k in 0..HIDDEN{let i=p*HIDDEN+k;let g=cpu::input_group(p,k/960,HIDDEN);
            let ro=(qold.codes[i]as f32)*qold.scales[g];let rn=(qnew.codes[i]as f32)*qnew.scales[g];
            a.add(value(bf(old,i)),ro as f64);b.add(value(bf(new,i)),rn as f64);restored.add(value(bf(old,i)),rn as f64*self.scale[k]);}}
        println!("I4_BALANCE_ACTIVATION {}",serde_json::json!({"round":round,"step":step,"original_coordinates":a.json(),"balanced_coordinates":b.json(),"balanced_reconstruction_restored_coordinates":restored.json(),"metric":"RN32(code*scale), exact F64 power2 compensation, F64 SSE; not exact dot error decomposition"}));
    }
}
struct Supplied<'a>{packed:safetensors::tensor::TensorView<'a>,scales:safetensors::tensor::TensorView<'a>}
impl<'a> Supplied<'a>{
    fn new(t:&'a safetensors::SafeTensors<'a>,projection:&str)->Result<Self>{
        assert!(["up_proj","gate_proj"].contains(&projection));let p=format!("model.language_model.layers.0.mlp.{projection}");
        let packed=t.tensor(&format!("{p}.weight_packed"))?;let scales=t.tensor(&format!("{p}.weight_scale"))?;
        assert_eq!(packed.dtype(),safetensors::Dtype::U8);assert_eq!(packed.shape(),[EXPANDED,HIDDEN/2]);
        assert_eq!(scales.dtype(),safetensors::Dtype::F8_E4M3);assert_eq!(scales.shape(),[EXPANDED,HIDDEN/16]);Ok(Self{packed,scales})
    }
    fn at(&self,row:usize,k:usize)->u16{let i=row*HIDDEN+k;let q=(self.packed.data()[i/2]>>(4*(i%2)))&15;
        bf16::from_f32(f4(q)*e4(self.scales.data()[i/16])).to_bits()}
}
pub fn weights(t:&safetensors::SafeTensors<'_>,projection:&str,base:&WeightCpu,b:&Balance)->Result<WeightCpu>{
    let src=Supplied::new(t,projection)?;assert_eq!((base.rows,base.cols),(EXPANDED,HIDDEN));
    let(mut failures,mut zeros,mut normals)=(0,0,0);let(mut min,mut max)=(f64::INFINITY,0f64);
    // Full range and inverse pass precedes any changed weight quantization.
    for row in 0..EXPANDED{for k in 0..HIDDEN{let original=src.at(row,k);let x=value(original);let exact=x*b.scale[k];let v=b16(exact);let y=value(v);
        let safe=normal_or_zero(x as f32)&&normal_or_zero(y as f32)&&y.is_finite()&&y==exact&&y/b.scale[k]==x&&b16(y/b.scale[k])==original;
        if !safe{if failures<8{println!("I4_BALANCE_WEIGHT_DIFF projection={projection} row={row} k={k} old={original:04x} new={v:04x} exponent={}",b.exponents[k]);}failures+=1;}
        if x==0.0{zeros+=1;}else{normals+=1;}min=min.min(y.abs());max=max.max(y.abs());}}
    println!("I4_BALANCE_WEIGHT_PROOF projection={projection} values={} failures={failures} original_zeros={zeros} original_nonzeros={normals} transformed_abs_min={min:.12e} transformed_abs_max={max:.12e} all_value_BF16_inverse={}",EXPANDED*HIDDEN,failures==0);
    if failures!=0{return Err("provided weight compensation prerequisite failed; no quantized interpretation".into());}
    let mut result=WeightCpu{rows:base.rows,cols:base.cols,packed:vec![0;base.packed.len()],dense:vec![0;base.dense.len()],scales:vec![f32::NAN;base.scales.len()],samples:BTreeMap::new(),global:base.global,source:base.source.clone()};
    let selected:BTreeMap<usize,usize>=(0..SAMPLES).map(|i|(cpu::sample_row(i,EXPANDED),i)).collect();assert_eq!(selected.len(),SAMPLES);
    let(mut olderr,mut newerr,mut restored)=(Error::default(),Error::default(),Error::default());
    for row in 0..EXPANDED{let mut sample=if selected.contains_key(&row){Some(SampleWeight{codes:Vec::new(),scales:Vec::new()})}else{None};
        for part in 0..HIDDEN/960{let originals:Vec<u16>=(0..960).map(|k|src.at(row,part*960+k)).collect();
            let values:Vec<u16>=originals.iter().enumerate().map(|(k,&v)|b16(value(v)*b.scale[part*960+k])).collect();
            let(q,s)=rational::reference(&values);let g=cpu::weight_group(row,part,HIDDEN);result.scales[g]=s;put(&mut result.packed,&mut result.dense,g*960,&q);
            if let Some(v)=sample.as_mut(){v.codes.extend(q.iter().map(|&x|x as i8));v.scales.push(s);}
            for k in 0..960{let old=(code(&base.packed,g*960+k)as f32)*base.scales[g];let new=(q[k]as f32)*s;
                olderr.add(value(originals[k]),old as f64);newerr.add(value(values[k]),new as f64);restored.add(value(originals[k]),new as f64/b.scale[part*960+k]);}
        }if let Some(s)=sample{result.samples.insert(row,s);}}
    assert_eq!(result.samples.len(),SAMPLES);assert!(result.scales.iter().all(|x|x.is_normal()&&*x>0.0));
    println!("I4_BALANCE_WEIGHT_METRIC {}",serde_json::json!({"projection":projection,"original_coordinates":olderr.json(),"balanced_coordinates":newerr.json(),"balanced_reconstruction_restored_coordinates":restored.json(),"metric":"RN32(code*scale), exact F64 power2 inverse compensation, F64 SSE; not exact dot error decomposition","source":base.source,"global_unchanged":true}));
    Ok(result)
}
pub fn products(t:&safetensors::SafeTensors<'_>,b:&Balance,old:&[u8],new:&[u8],round:usize,step:usize)->Result<usize>{
    let mut checked=0;for name in["up_proj","gate_proj"]{let src=Supplied::new(t,name)?;let mut sums=0;
        for i in 0..SAMPLES{let row=cpu::sample_row(i,EXPANDED);let p=i;let(mut a,mut z)=(0f64,0f64);
            for k in 0..HIDDEN{let w=value(src.at(row,k));let wp=value(b16(w*b.scale[k]));let x=value(bf(old,p*HIDDEN+k));let xp=value(bf(new,p*HIDDEN+k));
                let o=x*w;let n=xp*wp;assert_eq!(o,n,"unquantized F64 product {name} token{p} row{row} k{k}");
                let fo=(x as f32)*(w as f32);let fn_=(xp as f32)*(wp as f32);
                assert!(normal_or_zero(fo)&&normal_or_zero(fn_),"sample product FTZ/overflow boundary");assert_eq!(fo.to_bits(),fn_.to_bits(),"sample RN32 product bits");
                a+=o;z+=n;checked+=1;
            }assert_eq!(a.to_bits(),z.to_bits(),"same-order unquantized F64 dot");sums+=1;}
        println!("I4_BALANCE_PRODUCT_PROOF round={round} step={step} projection={name} dots={sums} products={} exact_term_and_same_order_F64_sum=true normal_or_zero_F32_products=true",SAMPLES*HIDDEN);
    }Ok(checked)
}

/// All three arms receive the same incoming original-production residual at
/// BOTH steps. This local reporter does not inherit the old diverged-feedback label.
pub fn drift(got:&[u8],reference:&[u8],label:&str,round:usize,step:usize){
    assert_eq!(got.len(),reference.len());assert_eq!(got.len()%2,0);
    let(mut e,mut den,mut max,mut changed)=(0f64,0f64,0f64,0usize);
    for i in 0..got.len()/2{let a=bf(got,i);let b=bf(reference,i);let x=value(a);let y=value(b);
        assert!(x.is_finite()&&y.is_finite(),"nonfinite drift {label}");
        e+=(x-y).powi(2);den+=y*y;max=max.max((x-y).abs());changed+=usize::from(a!=b);}
    println!("I4_BALANCE_DRIFT {}",serde_json::json!({"comparison":label,"round":round,"step":step,"bf16_values":got.len()/2,"changed":changed,"relative_l2":(e/den.max(1e-300)).sqrt(),"maxabs":max,"matched_incoming_inputs":true,"common_feedback":"original production residual","no_model_quality_claim":true}));
}

//! Approximate native E4 LM: full provided vocabulary, intended-quantized oracle.
//! Original BF16 control errors/selections are reported, never called bit-exact.
#[path="support/npu_profile.rs"] mod npu_profile;
use furiosa_opt_gemma4::{axes::{B,H,W},Chip,EPS,LOGIT_SOFTCAP,
 host::{load,sampling::{sample,SamplingConfig}},ops_b16_lm_storegroup as control,
 ops_b16_lm_fp8 as candidate,ops_query_b16_native_fp8 as quant,ops_norm_batch};
use furiosa_opt_std::prelude::*;
use rand::{SeedableRng,rngs::StdRng};
use std::{collections::HashSet,time::Instant};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn word(mut x:u64)->u64 {
    x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)
}
fn f32_at(raw:&[u8],i:usize)->f32 {
    f32::from_bits((u16::from_le_bytes([raw[2*i],raw[2*i+1]]) as u32)<<16)
}
fn bits_at(raw:&[u8],i:usize)->u16 {u16::from_le_bytes([raw[2*i],raw[2*i+1]])}
async fn read<E:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,E>)->Vec<u8> {
    x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()
}
fn fixture(case:usize,embedding:&[u8])->Vec<bf16> {
    (0..B::SIZE*H::SIZE).map(|i|{
        let r=i/H::SIZE;let h=i%H::SIZE;
        let u=((word((i+17+case*65537)as u64)>>40)as f32+0.37)/16777216.0;
        let v=if case==0 && r==0 {0.0} else if case==0 && r==1 {(u-0.5)*1.0e-17}
        else if case==2 {
            let n=(r*15331+107)%W::SIZE;
            f32_at(embedding,n*H::SIZE+h)*[0.71,-1.13,2.07,0.39][r%4]+(u-0.5)*0.017
        } else {(u-0.5)*[0.73,-1.37,2.19,-3.41][r%4]+(r as f32-7.1)*0.031};
        bf16::from_f32(v)
    }).collect()
}

fn f32_word(raw:&[u8],i:usize)->f32 {f32::from_le_bytes(raw[4*i..4*i+4].try_into().unwrap())}
// Independent scalar nearest-even E4 encoder, valid for finite row-normalized values.
fn e4_code(v:f32)->u8 {
 let sign=((v.to_bits()>>24)&128)as u8;let a=v.abs();
 assert!(a.is_finite()&&a<=448.0);
 let code=if a<0.015625 {(a*512.0).round_ties_even()as u32}
 else {let b=a.to_bits();((b+0x7ffff+((b>>20)&1))>>20)-960};
 sign|code.min(126)as u8
}
fn e4_value(code:u8)->f64 {
 let exponent=(code>>3)&15;let mantissa=code&7;
 let v=if exponent==0 {mantissa as f64/512.0}
 else {(1.0+mantissa as f64/8.0)*2f64.powi(exponent as i32-7)};
 if code&128!=0 {-v}else{v}
}
fn scale_for(raw:&[u8])->f32 {
 let max=raw.chunks_exact(2).map(|b|{
  let x=f32::from_bits((u16::from_le_bytes([b[0],b[1]])as u32)<<16);
  assert!(x.is_finite());x.abs()
 }).fold(1.925929944387236e-34f32,f32::max);
 f32::from_bits(max.to_bits()&0x7f80_0000)*0.0078125
}
fn selftest_encoding() {
 let mut count=0;
 for bits in 0..=65535u16 {for scale in [0.0078125f32,1.0,128.0] {
  let x=bf16::from_bits(bits).to_f32()/scale;
  if x.is_finite()&&x.abs()<=256.0 {
   assert_eq!((e4_value(e4_code(x))as f32).to_bits(),f8e4m3::from_f32(x).to_f32().to_bits(),"E4 CPU encoder bits={bits} scale={scale}");count+=1;
  }
 }}
 println!("LM_CPU_ENCODER checks={count} nearest_even_and_signed_zero=true");
}
fn check_quantized(original:&[u8],q:&[u8],scales:&[u8],rows:usize,stride:usize,label:&str) {
 assert_eq!(original.len(),rows*H::SIZE*2);assert_eq!(q.len(),rows*H::SIZE);
 assert_eq!(scales.len(),rows*stride*4);let start=Instant::now();
 for row in 0..rows {
  let source=&original[row*H::SIZE*2..(row+1)*H::SIZE*2];let s=scale_for(source);
  assert_eq!(f32_word(scales,row*stride).to_bits(),s.to_bits(),"scale {label} row={row}");
  for h in 0..H::SIZE {
   let expected=e4_code(f32_at(source,h)/s);let got=q[row*H::SIZE+h];
   assert_eq!(got,expected,"quant {label} row={row} h={h}");
  }
 }
 println!("LM_QUANT_CPU label={label} codes={} scales={rows} cpu_ms={:.3}",q.len(),start.elapsed().as_secs_f64()*1e3);
}
fn cpu_samples(input:&[u8],norm_input:&[bf16],norm_weight:&[u8],weight:&[u8],
 qx:&[u8],xs:&[u8],qw:&[u8],ws:&[u8],reference:&[u8],actual:&[u8],case:usize,permutation:usize)->usize {
 let mut failures=0;let mut differences=[0usize;2];let mut max=[0f32;2];
 for r in 0..B::SIZE {
  let row=&norm_input[r*H::SIZE..(r+1)*H::SIZE];
  let rms=(row.iter().map(|v|{let x=v.to_f32()as f64;x*x}).sum::<f64>()/H::SIZE as f64+EPS as f64).sqrt();
  for j in 0..8 {
   let h=(word((r*8+j+case*1009)as u64)as usize)%H::SIZE;
   let expect=bf16::from_f32((row[h].to_f32()as f64/rms*f32_at(norm_weight,h)as f64)as f32).to_f32();
   assert!((f32_at(input,r*H::SIZE+h)-expect).abs()<=0.016+0.01*expect.abs(),"RMS sample");
  }
  for j in 0..32 {
   let n=match j {0=>0,1=>W::SIZE-1,2=>8191,3=>8192,4=>16383,5=>16384,_=>(word((r*32+j+case*337)as u64)as usize)%W::SIZE};
   let olddot=(0..H::SIZE).map(|h|f32_at(input,r*H::SIZE+h)as f64*f32_at(weight,n*H::SIZE+h)as f64).sum::<f64>();
   let qdot=(0..H::SIZE).map(|h|e4_value(qx[r*H::SIZE+h])*e4_value(qw[n*H::SIZE+h])).sum::<f64>();
   let newdot=(qdot as f32*f32_word(xs,r*8))*f32_word(ws,n);
   for (v,(raw,dot)) in [(reference,olddot as f32),(actual,newdot)].into_iter().enumerate() {
    let rounded=bf16::from_f32(dot).to_f32();
    let expected=bf16::from_f32((rounded/LOGIT_SOFTCAP).tanh()*LOGIT_SOFTCAP).to_f32();
    let got=f32_at(raw,r*W::SIZE+n);let delta=(got-expected).abs();max[v]=max[v].max(delta);
    differences[v]+=usize::from(got.to_bits()!=expected.to_bits());
    // F64 is an independent numerical oracle, not an F32 association identity.
    if delta>0.0002+0.01*expected.abs() {
     if failures<8 {println!("LM_F64_DIFF case={case} permutation={permutation} variant={v} row={r} vocab={n} actual={got} expected={expected} rawdot64={qdot}");}failures+=1;
    }
   }
  }
 }
 println!("LM_F64 case={case} permutation={permutation} samples_per_variant=512 tolerance_failures={failures} rounded_differences={differences:?} max_abs={max:?} scope=independent_original_and_quantized_algorithm");failures
}
#[cfg(backend="npu")]
fn memory(d:&Device,label:&str)->Result<usize>{let m=d.memory()?;println!("LM_MEMORY stage={label} available={} largest={} used={}",m.available,m.largest,m.capacity-m.available);Ok(m.available)}
#[cfg(not(backend="npu"))]
fn memory(_d:&Device,_label:&str)->Result<usize>{Err("NPU only".into())}
fn compare(reference:&[u8],actual:&[u8],case:usize,permutation:usize,all_bans:&HashSet<usize>)->(usize,usize) {
    assert_eq!(reference.len(),B::SIZE*W::SIZE*2);assert_eq!(actual.len(),reference.len());
    let mut differences=0;let mut max_abs=0f32;let mut e2=0f64;let mut ref2=0f64;let mut selected_differences=0;
    for r in 0..B::SIZE {
        let a:Vec<f32>=(0..W::SIZE).map(|n|f32_at(actual,r*W::SIZE+n)).collect();
        let b:Vec<f32>=(0..W::SIZE).map(|n|f32_at(reference,r*W::SIZE+n)).collect();
        let mut row_differences=0;
        for n in 0..W::SIZE {
            assert!(a[n].is_finite()&&b[n].is_finite(),"nonfinite case={case} row={r} vocab={n}");
            if bits_at(actual,r*W::SIZE+n)!=bits_at(reference,r*W::SIZE+n) {
                if differences<16 {println!("LM_BITS_DIFF case={case} permutation={permutation} row={r} vocab={n} candidate=0x{:04x} control=0x{:04x} delta={}",bits_at(actual,r*W::SIZE+n),bits_at(reference,r*W::SIZE+n),a[n]-b[n]);}
                differences+=1;row_differences+=1;
            }
            let err=(a[n]as f64-b[n]as f64).abs();max_abs=max_abs.max(err as f32);e2+=err*err;ref2+=(b[n]as f64).powi(2);
        }
        let empty=HashSet::new();let default:HashSet<usize>=[258882,258883].into_iter().collect();
        let mut rng=StdRng::seed_from_u64(17);
        let winner=sample(&b,&SamplingConfig::DEFAULT,&empty,&mut rng);
        let mut arbitrary:HashSet<usize>=(0..37).map(|i|(word((i+r*67)as u64)as usize)%W::SIZE).collect();
        arbitrary.extend([winner,W::SIZE+79,usize::MAX]);
        for (policy,ban) in [&empty,&default,&arbitrary,all_bans].into_iter().enumerate() {
            let x=sample(&a,&SamplingConfig::DEFAULT,ban,&mut rng);let y=sample(&b,&SamplingConfig::DEFAULT,ban,&mut rng);
            if x!=y {selected_differences+=1;println!("LM_SELECTION_DIFF case={case} permutation={permutation} row={r} policy={policy} candidate={x} control={y}");}
        }
        println!("LM_ROW case={case} permutation={permutation} row={r} raw_bf16_differences={row_differences}");
    }
    println!("LM_ALL_VOCAB case={case} permutation={permutation} values={} raw_bf16_differences={differences} max_abs={max_abs} relative_l2={} selection_checks=64 selection_differences={selected_differences}",B::SIZE*W::SIZE,(e2/ref2.max(1e-300)).sqrt());
    (differences,selected_differences)
}

#[tokio::main]
async fn main()->Result<()> {
 selftest_encoding();
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone())?;
 let mut d=Device::new(candidate::logits_quantized.topology())?;
 let file=std::fs::File::open(load::model_dir()?.join("model.safetensors"))?;
 let mmap=unsafe{memmap2::Mmap::map(&file)?};let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
 let table=tensors.tensor("model.language_model.embed_tokens.weight")?;
 assert_eq!(table.shape(),[W::SIZE,H::SIZE]);assert_eq!(table.dtype(),safetensors::Dtype::BF16);
 let norm=tensors.tensor("model.language_model.norm.weight")?;
 assert_eq!(norm.shape(),[H::SIZE]);assert_eq!(norm.dtype(),safetensors::Dtype::BF16);
 let weight=HostTensor::<bf16,m![W,H]>::from_safetensors(&table)?.to_hbm(&mut d.pdma).await?;
 let norm_weight=HostTensor::<bf16,m![H]>::from_safetensors(&norm)?.to_hbm(&mut d.pdma).await?;
 let before=memory(&d,"original_retained")?;let start=Instant::now();
 let (qweight,wscale)=launch(candidate::quantize_embedding,(&mut d,&weight)).await?;
 let startup_ms=start.elapsed().as_secs_f64()*1e3;let after=memory(&d,"with_shadow")?;
 println!("LM_STARTUP ms={startup_ms} payload_bytes=1007681536 allocator_delta={} original_BF16_retained=true standalone_capacity_only=true",before-after);
 assert!(before-after<=1_100_000_000,"shadow persistent allocation cap");
 let qw=qweight.to_host::<m![W,H]>(&mut d.pdma).await?.into_inner().into_buf();
 let ws=wscale.to_host::<m![W]>(&mut d.pdma).await?.into_inner().into_buf();
 check_quantized(table.data(),&qw,&ws,W::SIZE,1,"embedding");
 println!("LM_PROVIDED embedding_bytes={} norm_bytes={} fixture=non_dyadic_then_original_device_finalRMS",table.data().len(),norm.data().len());
 let all_bans:HashSet<usize>=(0..W::SIZE).collect();let mut total=[0usize;3];let mut values=0usize;
 for case in 0..3 {
  let raw=fixture(case,table.data());let mut first:Option<(Vec<u8>,Vec<u8>)>=None;
  for permutation in 0..2 {
   let order:Vec<usize>=(0..B::SIZE).map(|r|if permutation==0 {r}else{(r*5+3)%B::SIZE}).collect();
   let host:Vec<bf16>=order.iter().flat_map(|&r|raw[r*H::SIZE..(r+1)*H::SIZE].iter().copied()).collect();
   let input=HostTensor::<bf16,m![B,H]>::from_vec(host.clone()).to_hbm(&mut d.pdma).await?;
   let mut normalized=HostTensor::<bf16,m![B,H]>::zero().to_hbm(&mut d.pdma).await?;
   launch(ops_norm_batch::rmsnorm_b16,(&mut d,&input,&norm_weight,&mut normalized)).await?;
   let cpu_input=read(&mut d,&normalized).await;
   let mut qinput=HostTensor::<f8e4m3,m![B,H]>::from_vec((0..B::SIZE*H::SIZE).map(|_|f8e4m3::from_f32(f32::NAN))).to_hbm(&mut d.pdma).await?;
   let mut xscale=HostTensor::<f32,m![B,1#8]>::from_vec([f32::NAN;B::SIZE*8]).to_hbm(&mut d.pdma).await?;
   let poison:Vec<u8>=(0..B::SIZE*W::SIZE).flat_map(|i|(if i%2==0{0x7fc1u16}else{0xffc3u16}).to_le_bytes()).collect();
   let mut output=HostTensor::<bf16,m![B,W]>::from_buf(poison.clone()).to_hbm(&mut d.pdma).await?;
   let mut reference=HostTensor::<bf16,m![B,W]>::from_buf(poison).to_hbm(&mut d.pdma).await?;
   launch(quant::quantize_input,(&mut d,&normalized,&mut qinput,&mut xscale)).await?;
   launch(candidate::logits_quantized,(&mut d,&qinput,&xscale,&qweight,&wscale,&mut output)).await?;
   let actual=read(&mut d,&output).await;
   let qx=qinput.to_host::<m![B,H]>(&mut d.pdma).await?.into_inner().into_buf();
   let xs=xscale.to_host::<m![B,1#8]>(&mut d.pdma).await?.into_inner().into_buf();
   check_quantized(&cpu_input,&qx,&xs,B::SIZE,8,"input");
   launch(control::logits_batch,(&mut d,&normalized,&weight,&mut reference)).await?;
   let expected=read(&mut d,&reference).await;
   let (diff,sel)=compare(&expected,&actual,case,permutation,&all_bans);total[0]+=diff;total[1]+=sel;
   total[2]+=cpu_samples(&cpu_input,&host,norm.data(),table.data(),&qx,&xs,&qw,&ws,&expected,&actual,case,permutation);
   values+=B::SIZE*W::SIZE;
   if let Some((orig_a,orig_b))=&first {
    for (row,&src) in order.iter().enumerate() {
     let r=row*W::SIZE*2..(row+1)*W::SIZE*2;let p=src*W::SIZE*2..(src+1)*W::SIZE*2;
     assert_eq!(&actual[r.clone()],&orig_a[p.clone()],"candidate row permutation case={case} row={row}");
     assert_eq!(&expected[r],&orig_b[p],"control row permutation case={case} row={row}");
    }
    println!("LM_PERMUTATION case={case} values={} candidate_and_control_exact=true",B::SIZE*W::SIZE);
   } else {first=Some((actual,expected));}
   assert_eq!(read(&mut d,&normalized).await,cpu_input,"normalized input immutable");
   let mut times=[Vec::new(),Vec::new()];
   for variant in [0,1,1,0] {
    let start=Instant::now();for _ in 0..4 {
     if variant==0 {launch(control::logits_batch,(&mut d,&normalized,&weight,&mut reference)).await?;}
     else {launch(quant::quantize_input,(&mut d,&normalized,&mut qinput,&mut xscale)).await?;
      launch(candidate::logits_quantized,(&mut d,&qinput,&xscale,&qweight,&wscale,&mut output)).await?;}
    }
    times[variant].push(start.elapsed().as_secs_f64()*1e6/4.0);
   }
   println!("LM_ABBA case={case} permutation={permutation} control_us={:?} candidate_us={:?} scope=completed_inputquant_plus_fullLM_canonicaloutput_no_startup_or_D2H",times[0],times[1]);
   if case==1&&permutation==0&&std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
    for variant in 0..2 {
     println!("PROFILE_VARIANT {variant}");collector.enable(true);
     if variant==0 {launch(control::logits_batch,(&mut d,&normalized,&weight,&mut reference)).await?;}
     else {launch(quant::quantize_input,(&mut d,&normalized,&mut qinput,&mut xscale)).await?;
      launch(candidate::logits_quantized,(&mut d,&qinput,&xscale,&qweight,&wscale,&mut output)).await?;}
     collector.enable(false);
    }
   }
  }
 }
 assert_eq!(read(&mut d,&weight).await,table.data(),"original embedding immutable");
 assert_eq!(qweight.to_host::<m![W,H]>(&mut d.pdma).await?.into_inner().into_buf(),qw,"shadow immutable");
 assert_eq!(wscale.to_host::<m![W]>(&mut d.pdma).await?.into_inner().into_buf(),ws,"scales immutable");
 println!("LM_APPROX_RESULT values={values} raw_bf16_differences={} selection_checks=384 selection_differences={} intended_f64_tolerance_failures={} no_host_or_model_change=true",total[0],total[1],total[2]);
 assert_eq!(total[2],0,"intended algorithm F64 oracle");
 println!("PASS intended quantization/algorithm, all384 approximation selections reported; original LM equivalence and task accuracy NOT claimed");Ok(())
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn cpu_encoder_matches_sdk_on_finite_bf16_power2_domain() {selftest_encoding();}
}

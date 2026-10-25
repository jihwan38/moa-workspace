//! Original scalar versus batched audio on provided weights, with independent CPU checks.
use furiosa_opt_gemma4::{Chip,EPS,axes::*,host::{load,runtime},ops_audio,ops_audio_batch as batch};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn frame(row:usize,seed:usize)->Vec<bf16>{
 (0..Aa::SIZE).map(|k|bf16::from_f32(match row {
  0=>0.0,
  1=>if k%2==0{1e-20}else{-1e-20},
  2=>0.375,
  3=>if k%2==0{1e3}else{-1e3},
  _=>{let v=((row*733+k*37+seed*97)%1023)as f32/512.0-511.0/512.0;v*[0.125,0.5,1.0,4.0][row%4]},
 })).collect()
}
async fn read<Mt:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mt>)->Result<Vec<bf16>>{
 let h:HostTensor<bf16,Mt>=x.to_host(&mut d.pdma).await?;Ok(h.into_vec())
}
fn exact(label:&str,a:&[bf16],b:&[bf16]){
 assert_eq!(a.len(),b.len());
 let diff=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
 println!("AUDIO_EXACT {label} values={} differences={diff}",a.len());assert_eq!(diff,0,"{label}");
}
async fn scalar_frame(d:&mut Device,w:&HbmTensor<bf16,Chip,m![H,Aa]>,frame:&[bf16])
 ->Result<HbmTensor<bf16,Chip,m![H]>>{
 // Exactly the allocation/upload/launch sequence of runtime::encode_audio_frame.
 let input=HostTensor::<bf16,m![Aa]>::from_vec(frame.to_vec()).to_hbm(&mut d.pdma).await?;
 let mut out=HostTensor::<bf16,m![H]>::zero().to_hbm(&mut d.pdma).await?;
 launch(ops_audio::audio_project_frame,(d,&input,w,&mut out)).await?;Ok(out)
}
fn cpu_checks(label:&str,frames:&[Vec<bf16>],norm:&[bf16],out:&[bf16],weight:&[bf16]){
 let mut nerr=0.0;let mut nenergy=0.0;let mut ndiff=0;
 for (r,x) in frames.iter().enumerate(){
  let rms=(x.iter().map(|v|(v.to_f32()as f64).powi(2)).sum::<f64>()/Aa::SIZE as f64+EPS as f64).sqrt();
  for k in 0..Aa::SIZE {let expected=x[k].to_f32()as f64/rms;let got=norm[r*Aa::SIZE+k].to_f32()as f64;
   assert!(got.is_finite());nerr+=(got-expected).powi(2);nenergy+=expected*expected;
   ndiff+=usize::from(norm[r*Aa::SIZE+k].to_bits()!=bf16::from_f32(expected as f32).to_bits());
  }
 }
 let mut err=0.0;let mut energy=0.0;let mut diff=0;let mut count=0;let mut max_abs=0.0f64;
 for r in 0..frames.len(){for h in [0,1,119,120,1919,1920,3838,3839]{
  let expected=(0..Aa::SIZE).map(|k|norm[r*Aa::SIZE+k].to_f32()as f64*weight[h*Aa::SIZE+k].to_f32()as f64).sum::<f64>();
  let got=out[r*H::SIZE+h].to_f32()as f64;assert!(got.is_finite());
  err+=(got-expected).powi(2);energy+=expected*expected;max_abs=max_abs.max((got-expected).abs());
  diff+=usize::from(out[r*H::SIZE+h].to_bits()!=bf16::from_f32(expected as f32).to_bits());count+=1;
 }}
 let norm_nrmse=(nerr/nenergy.max(1e-100)).sqrt();let dot_nrmse=(err/energy.max(1e-100)).sqrt();
 println!("AUDIO_CPU {label} norm_values={} norm_bf16_vs_f64_differences={ndiff} norm_nrmse={norm_nrmse:.9} dot_values={count} dot_bf16_vs_f64_differences={diff} dot_nrmse={dot_nrmse:.9} dot_max_abs={max_abs:.9}",norm.len());
 // F64 does not reproduce the baseline F32 reduction association. Scalar output
 // parity is exact; these independent checks bound RMS/dot formula and indexing.
 assert!(norm_nrmse<0.004);assert!(dot_nrmse<0.004);
}

async fn case_b16(d:&mut Device,w:&HbmTensor<bf16,Chip,m![H,Aa]>,weights:&[bf16],seed:usize)->Result<()> {
 let frames:Vec<_>=(0..B::SIZE).map(|r|frame(r,seed)).collect();
 let mut scalar=Vec::new();
 for order_id in 0..2 {
  let order:Vec<_>=(0..B::SIZE).map(|i|if order_id==0{i}else{(i*37+11)%B::SIZE}).collect();
  let ordered:Vec<_>=order.iter().map(|&i|frames[i].clone()).collect();
  let flat:Vec<_>=ordered.iter().flatten().copied().collect();
  let input=HostTensor::<bf16,m![B,Aa]>::from_vec(flat.clone()).to_hbm(&mut d.pdma).await?;
  let mut output=HostTensor::<bf16,m![B,H]>::from_vec(vec![bf16::from_f32(f32::NAN);B::SIZE*H::SIZE]).to_hbm(&mut d.pdma).await?;
  // Candidate first; outputs and norm outputs are independently poisoned.
  launch(batch::audio_project_b16_into,(d,&input,w,&mut output)).await?;
  let got=read(d,&output).await?;
  let mut normalized=HostTensor::<bf16,m![B,Aa]>::from_vec(vec![bf16::from_f32(f32::NAN);B::SIZE*Aa::SIZE]).to_hbm(&mut d.pdma).await?;
  launch(batch::normalize_b16,(d,&input,&mut normalized)).await?;
  let norm=read(d,&normalized).await?;
  exact("immutable_input",&flat,&read(d,&input).await?);
  if order_id==0 {for x in &frames {let out=scalar_frame(d,w,x).await?;scalar.push(read(d,&out).await?);}}
  let expected:Vec<_>=order.iter().flat_map(|&i|scalar[i].iter().copied()).collect();
  exact(&format!("B seed={seed} order={order_id} scalar_all"),&expected,&got);
  cpu_checks(&format!("B seed={seed} order={order_id}"),&ordered,&norm,&got,weights);
  let slices:Vec<_>=ordered.iter().map(Vec::as_slice).collect();
  let first=runtime::encode_audio_frames_b16(d,w,&slices).await?;
  let take=if B::SIZE==16{7}else{127};
  let second=runtime::encode_audio_frames_b16(d,w,&slices[..take]).await?;
  exact("returned_first_retains_ownership",&got,&read(d,&first).await?);
  let second=read(d,&second).await?;
  exact("partial_live_rows",&got[..take*H::SIZE],&second[..take*H::SIZE]);
  assert!(second[take*H::SIZE..].iter().all(|v|v.to_f32()==0.0),"inactive rows not zero");
  println!("AUDIO_PADDING B live={take} inactive_values={} zero=true",(B::SIZE-take)*H::SIZE);
 }
 if seed==31 {
  let slices:Vec<_>=frames.iter().map(Vec::as_slice).collect();
  for variant in [1,0,0,1] {
   let now=std::time::Instant::now();
   for _ in 0..3 {
    if variant==0 {for x in &frames {let _out=scalar_frame(d,w,x).await?;}}
    else {let _out=runtime::encode_audio_frames_b16(d,w,&slices).await?;}
   }
   println!("AUDIO_COMPLETE B variant={variant} frames={} warm_us={:.3} scope=host_pack_H2D_output_allocation_all_kernel_completions_no_read_or_scatter",B::SIZE,now.elapsed().as_secs_f64()*1e6/3.0);
  }
 }
 Ok(())
}

async fn case_p128(d:&mut Device,w:&HbmTensor<bf16,Chip,m![H,Aa]>,weights:&[bf16],seed:usize)->Result<()> {
 let frames:Vec<_>=(0..P128::SIZE).map(|r|frame(r,seed)).collect();
 let mut scalar=Vec::new();
 for order_id in 0..2 {
  let order:Vec<_>=(0..P128::SIZE).map(|i|if order_id==0{i}else{(i*37+11)%P128::SIZE}).collect();
  let ordered:Vec<_>=order.iter().map(|&i|frames[i].clone()).collect();
  let flat:Vec<_>=ordered.iter().flatten().copied().collect();
  let input=HostTensor::<bf16,m![P128,Aa]>::from_vec(flat.clone()).to_hbm(&mut d.pdma).await?;
  let mut output=HostTensor::<bf16,m![P128,H]>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*H::SIZE]).to_hbm(&mut d.pdma).await?;
  // Candidate first; outputs and norm outputs are independently poisoned.
  launch(batch::audio_project_p128_into,(d,&input,w,&mut output)).await?;
  let got=read(d,&output).await?;
  let mut normalized=HostTensor::<bf16,m![P128,Aa]>::from_vec(vec![bf16::from_f32(f32::NAN);P128::SIZE*Aa::SIZE]).to_hbm(&mut d.pdma).await?;
  launch(batch::normalize_p128,(d,&input,&mut normalized)).await?;
  let norm=read(d,&normalized).await?;
  exact("immutable_input",&flat,&read(d,&input).await?);
  if order_id==0 {for x in &frames {let out=scalar_frame(d,w,x).await?;scalar.push(read(d,&out).await?);}}
  let expected:Vec<_>=order.iter().flat_map(|&i|scalar[i].iter().copied()).collect();
  exact(&format!("P128 seed={seed} order={order_id} scalar_all"),&expected,&got);
  cpu_checks(&format!("P128 seed={seed} order={order_id}"),&ordered,&norm,&got,weights);
  let slices:Vec<_>=ordered.iter().map(Vec::as_slice).collect();
  let first=runtime::encode_audio_frames_p128(d,w,&slices).await?;
  let take=if P128::SIZE==16{7}else{127};
  let second=runtime::encode_audio_frames_p128(d,w,&slices[..take]).await?;
  exact("returned_first_retains_ownership",&got,&read(d,&first).await?);
  let second=read(d,&second).await?;
  exact("partial_live_rows",&got[..take*H::SIZE],&second[..take*H::SIZE]);
  assert!(second[take*H::SIZE..].iter().all(|v|v.to_f32()==0.0),"inactive rows not zero");
  println!("AUDIO_PADDING P128 live={take} inactive_values={} zero=true",(P128::SIZE-take)*H::SIZE);
 }
 if seed==31 {
  let slices:Vec<_>=frames.iter().map(Vec::as_slice).collect();
  for variant in [1,0,0,1] {
   let now=std::time::Instant::now();
   for _ in 0..3 {
    if variant==0 {for x in &frames {let _out=scalar_frame(d,w,x).await?;}}
    else {let _out=runtime::encode_audio_frames_p128(d,w,&slices).await?;}
   }
   println!("AUDIO_COMPLETE P128 variant={variant} frames={} warm_us={:.3} scope=host_pack_H2D_output_allocation_all_kernel_completions_no_read_or_scatter",P128::SIZE,now.elapsed().as_secs_f64()*1e6/3.0);
  }
 }
 Ok(())
}

#[tokio::main]
async fn main()->Result<()> {
 let path=load::model_dir()?.join("model.safetensors");
 let file=std::fs::File::open(path)?;let mmap=unsafe{memmap2::Mmap::map(&file)?};
 let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
 let weight=tensors.tensor("model.embed_audio.embedding_projection.weight")?;
 let host=HostTensor::<bf16,m![H,Aa]>::from_safetensors(&weight).map_err(|e|format!("audio weight: {e}"))?;
 let weights=host.into_vec();assert_eq!(weights.len(),H::SIZE*Aa::SIZE);
 let mut d=Device::new(batch::audio_project_b16.topology())?;
 let w=HostTensor::<bf16,m![H,Aa]>::from_vec(weights.clone()).to_hbm(&mut d.pdma).await?;
 for seed in [0,31,79] {case_b16(&mut d,&w,&weights,seed).await?;case_p128(&mut d,&w,&weights,seed).await?;}
 exact("immutable_provided_weight",&weights,&read(&mut d,&w).await?);
 println!("PASS batched audio original_scalar_bit_exact all_rows_permutations_zero_padding_returned_ownership_cpu_checks");Ok(())
}

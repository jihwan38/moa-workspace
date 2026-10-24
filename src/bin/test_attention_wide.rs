//! Deterministic NPU/CPU attention comparisons, page/ring boundaries, and future-key invariance.
use furiosa_opt_gemma4::{Chip, axes::*, ops_sliding_wide as wide, ops_sliding_prompt_clusters as sliding};
use furiosa_opt_std::prelude::*;
use sliding::Bk;
use wide::Pk128;
use std::time::Instant;

fn bf(x: f32) -> f32 {
    f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1))) & 0xffff0000)
}
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            bf((((x >> 40) as f32) / 16777216.0 * 2.0 - 1.0) * scale)
        })
        .collect()
}
async fn bf_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<bf16, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    let bytes: Vec<u8> = x
        .iter()
        .flat_map(|x| ((bf(*x).to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    HostTensor::<bf16, Mp>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn f_upload<Mp: M>(d: &mut Device, x: &[f32]) -> HbmTensor<f32, Chip, Mp> {
    assert_eq!(x.len(), Mp::SIZE);
    HostTensor::<f32, Mp>::from_buf(x.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn zero<D: ScalarBytes + MaterializableScalar, Mp: M>(
    d: &mut Device,
) -> HbmTensor<D, Chip, Mp> {
    HostTensor::<D, Mp>::from_buf(vec![0u8; Mp::SIZE * D::BITS / 8])
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) -> Vec<f32> {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    h.into_vec().into_iter().map(bf16::to_f32).collect()
}

fn old_position(start:usize,slot:usize)->Option<usize>{if start==0{return None;}let last=start-1;let p=last as isize-((last+1024-slot)%1024) as isize;(p>=0).then_some(p as usize)}
fn masks(start:usize,valid:usize,rows:usize,keys:usize)->(Vec<f32>,Vec<f32>){
 let old=(0..rows).flat_map(|r|(0..1024).map(move|t|if r<valid&&old_position(start,t).is_some_and(|p|p+1024>start+r){0.}else{-1e30})).collect();
 let new=(0..rows).flat_map(|r|(0..keys).map(move|t|if r<valid&&t<=r&&t<valid{0.}else{-1e30})).collect();(old,new)
}
fn metric(label:&str,a:&[f32],b:&[f32],bound:f64){assert_eq!(a.len(),b.len());let mut error=0f64;let mut norm=0f64;let mut max=0f32;let mut diffs=0usize;for(&x,&y)in a.iter().zip(b){assert!(x.is_finite()&&y.is_finite());error+=(x as f64-y as f64).powi(2);norm+=(y as f64).powi(2);max=max.max((x-y).abs());diffs+=usize::from(x.to_bits()!=y.to_bits());}let nrms=(error/norm.max(1e-30)).sqrt();println!("{label} n={} nrms={nrms:.8} max_abs={max} bit_differences={diffs}",a.len());assert!(nrms<bound,"{label} exceeds numerical gate");}
async fn native(d:&mut Device,q:&[f32],ok:&[f32],ov:&[f32],nk:&[f32],nv:&[f32],start:usize,valid:usize)->(Vec<f32>,f64){
 let q=bf_upload::<m![P128,Ns,Gs,Ds]>(d,q).await;let ok=bf_upload::<m![Ts,Ns,Ds]>(d,ok).await;let ov=bf_upload::<m![Ts,Ns,Ds]>(d,ov).await;let nk=bf_upload::<m![P128,Ns,Ds]>(d,nk).await;let nv=bf_upload::<m![P128,Ns,Ds]>(d,nv).await;
 let(om,nm)=masks(start,valid,128,128);let om=f_upload::<m![P128,Ts]>(d,&om).await;let nm=f_upload::<m![P128,Pk128]>(d,&nm).await;let mut out=zero::<bf16,m![P128,Ns,Gs,Ds]>(d).await;
 launch(wide::sliding_attention_p128,(d,&q,&ok,&ov,&nk,&nv,&om,&nm,&mut out)).await.unwrap();let begin=Instant::now();launch(wide::sliding_attention_p128,(d,&q,&ok,&ov,&nk,&nv,&om,&nm,&mut out)).await.unwrap();let us=begin.elapsed().as_secs_f64()*1e6;(read(d,&out).await,us)
}
async fn chronological(d:&mut Device,q:&[f32],ok:&[f32],ov:&[f32],nk:&[f32],nv:&[f32],start:usize,valid:usize)->(Vec<f32>,f64){
 let mut ringk=ok.to_vec();let mut ringv=ov.to_vec();let mut result=Vec::new();let mut total=0.;
 for row in (0..valid).step_by(16){let count=(valid-row).min(16);let qr=bf_upload::<m![B,Ns,Gs,Ds]>(d,&q[row*4096..(row+16)*4096]).await;let kr=bf_upload::<m![Ts,Ns,Ds]>(d,&ringk).await;let vr=bf_upload::<m![Ts,Ns,Ds]>(d,&ringv).await;let kn=bf_upload::<m![B,Ns,Ds]>(d,&nk[row*2048..(row+16)*2048]).await;let vn=bf_upload::<m![B,Ns,Ds]>(d,&nv[row*2048..(row+16)*2048]).await;
  let(om,nm)=masks(start+row,count,16,16);let om=f_upload::<m![B,Ts]>(d,&om).await;let nm=f_upload::<m![B,Bk]>(d,&nm).await;let mut out=zero::<bf16,m![B,Ns,Gs,Ds]>(d).await;
  launch(sliding::sliding_attention_b16,(d,&qr,&kr,&vr,&kn,&vn,&om,&nm,&mut out)).await.unwrap();let begin=Instant::now();launch(sliding::sliding_attention_b16,(d,&qr,&kr,&vr,&kn,&vn,&om,&nm,&mut out)).await.unwrap();total+=begin.elapsed().as_secs_f64()*1e6;result.extend_from_slice(&read(d,&out).await[..count*4096]);
  for i in row..row+count{let slot=(start+i)%1024;ringk[slot*2048..(slot+1)*2048].copy_from_slice(&nk[i*2048..(i+1)*2048]);ringv[slot*2048..(slot+1)*2048].copy_from_slice(&nv[i*2048..(i+1)*2048]);}
 }(result,total)
}
fn cpu_samples(q:&[f32],ok:&[f32],ov:&[f32],nk:&[f32],nv:&[f32],start:usize,valid:usize,got:&[f32]){
 let mut rows=vec![0,1,15,16,63,64,valid-1];rows.retain(|r|*r<valid);rows.sort();rows.dedup();let mut want=Vec::new();let mut selected=Vec::new();
 for r in rows {for h in [0usize,3,7]{for g in 0..2{let qi=(r*16+h*2+g)*256;let mut entries=Vec::new();
  for t in 0..1024{if old_position(start,t).is_some_and(|p|p+1024>start+r){let base=(t*8+h)*256;let score=bf((0..256).map(|j|q[qi+j] as f64*ok[base+j] as f64).sum::<f64>() as f32);entries.push((score,&ov[base..base+256]));}}
  for t in 0..=r{let base=(t*8+h)*256;let score=bf((0..256).map(|j|q[qi+j] as f64*nk[base+j] as f64).sum::<f64>() as f32);entries.push((score,&nv[base..base+256]));}
  let max=entries.iter().map(|(s,_)|*s).fold(f32::NEG_INFINITY,f32::max);let denom=entries.iter().map(|(s,_)|((*s-max) as f64).exp()).sum::<f64>()+1e-6;
  let mut sum=vec![0f64;256];for(s,v)in entries{let w=bf(((s-max) as f64).exp() as f32) as f64;for j in 0..256{sum[j]+=w*v[j] as f64;}}
  want.extend(sum.into_iter().map(|x|bf((x/denom) as f32)));selected.extend_from_slice(&got[qi..qi+256]);
 }}}
 metric(&format!("nativeCPU start={start} valid={valid}"),&selected,&want,0.008);
}
async fn test(d:&mut Device,start:usize,valid:usize){
 let q=values(128*4096,100+start as u64,0.2);let ok=values(1024*2048,200,0.2);let ov=values(1024*2048,300,1.);let nk=values(128*2048,400+start as u64,0.2);let nv=values(128*2048,500+start as u64,1.);
 let(got,native_us)=native(d,&q,&ok,&ov,&nk,&nv,start,valid).await;let(reference,control_us)=chronological(d,&q,&ok,&ov,&nk,&nv,start,valid).await;
 metric(&format!("native_vs_chronological start={start} valid={valid}"),&got[..valid*4096],&reference,0.015);cpu_samples(&q,&ok,&ov,&nk,&nv,start,valid,&got);
 let mut future=nk.clone();let mut futurev=nv.clone();for x in &mut future[(valid-1)*2048..valid*2048]{*x=bf(*x*128.+16.);}for x in &mut futurev[(valid-1)*2048..valid*2048]{*x=bf(*x*128.+1000.);}let(changed,_)=native(d,&q,&ok,&ov,&future,&futurev,start,valid).await;
 let differences=got[..(valid-1)*4096].iter().zip(&changed).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();assert_eq!(differences,0,"future key/value changedearlierquery");
 if start>=1024{let expired=start%1024;let mut deadk=ok.clone();let mut deadv=ov.clone();deadk[expired*2048..(expired+1)*2048].fill(128.);deadv[expired*2048..(expired+1)*2048].fill(1024.);let(changed,_)=native(d,&q,&deadk,&deadv,&nk,&nv,start,valid).await;let differences=got[..valid*4096].iter().zip(&changed).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();assert_eq!(differences,0,"expiredoldkey/value changedqueries");}
 println!("PASS boundary start={start} valid={valid} future_invariance_exact native_warm_us={native_us:.3} chronological_attention_warm_sum_us={control_us:.3} (excludeshostuploads/commits)");
}
#[tokio::main]
async fn main(){let mut d=Device::new(wide::sliding_attention_p128.topology()).unwrap();for(start,valid)in[(0,128),(384,128),(512,128),(896,128),(960,128),(1024,128),(1032,31)]{test(&mut d,start,valid).await;}println!("PASS:nativeP128sliding CPUoracle,chronologicalnumericalgate,future/expiredkeyinvariance; oldringnotmodified");}

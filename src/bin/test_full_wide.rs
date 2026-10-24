//! Deterministic NPU/CPU attention comparisons, page/ring boundaries, and future-key invariance.
use furiosa_opt_gemma4::{Chip, axes::*, ops_full_wide as wide, ops_attention as full};
use furiosa_opt_std::prelude::*;
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

async fn native(d:&mut Device,qv:&[f32],kv:&[f32],vv:&[f32],start:usize,valid:usize)->(Vec<f32>,f64){
 let q=bf_upload::<m![P128,Gf,Df]>(d,qv).await;let mut max=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut sum=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut out=zero::<bf16,m![P128,Gf,Df]>(d).await;let mut divided=zero::<bf16,m![P128,Gf,Df]>(d).await;let mut warm=0.;
 for pass in 0..2{let mut elapsed=0.;for page in 0..=(start+valid-1)/512{let a=page*512*512;let k=bf_upload::<m![Tf,Df]>(d,&kv[a..a+512*512]).await;let v=bf_upload::<m![Tf,Df]>(d,&vv[a..a+512*512]).await;
 let mask:Vec<_>=(0..128).flat_map(|r|(0..512).map(move|t|if r<valid&&page*512+t<=start+r{0.}else{-1e30})).collect();let mask=f_upload::<m![P128,Tf]>(d,&mask).await;
 let begin=Instant::now();if page==0{launch(wide::full_attention_first_page_p128,(d,&q,&k,&v,&mask,&mut max,&mut sum,&mut out)).await.unwrap();}else{launch(wide::full_attention_next_page_p128,(d,&q,&k,&v,&mask,&mut max,&mut sum,&mut out)).await.unwrap();}elapsed+=begin.elapsed().as_secs_f64()*1e6;
 }let begin=Instant::now();launch(wide::full_attention_divide_p128,(d,&out,&sum,&mut divided)).await.unwrap();elapsed+=begin.elapsed().as_secs_f64()*1e6;if pass==1{warm=elapsed;}}
 (read(d,&divided).await,warm)
}
async fn original(d:&mut Device,qv:&[f32],kv:&[f32],vv:&[f32],start:usize,valid:usize)->(Vec<f32>,f64){
 let q=bf_upload::<m![B,Gf,Df]>(d,qv).await;let mut max=zero::<f32,m![B,Gf,1#8]>(d).await;let mut sum=zero::<f32,m![B,Gf,1#8]>(d).await;let mut out=zero::<bf16,m![B,Gf,Df]>(d).await;let mut divided=zero::<bf16,m![B,Gf,Df]>(d).await;let mut warm=0.;
 for pass in 0..2{let mut elapsed=0.;for page in 0..=(start+valid-1)/512{let a=page*512*512;let k=bf_upload::<m![Tf,Df]>(d,&kv[a..a+512*512]).await;let v=bf_upload::<m![Tf,Df]>(d,&vv[a..a+512*512]).await;
 let mask:Vec<_>=(0..16).flat_map(|r|(0..512).map(move|t|if r<valid&&page*512+t<=start+r{0.}else{-1e30})).collect();let mask=f_upload::<m![B,Tf]>(d,&mask).await;
 let begin=Instant::now();if page==0{launch(full::full_attention_first_page_b16,(d,&q,&k,&v,&mask,&mut max,&mut sum,&mut out)).await.unwrap();}else{launch(full::full_attention_next_page_b16,(d,&q,&k,&v,&mask,&mut max,&mut sum,&mut out)).await.unwrap();}elapsed+=begin.elapsed().as_secs_f64()*1e6;
 }let begin=Instant::now();launch(full::full_attention_divide_b16,(d,&out,&sum,&mut divided)).await.unwrap();elapsed+=begin.elapsed().as_secs_f64()*1e6;if pass==1{warm=elapsed;}}
 (read(d,&divided).await,warm)
}
fn exact(label:&str,a:&[f32],b:&[f32]){assert_eq!(a.len(),b.len());let diffs=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();let max=a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);println!("{label} n={} bit_differences={diffs} max_abs={max}",a.len());assert_eq!(diffs,0,"{label}");}
fn cpu(q:&[f32],k:&[f32],v:&[f32],start:usize,valid:usize,got:&[f32]){let mut rows=vec![0,63,64,valid-1];rows.retain(|r|*r<valid);rows.sort();rows.dedup();let mut error=0f64;let mut norm=0f64;let mut maxabs=0.;let mut n=0;
 for r in rows{for h in [0usize,7,15]{let qi=(r*16+h)*512;let scores:Vec<_>=(0..=start+r).map(|t|bf((0..512).map(|j|q[qi+j] as f64*k[t*512+j] as f64).sum::<f64>() as f32)).collect();let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max);let denom=scores.iter().map(|s|((*s-max) as f64).exp()).sum::<f64>()+1e-6;let mut numerator=vec![0f64;512];for(t,s)in scores.iter().enumerate(){let w=bf(((*s-max) as f64).exp() as f32) as f64;for j in 0..512{numerator[j]+=w*v[t*512+j] as f64;}}
 for j in 0..512{let expected=bf((numerator[j]/denom) as f32) as f64;let delta=got[qi+j] as f64-expected;error+=delta*delta;norm+=expected*expected;maxabs=f64::max(maxabs,delta.abs());n+=1;}
 }}let nrms=(error/norm.max(1e-30)).sqrt();println!("CPU start={start} valid={valid} n={n} nrms={nrms:.8} max_abs={maxabs}");assert!(nrms<0.012);}
async fn test(d:&mut Device,start:usize,valid:usize){assert!(start%512+valid<=512);let size=((start+valid+511)/512)*512*512;let q=values(128*8192,1000+start as u64,0.2);let k=values(size,2000,0.2);let v=values(size,3000,1.);let(got,native_us)=native(d,&q,&k,&v,start,valid).await;let mut reference=Vec::new();let mut old_us=0.;for row in (0..valid).step_by(16){let count=(valid-row).min(16);let(x,us)=original(d,&q[row*8192..(row+16)*8192],&k,&v,start+row,count).await;reference.extend_from_slice(&x[..count*8192]);old_us+=us;}
 exact(&format!("full native versus8B16 start={start} valid={valid}"),&got[..valid*8192],&reference);cpu(&q,&k,&v,start,valid,&got);
 let mut k2=k.clone();let mut v2=v.clone();k2[(start+valid-1)*512..(start+valid)*512].fill(128.);v2[(start+valid-1)*512..(start+valid)*512].fill(1024.);let(changed,_)=native(d,&q,&k2,&v2,start,valid).await;exact("future K/V invariance",&got[..(valid-1)*8192],&changed[..(valid-1)*8192]);println!("PASS full start={start} valid={valid} native_allpages_divide_us={native_us:.3} sum8original_allpages_divide_us={old_us:.3}");}
#[tokio::main]
async fn main(){let mut d=Device::new(wide::full_attention_first_page_p128.topology()).unwrap();for(start,valid)in[(0,128),(384,128),(512,128),(896,128),(1024,128),(1008,16),(1024,31)]{test(&mut d,start,valid).await;}println!("PASS native P128 full attention exact B16 page recurrence, CPUoracle and future invariance");}

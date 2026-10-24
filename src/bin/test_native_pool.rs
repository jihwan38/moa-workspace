//! Deterministic NPU/CPU attention comparisons, page/ring boundaries, and future-key invariance.
use furiosa_opt_gemma4::{Chip, axes::*, ops_sliding_wide as owned_s, ops_sliding_wide_pool as pool_s, ops_full_wide as owned_f, ops_full_wide_pool as pool_f, ops_commit_wide_pool as commit, ops_request_attention as bridge, ops_request_sliding_r64 as bridge_s, ops_pool_views::Rc64};
use furiosa_opt_std::prelude::*;
use owned_s::Pk128;
use bridge::Fc;
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

async fn indices<Mp:M>(d:&mut Device,x:Vec<i32>)->HbmTensor<i32,Chip,Mp>{HostTensor::<i32,Mp>::from_vec(x).to_hbm(&mut d.pdma).await.unwrap()}
fn exact(label:&str,a:&[f32],b:&[f32]){assert_eq!(a.len(),b.len());let n=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();println!("{label}: n={} differences={n}",a.len());assert_eq!(n,0,"{label}");}
async fn sliding(d:&mut Device){
 let mut pk=zero::<bf16,m![Rc64,Ts,Ns,Ds]>(d).await;let mut pv=zero::<bf16,m![Rc64,Ts,Ns,Ds]>(d).await;
 for slot in [0usize,31,63]{let start=960;let qv=values(128*4096,100+slot as u64,0.2);let kv=values(1024*2048,200+slot as u64,0.2);let vv=values(1024*2048,300+slot as u64,1.);let nkv=values(128*2048,400,0.2);let nvv=values(128*2048,500,1.);
 let q=bf_upload::<m![P128,Ns,Gs,Ds]>(d,&qv).await;let k=bf_upload::<m![Ts,Ns,Ds]>(d,&kv).await;let v=bf_upload::<m![Ts,Ns,Ds]>(d,&vv).await;let nk=bf_upload::<m![P128,Ns,Ds]>(d,&nkv).await;let nv=bf_upload::<m![P128,Ns,Ds]>(d,&nvv).await;let base=indices::<m![1]>(d,vec![(slot*1024*2048*2) as i32]).await;
 launch(bridge_s::copy_sliding_cache_to_pool_rc64,(d,&k,&v,&base,&mut pk,&mut pv)).await.unwrap();
 for valid in [128usize,31]{let om:Vec<_>=(0..128).flat_map(|r|(0..1024).map(move|t|if r<valid&&t<start&&t+1024>start+r{0.}else{-1e30})).collect();let cm:Vec<_>=(0..128).flat_map(|r|(0..128).map(move|t|if r<valid&&t<=r{0.}else{-1e30})).collect();let om=f_upload::<m![P128,Ts]>(d,&om).await;let cm=f_upload::<m![P128,Pk128]>(d,&cm).await;
 let mut want=zero::<bf16,m![P128,Ns,Gs,Ds]>(d).await;let mut got=zero::<bf16,m![P128,Ns,Gs,Ds]>(d).await;
 launch(owned_s::sliding_attention_p128,(d,&q,&k,&v,&nk,&nv,&om,&cm,&mut want)).await.unwrap();launch(pool_s::sliding_attention_p128,(d,&q,&pk,&pv,&base,&nk,&nv,&om,&cm,&mut got)).await.unwrap();let now=Instant::now();launch(pool_s::sliding_attention_p128,(d,&q,&pk,&pv,&base,&nk,&nv,&om,&cm,&mut got)).await.unwrap();let us=now.elapsed().as_secs_f64()*1e6;exact(&format!("sliding pool slot={slot} valid={valid}"),&read(d,&got).await[..valid*4096],&read(d,&want).await[..valid*4096]);println!("sliding pool warm_us={us:.3}");}
 let offsets=indices::<m![P128]>(d,(0..128).map(|r|((slot*1024+(start+r)%1024)*2048*2) as i32).collect()).await;launch(commit::commit_sliding_kv_p128,(d,&nk,&nv,&offsets,&mut pk,&mut pv)).await.unwrap();let mut rk=zero::<bf16,m![Ts,Ns,Ds]>(d).await;let mut rv=zero::<bf16,m![Ts,Ns,Ds]>(d).await;launch(commit::copy_sliding_pool_to_cache_rc64,(d,&pk,&pv,&base,&mut rk,&mut rv)).await.unwrap();let mut wantk=kv;let mut wantv=vv;for r in 0..128{let t=(start+r)%1024;wantk[t*2048..(t+1)*2048].copy_from_slice(&nkv[r*2048..(r+1)*2048]);wantv[t*2048..(t+1)*2048].copy_from_slice(&nvv[r*2048..(r+1)*2048]);}exact("sliding committed K and untouched positions",&read(d,&rk).await,&wantk);exact("sliding committed V and untouched positions",&read(d,&rv).await,&wantv);
 let neighbor=if slot==63{62}else{slot+1};let neighbor=indices::<m![1]>(d,vec![(neighbor*1024*2048*2) as i32]).await;launch(commit::copy_sliding_pool_to_cache_rc64,(d,&pk,&pv,&neighbor,&mut rk,&mut rv)).await.unwrap();assert!(read(d,&rk).await.iter().chain(read(d,&rv).await.iter()).all(|v|*v==0.),"neighboring ring corrupted");
 }
}
async fn full(d:&mut Device){let mut pk=zero::<bf16,m![Fc,Tf,Df]>(d).await;let mut pv=zero::<bf16,m![Fc,Tf,Df]>(d).await;
 for page in [0usize,127,255]{let qv=values(128*8192,600+page as u64,0.2);let kv=values(512*512,700+page as u64,0.2);let vv=values(512*512,800+page as u64,1.);let nkv=values(128*512,900,0.2);let nvv=values(128*512,1000,1.);
 let q=bf_upload::<m![P128,Gf,Df]>(d,&qv).await;let k=bf_upload::<m![Tf,Df]>(d,&kv).await;let v=bf_upload::<m![Tf,Df]>(d,&vv).await;let base=indices::<m![1]>(d,vec![(page*512*512*2) as i32]).await;launch(bridge::copy_full_page_to_pool,(d,&k,&v,&base,&mut pk,&mut pv)).await.unwrap();
 let mut om=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut os=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut oo=zero::<bf16,m![P128,Gf,Df]>(d).await;let mut pm=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut ps=zero::<f32,m![P128,Gf,1#8]>(d).await;let mut po=zero::<bf16,m![P128,Gf,Df]>(d).await;
 for stage in 0..2{let mask:Vec<_>=(0..128).flat_map(|r|(0..512).map(move|t|if stage==0||t<=r{0.}else{-1e30})).collect();let mask=f_upload::<m![P128,Tf]>(d,&mask).await;
 if stage==0{launch(owned_f::full_attention_first_page_p128,(d,&q,&k,&v,&mask,&mut om,&mut os,&mut oo)).await.unwrap();launch(pool_f::full_attention_first_page_p128,(d,&q,&pk,&pv,&base,&mask,&mut pm,&mut ps,&mut po)).await.unwrap();let now=Instant::now();launch(pool_f::full_attention_first_page_p128,(d,&q,&pk,&pv,&base,&mask,&mut pm,&mut ps,&mut po)).await.unwrap();println!("full pool page={page} first_warm_us={:.3}",now.elapsed().as_secs_f64()*1e6);}else{launch(owned_f::full_attention_next_page_p128,(d,&q,&k,&v,&mask,&mut om,&mut os,&mut oo)).await.unwrap();launch(pool_f::full_attention_next_page_p128,(d,&q,&pk,&pv,&base,&mask,&mut pm,&mut ps,&mut po)).await.unwrap();}
 exact(&format!("full pool page={page} recurrence={stage}"),&read(d,&po).await,&read(d,&oo).await);
 }let mut od=zero::<bf16,m![P128,Gf,Df]>(d).await;let mut pd=zero::<bf16,m![P128,Gf,Df]>(d).await;launch(owned_f::full_attention_divide_p128,(d,&oo,&os,&mut od)).await.unwrap();launch(pool_f::full_attention_divide_p128,(d,&po,&ps,&mut pd)).await.unwrap();exact("full divided",&read(d,&pd).await,&read(d,&od).await);
 let nk=bf_upload::<m![P128,Df]>(d,&nkv).await;let nv=bf_upload::<m![P128,Df]>(d,&nvv).await;let offsets=indices::<m![P128]>(d,(0..128).map(|r|((page*512+384+r)*512*2) as i32).collect()).await;launch(commit::commit_full_kv_p128,(d,&nk,&nv,&offsets,&mut pk,&mut pv)).await.unwrap();let mut rk=zero::<bf16,m![Tf,Df]>(d).await;let mut rv=zero::<bf16,m![Tf,Df]>(d).await;launch(bridge::copy_full_pool_to_page,(d,&pk,&pv,&base,&mut rk,&mut rv)).await.unwrap();let mut wantk=kv;let mut wantv=vv;wantk[384*512..].copy_from_slice(&nkv);wantv[384*512..].copy_from_slice(&nvv);exact("full committedK and untouched positions",&read(d,&rk).await,&wantk);exact("full committedV and untouched positions",&read(d,&rv).await,&wantv);
 let neighbor=if page==255{254}else{page+1};let neighbor=indices::<m![1]>(d,vec![(neighbor*512*512*2) as i32]).await;launch(bridge::copy_full_pool_to_page,(d,&pk,&pv,&neighbor,&mut rk,&mut rv)).await.unwrap();assert!(read(d,&rk).await.iter().chain(read(d,&rv).await.iter()).all(|v|*v==0.),"neighboring fullpage corrupted");
 }
}
#[tokio::main]
async fn main(){let mut d=Device::new(pool_s::sliding_attention_p128.topology()).unwrap();sliding(&mut d).await;full(&mut d).await;println!("PASS native P128 wholepool byteoffset adapters, owned attention parity, physicalfirst/middle/lastslot, page recurrence, commits and neighboring-slot preservation");}

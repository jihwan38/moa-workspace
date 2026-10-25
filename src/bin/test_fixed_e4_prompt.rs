//! Exact compressed-pool versus q022 surrogate prompt gates; no server/model activation.
use furiosa_opt_gemma4::{Chip,axes::*,ops_pool_views::Rc64,host::{fixed_e4_codec as cpu,load},
 ops_fixed_e4_wide as ew,ops_sliding_wide_head_rows as bw,
 ops_fixed_e4_prompt as ep,ops_sliding_prompt_pool as bp,
 ops_fixed_e4_wide_original as eo,ops_sliding_wide_pool as bo,
 ops_fixed_e4_fused as ef,ops_prefill_head_attn_commit as bf,
 ops_fixed_e4_commit_wide as ec,ops_fixed_e4_prompt_diagnostic as setup,
 ops_request_sliding_r64 as bridge,ops_qkv_shared_blocked::{QueryOutput,KeyOutput},
 ops_sliding_wide::Pk128,ops_sliding_batch::Bk};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
#[path="support/npu_profile.rs"] mod npu_profile;
type PE=HbmTensor<f8e4m3,Chip,m![Rc64,Ts,Ns,Ds]>;
type PB=HbmTensor<bf16,Chip,m![Rc64,Ts,Ns,Ds]>;
const K:usize=2048; const Q:usize=4096; const RING:usize=1024*K;
async fn up<D:ScalarBytes+MaterializableScalar,Mp:M>(d:&mut Device,b:&[u8])->HbmTensor<D,Chip,Mp>{assert_eq!(b.len(),Mp::SIZE*D::BITS/8);HostTensor::<D,Mp>::from_buf(b.to_vec()).to_hbm(&mut d.pdma).await.unwrap()}
async fn down<D:ScalarBytes+MaterializableScalar,Mp:M>(d:&mut Device,x:&HbmTensor<D,Chip,Mp>)->Vec<u8>{x.to_host::<Mp>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
async fn idx<Mp:M>(d:&mut Device,v:Vec<i32>)->HbmTensor<i32,Chip,Mp>{HostTensor::<i32,Mp>::from_vec(v).to_hbm(&mut d.pdma).await.unwrap()}
async fn poison<Mp:M>(d:&mut Device)->HbmTensor<bf16,Chip,Mp>{up(d,&[0xc1,0x7f].repeat(Mp::SIZE)).await}
fn exact(label:&str,a:&[u8],b:&[u8])->usize{assert_eq!(a.len(),b.len());let n=a.iter().zip(b).filter(|(a,b)|a!=b).count();assert_eq!(n,0,"{label}: first={:?}",a.iter().zip(b).position(|(a,b)|a!=b));a.len()}
fn random(n:usize,seed:u64,scale:f32)->Vec<u8>{(0..n).flat_map(|i|{let mut x=(i as u64).wrapping_add(seed*997);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^=x>>31;let f=if i%509==0{-0.0}else{((x>>40)as f32/8388608.-1.)*scale};bf16::from_f32(f).to_bits().to_le_bytes()}).collect()}
fn rounded(b:&[u8])->Vec<u8>{b.chunks_exact(2).flat_map(|x|cpu::decode(cpu::encode(u16::from_le_bytes([x[0],x[1]]))).to_le_bytes()).collect()}
fn codes(b:&[u8])->Vec<u8>{b.chunks_exact(2).map(|x|cpu::encode(u16::from_le_bytes([x[0],x[1]]))).collect()}
fn flat(b:&[u8])->Vec<f32>{b.chunks_exact(2).map(|x|f32::from_bits((u16::from_le_bytes([x[0],x[1]])as u32)<<16)).collect()}
fn oldpos(start:usize,t:usize)->Option<usize>{if start==0{return None}let last=start-1;let p=last as isize-((last+1024-t)%1024)as isize;(p>=0).then_some(p as usize)}
fn masks(start:usize,valid:usize,rows:usize)->(Vec<u8>,Vec<u8>){let a=(0..rows).flat_map(|r|(0..1024).flat_map(move|t|{let f:f32=if oldpos(start,t).is_some_and(|p|p+1024>start+r.min(valid-1)){0.}else{-1e30};f.to_le_bytes()})).collect();let b=(0..rows).flat_map(|r|(0..rows).flat_map(move|t|{let f:f32=if t<valid&&t<=r.min(valid-1){0.}else{-1e30};f.to_le_bytes()})).collect();(a,b)}
fn update(pool:&mut[u8],slot:usize,start:usize,rows:&[u8]){assert_eq!(rows.len()%K,0);for (r,bytes) in rows.chunks_exact(K).enumerate(){let off=(slot*1024+(start+r)%1024)*K;pool[off..off+K].copy_from_slice(bytes)}}
struct Pools{ek:PE,ev:PE,bk:PB,bv:PB,k:Vec<u8>,v:Vec<u8>}
impl Pools{
 async fn new(d:&mut Device)->Self{assert_eq!(Rc64::SIZE,32);let n=Rc64::SIZE*RING;let k:Vec<_>=(0..n).map(|i|{let x=i.wrapping_mul(73)^(i/K).wrapping_mul(19)^(i/RING)*37; (8+x%32)as u8|if x&128!=0{128}else{0}}).collect();let v:Vec<_>=(0..n).map(|i|{let x=i.wrapping_mul(53)^(i/K).wrapping_mul(31)^(i/RING)*71; (8+x%48)as u8|if x&256!=0{128}else{0}}).collect();let ek=up(d,&k).await;let ev=up(d,&v).await;let bk=up(d,&cpu::decode_bytes(&k)).await;let bv=up(d,&cpu::decode_bytes(&v)).await;Self{ek,ev,bk,bv,k,v}}
 async fn verify(&self,d:&mut Device,label:&str)->usize{let mut count=exact(label,&down(d,&self.ek).await,&self.k)+exact(label,&down(d,&self.ev).await,&self.v);for (h,c) in [(&self.bk,&self.k),(&self.bv,&self.v)]{let b=down(d,h).await;assert_eq!(b.len(),c.len()*2);let mut n=0;for (a,&e) in b.chunks_exact(2).zip(c){n+=usize::from(u16::from_le_bytes([a[0],a[1]])!=cpu::decode(e));}assert_eq!(n,0,"{label} decoded BF16 pool");count+=b.len();}println!("POOL_EXACT label={label} bytes={count}");count}
 async fn restore_ring(&mut self,d:&mut Device,slot:usize,k:&[u8],v:&[u8]){let a=up::<f8e4m3,m![Ts,Ns,Ds]>(d,k).await;let b=up::<f8e4m3,m![Ts,Ns,Ds]>(d,v).await;let c=idx::<m![1]>(d,vec![(slot*RING)as i32]).await;launch(setup::copy_ring,(d,&a,&b,&c,&mut self.ek,&mut self.ev)).await.unwrap();let a=up::<bf16,m![Ts,Ns,Ds]>(d,&cpu::decode_bytes(k)).await;let b=up::<bf16,m![Ts,Ns,Ds]>(d,&cpu::decode_bytes(v)).await;let c=idx::<m![1]>(d,vec![(slot*RING*2)as i32]).await;launch(bridge::copy_sliding_cache_to_pool_rc64,(d,&a,&b,&c,&mut self.bk,&mut self.bv)).await.unwrap();}
}
async fn commit_gate(d:&mut Device,p:&mut Pools)->usize{
 let c:Vec<u8>=(0..128*K).map(|i|{let n=(i*73)%254;if n<127{n as u8}else{(n+1)as u8}}).collect();let v:Vec<u8>=c.iter().rev().copied().collect();let kb=cpu::decode_bytes(&c);let vb=cpu::decode_bytes(&v);assert_eq!(codes(&kb),c);assert_eq!(codes(&vb),v);
 let k=up::<bf16,m![P128,Ns,Ds]>(d,&kb).await;let val=up::<bf16,m![P128,Ns,Ds]>(d,&vb).await;
 for slot in 0..32{let start=[0,896,960,1023,1024,1920][slot%6];let e=idx::<m![P128]>(d,(0..128).map(|r|((slot*1024+(start+r)%1024)*K)as i32).collect()).await;let b=idx::<m![P128]>(d,(0..128).map(|r|((slot*1024+(start+r)%1024)*K*2)as i32).collect()).await;
 launch(ec::commit_sliding_kv_p128,(d,&k,&val,&e,&mut p.ek,&mut p.ev)).await.unwrap();launch(furiosa_opt_gemma4::ops_commit_wide_pool::commit_sliding_kv_p128,(d,&k,&val,&b,&mut p.bk,&mut p.bv)).await.unwrap();update(&mut p.k,slot,start,&c);update(&mut p.v,slot,start,&v);}
 let n=p.verify(d,"all32_ring_commits_finite254_signedzeros_wrap").await;exact("commit sourceK immutable",&down(d,&k).await,&kb);exact("commit sourceV immutable",&down(d,&val).await,&vb);n
}
// Independent flat F64 attention samples, secondary to mandatory all-bit control.
fn cpu_samples(q:&[u8],oldk:&[u8],oldv:&[u8],nk:&[u8],nv:&[u8],start:usize,got:&[u8]){
 let q=flat(q);let k=flat(&cpu::decode_bytes(oldk));let v=flat(&cpu::decode_bytes(oldv));let nk=flat(nk);let nv=flat(nv);let out=flat(got);let mut err=0f64;let mut norm=0f64;let mut count=0;
 for r in [0usize,15,63,127]{for h in [0usize,7]{for g in 0..2{let qi=(r*16+h*2+g)*256;let mut entries=Vec::new();for t in 0..1024{if oldpos(start,t).is_some_and(|p|p+1024>start+r){let off=(t*8+h)*256;let s=(0..256).map(|j|q[qi+j]as f64*k[off+j]as f64).sum::<f64>() as f32;entries.push((bf16::from_f32(s).to_f32(),false,off));}}for t in 0..=r{let off=(t*8+h)*256;let s=(0..256).map(|j|q[qi+j]as f64*nk[off+j]as f64).sum::<f64>()as f32;entries.push((bf16::from_f32(s).to_f32(),true,off));}
 let max=entries.iter().map(|x|x.0).fold(f32::NEG_INFINITY,f32::max);let den=entries.iter().map(|x|((x.0-max)as f64).exp()).sum::<f64>()+1e-6;for j in [0usize,17,127,255]{let sum=entries.iter().map(|(s,new,o)|bf16::from_f32(((*s-max)as f64).exp()as f32).to_f32()as f64 * if *new{nv[o+j]}else{v[o+j]}as f64).sum::<f64>()/den;let a=out[qi+j]as f64;err+=(a-sum).powi(2);norm+=sum.powi(2);count+=1;}}}}
 let rel=(err/norm.max(1e-30)).sqrt();println!("CPU_F64 samples={count} relative_l2={rel:.8} boundary=flat_secondary");assert!(rel<0.015);
}
async fn readers(d:&mut Device,p:&mut Pools,profile:&npu_profile::Collector)->usize{
 let mut checked=0;
 for slot in 0..32{let start=[0,128,384,896,960,1024,1920,8064][slot%8];let qb=random(128*Q,slot as u64+20,0.15);let kb=rounded(&random(128*K,slot as u64+30,0.2));let vb=rounded(&random(128*K,slot as u64+40,1.));let q=up::<bf16,m![P128,Ns,Gs,Ds]>(d,&qb).await;let k=up::<bf16,m![P128,Ns,Ds]>(d,&kb).await;let v=up::<bf16,m![P128,Ns,Ds]>(d,&vb).await;let (om,cm)=masks(start,128,128);let om=up::<f32,m![P128,Ts]>(d,&om).await;let cm=up::<f32,m![P128,Pk128]>(d,&cm).await;let eb=idx::<m![P128/64]>(d,vec![(slot*RING)as i32;2]).await;let bb=idx::<m![P128/64]>(d,vec![(slot*RING*2)as i32;2]).await;let mut a=poison::<m![P128,Ns,Gs,Ds]>(d).await;let mut b=poison::<m![P128,Ns,Gs,Ds]>(d).await;
 launch(ew::sliding_attention_p128,(d,&q,&p.ek,&p.ev,&eb,&k,&v,&om,&cm,&mut a)).await.unwrap();let got=down(d,&a).await;
 launch(bw::sliding_attention_p128,(d,&q,&p.bk,&p.bv,&bb,&k,&v,&om,&cm,&mut b)).await.unwrap();checked+=exact("wide all128 candidate-first",&got,&down(d,&b).await);
 if start==0{launch(ew::sliding_attention_first_p128,(d,&q,&k,&v,&cm,&mut a)).await.unwrap();checked+=exact("first128",&down(d,&a).await,&got);}
 if slot==0||slot==31{cpu_samples(&qb,&p.k[slot*RING..(slot+1)*RING],&p.v[slot*RING..(slot+1)*RING],&kb,&vb,start,&got);}
 let mut fk=kb.clone();let mut fv=vb.clone();for x in fk[(127*K*2)..].chunks_exact_mut(2){x.copy_from_slice(&cpu::decode(0x70).to_le_bytes())}for x in fv[(127*K*2)..].chunks_exact_mut(2){x.copy_from_slice(&cpu::decode(0xf0).to_le_bytes())}let fkh=up::<bf16,m![P128,Ns,Ds]>(d,&fk).await;let fvh=up::<bf16,m![P128,Ns,Ds]>(d,&fv).await;
 launch(ew::sliding_attention_p128,(d,&q,&p.ek,&p.ev,&eb,&fkh,&fvh,&om,&cm,&mut a)).await.unwrap();checked+=exact("future row127 invariant",&down(d,&a).await[..127*Q*2],&got[..127*Q*2]);
 if slot==31{let eb1=idx::<m![1]>(d,vec![(slot*RING)as i32]).await;let bb1=idx::<m![1]>(d,vec![(slot*RING*2)as i32]).await;launch(eo::sliding_attention_p128,(d,&q,&p.ek,&p.ev,&eb1,&k,&v,&om,&cm,&mut a)).await.unwrap();let x=down(d,&a).await;launch(bo::sliding_attention_p128,(d,&q,&p.bk,&p.bv,&bb1,&k,&v,&om,&cm,&mut b)).await.unwrap();checked+=exact("original diagnostic reader",&x,&down(d,&b).await);
 let mut times=[Vec::new(),Vec::new()];for choose in [false,true,true,false,false,true,true,false]{let t=Instant::now();if choose{launch(ew::sliding_attention_p128,(d,&q,&p.ek,&p.ev,&eb,&k,&v,&om,&cm,&mut a)).await.unwrap();}else{launch(bw::sliding_attention_p128,(d,&q,&p.bk,&p.bv,&bb,&k,&v,&om,&cm,&mut b)).await.unwrap();}times[choose as usize].push(t.elapsed().as_secs_f64()*1e6);}println!("TIMING wide control_us={:?} e4_us={:?}",times[0],times[1]);
 if std::env::args().any(|a|a=="--profile"){profile.enable(true);println!("PROFILE_WIDE_CONTROL");launch(bw::sliding_attention_p128,(d,&q,&p.bk,&p.bv,&bb,&k,&v,&om,&cm,&mut b)).await.unwrap();println!("PROFILE_WIDE_E4");launch(ew::sliding_attention_p128,(d,&q,&p.ek,&p.ev,&eb,&k,&v,&om,&cm,&mut a)).await.unwrap();profile.enable(false);}}
 checked+=exact("query immutable",&down(d,&q).await,&qb)+exact("chunkK immutable",&down(d,&k).await,&kb)+exact("chunkV immutable",&down(d,&v).await,&vb);
 println!("WIDE_PASS slot={slot} start={start} checked_bytes={checked}");}
 checked+=p.verify(d,"all_readers_immutable").await;checked
}
async fn partial(d:&mut Device,p:&Pools)->usize{let qb=random(16*Q,111,0.15);let kb=rounded(&random(16*K,112,0.2));let vb=rounded(&random(16*K,113,1.));let q=up::<bf16,m![B,Ns,Gs,Ds]>(d,&qb).await;let k=up::<bf16,m![B,Ns,Ds]>(d,&kb).await;let v=up::<bf16,m![B,Ns,Ds]>(d,&vb).await;let eb=idx::<m![1]>(d,vec![(31*RING)as i32]).await;let bb=idx::<m![1]>(d,vec![(31*RING*2)as i32]).await;let mut a=poison::<m![B,Ns,Gs,Ds]>(d).await;let mut b=poison::<m![B,Ns,Gs,Ds]>(d).await;let mut n=0;
 for valid in 1..=16{let start=if valid%2==0{1024}else{1008};let (om,cm)=masks(start,valid,16);let om=up::<f32,m![B,Ts]>(d,&om).await;let cm=up::<f32,m![B,Bk]>(d,&cm).await;launch(ep::sliding_attention_b16,(d,&q,&p.ek,&p.ev,&eb,&k,&v,&om,&cm,&mut a)).await.unwrap();let got=down(d,&a).await;launch(bp::sliding_attention_b16,(d,&q,&p.bk,&p.bv,&bb,&k,&v,&om,&cm,&mut b)).await.unwrap();n+=exact("partial live rows",&got[..valid*Q*2],&down(d,&b).await[..valid*Q*2]);
 let mut futurek=kb.clone();let mut futurev=vb.clone();for x in futurek[(valid-1)*K*2..].chunks_exact_mut(2){x.copy_from_slice(&cpu::decode(0x70).to_le_bytes())}for x in futurev[(valid-1)*K*2..].chunks_exact_mut(2){x.copy_from_slice(&cpu::decode(0xf0).to_le_bytes())}let fk=up::<bf16,m![B,Ns,Ds]>(d,&futurek).await;let fv=up::<bf16,m![B,Ns,Ds]>(d,&futurev).await;launch(ep::sliding_attention_b16,(d,&q,&p.ek,&p.ev,&eb,&fk,&fv,&om,&cm,&mut a)).await.unwrap();n+=exact("partial future/padding invariant",&down(d,&a).await[..(valid-1)*Q*2],&got[..(valid-1)*Q*2]);}
 n+=exact("partial Q immutable",&down(d,&q).await,&qb)+exact("partial K immutable",&down(d,&k).await,&kb)+exact("partial V immutable",&down(d,&v).await,&vb);println!("PARTIAL_PASS live1..16 checked_bytes={n}");n}
fn block(bytes:&[u8],channels:usize,tile:usize)->Vec<u8>{let mut out=vec![0;bytes.len()];let mut seen=vec![false;128*channels];for r in 0..128{for c in 0..channels{let from=r*channels+c;let to=(((r/64)*(channels/tile)+c/tile)*64+r%64)*tile+c%tile;assert!(!seen[to]);seen[to]=true;out[to*2..to*2+2].copy_from_slice(&bytes[from*2..from*2+2]);}}assert!(seen.into_iter().all(|x|x));out}
struct Raw{q:HbmTensor<bf16,Chip,QueryOutput>,k:HbmTensor<bf16,Chip,KeyOutput>,v:HbmTensor<bf16,Chip,KeyOutput>,qb:Vec<u8>,kb:Vec<u8>,vb:Vec<u8>}
impl Raw{async fn new(d:&mut Device,seed:u64)->Self{let qb=block(&random(128*Q,seed,2.),Q,64);let kb=block(&random(128*K,seed+1,2.),K,32);let vb=block(&random(128*K,seed+2,2.),K,32);Self{q:up(d,&qb).await,k:up(d,&kb).await,v:up(d,&vb).await,qb,kb,vb}}}
struct Out{k:HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,v:HbmTensor<bf16,Chip,m![P128,Ns,Ds]>,a:HbmTensor<bf16,Chip,m![P128,Ns,Gs,Ds]>}
impl Out{async fn new(d:&mut Device)->Self{Self{k:poison(d).await,v:poison(d).await,a:poison(d).await}}}
struct Meta{qn:HbmTensor<bf16,Chip,m![Ds]>,kn:HbmTensor<bf16,Chip,m![Ds]>,cos:HbmTensor<bf16,Chip,m![E,Ds]>,sin:HbmTensor<bf16,Chip,m![E,Ds]>,rope:HbmTensor<i32,Chip,m![P128]>,eb:HbmTensor<i32,Chip,m![P128/64]>,bb:HbmTensor<i32,Chip,m![P128/64]>,eo:HbmTensor<i32,Chip,m![P128]>,bo:HbmTensor<i32,Chip,m![P128]>,om:HbmTensor<f32,Chip,m![P128,Ts]>,cm:HbmTensor<f32,Chip,m![P128,Pk128]>}
impl Meta{async fn new(d:&mut Device,start:usize,slot:usize,layer:usize)->Self{
 let f=std::fs::File::open(load::model_dir().unwrap().join("model.safetensors")).unwrap();let mmap=unsafe{memmap2::Mmap::map(&f).unwrap()};let tensors=safetensors::SafeTensors::deserialize(&mmap).unwrap();let prefix=format!("model.language_model.layers.{layer}.self_attn");let qn=up(d,tensors.tensor(&format!("{prefix}.q_norm.weight")).unwrap().data()).await;let kn=up(d,tensors.tensor(&format!("{prefix}.k_norm.weight")).unwrap().data()).await;
 let mut cos=vec![0u8;E::SIZE*256*2];let mut sin=cos.clone();for p in start..start+128{assert!(p<E::SIZE);for c in 0..256{let ang=p as f32/10000f32.powf(2.*(c%128)as f32/256.);let off=(p*256+c)*2;cos[off..off+2].copy_from_slice(&bf16::from_f32(ang.cos()).to_bits().to_le_bytes());sin[off..off+2].copy_from_slice(&bf16::from_f32(ang.sin()*if c<128{-1.}else{1.}).to_bits().to_le_bytes());}}
 let cos=up(d,&cos).await;let sin=up(d,&sin).await;let rope=idx(d,(start..start+128).map(|p|(p*256*2)as i32).collect()).await;let eb=idx(d,vec![(slot*RING)as i32;2]).await;let bb=idx(d,vec![(slot*RING*2)as i32;2]).await;let eo=idx(d,(0..128).map(|r|((slot*1024+(start+r)%1024)*K)as i32).collect()).await;let bo=idx(d,(0..128).map(|r|((slot*1024+(start+r)%1024)*K*2)as i32).collect()).await;let(om,cm)=masks(start,128,128);Self{qn,kn,cos,sin,rope,eb,bb,eo,bo,om:up(d,&om).await,cm:up(d,&cm).await}
}}
async fn fused_call(d:&mut Device,candidate:bool,first:bool,raw:&Raw,m:&Meta,p:&mut Pools,o:&mut Out){
 if candidate{if first{launch(ef::sliding_fused_first_p128,(d,&raw.q,&raw.k,&raw.v,&m.qn,&m.kn,&m.rope,&m.cos,&m.sin,&mut p.ek,&mut p.ev,&m.eb,&m.om,&m.cm,&m.eo,&mut o.k,&mut o.v,&mut o.a)).await.unwrap();}else{launch(ef::sliding_fused_p128,(d,&raw.q,&raw.k,&raw.v,&m.qn,&m.kn,&m.rope,&m.cos,&m.sin,&mut p.ek,&mut p.ev,&m.eb,&m.om,&m.cm,&m.eo,&mut o.k,&mut o.v,&mut o.a)).await.unwrap();}}
 else if first{launch(bf::sliding_fused_first_p128,(d,&raw.q,&raw.k,&raw.v,&m.qn,&m.kn,&m.rope,&m.cos,&m.sin,&mut p.bk,&mut p.bv,&m.bb,&m.om,&m.cm,&m.bo,&mut o.k,&mut o.v,&mut o.a)).await.unwrap();}else{launch(bf::sliding_fused_p128,(d,&raw.q,&raw.k,&raw.v,&m.qn,&m.kn,&m.rope,&m.cos,&m.sin,&mut p.bk,&mut p.bv,&m.bb,&m.om,&m.cm,&m.bo,&mut o.k,&mut o.v,&mut o.a)).await.unwrap();}
}
async fn fused(d:&mut Device,p:&mut Pools,profile:&npu_profile::Collector)->usize{let mut n=0;
 for (case,(slot,start,layer)) in [(0usize,0usize,0usize),(31,128,6),(15,896,46),(15,1024,0),(31,1920,6),(0,8064,46)].into_iter().enumerate(){let raw=Raw::new(d,801+case as u64*73).await;let m=Meta::new(d,start,slot,layer).await;let mut c=Out::new(d).await;let mut r=Out::new(d).await;let oldk=p.k[slot*RING..(slot+1)*RING].to_vec();let oldv=p.v[slot*RING..(slot+1)*RING].to_vec();
 fused_call(d,true,start==0,&raw,&m,p,&mut c).await;let ck=down(d,&c.k).await;let cv=down(d,&c.v).await;let ca=down(d,&c.a).await;
 fused_call(d,false,start==0,&raw,&m,p,&mut r).await;n+=exact("fused K scratch",&ck,&down(d,&r.k).await)+exact("fused V scratch",&cv,&down(d,&r.v).await)+exact("fused attention",&ca,&down(d,&r.a).await);
 // Independent CPU codec also proves the stored BF16 scratch is already representable.
 let kc=codes(&ck);let vc=codes(&cv);n+=exact("headed K codec idempotence",&cpu::decode_bytes(&kc),&ck)+exact("headed V codec idempotence",&cpu::decode_bytes(&vc),&cv);
 let mut times=[Vec::new(),Vec::new()];for choose in [false,true,true,false,false,true,true,false]{p.restore_ring(d,slot,&oldk,&oldv).await;let t=Instant::now();if choose{fused_call(d,true,start==0,&raw,&m,p,&mut c).await;}else{fused_call(d,false,start==0,&raw,&m,p,&mut r).await;}times[choose as usize].push(t.elapsed().as_secs_f64()*1e6);}
 println!("TIMING fused case={case} slot={slot} start={start} control_us={:?} e4_us={:?} reset_outside=true",times[0],times[1]);
 p.restore_ring(d,slot,&oldk,&oldv).await;
 if case==5&&std::env::args().any(|x|x=="--profile"){profile.enable(true);println!("PROFILE_FUSED_E4");}
 fused_call(d,true,start==0,&raw,&m,p,&mut c).await;profile.enable(false);
 if case==5&&std::env::args().any(|x|x=="--profile"){profile.enable(true);println!("PROFILE_FUSED_CONTROL");}
 fused_call(d,false,start==0,&raw,&m,p,&mut r).await;profile.enable(false);
 n+=exact("fused reset/replay candidate",&down(d,&c.a).await,&ca)+exact("fused reset/replay control",&down(d,&r.a).await,&ca);
 update(&mut p.k,slot,start,&kc);update(&mut p.v,slot,start,&vc);
 n+=exact("rawQ immutable",&down(d,&raw.q).await,&raw.qb)+exact("rawK immutable",&down(d,&raw.k).await,&raw.kb)+exact("rawV immutable",&down(d,&raw.v).await,&raw.vb);
 // Full pools, including every untouched slot, are checked after every changing case.
 n+=p.verify(d,&format!("fused{case}_all_pools_CPU_affine")).await;println!("FUSED_PASS case={case} slot={slot} start={start} layer={layer} checked_bytes={n}");}
 n}
#[tokio::main]
async fn main(){let profile=npu_profile::Collector::default();tracing::subscriber::set_global_default(profile.clone()).unwrap();let mut d=Device::new(ew::sliding_attention_p128.topology()).unwrap();let mut n=0;{
 let mut p=Pools::new(&mut d).await;n+=commit_gate(&mut d,&mut p).await;
 }let mut p=Pools::new(&mut d).await;n+=readers(&mut d,&mut p,&profile).await;n+=partial(&mut d,&p).await;n+=p.verify(&mut d,"partial_immutable").await;n+=fused(&mut d,&mut p,&profile).await;
 println!("PASS FIXED_E4_PROMPT checked_bytes={n} physical_rings=32 readers_all_rings=true partial_live=1..16 fused_cases=6 full_cache_cpu=true control=q022_BF16_surrogate current_chunks_BF16=true");}

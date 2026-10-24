use furiosa_opt_gemma4::{Chip,axes::*,ops_commit_wide as wide};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
fn bits(n:usize,seed:u32)->Vec<u16>{(0..n).map(|i|((i as u32).wrapping_mul(0x9e3779b9).wrapping_add(seed)>>8) as u16).collect()}
async fn upload<Mp:M>(d:&mut Device,x:&[u16])->HbmTensor<bf16,Chip,Mp>{assert_eq!(x.len(),Mp::SIZE);HostTensor::<bf16,Mp>::from_buf(x.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap()}
async fn read<Mp:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mp>)->Vec<u16>{let h:HostTensor<bf16,Mp>=x.to_host(&mut d.pdma).await.unwrap();h.into_vec().into_iter().map(|x|x.to_bits()).collect()}
async fn offsets(d:&mut Device,start:usize,rows:usize,width:usize)->HbmTensor<i32,Chip,m![P128]>{HostTensor::<i32,m![P128]>::from_vec((0..128).map(|i|(((start+i)%rows)*width*2) as i32).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap()}
fn expected(mut cache:Vec<u16>,x:&[u16],start:usize,rows:usize,width:usize)->Vec<u16>{for i in 0..128{let slot=(start+i)%rows;cache[slot*width..(slot+1)*width].copy_from_slice(&x[i*width..(i+1)*width]);}cache}
fn exact(label:&str,a:&[u16],b:&[u16]){assert_eq!(a.len(),b.len());let count=a.iter().zip(b).filter(|(a,b)|a!=b).count();if count>0{let i=a.iter().zip(b).position(|(a,b)|a!=b).unwrap();println!("{label}: first_difference={i} got={:04x} expected={:04x}",a[i],b[i]);}assert_eq!(count,0,"{label}");println!("PASS {label} all{} values and nonselected cache positions exact",a.len());}
#[tokio::main]
async fn main(){let mut d=Device::new(wide::commit_sliding_kv_p128.topology()).unwrap();
 let nk=bits(128*2048,100);let nv=bits(128*2048,200);let oldk=bits(1024*2048,300);let oldv=bits(1024*2048,400);let k=upload::<m![P128,Ns,Ds]>(&mut d,&nk).await;let v=upload::<m![P128,Ns,Ds]>(&mut d,&nv).await;
 for start in [0,384,896,960,1024,1032]{let mut kc=upload::<m![Ts,Ns,Ds]>(&mut d,&oldk).await;let mut vc=upload::<m![Ts,Ns,Ds]>(&mut d,&oldv).await;let offsets=offsets(&mut d,start,1024,2048).await;
 launch(wide::commit_sliding_kv_p128,(&mut d,&k,&v,&offsets,&mut kc,&mut vc)).await.unwrap();let now=Instant::now();launch(wide::commit_sliding_kv_p128,(&mut d,&k,&v,&offsets,&mut kc,&mut vc)).await.unwrap();let us=now.elapsed().as_secs_f64()*1e6;
 exact(&format!("sliding K start={start}"),&read(&mut d,&kc).await,&expected(oldk.clone(),&nk,start,1024,2048));exact(&format!("sliding V start={start}"),&read(&mut d,&vc).await,&expected(oldv.clone(),&nv,start,1024,2048));println!("sliding commit warm_us={us:.3}");}
 let nk=bits(128*512,500);let nv=bits(128*512,600);let oldk=bits(512*512,700);let oldv=bits(512*512,800);let k=upload::<m![P128,Df]>(&mut d,&nk).await;let v=upload::<m![P128,Df]>(&mut d,&nv).await;
 for start in [0,128,256,384]{let mut kc=upload::<m![Tf,Df]>(&mut d,&oldk).await;let mut vc=upload::<m![Tf,Df]>(&mut d,&oldv).await;let offsets=offsets(&mut d,start,512,512).await;
 launch(wide::commit_full_kv_p128,(&mut d,&k,&v,&offsets,&mut kc,&mut vc)).await.unwrap();let now=Instant::now();launch(wide::commit_full_kv_p128,(&mut d,&k,&v,&offsets,&mut kc,&mut vc)).await.unwrap();let us=now.elapsed().as_secs_f64()*1e6;
 exact(&format!("full K start={start}"),&read(&mut d,&kc).await,&expected(oldk.clone(),&nk,start,512,512));exact(&format!("full V start={start}"),&read(&mut d,&vc).await,&expected(oldv.clone(),&nv,start,512,512));println!("full commit warm_us={us:.3}");}
 println!("PASS production128 KV commits, sliding wrap, full page end, arbitrary bits and untouched positions; partial chunks must use existing B16 live-prefix commits");}

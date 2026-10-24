use furiosa_opt_gemma4::{Chip,axes::*,ops_commit_r64 as op,ops_request_attention as old,ops_pool_views::Rc64};
use old::Fc;
use furiosa_opt_std::prelude::*;
fn bits(n:usize,seed:u32)->Vec<u16>{(0..n).map(|i|((i as u32).wrapping_mul(0x9e3779b9).wrapping_add(seed)>>8) as u16).collect()}
async fn upload<Mp:M>(d:&mut Device,x:&[u16])->HbmTensor<bf16,Chip,Mp>{assert_eq!(x.len(),Mp::SIZE);HostTensor::<bf16,Mp>::from_buf(x.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>()).to_hbm(&mut d.pdma).await.unwrap()}
async fn zero<Mp:M>(d:&mut Device)->HbmTensor<bf16,Chip,Mp>{upload::<Mp>(d,&vec![0u16;Mp::SIZE]).await}
async fn read<Mp:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mp>)->Vec<u16>{let h:HostTensor<bf16,Mp>=x.to_host(&mut d.pdma).await.unwrap();h.into_vec().into_iter().map(|x|x.to_bits()).collect()}
async fn indices<Mp:M>(d:&mut Device,x:Vec<i32>)->HbmTensor<i32,Chip,Mp>{HostTensor::<i32,Mp>::from_vec(x).to_hbm(&mut d.pdma).await.unwrap()}
fn exact(label:&str,a:&[u16],b:&[u16]){assert_eq!(a.len(),b.len());let n=a.iter().zip(b).filter(|(a,b)|a!=b).count();if n>0{let i=a.iter().zip(b).position(|(a,b)|a!=b).unwrap();println!("{label}: first={i} got={:04x} wanted={:04x}",a[i],b[i]);}assert_eq!(n,0,"{label}");}
fn position(r:usize,rows:usize)->usize{[0,1,rows-1,rows-2,rows/2-1,rows/2,17,31][r%8]}
async fn test_sliding(d:&mut Device){
 let nkv=bits(64*2048,100);let nvv=bits(64*2048,200);let nk=upload::<m![R64,Ns,Ds]>(d,&nkv).await;let nv=upload::<m![R64,Ns,Ds]>(d,&nvv).await;
 let offsets:Vec<i32>=(0..64).map(|r|((((r*5+3)%64)*1024+position(r,1024))*2048*2) as i32).collect();
 let neighbors:Vec<i32>=(0..64).map(|r|((((r*5+3)%64)*1024+(position(r,1024)+1)%1024)*2048*2) as i32).collect();
 for count in [1usize,7,17,31,63,64]{let mut pk=zero::<m![Rc64,Ts,Ns,Ds]>(d).await;let mut pv=zero::<m![Rc64,Ts,Ns,Ds]>(d).await;
 if count==64{let offsets=indices::<m![R64]>(d,offsets.clone()).await;launch(op::commit_sliding_kv_r64,(d,&nk,&nv,&offsets,&mut pk,&mut pv)).await.unwrap();}else{
 for first in (0..count).step_by(16){let valid=(count-first).min(16);let tile=indices::<m![1]>(d,vec![(first/16) as i32]).await;let mut bk=zero::<m![B,Ns,Ds]>(d).await;let mut bv=zero::<m![B,Ns,Ds]>(d).await;
 launch(op::load_sliding_kv_tile,(d,&nk,&nv,&tile,&mut bk,&mut bv)).await.unwrap();exact("sliding K tile",&read(d,&bk).await,&nkv[first*2048..(first+16)*2048]);exact("sliding V tile",&read(d,&bv).await,&nvv[first*2048..(first+16)*2048]);
 match valid{
1=>{let offsets=indices::<m![B=1]>(d,offsets[first..first+1].to_vec()).await;launch(op::commit_sliding_pool_1,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
2=>{let offsets=indices::<m![B=2]>(d,offsets[first..first+2].to_vec()).await;launch(op::commit_sliding_pool_2,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
3=>{let offsets=indices::<m![B=3]>(d,offsets[first..first+3].to_vec()).await;launch(op::commit_sliding_pool_3,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
4=>{let offsets=indices::<m![B=4]>(d,offsets[first..first+4].to_vec()).await;launch(op::commit_sliding_pool_4,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
5=>{let offsets=indices::<m![B=5]>(d,offsets[first..first+5].to_vec()).await;launch(op::commit_sliding_pool_5,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
6=>{let offsets=indices::<m![B=6]>(d,offsets[first..first+6].to_vec()).await;launch(op::commit_sliding_pool_6,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
7=>{let offsets=indices::<m![B=7]>(d,offsets[first..first+7].to_vec()).await;launch(op::commit_sliding_pool_7,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
8=>{let offsets=indices::<m![B=8]>(d,offsets[first..first+8].to_vec()).await;launch(op::commit_sliding_pool_8,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
9=>{let offsets=indices::<m![B=9]>(d,offsets[first..first+9].to_vec()).await;launch(op::commit_sliding_pool_9,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
10=>{let offsets=indices::<m![B=10]>(d,offsets[first..first+10].to_vec()).await;launch(op::commit_sliding_pool_10,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
11=>{let offsets=indices::<m![B=11]>(d,offsets[first..first+11].to_vec()).await;launch(op::commit_sliding_pool_11,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
12=>{let offsets=indices::<m![B=12]>(d,offsets[first..first+12].to_vec()).await;launch(op::commit_sliding_pool_12,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
13=>{let offsets=indices::<m![B=13]>(d,offsets[first..first+13].to_vec()).await;launch(op::commit_sliding_pool_13,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
14=>{let offsets=indices::<m![B=14]>(d,offsets[first..first+14].to_vec()).await;launch(op::commit_sliding_pool_14,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
15=>{let offsets=indices::<m![B=15]>(d,offsets[first..first+15].to_vec()).await;launch(op::commit_sliding_pool_15,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
16=>{let offsets=indices::<m![B=16]>(d,offsets[first..first+16].to_vec()).await;launch(op::commit_sliding_pool_16,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
_=>unreachable!(),}}}
 let offsets=indices::<m![R64]>(d,offsets.clone()).await;let mut rk=zero::<m![R64,Ns,Ds]>(d).await;let mut rv=zero::<m![R64,Ns,Ds]>(d).await;launch(op::gather_sliding_rows_r64,(d,&pk,&pv,&offsets,&mut rk,&mut rv)).await.unwrap();let mut expectedk=vec![0u16;64*2048];let mut expectedv=expectedk.clone();expectedk[..count*2048].copy_from_slice(&nkv[..count*2048]);expectedv[..count*2048].copy_from_slice(&nvv[..count*2048]);exact("sliding K live and inactive rows",&read(d,&rk).await,&expectedk);exact("sliding V live and inactive rows",&read(d,&rv).await,&expectedv);
 let neighbors=indices::<m![R64]>(d,neighbors.clone()).await;launch(op::gather_sliding_rows_r64,(d,&pk,&pv,&neighbors,&mut rk,&mut rv)).await.unwrap();assert!(read(d,&rk).await.iter().chain(read(d,&rv).await.iter()).all(|v|*v==0),"sliding neighboring tokens changed");println!("PASS sliding count={count} exact live/inactive positions, all tile payloads and neighboring tokens");
 }
}
async fn test_full(d:&mut Device){
 let nkv=bits(64*512,100);let nvv=bits(64*512,200);let nk=upload::<m![R64,Df]>(d,&nkv).await;let nv=upload::<m![R64,Df]>(d,&nvv).await;
 let offsets:Vec<i32>=(0..64).map(|r|((((r*5+255)%256)*512+position(r,512))*512*2) as i32).collect();
 let neighbors:Vec<i32>=(0..64).map(|r|((((r*5+255)%256)*512+(position(r,512)+1)%512)*512*2) as i32).collect();
 for count in [1usize,7,17,31,63,64]{let mut pk=zero::<m![Fc,Tf,Df]>(d).await;let mut pv=zero::<m![Fc,Tf,Df]>(d).await;
 if count==64{let offsets=indices::<m![R64]>(d,offsets.clone()).await;launch(op::commit_full_kv_r64,(d,&nk,&nv,&offsets,&mut pk,&mut pv)).await.unwrap();}else{
 for first in (0..count).step_by(16){let valid=(count-first).min(16);let tile=indices::<m![1]>(d,vec![(first/16) as i32]).await;let mut bk=zero::<m![B,Df]>(d).await;let mut bv=zero::<m![B,Df]>(d).await;
 launch(op::load_full_kv_tile,(d,&nk,&nv,&tile,&mut bk,&mut bv)).await.unwrap();exact("full K tile",&read(d,&bk).await,&nkv[first*512..(first+16)*512]);exact("full V tile",&read(d,&bv).await,&nvv[first*512..(first+16)*512]);
 match valid{
1=>{let offsets=indices::<m![B=1]>(d,offsets[first..first+1].to_vec()).await;launch(old::commit_full_pool_1,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
2=>{let offsets=indices::<m![B=2]>(d,offsets[first..first+2].to_vec()).await;launch(old::commit_full_pool_2,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
3=>{let offsets=indices::<m![B=3]>(d,offsets[first..first+3].to_vec()).await;launch(old::commit_full_pool_3,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
4=>{let offsets=indices::<m![B=4]>(d,offsets[first..first+4].to_vec()).await;launch(old::commit_full_pool_4,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
5=>{let offsets=indices::<m![B=5]>(d,offsets[first..first+5].to_vec()).await;launch(old::commit_full_pool_5,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
6=>{let offsets=indices::<m![B=6]>(d,offsets[first..first+6].to_vec()).await;launch(old::commit_full_pool_6,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
7=>{let offsets=indices::<m![B=7]>(d,offsets[first..first+7].to_vec()).await;launch(old::commit_full_pool_7,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
8=>{let offsets=indices::<m![B=8]>(d,offsets[first..first+8].to_vec()).await;launch(old::commit_full_pool_8,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
9=>{let offsets=indices::<m![B=9]>(d,offsets[first..first+9].to_vec()).await;launch(old::commit_full_pool_9,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
10=>{let offsets=indices::<m![B=10]>(d,offsets[first..first+10].to_vec()).await;launch(old::commit_full_pool_10,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
11=>{let offsets=indices::<m![B=11]>(d,offsets[first..first+11].to_vec()).await;launch(old::commit_full_pool_11,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
12=>{let offsets=indices::<m![B=12]>(d,offsets[first..first+12].to_vec()).await;launch(old::commit_full_pool_12,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
13=>{let offsets=indices::<m![B=13]>(d,offsets[first..first+13].to_vec()).await;launch(old::commit_full_pool_13,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
14=>{let offsets=indices::<m![B=14]>(d,offsets[first..first+14].to_vec()).await;launch(old::commit_full_pool_14,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
15=>{let offsets=indices::<m![B=15]>(d,offsets[first..first+15].to_vec()).await;launch(old::commit_full_pool_15,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
16=>{let offsets=indices::<m![B=16]>(d,offsets[first..first+16].to_vec()).await;launch(old::commit_full_pool_16,(d,&bk,&bv,&offsets,&mut pk,&mut pv)).await.unwrap();},
_=>unreachable!(),}}}
 let offsets=indices::<m![R64]>(d,offsets.clone()).await;let mut rk=zero::<m![R64,Df]>(d).await;let mut rv=zero::<m![R64,Df]>(d).await;launch(op::gather_full_rows_r64,(d,&pk,&pv,&offsets,&mut rk,&mut rv)).await.unwrap();let mut expectedk=vec![0u16;64*512];let mut expectedv=expectedk.clone();expectedk[..count*512].copy_from_slice(&nkv[..count*512]);expectedv[..count*512].copy_from_slice(&nvv[..count*512]);exact("full K live and inactive rows",&read(d,&rk).await,&expectedk);exact("full V live and inactive rows",&read(d,&rv).await,&expectedv);
 let neighbors=indices::<m![R64]>(d,neighbors.clone()).await;launch(op::gather_full_rows_r64,(d,&pk,&pv,&neighbors,&mut rk,&mut rv)).await.unwrap();assert!(read(d,&rk).await.iter().chain(read(d,&rv).await.iter()).all(|v|*v==0),"full neighboring tokens changed");println!("PASS full count={count} exact live/inactive positions, all tile payloads and neighboring tokens");
 }
}
#[tokio::main]
async fn main(){let mut d=Device::new(op::commit_sliding_kv_r64.topology()).unwrap();test_sliding(&mut d).await;test_full(&mut d).await;println!("PASS R64 live64 and1/7/17/31/63 prefix commits, arbitraryu16 bitpatterns, inactive rows, neighbor tokens, physicalslot63/fullpage255 and cache boundaries");}

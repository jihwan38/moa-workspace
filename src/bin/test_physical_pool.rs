//! Same fixed32 live slots on physical64 control or compact32 candidate.
//! CPU affine address oracle covers every K/V cell; no model arithmetic claim.
use furiosa_opt_gemma4::{Chip,axes::*,ops_pool_views::Rc64,ops_pool_offsets as pool,ops_zero_pool as zero,ops_commit_r64 as commit,ops_request_attention::Fc};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn value(i:usize,salt:usize)->bf16 {bf16::from_bits((0x3d00+(i.wrapping_mul(73)+salt*31)%1536)as u16)}
async fn raw<Mp:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mp>)->Result<Vec<bf16>> {Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_vec())}
fn exact(label:&str,a:&[bf16],b:&[bf16]) {assert_eq!(a.len(),b.len());let dif=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();println!("PHYSICAL_POOL_EXACT label={label} values={} diff={dif}",a.len());assert_eq!(dif,0);}
#[tokio::main] async fn main()->Result<()> {
 assert!(Rc64::SIZE>=32);assert_eq!(R64::SIZE,64);assert_eq!(Fc::SIZE,256);
 let mut d=Device::new(zero::zero_sliding_pool.topology())?;
 let mut k=launch(zero::zero_sliding_pool,&mut d).await?;let mut v=launch(zero::zero_sliding_pool,&mut d).await?;
 let ring=Ts::SIZE*Ns::SIZE*Ds::SIZE;let mut ek=vec![bf16::from_bits(0);Rc64::SIZE*ring];let mut ev=ek.clone();
 exact("zero_k",&raw(&mut d,&k).await?,&ek);exact("zero_v",&raw(&mut d,&v).await?,&ev);
 let mut out=HostTensor::<bf16,m![Ts,Ns,Ds]>::zero().to_hbm(&mut d.pdma).await?;
 for s in (0..32).map(|i|(i*13+31)%32) {
  let offset=HostTensor::<i32,m![1]>::from_vec([(s*ring*2)as i32]).to_hbm(&mut d.pdma).await?;
  for which in 0..2 {
   let vals:Vec<_>=(0..ring).map(|i|value(i,s+which*101)).collect();
   let input=HostTensor::<bf16,m![Ts,Ns,Ds]>::from_vec(vals.clone()).to_hbm(&mut d.pdma).await?;
   let (target,expected)=if which==0{(&mut k,&mut ek)}else{(&mut v,&mut ev)};
   launch(pool::write_ring,(&mut d,&input,&offset,&mut *target)).await?;
   launch(pool::read_ring,(&mut d,&*target,&offset,&mut out)).await?;
   exact(&format!("slot{s}_kv{which}"),&raw(&mut d,&out).await?,&vals);
   expected[s*ring..(s+1)*ring].copy_from_slice(&vals);
  }
 }
 let sourcek:Vec<_>=(0..B::SIZE*Ns::SIZE*Ds::SIZE).map(|i|value(i,601)).collect();
 let sourcev:Vec<_>=(0..B::SIZE*Ns::SIZE*Ds::SIZE).map(|i|value(i,701)).collect();
 let sk=HostTensor::<bf16,m![B,Ns,Ds]>::from_vec(sourcek.clone()).to_hbm(&mut d.pdma).await?;
 let sv=HostTensor::<bf16,m![B,Ns,Ds]>::from_vec(sourcev.clone()).to_hbm(&mut d.pdma).await?;
 macro_rules! live {($n:literal,$fn:ident)=>{{
  let offsets:Vec<i32>=(0..$n).map(|r|{let s=(r*7+31)%32;let t=(r*131+$n*17)%Ts::SIZE;((s*Ts::SIZE+t)*Ns::SIZE*Ds::SIZE*2)as i32}).collect();
  let off=HostTensor::<i32,m![B=$n]>::from_vec(offsets.clone()).to_hbm(&mut d.pdma).await?;
  launch(commit::$fn,(&mut d,&sk,&sv,&off,&mut k,&mut v)).await?;
  for (r,&o) in offsets.iter().enumerate(){let a=o as usize/2;let b=r*Ns::SIZE*Ds::SIZE;ek[a..a+Ns::SIZE*Ds::SIZE].copy_from_slice(&sourcek[b..b+Ns::SIZE*Ds::SIZE]);ev[a..a+Ns::SIZE*Ds::SIZE].copy_from_slice(&sourcev[b..b+Ns::SIZE*Ds::SIZE]);}
 }};}
 live!(1,commit_sliding_pool_1);live!(2,commit_sliding_pool_2);live!(3,commit_sliding_pool_3);live!(4,commit_sliding_pool_4);
 live!(5,commit_sliding_pool_5);live!(6,commit_sliding_pool_6);live!(7,commit_sliding_pool_7);live!(8,commit_sliding_pool_8);
 live!(9,commit_sliding_pool_9);live!(10,commit_sliding_pool_10);live!(11,commit_sliding_pool_11);live!(12,commit_sliding_pool_12);
 live!(13,commit_sliding_pool_13);live!(14,commit_sliding_pool_14);live!(15,commit_sliding_pool_15);live!(16,commit_sliding_pool_16);
 exact("all_slots_k_after_live1to16",&raw(&mut d,&k).await?,&ek);exact("all_slots_v_after_live1to16",&raw(&mut d,&v).await?,&ev);
 exact("source_k_immutable",&raw(&mut d,&sk).await?,&sourcek);exact("source_v_immutable",&raw(&mut d,&sv).await?,&sourcev);
 // Full pages keep their original capacity, physical addresses and final page255.
 let mut full=launch(zero::zero_full_pool,&mut d).await?;let mut expected=vec![bf16::from_bits(0);Fc::SIZE*Tf::SIZE*Df::SIZE];
 let page:Vec<_>=(0..Tf::SIZE*Df::SIZE).map(|i|value(i,900)).collect();
 let input=HostTensor::<bf16,m![Tf,Df]>::from_vec(page.clone()).to_hbm(&mut d.pdma).await?;
 let offset=HostTensor::<i32,m![1]>::from_vec([(255*Tf::SIZE*Df::SIZE*2)as i32]).to_hbm(&mut d.pdma).await?;
 launch(pool::write_page,(&mut d,&input,&offset,&mut full)).await?;expected[255*Tf::SIZE*Df::SIZE..].copy_from_slice(&page);
 exact("all_full_pages_final255",&raw(&mut d,&full).await?,&expected);
 println!("PASS physical_pool={} unchanged_query64 fixed32_live_rings all_cells_zero_read_write_live_prefix_neighbors_source_full255",Rc64::SIZE);Ok(())
}

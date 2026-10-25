use furiosa_opt_gemma4::{Chip,axes::*,ops_output_r64_parallel_gather as control,ops_output_r64_trf32 as op};
use furiosa_opt_std::prelude::*;
fn hash(i:usize,seed:usize)->usize {let mut x=(i as u64).wrapping_add(seed as u64);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);(x^(x>>31))as usize}
async fn read<Mt:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mt>)->Vec<bf16>{let h:HostTensor<bf16,Mt>=x.to_host(&mut d.pdma).await.unwrap();h.into_vec()}
fn check(label:&str,a:&[bf16],b:&[bf16]){assert_eq!(a.len(),b.len());let n=a.iter().zip(b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();println!("R64_O32_CHECK {label} values={} differences={n}",a.len());assert_eq!(n,0);}
macro_rules! fixture {($d:ident,$seed:ident,$input:ty,$axis:ident,$project:ident,$partial:ident)=>{{
 let values:Vec<_>=(0..R64::SIZE*$axis::SIZE).map(|i|{
  let row=i/$axis::SIZE;let col=i%$axis::SIZE;let row=if $seed==79 {R64::SIZE-1-row}else{row};
  bf16::from_f32(if row==0 {0.0}else{((hash(row*$axis::SIZE+col,$seed)%255)as f32/128.0-127.0/128.0)*2.0f32.powi((col%9)as i32-4)})
 }).collect();
 let x=HostTensor::<bf16,$input>::from_vec(values).to_hbm(&mut $d.pdma).await.unwrap();
 let w=HostTensor::<f8e4m3,m![H,$axis]>::from_vec((0..H::SIZE*$axis::SIZE).map(|i|f8e4m3::from_f32((hash(i,$seed+739)%31)as f32/16.0-15.0/16.0)).collect::<Vec<_>>()).to_hbm(&mut $d.pdma).await.unwrap();
 let s=HostTensor::<bf16,m![H]>::from_vec((0..H::SIZE).map(|i|bf16::from_f32([0.0,0.25,-0.5,1.0,1.5,2.0,-4.0][i%7])).collect::<Vec<_>>()).to_hbm(&mut $d.pdma).await.unwrap();
 let mut out=HostTensor::<bf16,m![R64,H]>::zero().to_hbm(&mut $d.pdma).await.unwrap();
 let mut raw=HostTensor::<bf16,m![R64,$axis/1024,H]>::zero().to_hbm(&mut $d.pdma).await.unwrap();
 launch(control::$partial,(&mut $d,&x,&w,&mut raw)).await.unwrap();let expected=read(&mut $d,&raw).await;
 launch(op::$partial,(&mut $d,&x,&w,&mut raw)).await.unwrap();check(concat!(stringify!($axis)," partial"),&expected,&read(&mut $d,&raw).await);
 launch(control::$project,(&mut $d,&x,&w,&s,&mut out)).await.unwrap();let expected=read(&mut $d,&out).await;
 launch(op::$project,(&mut $d,&x,&w,&s,&mut out)).await.unwrap();check(concat!(stringify!($axis)," final"),&expected,&read(&mut $d,&out).await);
 if $seed==31 {for variant in [0,1,1,0] {let start=std::time::Instant::now();for _ in 0..20 {if variant==0 {launch(control::$project,(&mut $d,&x,&w,&s,&mut out)).await.unwrap();}else{launch(op::$project,(&mut $d,&x,&w,&s,&mut out)).await.unwrap();}}println!("R64_O32_ABBA axis={} variant={variant} ms={} scope=20completedcalls_nohostread",stringify!($axis),start.elapsed().as_secs_f64()*1000./20.);}}
}};}
#[tokio::main]
async fn main(){let mut d=Device::new(op::project_sliding_output_batch.topology()).unwrap();for seed in [31,79,103]{
 fixture!(d,seed,m![R64,Ns,Gs,Ds],Qs,project_sliding_output_batch,project_sliding_partial_dots);
 fixture!(d,seed,m![R64,Gf,Df],Qf,project_full_output_batch,project_full_partial_dots);
}println!("PASS R64 O TRF32 exact partials and outputs");}

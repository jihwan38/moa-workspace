//! Calls the unchanged production QKV helper before head RMS; no performance claim.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{axes::*, Chip, probe};
use std::io::Write;
const NAMES: [&str;9]=["signed_zero","isolated_k959","isolated_k960","isolated_k3839","multi_k_boundaries","row_group_boundaries","tiny_correlated_cancellation","scale_before_bf16","global_scale_fourth_term_active"];
fn checksum(b:&[u8])->u32 {assert_eq!(b.len()%4,0);b.chunks_exact(4).enumerate().fold(0u32,|a,(i,p)|a.wrapping_add(u32::from_le_bytes(p.try_into().unwrap()).wrapping_mul((2*i+1)as u32)))}
fn weights(case:usize,rows:usize,which:usize)->Vec<u8>{
 let local=rows/128;let mut w=Vec::with_capacity(rows*H::SIZE);
 for r in 0..rows {for k in 0..H::SIZE {
  let mut c=(0x38+((r*3+r/local+r/(rows/2)+k+k/960+which)%16))|(((r+k/960+which)%2)<<7);
  if case==5 && ![0,1,local-2,local-1].contains(&(r%local)){c=0;}
  if case==6 {c=if k%960<2 {0x38}else{0x77};}
  if case==7 {c=0x38+8*((r+which)%2);}
  if case==8 {c=if k<2 {0x38}else{0x77|(((r/(rows/2)+r%2+which)%2)<<7)};}
  w.push(c as u8);
 }}w
}
fn take<'a>(d:&'a[u8],p:&mut usize,n:usize)->&'a[u8]{let v=&d[*p..*p+n];*p+=n;v}
async fn upload<D:ScalarBytes+MaterializableScalar,T:M>(d:&mut Device,b:Vec<u8>)->HbmTensor<D,Chip,T>{assert_eq!(b.len(),T::SIZE*D::BITS/8);HostTensor::<D,T>::from_buf(b).to_hbm(&mut d.pdma).await.unwrap()}
async fn read<T:M>(d:&mut Device,t:&HbmTensor<bf16,Chip,T>)->Vec<bf16>{let h:HostTensor<bf16,T>=t.to_host(&mut d.pdma).await.unwrap();h.into_vec()}
fn compare(case:usize,key:&str,got:&[bf16],want:&[u16])->usize{
 assert_eq!(got.len(),want.len());let(mut fail,mut nonfinite)=(0,0);
 for(r,(&g,&w))in got.iter().zip(want).enumerate(){let b=g.to_bits();let equal=b==w||((b&0x7fff)==0&&(w&0x7fff)==0);if !g.to_f32().is_finite(){nonfinite+=1;}if !equal{if fail<16{println!("RAW_MISMATCH case={case} output={key} row={r} got={b:04x} want={w:04x}");}fail+=1;}}
 println!("RAW_RESULT case={case} name={} output={key} checked={} failures={fail} nonfinite={nonfinite} status={}",NAMES[case],got.len(),if fail+nonfinite==0{"PASS"}else{"FAIL"});fail+nonfinite
}
#[tokio::main]
async fn main(){
 let args:Vec<_>=std::env::args().collect();let host_only=args.iter().any(|a|a=="--host-check-only");let dump=args.windows(2).find(|a|a[0]=="--dump-generated").map(|a|a[1].clone());
 let path=args.windows(2).find(|a|a[0]=="--fixture").map(|a|a[1].as_str()).unwrap_or("probe-fixture.bin");
 if let Some(p)=&dump{assert!(host_only);std::fs::create_dir_all(p).unwrap();}
 let data=std::fs::read(path).unwrap();assert_eq!(&data[..8],b"E138RAW1");let mut p=8;
 let mut device=if host_only{None}else{Some(Device::new(probe::qkv_before_head_rms.topology()).unwrap())};let(mut completed,mut failed)=(0,0);
 println!("RAW_BEGIN cases=9 values_per_case=8192 bitwise_bf16=true signed_zeros_equivalent=true host_only={host_only} performance=false");
 for case in 0..9 {
  let checks:Vec<_>=take(&data,&mut p,28).chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
  let x=take(&data,&mut p,H::SIZE*2).to_vec();let qs=take(&data,&mut p,Qs::SIZE*2).to_vec();let ks=take(&data,&mut p,Ps::SIZE*2).to_vec();let vs=take(&data,&mut p,Ps::SIZE*2).to_vec();
  let mut want=Vec::new();for rows in [Qs::SIZE,Ps::SIZE,Ps::SIZE]{want.push(take(&data,&mut p,rows*2).chunks_exact(2).map(|b|u16::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>());}
  let qw=weights(case,Qs::SIZE,0);let kw=weights(case,Ps::SIZE,1);let vw=weights(case,Ps::SIZE,2);
  for(i,(n,b))in[("x",&x),("q_scale",&qs),("k_scale",&ks),("v_scale",&vs),("q_weight",&qw),("k_weight",&kw),("v_weight",&vw)].into_iter().enumerate(){let c=checksum(b);assert_eq!(c,checks[i],"case {case} input {n}");println!("RAW_INPUT case={case} name={n} bytes={} checksum={c:08x} status=PASS",b.len());if let Some(dir)=&dump{std::fs::write(format!("{dir}/case{case}.{n}.bin"),b).unwrap();}}
  if host_only{completed+=1;continue;}
  let d=device.as_mut().unwrap();
  let x:HbmTensor<bf16,Chip,m![H]>=upload(d,x).await;
  let qs:HbmTensor<bf16,Chip,m![Qs]>=upload(d,qs).await;let ks:HbmTensor<bf16,Chip,m![Ps]>=upload(d,ks).await;let vs:HbmTensor<bf16,Chip,m![Ps]>=upload(d,vs).await;
  let qw:HbmTensor<f8e4m3,Chip,m![Qs,H]>=upload(d,qw).await;let kw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(d,kw).await;let vw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(d,vw).await;
  let mut qo:HbmTensor<bf16,Chip,m![Ns,Gs,Ds]>=upload(d,[0xc1u8,0x7f].repeat(Qs::SIZE)).await;
  let mut ko:HbmTensor<bf16,Chip,m![Ns,Ds]>=upload(d,[0xc1u8,0x7f].repeat(Ps::SIZE)).await;let mut vo:HbmTensor<bf16,Chip,m![Ns,Ds]>=upload(d,[0xc1u8,0x7f].repeat(Ps::SIZE)).await;
  println!("RAW_LAUNCH case={case} name={}",NAMES[case]);std::io::stdout().flush().unwrap();
  launch(probe::qkv_before_head_rms,(&mut *d,&x,&qw,&kw,&vw,&qs,&ks,&vs,&mut qo,&mut ko,&mut vo)).await.expect("actual QKV projection helper launch");
  failed+=compare(case,"q",&read(d,&qo).await,&want[0]);failed+=compare(case,"k",&read(d,&ko).await,&want[1]);failed+=compare(case,"v",&read(d,&vo).await,&want[2]);completed+=1;
 }
 assert_eq!(p,data.len());println!("RAW_COMPLETE completed={completed} planned=9 failed_values={failed} host_only={host_only} status={}",if failed==0{"PASS"}else{"FAIL"});if failed!=0{std::process::exit(1);}
}

//! Full four-input QKV comparison, including all real joins and stores.
#[path="support/npu_profile.rs"] mod npu_profile;
thread_local! {static PROFILE:npu_profile::Collector=npu_profile::Collector::default();}
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip,axes::*,ops_qkv_weight_stationary as op,ops_qkv_shared_ingress as control};
fn hash(mut v:u64)->u64 {v=v.wrapping_add(0x9e3779b97f4a7c15);v=(v^(v>>30)).wrapping_mul(0xbf58476d1ce4e5b9);v=(v^(v>>27)).wrapping_mul(0x94d049bb133111eb);v^(v>>31)}
fn value(i:usize,seed:usize,broad:bool)->f8e4m3 {let h=hash(i as u64+(seed as u64)*719);if broad {f8e4m3::from_bits(h as u8 & 0xfe)}else{f8e4m3::from_f32((h%17)as f32/8.0-1.0)}}
fn blocked(row:usize,col:usize,width:usize,tile:usize)->usize {((row/64*(width/tile)+col/tile)*64+row%64)*tile+col%tile}
fn provided<E:M>(layer:usize,name:&str)->(Vec<f8e4m3>,Vec<bf16>){
 let file=std::fs::File::open(furiosa_opt_gemma4::host::load::model_dir().unwrap().join("model.safetensors")).unwrap();let map=unsafe{memmap2::Mmap::map(&file).unwrap()};let s=safetensors::SafeTensors::deserialize(&map).unwrap();let p=format!("model.language_model.layers.{layer}.self_attn.{name}_proj");
 let w=HostTensor::<f8e4m3,m![E,H]>::from_safetensors(&s.tensor(&format!("{p}.weight")).unwrap()).unwrap().into_vec();let scale=s.tensor(&format!("{p}.weight_scale")).unwrap();assert_eq!(scale.dtype(),safetensors::Dtype::BF16);assert_eq!(scale.shape(),[E::SIZE,1]);(w,HostTensor::<bf16,E>::from_buf(scale.data().to_vec()).into_vec())
}
async fn read_bf<E:M>(d:&mut Device,t:&HbmTensor<bf16,Chip,E>)->Vec<bf16>{let v:HostTensor<bf16,E>=t.to_host(&mut d.pdma).await.unwrap();v.into_vec()}
async fn immutable_e4<E:M>(d:&mut Device,t:&HbmTensor<f8e4m3,Chip,E>,expected:&[f8e4m3]){let v:HostTensor<f8e4m3,E>=t.to_host(&mut d.pdma).await.unwrap();assert!(v.into_vec().iter().zip(expected).all(|(a,b)|a.to_bits()==b.to_bits()));}
async fn poison<E:M>(d:&mut Device)->HbmTensor<bf16,Chip,E>{HostTensor::<bf16,E>::from_vec(vec![bf16::from_f32(f32::NAN);E::SIZE]).to_hbm(&mut d.pdma).await.unwrap()}
macro_rules! call {($d:ident,$f:path,$x:ident,$xs:ident,$qw:ident,$kw:ident,$vw:ident,$qs:ident,$ks:ident,$vs:ident,$out:ident,$order:ident)=>{{
 let [a,b,c,e]=$out.as_mut_slice() else{unreachable!()};
 launch($f,(&mut $d,&$x[$order[0]],&$xs[$order[0]],&$x[$order[1]],&$xs[$order[1]],&$x[$order[2]],&$xs[$order[2]],&$x[$order[3]],&$xs[$order[3]],&$qw,&$kw,&$vw,&$qs,&$ks,&$vs,(&mut a.0,&mut a.1,&mut a.2),(&mut b.0,&mut b.1,&mut b.2),(&mut c.0,&mut c.1,&mut c.2),(&mut e.0,&mut e.1,&mut e.2))).await.unwrap();
}};}
async fn fixture(case:usize){
 let mut d=Device::new(op::project_sliding_qkv_four.topology()).unwrap();
 let synthetic=|n:usize,seed:usize|( (0..n*3840).map(|i|value(i,seed,case==2)).collect::<Vec<_>>(),(0..n).map(|i|bf16::from_f32([0.0,0.5,-0.5,1.0,-1.0,1.5,2.0][i%7])).collect::<Vec<_>>());
 let (cq,cqs)=if case<3{synthetic(4096,case+7)}else{provided::<Qs>([0,6,46][case-3],"q")};
 let (ck,cks)=if case<3{synthetic(2048,case+23)}else{provided::<Ps>([0,6,46][case-3],"k")};
 let (cv,cvs)=if case<3{synthetic(2048,case+47)}else{provided::<Ps>([0,6,46][case-3],"v")};
 let qw=HostTensor::<f8e4m3,m![Qs,H]>::from_vec(cq.clone()).to_hbm(&mut d.pdma).await.unwrap();let kw=HostTensor::<f8e4m3,m![Ps,H]>::from_vec(ck.clone()).to_hbm(&mut d.pdma).await.unwrap();let vw=HostTensor::<f8e4m3,m![Ps,H]>::from_vec(cv.clone()).to_hbm(&mut d.pdma).await.unwrap();
 let qs=HostTensor::<bf16,m![Qs]>::from_vec(cqs.clone()).to_hbm(&mut d.pdma).await.unwrap();let ks=HostTensor::<bf16,m![Ps]>::from_vec(cks.clone()).to_hbm(&mut d.pdma).await.unwrap();let vs=HostTensor::<bf16,m![Ps]>::from_vec(cvs.clone()).to_hbm(&mut d.pdma).await.unwrap();
 let mut x=Vec::new();let mut xs=Vec::new();let mut cx=Vec::new();let mut cxs=Vec::new();let mut out=Vec::new();let mut reference=Vec::new();
 for g in 0..4 {let v=(0..128*3840).map(|i|if i/3840==0 {f8e4m3::from_f32(0.0)}else{value(i,case*131+g*19+73,case==2)}).collect::<Vec<_>>();let mut scales=vec![0f32;128*8];for row in 0..128{scales[row*8]=[0.0,0.00390625,0.03125,0.5,1.0,2.0,8.0][(row+g)%7];}
  x.push(HostTensor::<f8e4m3,m![P128,H]>::from_vec(v.clone()).to_hbm(&mut d.pdma).await.unwrap());xs.push(HostTensor::<f32,m![P128,1#8]>::from_vec(scales.clone()).to_hbm(&mut d.pdma).await.unwrap());cx.push(v);cxs.push(scales);
  out.push((poison::<op::QueryOutput>(&mut d).await,poison::<op::KeyOutput>(&mut d).await,poison::<op::KeyOutput>(&mut d).await));reference.push((poison::<control::QueryOutput>(&mut d).await,poison::<control::KeyOutput>(&mut d).await,poison::<control::KeyOutput>(&mut d).await));
 }
 for order in [[0usize,1,2,3],[3,1,0,2]] {
  call!(d,op::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,out,order);
  call!(d,control::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,reference,order);
  let mut differences=0usize;let mut samples=0usize;
  for g in 0..4 {
   let got=[read_bf(&mut d,&out[g].0).await,read_bf(&mut d,&out[g].1).await,read_bf(&mut d,&out[g].2).await];let expected=[read_bf(&mut d,&reference[g].0).await,read_bf(&mut d,&reference[g].1).await,read_bf(&mut d,&reference[g].2).await];
   for kind in 0..3 {for (i,(a,b)) in got[kind].iter().zip(&expected[kind]).enumerate(){if a.to_bits()!=b.to_bits(){if differences<8{println!("QKV_DIFF case={case} group={g} kind={kind} physical={i} got={} control={}",a.to_f32(),b.to_f32());}differences+=1;}}
    if case<2 {let width=if kind==0{4096}else{2048};let tile=if kind==0{64}else{32};let weights=[&cq,&ck,&cv][kind];let scales=[&cqs,&cks,&cvs][kind];for row in [0,1,17,63,64,127]{for col in [0,1,7,8,31,32,width/2-1,width/2,width-1]{let mut partial=[0f32;4];for block in 0..4{let mut sum=0f64;for k in block*960..(block+1)*960{sum+=cx[order[g]][row*3840+k].to_f32() as f64*weights[col*3840+k].to_f32() as f64;}partial[block]=sum as f32;}let sum=((partial[0]+partial[1])+partial[2])+partial[3];let dot=bf16::from_f32(sum*cxs[order[g]][row*8]);let result=bf16::from_f32(dot.to_f32()*scales[col].to_f32());assert_eq!(got[kind][blocked(row,col,width,tile)].to_bits(),result.to_bits(),"CPU case={case} group={g} kind={kind} row={row} col={col}");samples+=1;}}}
   }
  }
  println!("QKV_ORACLE case={case} order={order:?} values={} bitdiff={differences} independent_f64={samples}",4*128*8192);assert_eq!(differences,0);
 }
 let order=[0usize,1,2,3];let mut times=[Vec::new(),Vec::new()];
 for which in [0,1,1,0,0,1,1,0]{let now=std::time::Instant::now();for _ in 0..8{if which==0{call!(d,control::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,reference,order);}else{call!(d,op::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,out,order);}}times[which].push(now.elapsed().as_secs_f64()*1e6/8.0);}
 println!("QKV_TIMING case={case} control_us={:?} stationary_us={:?}",times[0],times[1]);
 for which in 0..2{println!("QKV_PROFILE_BEGIN case={case} which={which}");PROFILE.with(|p|p.enable(true));if which==0{call!(d,control::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,reference,order);}else{call!(d,op::project_sliding_qkv_four,x,xs,qw,kw,vw,qs,ks,vs,out,order);}PROFILE.with(|p|p.enable(false));println!("QKV_PROFILE_END case={case} which={which}");}
 immutable_e4(&mut d,&qw,&cq).await;immutable_e4(&mut d,&kw,&ck).await;immutable_e4(&mut d,&vw,&cv).await;
 assert!(read_bf(&mut d,&qs).await.iter().zip(&cqs).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(read_bf(&mut d,&ks).await.iter().zip(&cks).all(|(a,b)|a.to_bits()==b.to_bits()));assert!(read_bf(&mut d,&vs).await.iter().zip(&cvs).all(|(a,b)|a.to_bits()==b.to_bits()));
 for g in 0..4{immutable_e4(&mut d,&x[g],&cx[g]).await;let v:HostTensor<f32,m![P128,1#8]>=xs[g].to_host(&mut d.pdma).await.unwrap();assert!(v.into_vec().iter().zip(&cxs[g]).all(|(a,b)|a.to_bits()==b.to_bits()));}
 println!("QKV_CASE_PASS case={case} immutable_weights_inputs_scales=true");
}
#[tokio::main(flavor="current_thread")]
async fn main(){let p=PROFILE.with(Clone::clone);tracing::subscriber::set_global_default(p).unwrap();let args=std::env::args().collect::<Vec<_>>();let start:usize=args.get(1).map_or("0",String::as_str).parse().unwrap();let end:usize=args.get(2).map_or("6",String::as_str).parse().unwrap();assert!(start<end&&end<=6);for case in start..end{fixture(case).await;}println!("PASS exact current fourP128 QKV vs retained joined weight TRF; fullmodel pending");}

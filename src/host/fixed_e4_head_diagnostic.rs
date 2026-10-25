//! Same-input proof for every active sliding K/V producer; no timing claim.
use crate::{Chip,axes::*,ops_fixed_e4_quality as quality,ops_fixed_e4_quality_diagnostic as diag,
 ops_head_batch as head,ops_pool_views::Rc64};
use super::{load,fixed_e4_codec as codec};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
const POS:[usize;16]=[0,1,15,16,127,255,511,512,896,1023,1024,2047,4095,8191,16383,32767];
const Q:usize=4096; const K:usize=2048;
async fn ub<Mp:M>(d:&mut Device,bytes:&[u8])->Result<HbmTensor<bf16,Chip,Mp>>{
 assert_eq!(bytes.len(),Mp::SIZE*2);Ok(HostTensor::<bf16,Mp>::from_buf(bytes.to_vec()).to_hbm(&mut d.pdma).await?)
}
async fn rb<Mp:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,Mp>)->Result<Vec<u8>>{
 Ok(x.to_host::<Mp>(&mut d.pdma).await?.into_inner().into_buf())
}
async fn poison<Mp:M>(d:&mut Device)->Result<HbmTensor<bf16,Chip,Mp>>{ub(d,&[0xc1,0x7f].repeat(Mp::SIZE)).await}
async fn offsets<Mp:M>(d:&mut Device,rows:usize)->Result<HbmTensor<i32,Chip,Mp>>{
 Ok(HostTensor::<i32,Mp>::from_vec((0..rows).map(|r|(POS[r%16]*256*2)as i32)).to_hbm(&mut d.pdma).await?)
}
fn random(n:usize,seed:u64)->Vec<u8>{(0..n).flat_map(|i|{
 let mut x=(i as u64).wrapping_add(seed*997);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^=x>>31;
 let v=if i%257==0{0.0}else if i%257==1{-0.0}else if i%257==2{1e-20}else{((x>>40)as f32/8388608.-1.)*4.};
 bf16::from_f32(v).to_bits().to_le_bytes()}).collect()}
fn exact(label:&str,a:&[u8],b:&[u8])->usize{
 assert_eq!(a.len(),b.len(),"{label}");let diffs=a.iter().zip(b).filter(|(a,b)|a!=b).count();
 assert_eq!(diffs,0,"{label}: first={:?}",a.iter().zip(b).position(|(a,b)|a!=b));a.len()
}
fn expected(bytes:&[u8],lut:&[u8])->Vec<u8>{if quality::ENABLED{codec::decode_bytes(&codec::encode_bytes(bytes,lut))}else{bytes.to_vec()}}
fn block(bytes:&[u8],rows:usize,channels:usize,rtile:usize,ctile:usize)->Vec<u8>{
 assert_eq!(bytes.len(),rows*channels*2);let mut out=Vec::with_capacity(bytes.len());
 for ro in 0..rows/rtile{for co in 0..channels/ctile{for r in 0..rtile{for c in 0..ctile{
 let i=((ro*rtile+r)*channels+co*ctile+c)*2;out.extend_from_slice(&bytes[i..i+2]);}}}}out
}
struct Weights{
 qw:HbmTensor<f8e4m3,Chip,m![Qs,H]>,kw:HbmTensor<f8e4m3,Chip,m![Ps,H]>,vw:HbmTensor<f8e4m3,Chip,m![Ps,H]>,
 qs:HbmTensor<bf16,Chip,m![Qs]>,ks:HbmTensor<bf16,Chip,m![Ps]>,vs:HbmTensor<bf16,Chip,m![Ps]>,
 norm:HbmTensor<bf16,Chip,m![H]>,qn:HbmTensor<bf16,Chip,m![Ds]>,kn:HbmTensor<bf16,Chip,m![Ds]>,
 cos:HbmTensor<bf16,Chip,m![E,Ds]>,sin:HbmTensor<bf16,Chip,m![E,Ds]>,
}
impl Weights{
 async fn new(d:&mut Device,layer:usize)->Result<Self>{
 let f=std::fs::File::open(load::model_dir()?.join("model.safetensors"))?;
 let map=unsafe{memmap2::Mmap::map(&f)?};let tensors=safetensors::SafeTensors::deserialize(&map)?;
 let prefix=format!("model.language_model.layers.{layer}");
 macro_rules! tensor{($dtype:ty,$shape:ty,$suffix:expr)=>{{
  let t=tensors.tensor(&format!("{prefix}.{}",$suffix))?;
  HostTensor::<$dtype,$shape>::from_safetensors(&t)?.to_hbm(&mut d.pdma).await?
 }}}
 // Weight-scale columns use their exact underlying BF16 bytes, not a shape reinterpret.
 macro_rules! column{($shape:ty,$suffix:expr)=>{{let t=tensors.tensor(&format!("{prefix}.{}",$suffix))?;ub::<$shape>(d,t.data()).await?}}}
 let mut cos=vec![0u8;E::SIZE*256*2];let mut sin=cos.clone();
 for &p in &POS{for c in 0..256{let j=c%128;let a=p as f32/10000f32.powf(2.*j as f32/256.);
  let cb=bf16::from_f32(a.cos()).to_bits().to_le_bytes();let sb=bf16::from_f32(a.sin()*if c<128{-1.}else{1.}).to_bits().to_le_bytes();
  let i=(p*256+c)*2;cos[i..i+2].copy_from_slice(&cb);sin[i..i+2].copy_from_slice(&sb);
 }}
 Ok(Self{qw:tensor!(f8e4m3,m![Qs,H],"self_attn.q_proj.weight"),kw:tensor!(f8e4m3,m![Ps,H],"self_attn.k_proj.weight"),vw:tensor!(f8e4m3,m![Ps,H],"self_attn.v_proj.weight"),
 qs:column!(m![Qs],"self_attn.q_proj.weight_scale"),ks:column!(m![Ps],"self_attn.k_proj.weight_scale"),vs:column!(m![Ps],"self_attn.v_proj.weight_scale"),
 norm:tensor!(bf16,m![H],"input_layernorm.weight"),qn:tensor!(bf16,m![Ds],"self_attn.q_norm.weight"),kn:tensor!(bf16,m![Ds],"self_attn.k_norm.weight"),cos:ub(d,&cos).await?,sin:ub(d,&sin).await?})
 }
}
async fn codec_shapes(d:&mut Device,lut:&[u8])->Result<usize>{
 let mut bytes=0;
 macro_rules! gate{($axis:ident,$kernel:path,$salt:expr)=>{{
 let data:Vec<u8>=(0..$axis::SIZE*K).flat_map(|i|{let mut b=((i*73+$salt)%65536)as u16;if b&0x7f80==0x7f80{b=0;}b.to_le_bytes()}).collect();
 let x=ub::<m![$axis,Ns,Ds]>(d,&data).await?;let mut y=poison::<m![$axis,Ns,Ds]>(d).await?;
 launch($kernel,(d,&x,&mut y)).await?;bytes+=exact(stringify!($kernel),&rb(d,&y).await?,&codec::decode_bytes(&codec::encode_bytes(&data,lut)));
 bytes+=exact("codec input immutable",&rb(d,&x).await?,&data);
 }}}
 gate!(B,diag::codec_b16,0);gate!(B,diag::codec_b16,32768);gate!(R64,diag::codec_r64,19);gate!(P128,diag::codec_p128,37);
 println!("FIXED_E4_CODEC_SHAPES bytes={bytes} finite_only=true source_flag={} unconditional_helper=true",quality::ENABLED);Ok(bytes)
}
async fn heads(d:&mut Device,w:&Weights,lut:&[u8])->Result<usize>{
 let q= random(128*Q,11);let k=random(128*K,17);let v=random(128*K,23);
 let(mut eq,mut ek,mut ev)=(Vec::new(),Vec::new(),Vec::new());
 for r in 0..128{
  let qr=ub::<m![Qs]>(d,&q[r*Q*2..(r+1)*Q*2]).await?;let kr=ub::<m![Ps]>(d,&k[r*K*2..(r+1)*K*2]).await?;let vr=ub::<m![Ps]>(d,&v[r*K*2..(r+1)*K*2]).await?;
  let off=HostTensor::<i32,m![1]>::from_vec(vec![(POS[r%16]*256*2)as i32]).to_hbm(&mut d.pdma).await?;
  let mut qo=poison::<m![Ns,Gs,Ds]>(d).await?;let mut ko=poison::<m![Ns,Ds]>(d).await?;let mut vo=poison::<m![Ns,Ds]>(d).await?;
  launch(head::reference_sliding_head_row,(d,&qr,&kr,&vr,&w.qn,&w.kn,&off,&w.cos,&w.sin,&mut qo,&mut ko,&mut vo)).await?;
  eq.extend(rb(d,&qo).await?);ek.extend(rb(d,&ko).await?);ev.extend(rb(d,&vo).await?);
 }
 let ek=expected(&ek,lut);let ev=expected(&ev,lut);let mut checked=0;
 macro_rules! check_head{($axis:ident,$kernel:path,$qr:expr,$kr:expr,$vr:expr,$base:expr)=>{{
  let off=offsets::<m![$axis]>(d,$axis::SIZE).await?;
  let mut qo=poison::<m![$axis,Ns,Gs,Ds]>(d).await?;let mut ko=poison::<m![$axis,Ns,Ds]>(d).await?;let mut vo=poison::<m![$axis,Ns,Ds]>(d).await?;
  launch($kernel,(d,$qr,$kr,$vr,&w.qn,&w.kn,&off,&w.cos,&w.sin,&mut qo,&mut ko,&mut vo)).await?;
  let a=$base;let b=a+$axis::SIZE;
  checked+=exact(concat!(stringify!($kernel)," Q"),&rb(d,&qo).await?,&eq[a*Q*2..b*Q*2]);
  checked+=exact(concat!(stringify!($kernel)," K"),&rb(d,&ko).await?,&ek[a*K*2..b*K*2]);
  checked+=exact(concat!(stringify!($kernel)," V"),&rb(d,&vo).await?,&ev[a*K*2..b*K*2]);
  println!("FIXED_E4_HEAD_PASS name={} rows={} byte_count={} enabled={}",stringify!($kernel),$axis::SIZE,checked,quality::ENABLED);
 }}}
 let qb=ub::<m![B,Qs]>(d,&q[..16*Q*2]).await?;let kb=ub::<m![B,Ps]>(d,&k[..16*K*2]).await?;let vb=ub::<m![B,Ps]>(d,&v[..16*K*2]).await?;
 check_head!(B,head::sliding_heads_b16,&qb,&kb,&vb,0);
 let qr=ub::<m![R64,Qs]>(d,&q[..64*Q*2]).await?;let kr=ub::<m![R64,Ps]>(d,&k[..64*K*2]).await?;let vr=ub::<m![R64,Ps]>(d,&v[..64*K*2]).await?;
 check_head!(R64,crate::ops_head_r64::sliding_heads_r64,&qr,&kr,&vr,0);
 let qp=ub::<m![P128,Qs]>(d,&q).await?;let kp=ub::<m![P128,Ps]>(d,&k).await?;let vp=ub::<m![P128,Ps]>(d,&v).await?;
 check_head!(P128,crate::ops_head_wide::sliding_heads_p128,&qp,&kp,&vp,0);
 let bq=ub::<crate::ops_qkv_blocked_b16::QueryOutput>(d,&block(&q[..16*Q*2],16,Q,8,64)).await?;
 let bk=ub::<crate::ops_qkv_blocked_b16::KeyOutput>(d,&block(&k[..16*K*2],16,K,8,32)).await?;
 let bv=ub::<crate::ops_qkv_blocked_b16::KeyOutput>(d,&block(&v[..16*K*2],16,K,8,32)).await?;
 check_head!(B,crate::ops_head_b16_blocked::sliding_heads_b16,&bq,&bk,&bv,0);
 let rq=ub::<crate::ops_qkv_blocked_r64::QueryOutput>(d,&block(&q[..64*Q*2],64,Q,32,64)).await?;
 let rk=ub::<crate::ops_qkv_blocked_r64::KeyOutput>(d,&block(&k[..64*K*2],64,K,32,32)).await?;
 let rv=ub::<crate::ops_qkv_blocked_r64::KeyOutput>(d,&block(&v[..64*K*2],64,K,32,32)).await?;
 check_head!(R64,crate::ops_head_r64_blocked::sliding_heads_r64,&rq,&rk,&rv,0);
 let pq=ub::<crate::ops_qkv_shared_blocked::QueryOutput>(d,&block(&q,128,Q,64,64)).await?;
 let pk=ub::<crate::ops_qkv_shared_blocked::KeyOutput>(d,&block(&k,128,K,64,32)).await?;
 let pv=ub::<crate::ops_qkv_shared_blocked::KeyOutput>(d,&block(&v,128,K,64,32)).await?;
 check_head!(P128,crate::ops_head_wide_blocked::sliding_heads_p128,&pq,&pk,&pv,0);
 check_head!(P128,crate::ops_prefill_head_attn_commit::headed_outputs,&pq,&pk,&pv,0);
 let tile=HostTensor::<i32,m![1]>::from_vec(vec![3]).to_hbm(&mut d.pdma).await?;
 let off=offsets::<m![B]>(d,16).await?;
 let mut qo=poison::<m![B,Ns,Gs,Ds]>(d).await?;let mut ko=poison::<m![B,Ns,Ds]>(d).await?;let mut vo=poison::<m![B,Ns,Ds]>(d).await?;
 launch(crate::ops_head_wide_tile::sliding_heads_from_wide_tile,(d,&qp,&kp,&vp,&tile,&w.qn,&w.kn,&off,&w.cos,&w.sin,&mut qo,&mut ko,&mut vo)).await?;
 checked+=exact("tile Q",&rb(d,&qo).await?,&eq[48*Q*2..64*Q*2]);checked+=exact("tile K",&rb(d,&ko).await?,&ek[48*K*2..64*K*2]);checked+=exact("tile V",&rb(d,&vo).await?,&ev[48*K*2..64*K*2]);
 checked+=exact("raw Q immutable",&rb(d,&qp).await?,&q)+exact("raw K immutable",&rb(d,&kp).await?,&k)+exact("raw V immutable",&rb(d,&vp).await?,&v);
 println!("PASS FIXED_E4_HEADS producers=8 checked_bytes={checked} scalar_reference_unchanged=true Q_exact=true KV_cpu_codec_exact=true enabled={}",quality::ENABLED);Ok(checked)
}
async fn scalar(d:&mut Device,w:&Weights,lut:&[u8])->Result<usize>{
 let xbytes=random(H::SIZE,97);let x=ub::<m![H]>(d,&xbytes).await?;let mut checked=0;
 macro_rules! scalar_gate{($shape:ty,$candidate:path,$reference:path,$slots:expr)=>{{
 let seed=[0x00,0x3e].repeat(<$shape>::SIZE);let mut ck=ub::<$shape>(d,&seed).await?;let mut cv=ub::<$shape>(d,&seed).await?;let mut rk=ub::<$shape>(d,&seed).await?;let mut rv=ub::<$shape>(d,&seed).await?;
 for (slot,pos) in $slots{
  let offset=((slot*1024+pos%1024)*K*2)as i32;let off=HostTensor::<i32,m![1]>::from_vec(vec![offset]).to_hbm(&mut d.pdma).await?;
  let rope=HostTensor::<i32,m![1]>::from_vec(vec![(pos*256*2)as i32]).to_hbm(&mut d.pdma).await?;
  let mut cq=poison::<m![Ns,Gs,Ds]>(d).await?;let mut rq=poison::<m![Ns,Gs,Ds]>(d).await?;
  launch($candidate,(d,&x,&w.qw,&w.kw,&w.vw,&w.qs,&w.ks,&w.vs,&w.norm,&w.qn,&w.kn,&off,&rope,&w.cos,&w.sin,&mut ck,&mut cv,&mut cq)).await?;
  let gotk=rb(d,&ck).await?;let gotv=rb(d,&cv).await?;let gotq=rb(d,&cq).await?;
  launch($reference,(d,&x,&w.qw,&w.kw,&w.vw,&w.qs,&w.ks,&w.vs,&w.norm,&w.qn,&w.kn,&off,&rope,&w.cos,&w.sin,&mut rk,&mut rv,&mut rq)).await?;
  let mut ek=rb(d,&rk).await?;let mut ev=rb(d,&rv).await?;
  // Every previously written location must be rounded too; seed is exactly E4.
  if quality::ENABLED{ek=expected(&ek,lut);ev=expected(&ev,lut);}
  checked+=exact("scalar complete K",&gotk,&ek)+exact("scalar complete V",&gotv,&ev)+exact("scalar Q",&gotq,&rb(d,&rq).await?);
  println!("FIXED_E4_SCALAR_PASS kernel={} slot={slot} position={pos} checked_bytes={checked}",stringify!($candidate));
 }
 }}}
 scalar_gate!(m![Rc64,Ts,Ns,Ds],crate::ops_cache_pools::sliding_project_qkv,diag::scalar_canonical,[(31usize,1023usize),(0,0),(31,1024)]);
 scalar_gate!(m![Ts,Ns,Ds],crate::ops::sliding_project_qkv,diag::scalar_legacy,[(0usize,1023usize),(0,0),(0,1024)]);
 checked+=exact("scalar input immutable",&rb(d,&x).await?,&xbytes);
 println!("PASS FIXED_E4_SCALAR producers=2 checked_bytes={checked} enabled={}",quality::ENABLED);Ok(checked)
}
pub async fn run()->Result<()>{
 let args:Vec<_>=std::env::args().skip(1).collect();let mode=args.first().map(String::as_str).unwrap_or("heads");let layer=args.get(1).map(|s|s.parse::<usize>()).transpose()?.unwrap_or(0);
 assert!([0,6,46].contains(&layer));let mut d=Device::new(diag::codec_b16.topology())?;let lut=codec::table();
 let w=Weights::new(&mut d,layer).await?;
 let bytes=match mode{"heads"=>codec_shapes(&mut d,&lut).await?+heads(&mut d,&w,&lut).await?,"scalar"=>scalar(&mut d,&w,&lut).await?,_=>return Err("mode heads|scalar".into())};
 println!("PASS FIXED_E4_HEAD_CODEC mode={mode} layer={layer} bytes={bytes} enabled={} no_compression_timing=true",quality::ENABLED);Ok(())
}

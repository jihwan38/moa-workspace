//! Intended fixed-code parity and approximate current-E4 complete P512 oracle.
use furiosa_opt_std::prelude::*;
use crate::{Chip,EPS,axes::{H,L,P128,P512},ops_batch_layouts::{P512H960,P512L960},
 ops_norm_blocked_residual::Residual, ops_native_fp8_p512_downblocked::Output,
 ops_output_p128_h64_blocked::Output as SlidingOutput,ops_norm_blocked_chain as norm,
 ops_group_ffn_projection_storegroup as original,ops_group_ffn_f4_projection as candidate,
 ops_group_ffn_f4_projection_dense as intended,ops_ffn_f4_projection as k,
 ops_ffn_projection_clusters::{JoinedWeight,JoinedScale,JoinedOutput}};
use super::{load::{self,Layer,MlpWeights},f4_group960::{quantize_blocked_e4,PackedF4Groups}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
type Hidden=HbmTensor<bf16,Chip,m![P128,H]>;
fn residual_index(p:usize,h:usize)->usize {
    (((((p/64%2)*32+p/2%32)*8+h/480)*4+p/128)*2+p%2)*480+h%480
}
fn output_index(p:usize,h:usize)->usize {(((p/256*16+h/240)*256+p%256)*240)+h%240}
fn sliding_index(p:usize,h:usize)->usize {(((p/64*60+h/64)*64+p%64)*64)+h%64}
fn perm_row(row:usize,live:usize,permuted:bool)->usize {
 if permuted && row<live {(row*73+127)%live}else{row}
}
fn data(seed:usize,kind:usize,live:usize,permuted:bool)->Vec<bf16> {
 (0..P512::SIZE*H::SIZE).map(|i|{let row=i/H::SIZE;let col=i%H::SIZE;
 let mapped=perm_row(row,live,permuted);let j=mapped*H::SIZE+col;
 let x=((j*73+seed*131+kind*19)%1009)as f32/504.-1.;
 bf16::from_f32(if mapped>=live||mapped%128==0 {0.}else if mapped%128==1 {
 if col%2==0 {1e-8}else{-1e-8}}else{x*2f32.powi((mapped%7)as i32-3)})}).collect()
}
async fn upload<D:MaterializableScalar,Mp:M>(d:&mut Device,bytes:Vec<u8>)->Result<HbmTensor<D,Chip,Mp>> {
    assert_eq!(bytes.len()*8,Mp::SIZE*D::BITS);
    Ok(HostTensor::<D,Mp>::from_buf(bytes).to_hbm(&mut d.pdma).await?)
}
fn raw(values:&[bf16])->Vec<u8> {values.iter().flat_map(|x|x.to_bits().to_le_bytes()).collect()}
fn packed_residual(v:&[bf16])->Vec<u8> {
    let mut bytes=vec![0u8;P512::SIZE*H::SIZE*2];let mut seen=vec![false;P512::SIZE*H::SIZE];
    for p in 0..P512::SIZE {for h in 0..H::SIZE {let at=residual_index(p,h);assert!(!seen[at]);seen[at]=true;bytes[2*at..2*at+2].copy_from_slice(&v[p*H::SIZE+h].to_bits().to_le_bytes());}}
    assert!(seen.iter().all(|x|*x));bytes
}
struct Inputs {canonical:[Hidden;4],sliding:[HbmTensor<bf16,Chip,SlidingOutput>;4],raw_canonical:[Vec<u8>;4],raw_sliding:[Vec<u8>;4]}
impl Inputs {
    async fn new(d:&mut Device,seed:usize,live:usize,permuted:bool)->Result<Self>{
        let values=data(seed,0,live,permuted);
        let raw_canonical=std::array::from_fn(|q|raw(&values[q*P128::SIZE*H::SIZE..(q+1)*P128::SIZE*H::SIZE]));
        let raw_sliding=std::array::from_fn(|q|{let mut b=vec![0;P128::SIZE*H::SIZE*2];let mut seen=vec![false;P128::SIZE*H::SIZE];
            for p in 0..P128::SIZE {for h in 0..H::SIZE {let i=sliding_index(p,h);assert!(!seen[i]);seen[i]=true;b[2*i..2*i+2].copy_from_slice(&values[(q*P128::SIZE+p)*H::SIZE+h].to_bits().to_le_bytes());}}assert!(seen.iter().all(|v|*v));b});
        let canonical=[upload(d,raw_canonical[0].clone()).await?,upload(d,raw_canonical[1].clone()).await?,upload(d,raw_canonical[2].clone()).await?,upload(d,raw_canonical[3].clone()).await?];
        let sliding=[upload(d,raw_sliding[0].clone()).await?,upload(d,raw_sliding[1].clone()).await?,upload(d,raw_sliding[2].clone()).await?,upload(d,raw_sliding[3].clone()).await?];
        Ok(Self{canonical,sliding,raw_canonical,raw_sliding})
    }
    async fn check_immutable(&self,d:&mut Device)->Result<()> {
        for q in 0..4 {assert_eq!(self.canonical[q].to_host::<m![P128,H]>(&mut d.pdma).await?.into_inner().into_buf(),self.raw_canonical[q]);assert_eq!(self.sliding[q].to_host::<SlidingOutput>(&mut d.pdma).await?.into_inner().into_buf(),self.raw_sliding[q]);}Ok(())
    }
}
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Result<Vec<u8>> {
 Ok(x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf())
}
fn f32s(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect()}
fn bf_at(b:&[u8],i:usize)->f64{f32::from_bits((u16::from_le_bytes([b[2*i],b[2*i+1]])as u32)<<16)as f64}
fn round_bf(x:f32)->f32 {let b=x.to_bits();f32::from_bits((b.wrapping_add(0x7fff+((b>>16)&1)))&0xffff0000)}
fn e4(b:u8)->f64 {let s=if b&128!=0{-1.}else{1.};let e=(b>>3)&15;let m=b&7;
 assert!(!(e==15&&m==7));s*if e==0{m as f64/512.}else{(1.+m as f64/8.)*2f64.powi(e as i32-7)}}
fn f4(b:u8)->f64 {let v=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8!=0{-v}else{v}}
fn code(q:&PackedF4Groups,i:usize)->f64 {f4((q.packed_f4[i/2]>>(4*(i%2)))&15)}
fn group_index(row:usize,part:usize,cols:usize)->usize{((row/240)*(cols/960)+part)*240+row%240}
struct Projection<E:M,S:M>{packed:HbmTensor<f4e2m1,Chip,E>,dense:HbmTensor<f8e4m3,Chip,E>,scales:HbmTensor<f32,Chip,S>,cpu:PackedF4Groups,old:Vec<u8>}
impl<E:M,S:M> Projection<E,S>{
 async fn new(d:&mut Device,x:&HbmTensor<f8e4m3,Chip,E>,rows:usize,cols:usize,label:&str,row_scale:&[f32])->Result<Self>{
  let old=read(d,x).await?;let start=std::time::Instant::now();let cpu=quantize_blocked_e4(&old,rows,cols)?;
  assert_eq!(cpu.packed_f4.len()*2,E::SIZE);assert_eq!(cpu.dense_e4_codes.len(),E::SIZE);assert_eq!(cpu.group_scales.len(),S::SIZE);
  let mut e2=0.;let mut v2=0.;let mut we2=0.;let mut wv2=0.;let mut max=0f64;
  for (g,(&scale,chunk)) in cpu.group_scales.iter().zip(old.chunks_exact(960)).enumerate(){
   let row=(g/(cols/960)/240)*240+g%240;
   for (h,&b) in chunk.iter().enumerate(){let x=e4(b);let y=code(&cpu,g*960+h)*scale as f64;let e=y-x;
    assert_eq!(e4(cpu.dense_e4_codes[g*960+h]),code(&cpu,g*960+h));
    e2+=e*e;v2+=x*x;let w=row_scale[row]as f64;we2+=(e*w).powi(2);wv2+=(x*w).powi(2);max=max.max(e.abs());}
  }
  println!("F4_WEIGHT label={label} values={} relative_l2={} row_dequant_l2={} max_abs={max} cpu_encode_and_audit_ms={} baseline=current_native_E4_not_original_NVFP4",E::SIZE,(e2/v2.max(1e-300)).sqrt(),(we2/wv2.max(1e-300)).sqrt(),start.elapsed().as_secs_f64()*1000.);
  let packed=upload(d,cpu.packed_f4.clone()).await?;let dense=upload(d,cpu.dense_e4_codes.clone()).await?;
  let scales=HostTensor::<f32,S>::from_vec(cpu.group_scales.clone()).to_hbm(&mut d.pdma).await?;
  Ok(Self{packed,dense,scales,cpu,old})
 }
 async fn immutable(&self,d:&mut Device,original:&HbmTensor<f8e4m3,Chip,E>)->Result<usize>{
  assert_eq!(read(d,original).await?,self.old);assert_eq!(read(d,&self.packed).await?,self.cpu.packed_f4);
  assert_eq!(read(d,&self.dense).await?,self.cpu.dense_e4_codes);
  assert_eq!(read(d,&self.scales).await?,self.cpu.group_scales.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>());
  Ok(self.old.len()+self.cpu.packed_f4.len()+self.cpu.dense_e4_codes.len()+self.cpu.group_scales.len()*4)
 }
}
struct Scratch {
    residual:HbmTensor<bf16,Chip,Residual>,qinput:HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:HbmTensor<f32,Chip,m![P512]>,joined:HbmTensor<bf16,Chip,JoinedOutput>,
    qhidden:HbmTensor<f8e4m3,Chip,P512L960>,hscale:HbmTensor<f32,Chip,m![P512]>,output:HbmTensor<bf16,Chip,Output>,
}
impl Scratch {
    async fn new(d:&mut Device,incoming:&[u8])->Result<Self>{
        macro_rules! poison {($t:ty,$m:ty,$v:expr)=>{HostTensor::<$t,$m>::from_vec((0..<$m>::SIZE).map(|_|$v)).to_hbm(&mut d.pdma).await?};}
        Ok(Self {residual:upload(d,incoming.to_vec()).await?,
            qinput:poison!(f8e4m3,P512H960,f8e4m3::from_f32(f32::NAN)),xscale:poison!(f32,m![P512],f32::NAN),
            joined:poison!(bf16,JoinedOutput,bf16::from_bits(0x7fc1)),
            qhidden:poison!(f8e4m3,P512L960,f8e4m3::from_f32(f32::NAN)),hscale:poison!(f32,m![P512],f32::NAN),output:poison!(bf16,Output,bf16::from_bits(0x7fc1))})
    }
    async fn reset(&mut self,d:&mut Device,incoming:&[u8])->Result<()> {
        self.residual=upload(d,incoming.to_vec()).await?;Ok(())
    }
    async fn snapshots(&self,d:&mut Device)->Result<Vec<Vec<u8>>> {
        macro_rules! get {($f:ident,$m:ty)=>{self.$f.to_host::<$m>(&mut d.pdma).await?.into_inner().into_buf()};}
        Ok(vec![get!(residual,Residual),get!(qinput,P512H960),get!(xscale,m![P512]),get!(joined,JoinedOutput),get!(qhidden,P512L960),get!(hscale,m![P512]),get!(output,Output)])
    }
}
// Untimed same-norm reference provides the retained pre-FFN residual for the
// independent F64 final-normalization oracle, without replacing the timed control.
async fn reference_mid(d:&mut Device,input:&Inputs,incoming:Vec<u8>,
    pa:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,sliding:bool,
    expected_q:&[u8],expected_s:&[u8])->Result<Vec<u8>> {
    let mut r:HbmTensor<bf16,Chip,Residual>=upload(d,incoming).await?;
    let mut q=HostTensor::<f8e4m3,P512H960>::zero().to_hbm(&mut d.pdma).await?;
    let mut scale=HostTensor::<f32,m![P512]>::zero().to_hbm(&mut d.pdma).await?;
    if sliding {launch(norm::post_attention_sliding,(d,&input.sliding[0],&input.sliding[1],&input.sliding[2],&input.sliding[3],pa,pre,&mut r,&mut q,&mut scale)).await?;}
    else {launch(norm::post_attention_full,(d,&input.canonical[0],&input.canonical[1],&input.canonical[2],&input.canonical[3],pa,pre,&mut r,&mut q,&mut scale)).await?;}
    assert_eq!(q.to_host::<P512H960>(&mut d.pdma).await?.into_inner().into_buf(),expected_q);
    assert_eq!(scale.to_host::<m![P512]>(&mut d.pdma).await?.into_inner().into_buf(),expected_s);
    Ok(r.to_host::<Residual>(&mut d.pdma).await?.into_inner().into_buf())
}
fn bf(bytes:&[u8],index:usize)->f32 {f32::from_bits((u16::from_le_bytes([bytes[index*2],bytes[index*2+1]])as u32)<<16)}
fn cpu_tail(case:usize,mid:&[u8],output:&[u8],actual:&[u8],weight:&[bf16],gate:f32){
    let mut square=0f64;let mut norm=0f64;let mut max_abs=0f64;
    for row in [0usize,1,15,16,63,64,127,128,255,256,383,384,511] {
        let rms=((0..H::SIZE).map(|h|{let x=bf(output,output_index(row,h))as f64;x*x}).sum::<f64>()/H::SIZE as f64+EPS as f64).sqrt();
        for h in 0..H::SIZE {
            let n=bf16::from_f32(((bf(output,output_index(row,h))as f64/rms)as f32)*weight[h].to_f32()).to_f32();
            let add=bf16::from_f32(n+bf(mid,residual_index(row,h))).to_f32();
            let expected=bf16::from_f32(add*gate).to_f32()as f64;
            let error=bf(actual,residual_index(row,h))as f64-expected;square+=error*error;norm+=expected*expected;max_abs=max_abs.max(error.abs());
        }
    }
    let nrmse=(square/norm.max(1e-30)).sqrt();println!("NORM_STOREGROUP_CPU case={case} rows=13 nrmse={nrmse:.9} max_abs={max_abs}");assert!(nrmse<0.001,"independent F64 postnorm disagrees");
}

struct Packed {
 up:Projection<crate::ops_native_fp8_blocked::UpWeight,m![L/240,H/960,L%240]>,
 gate:Projection<crate::ops_native_fp8_blocked::UpWeight,m![L/240,H/960,L%240]>,
 down:Projection<crate::ops_native_fp8_blocked::DownWeight,k::DownBlockScale>,
 joined:HbmTensor<f4e2m1,Chip,JoinedWeight>,
 dense_joined:HbmTensor<f8e4m3,Chip,JoinedWeight>,
 joined_block:HbmTensor<f32,Chip,k::JoinedBlockScale>,
}
impl Packed {
 async fn new(d:&mut Device,w:&MlpWeights)->Result<Self>{
  let n=&w.native_fp8;
  let us=f32s(&read(d,&n.up_dequant).await?);let gs=f32s(&read(d,&n.gate_dequant).await?);
  let ds=f32s(&read(d,&n.down_dequant).await?);
  let up=Projection::new(d,&n.up,L::SIZE,H::SIZE,"up",&us).await?;
  let gate=Projection::new(d,&n.gate,L::SIZE,H::SIZE,"gate",&gs).await?;
  let down=Projection::new(d,&n.down,H::SIZE,L::SIZE,"down",&ds).await?;
  let joined=upload(d,[up.cpu.packed_f4.as_slice(),gate.cpu.packed_f4.as_slice()].concat()).await?;
  let dense_joined=upload(d,[up.cpu.dense_e4_codes.as_slice(),gate.cpu.dense_e4_codes.as_slice()].concat()).await?;
  let joined_block=HostTensor::<f32,k::JoinedBlockScale>::from_vec(
   [up.cpu.group_scales.as_slice(),gate.cpu.group_scales.as_slice()].concat()).to_hbm(&mut d.pdma).await?;
  Ok(Self{up,gate,down,joined,dense_joined,joined_block})
 }
 async fn immutable(&self,d:&mut Device,w:&MlpWeights)->Result<usize>{
  let n=&w.native_fp8;
  let mut bytes=self.up.immutable(d,&n.up).await?+self.gate.immutable(d,&n.gate).await?+self.down.immutable(d,&n.down).await?;
  let expected=[self.up.cpu.packed_f4.as_slice(),self.gate.cpu.packed_f4.as_slice()].concat();
  assert_eq!(read(d,&self.joined).await?,expected);bytes+=expected.len();
  let expected=[self.up.cpu.dense_e4_codes.as_slice(),self.gate.cpu.dense_e4_codes.as_slice()].concat();
  assert_eq!(read(d,&self.dense_joined).await?,expected);bytes+=expected.len();
  let expected=self.up.cpu.group_scales.iter().chain(&self.gate.cpu.group_scales).flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
  assert_eq!(read(d,&self.joined_block).await?,expected);bytes+=expected.len();Ok(bytes)
 }
}
async fn chain(d:&mut Device,s:&mut Scratch,input:&Inputs,w:&MlpWeights,q:&Packed,
 old_joined:&HbmTensor<f8e4m3,Chip,JoinedWeight>,row_scales:&HbmTensor<f32,Chip,JoinedScale>,
 pa:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,post:&HbmTensor<bf16,Chip,m![H]>,lg:&HbmTensor<bf16,Chip,m![1#8]>,
 sliding:bool,mode:u8)->Result<()> {
 let n=&w.native_fp8;
 macro_rules! quant {($kernel:expr,$input:expr,$up:expr,$down:expr)=>{launch($kernel,
  (d,(&$input[0],&$input[1],&$input[2],&$input[3]),(pa,pre,post,lg),&mut s.residual,
   $up,row_scales,$down,(&q.joined_block,&q.down.scales),&n.down_dequant,
   &w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
   &mut s.qinput,&mut s.xscale,&mut s.joined,&mut s.qhidden,&mut s.hscale,&mut s.output)).await?};}
 macro_rules! old {($kernel:expr,$input:expr)=>{launch($kernel,
  (d,(&$input[0],&$input[1],&$input[2],&$input[3]),(pa,pre,post,lg),&mut s.residual,
   old_joined,row_scales,&n.down,&n.down_dequant,&w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
   &mut s.qinput,&mut s.xscale,&mut s.joined,&mut s.qhidden,&mut s.hscale,&mut s.output)).await?};}
 match (sliding,mode) {
  (true,2)=>quant!(candidate::sliding_projection_owned,input.sliding,&q.joined,&q.down.packed),
  (false,2)=>quant!(candidate::full_projection_owned,input.canonical,&q.joined,&q.down.packed),
  (true,1)=>quant!(intended::sliding_projection_owned,input.sliding,&q.dense_joined,&q.down.dense),
  (false,1)=>quant!(intended::full_projection_owned,input.canonical,&q.dense_joined,&q.down.dense),
  (true,_)=>old!(original::sliding_projection_owned,input.sliding),
  (false,_)=>old!(original::full_projection_owned,input.canonical),
 }
 Ok(())
}
fn input_index(p:usize,h:usize,cols:usize)->usize {(((p/256*(cols/960)+h/960)*256+p%256)*960)+h%960}
fn joined_index(proj:usize,p:usize,l:usize)->usize {proj*P512::SIZE*L::SIZE+(((p/256*(L::SIZE/240)+l/240)*256+p%256)*240)+l%240}
fn stage_index(stage:usize,p:usize,col:usize)->usize {
 match stage {
  0=>residual_index(p,col),1=>input_index(p,col,H::SIZE),2|5=>p,
  3=>joined_index(col/L::SIZE,p,col%L::SIZE),4=>input_index(p,col,L::SIZE),
  6=>output_index(p,col),_=>unreachable!(),
 }
}
fn stage_cols(stage:usize)->usize {match stage{0|1|6=>H::SIZE,2|5=>1,3=>2*L::SIZE,4=>L::SIZE,_=>unreachable!()}}
fn stage_element_bytes(stage:usize)->usize{match stage{1|4=>1,2|5=>4,_=>2}}
fn permutation(reference:&[Vec<u8>],actual:&[Vec<u8>],live:usize){
 for stage in 0..7 {let size=stage_element_bytes(stage);for row in 0..P512::SIZE {let p=perm_row(row,live,true);
  for col in 0..stage_cols(stage){let a=stage_index(stage,p,col)*size;let b=stage_index(stage,row,col)*size;
   assert_eq!(&reference[stage][a..a+size],&actual[stage][b..b+size],"stage={stage} row={row} col={col} permutation");
  }
 }}
}
fn stage_error(stage:usize,a:&[u8],b:&[u8])->(usize,f64,f64){
 assert_eq!(a.len(),b.len());let diff=a.iter().zip(b).filter(|(x,y)|x!=y).count();
 let mut e2=0.;let mut v2=0.;let mut max=0f64;
 for i in 0..a.len()/stage_element_bytes(stage){let get=|x:&[u8]|match stage {
  1|4=>e4(x[i]),2|5=>f32::from_le_bytes(x[4*i..4*i+4].try_into().unwrap())as f64,_=>bf_at(x,i)};
  let x=get(a);let y=get(b);assert!(x.is_finite()&&y.is_finite());e2+=(x-y).powi(2);v2+=x*x;max=max.max((x-y).abs());
 }
 (diff,(e2/v2.max(1e-300)).sqrt(),max)
}
fn partial(q:&PackedF4Groups,x:&[u8],cols:usize,p:usize,row:usize,part:usize)->(f64,f64){
 let g=group_index(row,part,cols);let(mut sum,mut abs)=(0.,0.);
 for h in 0..960{let z=e4(x[input_index(p,part*960+h,cols)])*code(q,g*960+h);sum+=z;abs+=z.abs();}
 (sum*q.group_scales[g]as f64,abs*(q.group_scales[g]as f64).abs())
}
fn cpu_projection(label:&str,q:&PackedF4Groups,x:&[u8],xs:&[f32],ws:&[f32],rows:usize,cols:usize,
 output:&[u8],projection:Option<usize>,global:f32)->Result<()> {
 let mut max=0f64;
 for i in 0..128 {let p=(i*73+31)%P512::SIZE;let row=(i*317+19)%rows;let(mut sum,mut abs)=(0.,0.);
  for part in 0..cols/960{let(s,a)=partial(q,x,cols,p,row,part);sum+=s;abs+=a;}
  let first=round_bf((sum as f32)*xs[p]*ws[row]);
  let expected=if projection.is_some(){first}else{round_bf(first*global)}as f64;
  let at=match projection{Some(which)=>joined_index(which,p,row),None=>output_index(p,row)};
  let actual=bf_at(output,at);let error=(expected-actual).abs();max=max.max(error);
  let exponent=(((expected as f32).abs().to_bits()>>23)&255)as i32;
  let ulp=if exponent==0{2f64.powi(-133)}else{2f64.powi(exponent-127-7)};
  let applied_global=if projection.is_some(){1.}else{global as f64};
  let limit=1.1*ulp+2e-5*abs*(xs[p]as f64*ws[row]as f64*applied_global).abs()+1e-5;
  assert!(error<=limit,"F64 {label} row={row} p={p} actual={actual} expected={expected} tolerance={limit}");
 }
 println!("P512_F4_CPU label={label} samples=128 max_abs={max} F64_group_scaled_partials_and_BF16_boundaries=true toleranced=true");Ok(())
}
pub async fn run(mut profile:impl FnMut(bool))->Result<()> {
 let start_case=std::env::args().nth(1).unwrap_or_else(||"0".into()).parse::<usize>()?;
 let end_case=std::env::args().nth(2).unwrap_or_else(||"4".into()).parse::<usize>()?;
 assert!(start_case<end_case&&end_case<=4);
 let mut d=Device::new(candidate::sliding_projection_owned.topology())?;
 let weights=load::load_weights(&mut d).await?;
 let signed=HostTensor::<bf16,m![1#8]>::from_vec([bf16::from_f32(-0.75);8]).to_hbm(&mut d.pdma).await?;
 let(mut exact_bytes,mut immutable_bytes)=(0usize,0usize);let mut times=[Vec::<f64>::new(),Vec::new()];
 for case in start_case..end_case {
  let index=[0usize,5,6,47][case];let live=[512usize,256,384,512][case];
  let(w,pa,pre,post,lg,sliding)=match &weights.layers[index]{
   Layer::Sliding(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar,true),
   Layer::Full(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar,false)};
  let lg=if case==3{&signed}else{lg};let n=&w.native_fp8;
  let(old_joined,row_scales)=launch(crate::ops_ffn_projection_join::join,(&mut d,&n.up,&n.gate,&n.up_dequant,&n.gate_dequant)).await?;
  let old_joined_bytes=read(&mut d,&old_joined).await?;let row_scale_bytes=read(&mut d,&row_scales).await?;
  let q=Packed::new(&mut d,w).await?;
  let statics=vec![read(&mut d,pa).await?,read(&mut d,pre).await?,read(&mut d,post).await?,read(&mut d,lg).await?,
   read(&mut d,&n.up_dequant).await?,read(&mut d,&n.gate_dequant).await?,read(&mut d,&n.down_dequant).await?,
   read(&mut d,&w.up_global_scale).await?,read(&mut d,&w.gate_global_scale).await?,read(&mut d,&w.down_global_scale).await?];
  let post_cpu=statics[2].chunks_exact(2).map(|b|bf16::from_bits(u16::from_le_bytes([b[0],b[1]]))).collect::<Vec<_>>();
  let gate_cpu=bf(&statics[3],0);
  let mut unpermuted=Vec::<Vec<Vec<u8>>>::new();
  for perm in [false,true] {
   let input=Inputs::new(&mut d,case+1,live,perm).await?;
   let initial=packed_residual(&data(case+1,1,live,perm));let mut incoming=initial.clone();
   let mut a=Scratch::new(&mut d,&initial).await?;let mut b=Scratch::new(&mut d,&initial).await?;let mut c=Scratch::new(&mut d,&initial).await?;
   for feedback in 0..2 {
    a=Scratch::new(&mut d,&incoming).await?;b=Scratch::new(&mut d,&incoming).await?;c=Scratch::new(&mut d,&incoming).await?;
    chain(&mut d,&mut c,&input,w,&q,&old_joined,&row_scales,pa,pre,post,lg,sliding,2).await?;let cand=c.snapshots(&mut d).await?;
    chain(&mut d,&mut b,&input,w,&q,&old_joined,&row_scales,pa,pre,post,lg,sliding,1).await?;let reference=b.snapshots(&mut d).await?;
    chain(&mut d,&mut a,&input,w,&q,&old_joined,&row_scales,pa,pre,post,lg,sliding,0).await?;let old=a.snapshots(&mut d).await?;
    for stage in 0..7 {
     assert_eq!(cand[stage],reference[stage],"intended case={case} perm={perm} feedback={feedback} stage={stage}");exact_bytes+=cand[stage].len();
     let(diff,l2,max)=stage_error(stage,&old[stage],&cand[stage]);if matches!(stage,1|2){assert_eq!(diff,0,"weight-independent prefix changed");}
     println!("P512_F4_STAGE case={case} layer={index} live={live} sliding={sliding} perm={perm} feedback={feedback} stage={stage} bytes={} intended_bitexact=true original_byte_diff={diff} original_relative_l2={l2} original_max_abs={max}",cand[stage].len());
    }
    if perm{permutation(&unpermuted[feedback],&cand,live);}else{unpermuted.push(cand.clone());}
    if !perm&&feedback==0 {
     let xs=f32s(&cand[2]);let hs=f32s(&cand[5]);
     cpu_projection("up",&q.up.cpu,&cand[1],&xs,&f32s(&statics[4]),L::SIZE,H::SIZE,&cand[3],Some(0),1.)?;
     cpu_projection("gate",&q.gate.cpu,&cand[1],&xs,&f32s(&statics[5]),L::SIZE,H::SIZE,&cand[3],Some(1),1.)?;
     cpu_projection("down",&q.down.cpu,&cand[4],&hs,&f32s(&statics[6]),H::SIZE,L::SIZE,&cand[6],None,f32s(&statics[9])[0])?;
     let mid=reference_mid(&mut d,&input,incoming.clone(),pa,pre,sliding,&cand[1],&cand[2]).await?;
     cpu_tail(case,&mid,&cand[6],&cand[0],&post_cpu,gate_cpu);
    }
    incoming=cand[0].clone();
   }
   input.check_immutable(&mut d).await?;immutable_bytes+=input.raw_canonical.iter().chain(&input.raw_sliding).map(Vec::len).sum::<usize>();
   for(round,mode)in [0u8,2,2,0].into_iter().enumerate(){let s=if mode==0{&mut a}else{&mut c};let mut us=0.;
    for _ in 0..5 {s.reset(&mut d,&initial).await?;let now=std::time::Instant::now();chain(&mut d,s,&input,w,&q,&old_joined,&row_scales,pa,pre,post,lg,sliding,mode).await?;us+=now.elapsed().as_secs_f64()*1e6;}
    let mean=us/5.;times[usize::from(mode==2)].push(mean);
    println!("P512_F4_TIME case={case} layer={index} sliding={sliding} perm={perm} round={round} candidate={} complete_us={mean} samples=5 upload_download_excluded=true",mode==2);
   }
   if !perm&&std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){for mode in [0u8,2]{let s=if mode==0{&mut a}else{&mut c};s.reset(&mut d,&initial).await?;
    println!("P512_F4_PROFILE case={case} layer={index} candidate={}",mode==2);profile(true);chain(&mut d,s,&input,w,&q,&old_joined,&row_scales,pa,pre,post,lg,sliding,mode).await?;profile(false);
   }}
  }
  immutable_bytes+=q.immutable(&mut d,w).await?;
  assert_eq!(read(&mut d,&old_joined).await?,old_joined_bytes);assert_eq!(read(&mut d,&row_scales).await?,row_scale_bytes);
  immutable_bytes+=old_joined_bytes.len()+row_scale_bytes.len();
  let after=vec![read(&mut d,pa).await?,read(&mut d,pre).await?,read(&mut d,post).await?,read(&mut d,lg).await?,
   read(&mut d,&n.up_dequant).await?,read(&mut d,&n.gate_dequant).await?,read(&mut d,&n.down_dequant).await?,
   read(&mut d,&w.up_global_scale).await?,read(&mut d,&w.gate_global_scale).await?,read(&mut d,&w.down_global_scale).await?];
  assert_eq!(statics,after);immutable_bytes+=statics.iter().map(Vec::len).sum::<usize>();
  println!("P512_F4_IMMUTABLE case={case} all_original_quantized_joined_weights_scales_and_inputs=true");
 }
 let means=times.iter().map(|v|v.iter().sum::<f64>()/v.len()as f64).collect::<Vec<_>>();let gain=1.-means[1]/means[0];
 println!("PASS P512_F4 intended_exact_bytes={exact_bytes} immutable_bytes={immutable_bytes} case_start={start_case} case_end={end_case} original_mean_us={} candidate_mean_us={} relative_gain={gain} speed_gate_10pct={} approximation_reported=true no_model_quality_claim=true",means[0],means[1],gain>=0.10);
 Ok(())
}

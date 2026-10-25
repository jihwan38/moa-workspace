//! Strict fixed-code/partial-scale oracle plus approximate current-E4 comparison.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H,L},ops_b16_f4_partialscale as k,
 ops_b16_f4_partialscale_epilogue as candidate,ops_b16_f4_partialscale_dense_epilogue as intended,
 ops_b16_ingress_epilogue as original};
use super::{load::{self,Layer,MlpWeights},f4_group960::PackedF4Groups,f4_group960_mse::quantize_blocked_e4_mse};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Result<Vec<u8>> {
 Ok(x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf())
}
async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,b:Vec<u8>)->Result<HbmTensor<D,Chip,E>> {
 Ok(HostTensor::<D,E>::from_buf(b).to_hbm(&mut d.pdma).await?)
}
fn f32s(b:&[u8])->Vec<f32>{b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect()}
fn bf_at(b:&[u8],i:usize)->f64{f32::from_bits((u16::from_le_bytes([b[2*i],b[2*i+1]])as u32)<<16)as f64}
fn bf(x:f32)->f32 {let b=x.to_bits();f32::from_bits((b.wrapping_add(0x7fff+((b>>16)&1)))&0xffff0000)}
fn e4(b:u8)->f64 {let s=if b&128!=0{-1.}else{1.};let e=(b>>3)&15;let m=b&7;
 assert!(!(e==15&&m==7));s*if e==0{m as f64/512.}else{(1.+m as f64/8.)*2f64.powi(e as i32-7)}}
fn f4(b:u8)->f64 {let v=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8!=0{-v}else{v}}
fn code(q:&PackedF4Groups,i:usize)->f64 {f4((q.packed_f4[i/2]>>(4*(i%2)))&15)}
fn group_index(row:usize,part:usize,cols:usize)->usize{((row/240)*(cols/960)+part)*240+row%240}
struct Projection<E:M,S:M>{packed:HbmTensor<f4e2m1,Chip,E>,dense:HbmTensor<f8e4m3,Chip,E>,scales:HbmTensor<f32,Chip,S>,cpu:PackedF4Groups,old:Vec<u8>}
impl<E:M,S:M> Projection<E,S>{
 async fn new(d:&mut Device,x:&HbmTensor<f8e4m3,Chip,E>,rows:usize,cols:usize,label:&str,row_scale:&[f32])->Result<Self>{
  let old=read(d,x).await?;let start=std::time::Instant::now();let fitted=quantize_blocked_e4_mse(&old,rows,cols)?;
  // Exact fixed-LUT E2M1 expansion; the fitted group scales remain separate.
  let e4_codes=[0x00u8,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc];
  let dense_e4_codes=fitted.packed_f4.iter().flat_map(|&b|[e4_codes[(b&15)as usize],e4_codes[(b>>4)as usize]]).collect();
  let cpu=PackedF4Groups{packed_f4:fitted.packed_f4,group_scales:fitted.group_scales,dense_e4_codes};
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
struct Packed {up:Projection<k::UpWeight,k::UpBlockScale>,gate:Projection<k::UpWeight,k::UpBlockScale>,down:Projection<k::DownWeight,k::DownBlockScale>}
struct Scratch {projected:HbmTensor<bf16,Chip,m![B,H]>,residual:HbmTensor<bf16,Chip,m![B,H]>,norm:HbmTensor<bf16,Chip,m![B,H]>,qinput:HbmTensor<f8e4m3,Chip,m![B,H]>,xscale:HbmTensor<f32,Chip,m![B]>,hidden:HbmTensor<bf16,Chip,m![B,L]>,qhidden:HbmTensor<f8e4m3,Chip,m![B,L]>,hscale:HbmTensor<f32,Chip,m![B]>}
fn fixture(seed:usize,permuted:bool,kind:usize)->Vec<u8>{(0..B::SIZE*H::SIZE).flat_map(|i|{
 let row=if permuted{(i/H::SIZE*5+3)%B::SIZE}else{i/H::SIZE};let j=row*H::SIZE+i%H::SIZE;
 let x=if row==0{0.}else{(((j*73+seed*131+kind*19)%1009)as f32/504.-1.)*2f32.powi((row%7)as i32-3)};
 bf16::from_f32(x).to_bits().to_le_bytes()
}).collect()}
impl Scratch {
 async fn new(d:&mut Device,p:&[u8],r:&[u8])->Result<Self>{
  macro_rules! poison {($ty:ty,$m:ty,$v:expr)=>{HostTensor::<$ty,$m>::from_vec(std::iter::repeat_n($v,<$m>::SIZE)).to_hbm(&mut d.pdma).await?};}
  Ok(Self{projected:upload(d,p.to_vec()).await?,residual:upload(d,r.to_vec()).await?,norm:poison!(bf16,m![B,H],bf16::from_bits(0x7fc1)),qinput:poison!(f8e4m3,m![B,H],f8e4m3::from_f32(f32::NAN)),xscale:poison!(f32,m![B],f32::NAN),hidden:poison!(bf16,m![B,L],bf16::from_bits(0x7fc1)),qhidden:poison!(f8e4m3,m![B,L],f8e4m3::from_f32(f32::NAN)),hscale:poison!(f32,m![B],f32::NAN)})
 }
 async fn reset(&mut self,d:&mut Device,p:&[u8],r:&[u8])->Result<()> {self.projected=upload(d,p.to_vec()).await?;self.residual=upload(d,r.to_vec()).await?;Ok(())}
 async fn snapshot(&self,d:&mut Device)->Result<Vec<Vec<u8>>>{Ok(vec![read(d,&self.projected).await?,read(d,&self.residual).await?,read(d,&self.norm).await?,read(d,&self.qinput).await?,read(d,&self.xscale).await?,read(d,&self.hidden).await?,read(d,&self.qhidden).await?,read(d,&self.hscale).await?])}
}
async fn chain(d:&mut Device,s:&mut Scratch,w:&MlpWeights,q:&Packed,pa:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,post:&HbmTensor<bf16,Chip,m![H]>,lg:&HbmTensor<bf16,Chip,m![1#8]>,mode:u8)->Result<()> {
 let n=&w.native_fp8;
 macro_rules! args {($up:expr,$gate:expr,$down:expr)=>{(d,&mut s.projected,&mut s.residual,pa,pre,post,lg,&mut s.norm,$up,$gate,$down,(&q.up.scales,&q.gate.scales,&q.down.scales),&n.up_dequant,&n.gate_dequant,&n.down_dequant,&w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,&mut s.qinput,&mut s.xscale,&mut s.hidden,&mut s.qhidden,&mut s.hscale)};}
 match mode {
  2=>launch(candidate::post_attention_ffn,args!(&q.up.packed,&q.gate.packed,&q.down.packed)).await?,
  1=>launch(intended::post_attention_ffn,args!(&q.up.dense,&q.gate.dense,&q.down.dense)).await?,
  _=>launch(original::post_attention_ffn,(d,&mut s.projected,&mut s.residual,pa,pre,post,lg,&mut s.norm,&n.up,&n.gate,&n.down,&n.up_dequant,&n.gate_dequant,&n.down_dequant,&w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,&mut s.qinput,&mut s.xscale,&mut s.hidden,&mut s.qhidden,&mut s.hscale)).await?,
 }
 Ok(())
}
fn stage_error(stage:usize,a:&[u8],b:&[u8])->(usize,f64,f64){assert_eq!(a.len(),b.len());let diff=a.iter().zip(b).filter(|(x,y)|x!=y).count();let n=if matches!(stage,3|6){a.len()}else if matches!(stage,4|7){a.len()/4}else{a.len()/2};let mut e2=0.;let mut v2=0.;let mut max=0f64;
 for i in 0..n{let get=|x:&[u8]|if matches!(stage,3|6){e4(x[i])}else if matches!(stage,4|7){f32::from_le_bytes(x[4*i..4*i+4].try_into().unwrap())as f64}else{bf_at(x,i)};let x=get(a);let y=get(b);assert!(x.is_finite()&&y.is_finite());e2+=(x-y).powi(2);v2+=x*x;max=max.max((x-y).abs());}
 (diff,(e2/v2.max(1e-300)).sqrt(),max)
}
fn permutation(original:&[Vec<u8>],permuted:&[Vec<u8>]){for(a,b)in original.iter().zip(permuted){let stride=a.len()/B::SIZE;for r in 0..B::SIZE{let p=(r*5+3)%B::SIZE;assert_eq!(&b[r*stride..(r+1)*stride],&a[p*stride..(p+1)*stride],"row permutation");}}}
fn partial_sum(q:&PackedF4Groups,x:&[u8],cols:usize,b:usize,row:usize,part:usize)->(f64,f64){let g=group_index(row,part,cols);let mut sum=0.;let mut abs=0.;for h in 0..960{let p=e4(x[b*cols+part*960+h])*code(q,g*960+h);sum+=p;abs+=p.abs();}let scale=q.group_scales[g]as f64;(sum*scale,abs*scale.abs())}
fn partial_check(label:&str,q:&PackedF4Groups,x:&[u8],rows:usize,cols:usize,found:&[u8]){let y=f32s(found);assert_eq!(y.len(),B::SIZE*rows*(cols/960));assert!(y.iter().all(|v|v.is_finite()));let mut max=0f64;for i in 0..128{let b=(i*7+1)%B::SIZE;let row=(i*317+31)%rows;let part=(i*7+i/16)%(cols/960);let(expected,abs)=partial_sum(q,x,cols,b,row,part);let actual=y[(b*rows+row)*(cols/960)+part]as f64;let error=(actual-expected).abs();max=max.max(error);assert!(error<=1e-5*abs+1e-5,"F64partial {label} row={row} b={b} part={part} actual={actual} expected={expected} abs={abs}");}println!("F4_CPU_PARTIAL label={label} samples=128 max_abs={max} tolerance=1e-5_abs_product_sum_plus_1e-5");}
async fn cpu_checks(d:&mut Device,s:&Scratch,snap:&[Vec<u8>],q:&Packed,w:&MlpWeights,label:&str)->Result<()> {
 let mut up:HbmTensor<f32,Chip,m![B,L,H/960]>=HostTensor::<f32,m![B,L,H/960]>::from_vec(std::iter::repeat_n(f32::NAN,B::SIZE*L::SIZE*(H::SIZE/960))).to_hbm(&mut d.pdma).await?;
 for (name,p) in [("up",&q.up),("gate",&q.gate)]{launch(k::debug_up_partials,(d,&s.qinput,&p.packed,&p.scales,&mut up)).await?;partial_check(name,&p.cpu,&snap[3],L::SIZE,H::SIZE,&read(d,&up).await?);}
 let mut down:HbmTensor<f32,Chip,m![B,H,L/960]>=HostTensor::<f32,m![B,H,L/960]>::from_vec(std::iter::repeat_n(f32::NAN,B::SIZE*H::SIZE*(L::SIZE/960))).to_hbm(&mut d.pdma).await?;
 launch(k::debug_down_partials,(d,&s.qhidden,&q.down.packed,&q.down.scales,&mut down)).await?;
 partial_check("down",&q.down.cpu,&snap[6],H::SIZE,L::SIZE,&read(d,&down).await?);
 let xs=f32s(&snap[7]);let ws=f32s(&read(d,&w.native_fp8.down_dequant).await?);let global=f32s(&read(d,&w.down_global_scale).await?)[0];let mut max=0f64;
 for i in 0..64 {let b=(i*7+1)%B::SIZE;let row=(i*317+31)%H::SIZE;let(mut sum,mut abs)=(0.,0.);for part in 0..L::SIZE/960{let(s,a)=partial_sum(&q.down.cpu,&snap[6],L::SIZE,b,row,part);sum+=s;abs+=a;}
  let expected=bf(bf((sum as f32)*xs[b]*ws[row])*global)as f64;let actual=bf_at(&snap[0],b*H::SIZE+row);let error=(expected-actual).abs();max=max.max(error);
  let bits=(expected as f32).abs().to_bits();let exponent=((bits>>23)&255)as i32;let ulp=if exponent==0{2f64.powi(-133)}else{2f64.powi(exponent-127-7)};
  let limit=1.1*ulp+2e-5*abs*(xs[b]as f64*ws[row]as f64*global as f64).abs()+1e-5;
  assert!(error<=limit,"F64final {label} row={row} b={b} actual={actual} expected={expected} tolerance={limit}");
 }
 println!("F4_CPU_FINAL label={label} samples=64 max_abs={max} original_BF16_dot_and_global_boundaries=true");Ok(())
}
pub async fn run(profile:&impl Fn(bool))->Result<()> {
 println!("F4_QUANTIZER mode=MSE_dyadic25 source=8e7ce99e original_ceilmax_gate=r1204_r1205 separate_group_scale=true");
 let start_case=std::env::args().nth(1).unwrap_or_else(||"0".into()).parse::<usize>()?;let end_case=std::env::args().nth(2).unwrap_or_else(||"4".into()).parse::<usize>()?;assert!(start_case<end_case&&end_case<=4);
 let mut d=Device::new(candidate::post_attention_ffn.topology())?;let weights=load::load_weights(&mut d).await?;let signed=HostTensor::<bf16,m![1#8]>::from_vec([bf16::from_f32(-0.75);8]).to_hbm(&mut d.pdma).await?;
 let mut exact_bytes=0usize;let mut immutable_bytes=0usize;let mut times=[Vec::<f64>::new(),Vec::new()];
 for case in start_case..end_case {let index=[0usize,5,6,47][case];let(w,pa,pre,post,lg)=match &weights.layers[index]{Layer::Sliding(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar),Layer::Full(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar)};let lg=if case==3{&signed}else{lg};
  let n=&w.native_fp8;let us=f32s(&read(&mut d,&n.up_dequant).await?);let gs=f32s(&read(&mut d,&n.gate_dequant).await?);let ds=f32s(&read(&mut d,&n.down_dequant).await?);
  let q=Packed{up:Projection::new(&mut d,&n.up,L::SIZE,H::SIZE,"up",&us).await?,gate:Projection::new(&mut d,&n.gate,L::SIZE,H::SIZE,"gate",&gs).await?,down:Projection::new(&mut d,&n.down,H::SIZE,L::SIZE,"down",&ds).await?};
  let statics=vec![read(&mut d,pa).await?,read(&mut d,pre).await?,read(&mut d,post).await?,read(&mut d,lg).await?,read(&mut d,&n.up_dequant).await?,read(&mut d,&n.gate_dequant).await?,read(&mut d,&n.down_dequant).await?,read(&mut d,&w.up_global_scale).await?,read(&mut d,&w.gate_global_scale).await?,read(&mut d,&w.down_global_scale).await?];
  let mut unpermuted:Vec<Vec<Vec<u8>>>=Vec::new();
  for perm in [false,true]{let p=fixture(case+1,perm,0);let r=fixture(case+1,perm,1);let mut a=Scratch::new(&mut d,&p,&r).await?;let mut b=Scratch::new(&mut d,&p,&r).await?;let mut c=Scratch::new(&mut d,&p,&r).await?;
   let(mut feedback_p,mut feedback_r)=(p.clone(),r.clone());
   for step in 0..2 {a=Scratch::new(&mut d,&feedback_p,&feedback_r).await?;b=Scratch::new(&mut d,&feedback_p,&feedback_r).await?;c=Scratch::new(&mut d,&feedback_p,&feedback_r).await?;
    chain(&mut d,&mut c,w,&q,pa,pre,post,lg,2).await?;let cand=c.snapshot(&mut d).await?;
    chain(&mut d,&mut b,w,&q,pa,pre,post,lg,1).await?;let reference=b.snapshot(&mut d).await?;
    chain(&mut d,&mut a,w,&q,pa,pre,post,lg,0).await?;let old=a.snapshot(&mut d).await?;
    for stage in 0..8 {assert_eq!(cand[stage],reference[stage],"intended dense-code parity case={case} perm={perm} step={step} stage={stage}");exact_bytes+=cand[stage].len();let(diff,l2,max)=stage_error(stage,&old[stage],&cand[stage]);if matches!(stage,2|3|4){assert_eq!(diff,0,"weight-independent prefix changed");}println!("F4_STAGE case={case} layer={index} perm={perm} feedback={step} stage={stage} bytes={} intended_bitexact=true original_byte_diff={diff} original_relative_l2={l2} original_max_abs={max}",cand[stage].len());}
    if perm{permutation(&unpermuted[step],&cand);}else{unpermuted.push(cand.clone());}
    if step==0&&!perm{cpu_checks(&mut d,&c,&cand,&q,w,&format!("layer{index}")).await?;}
    feedback_p=cand[0].clone();feedback_r=cand[1].clone();
   }
   for(round,mode)in [0u8,2,2,0].into_iter().enumerate(){let s=if mode==0{&mut a}else{&mut c};let mut us=0.;for _ in 0..8{s.reset(&mut d,&p,&r).await?;let start=std::time::Instant::now();chain(&mut d,s,w,&q,pa,pre,post,lg,mode).await?;us+=start.elapsed().as_secs_f64()*1e6;}
    let mean=us/8.;times[usize::from(mode==2)].push(mean);println!("F4_TIME case={case} layer={index} perm={perm} round={round} candidate={} complete_us={mean} samples=8 upload_download_excluded=true",mode==2);}
   if !perm {for mode in [0u8,2]{let s=if mode==0{&mut a}else{&mut c};s.reset(&mut d,&p,&r).await?;println!("F4_PROFILE case={case} layer={index} candidate={}",mode==2);profile(true);chain(&mut d,s,w,&q,pa,pre,post,lg,mode).await?;profile(false);}}
  }
  immutable_bytes+=q.up.immutable(&mut d,&n.up).await?+q.gate.immutable(&mut d,&n.gate).await?+q.down.immutable(&mut d,&n.down).await?;
  let after=vec![read(&mut d,pa).await?,read(&mut d,pre).await?,read(&mut d,post).await?,read(&mut d,lg).await?,read(&mut d,&n.up_dequant).await?,read(&mut d,&n.gate_dequant).await?,read(&mut d,&n.down_dequant).await?,read(&mut d,&w.up_global_scale).await?,read(&mut d,&w.gate_global_scale).await?,read(&mut d,&w.down_global_scale).await?];assert_eq!(statics,after);immutable_bytes+=statics.iter().map(Vec::len).sum::<usize>();
  println!("F4_IMMUTABLE case={case} all_original_and_packed_dense_weights_scales=true");
 }
 let means:Vec<f64>=times.iter().map(|v|v.iter().sum::<f64>()/v.len()as f64).collect();let gain=1.-means[1]/means[0];
 println!("PASS F4_PARTIALSCALE intended_exact_bytes={exact_bytes} immutable_bytes={immutable_bytes} case_start={start_case} case_end={end_case} original_mean_us={} candidate_mean_us={} relative_gain={gain} speed_gate_10pct={} approximation_reported=true no_model_quality_or_selection_claim=true",means[0],means[1],gain>=0.10);
 Ok(())
}

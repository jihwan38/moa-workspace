//! Complete current three-call oracle; all actual supplied FFN/norm weights.
use furiosa_opt_std::prelude::*;
use crate::{Chip, EPS, axes::{H,P128,P512}, ops_batch_layouts::{P512H960,P512L240,P512L960},
    ops_norm_blocked_residual::{self as residual, Residual},
    ops_native_fp8_p512_downblocked::Output, ops_output_p128_h64_blocked::Output as SlidingOutput,
    ops_norm_blocked_chain as norm, ops_native_fp8_p512_core_fused as core,
    ops_group_ffn_epilogue as fused};
use super::load::{self,Layer,MlpWeights};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
type Hidden = HbmTensor<bf16,Chip,m![P128,H]>;
fn residual_index(p:usize,h:usize)->usize {
    (((((p/64%2)*32+p/2%32)*8+h/480)*4+p/128)*2+p%2)*480+h%480
}
fn output_index(p:usize,h:usize)->usize {(((p/256*16+h/240)*256+p%256)*240)+h%240}
fn sliding_index(p:usize,h:usize)->usize {(((p/64*60+h/64)*64+p%64)*64)+h%64}
fn data(seed:usize,kind:usize,live:usize,permuted:bool)->Vec<bf16> {
    (0..P512::SIZE*H::SIZE).map(|i| {
        let row=i/H::SIZE;let col=i%H::SIZE;
        let mapped=if permuted {(row*73+127)%P512::SIZE}else{row};
        let j=mapped*H::SIZE+col;
        let x=((j*73+seed*131+kind*19)%1009) as f32/504.0-1.0;
        bf16::from_f32(if row>=live || row%128==0 {0.0}
            else if row%128==1 {if col%2==0 {1e-8}else{-1e-8}}
            else {x*2f32.powi((mapped%7) as i32-3)})
    }).collect()
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
struct Scratch {
    residual:HbmTensor<bf16,Chip,Residual>,qinput:HbmTensor<f8e4m3,Chip,P512H960>,
    xscale:HbmTensor<f32,Chip,m![P512]>,up:HbmTensor<bf16,Chip,P512L240>,gate:HbmTensor<bf16,Chip,P512L240>,
    qhidden:HbmTensor<f8e4m3,Chip,P512L960>,hscale:HbmTensor<f32,Chip,m![P512]>,output:HbmTensor<bf16,Chip,Output>,
}
impl Scratch {
    async fn new(d:&mut Device,seed:usize,live:usize,permuted:bool)->Result<Self>{
        macro_rules! poison {($t:ty,$m:ty,$v:expr)=>{HostTensor::<$t,$m>::from_vec((0..<$m>::SIZE).map(|_|$v)).to_hbm(&mut d.pdma).await?};}
        Ok(Self {residual:upload(d,packed_residual(&data(seed,1,live,permuted))).await?,
            qinput:poison!(f8e4m3,P512H960,f8e4m3::from_f32(f32::NAN)),xscale:poison!(f32,m![P512],f32::NAN),
            up:poison!(bf16,P512L240,bf16::from_bits(0x7fc1)),gate:poison!(bf16,P512L240,bf16::from_bits(0x7fc2)),
            qhidden:poison!(f8e4m3,P512L960,f8e4m3::from_f32(f32::NAN)),hscale:poison!(f32,m![P512],f32::NAN),output:poison!(bf16,Output,bf16::from_bits(0x7fc1))})
    }
    async fn reset(&mut self,d:&mut Device,seed:usize,live:usize,permuted:bool)->Result<()> {
        self.residual=upload(d,packed_residual(&data(seed,1,live,permuted))).await?;Ok(())
    }
    async fn snapshots(&self,d:&mut Device)->Result<Vec<Vec<u8>>> {
        macro_rules! get {($f:ident,$m:ty)=>{self.$f.to_host::<$m>(&mut d.pdma).await?.into_inner().into_buf()};}
        Ok(vec![get!(residual,Residual),get!(qinput,P512H960),get!(xscale,m![P512]),get!(up,P512L240),get!(gate,P512L240),get!(qhidden,P512L960),get!(hscale,m![P512]),get!(output,Output)])
    }
}
async fn chain(d:&mut Device,s:&mut Scratch,input:&Inputs,w:&MlpWeights,
    pa:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,post:&HbmTensor<bf16,Chip,m![H]>,lg:&HbmTensor<bf16,Chip,m![1#8]>,
    sliding:bool,candidate:bool,save_mid:bool)->Result<Option<Vec<u8>>> {
    let n=&w.native_fp8;
    if candidate {
        macro_rules! run {($kernel:expr,$a:expr)=>{launch($kernel,(d,(&$a[0],&$a[1],&$a[2],&$a[3]),(pa,pre,post,lg),&mut s.residual,
            &n.up,&n.gate,&n.down,&n.up_dequant,&n.gate_dequant,&n.down_dequant,&w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
            &mut s.qinput,&mut s.xscale,&mut s.up,&mut s.gate,&mut s.qhidden,&mut s.hscale,&mut s.output)).await?};}
        if sliding {run!(fused::sliding,input.sliding);}else{run!(fused::full,input.canonical);}Ok(None)
    } else {
        if sliding {launch(norm::post_attention_sliding,(d,&input.sliding[0],&input.sliding[1],&input.sliding[2],&input.sliding[3],pa,pre,&mut s.residual,&mut s.qinput,&mut s.xscale)).await?;}
        else {launch(norm::post_attention_full,(d,&input.canonical[0],&input.canonical[1],&input.canonical[2],&input.canonical[3],pa,pre,&mut s.residual,&mut s.qinput,&mut s.xscale)).await?;}
        let mid=if save_mid {Some(s.residual.to_host::<Residual>(&mut d.pdma).await?.into_inner().into_buf())}else{None};
        launch(core::ffn_core,(d,&s.qinput,&s.xscale,&n.up,&n.gate,&n.down,&n.up_dequant,&n.gate_dequant,&n.down_dequant,&w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
            &mut s.up,&mut s.gate,&mut s.qhidden,&mut s.hscale,&mut s.output)).await?;
        launch(residual::post_feedforward,(d,&s.output,post,lg,&mut s.residual)).await?;Ok(mid)
    }
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
    let nrmse=(square/norm.max(1e-30)).sqrt();println!("GROUP_EPILOGUE_CPU case={case} rows=13 nrmse={nrmse:.9} max_abs={max_abs}");assert!(nrmse<0.001,"independent F64 postnorm disagrees");
}
pub async fn run(mut profile:impl FnMut(bool))->Result<()> {
    let mut d=Device::new(fused::sliding.topology())?;let weights=load::load_weights(&mut d).await?;
    let signed=HostTensor::<bf16,m![1#8]>::from_vec([bf16::from_f32(-0.75);8]).to_hbm(&mut d.pdma).await?;
    let mut total=0usize;
    for (case,(index,live,permuted)) in [(0usize,512,false),(5,256,true),(6,384,true)].into_iter().enumerate(){
        let (w,pa,pre,post,lg,sliding)=match &weights.layers[index] {Layer::Sliding(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar,true),Layer::Full(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar,false)};
        let lg=if case==2{&signed}else{lg};let input=Inputs::new(&mut d,case+1,live,permuted).await?;
        let post_cpu=post.to_host::<m![H]>(&mut d.pdma).await?.into_vec();let gate_cpu=lg.to_host::<m![1#8]>(&mut d.pdma).await?.into_vec()[0].to_f32();
        let mut a=Scratch::new(&mut d,case+1,live,permuted).await?;let mut b=Scratch::new(&mut d,case+1,live,permuted).await?;
        for feedback in 0..2 {
            chain(&mut d,&mut b,&input,w,pa,pre,post,lg,sliding,true,false).await?;let candidate=b.snapshots(&mut d).await?;
            input.check_immutable(&mut d).await?;
            let mid=chain(&mut d,&mut a,&input,w,pa,pre,post,lg,sliding,false,true).await?.unwrap();let control=a.snapshots(&mut d).await?;
            for (stage,(x,y)) in control.iter().zip(&candidate).enumerate(){assert_eq!(x.len(),y.len());let diff=x.iter().zip(y).filter(|(x,y)|x!=y).count();total+=x.len();
                println!("GROUP_EPILOGUE_EXACT case={case} layer={index} live={live} feedback={feedback} stage={stage} bytes={} byte_diff={diff}",x.len());assert_eq!(diff,0,"stage{stage}");
                if matches!(stage,0|3|4|7){assert!(y.chunks_exact(2).all(|v|u16::from_le_bytes([v[0],v[1]])&0x7f80!=0x7f80));}}
            cpu_tail(case,&mid,&candidate[7],&candidate[0],&post_cpu,gate_cpu);
        }
        input.check_immutable(&mut d).await?;
        for (round,candidate) in [false,true,true,false].into_iter().enumerate(){let s=if candidate{&mut b}else{&mut a};let mut total_ms=0.;
            for _ in 0..5 {s.reset(&mut d,case+1,live,permuted).await?;let now=std::time::Instant::now();chain(&mut d,s,&input,w,pa,pre,post,lg,sliding,candidate,false).await?;total_ms+=now.elapsed().as_secs_f64()*1000.;}
            println!("GROUP_EPILOGUE_TIME case={case} sliding={sliding} round={round} candidate={candidate} complete_chain_ms={:.6} samples=5 excludes_equal_residual_upload_and_all_readback=true",total_ms/5.);}
        if std::env::var_os("FURIOSA_OPT_PROFILE").is_some() && case<2 {for candidate in [false,true] {let s=if candidate{&mut b}else{&mut a};s.reset(&mut d,case+1,live,permuted).await?;println!("PROFILE_GROUP_EPILOGUE case={case} candidate={candidate}");profile(true);chain(&mut d,s,&input,w,pa,pre,post,lg,sliding,candidate,false).await?;profile(false);}}
    }
    println!("PASS exact grouped3-to1epilogue bytes={total}; all512rows, supplied sliding/full weights,256/384 zero padding, feedback, signedgate, CPUtail and immutableinputs; model pending");Ok(())
}

//! Exact B16 media frontend comparison using only the provided vision weights.
use furiosa_opt_gemma4::{Chip, axes::*, host::load, ops_vision as scalar, ops_vision_b16 as batch};
use furiosa_opt_std::prelude::*;
use std::{error::Error as StdError, fs::File, time::Instant};
#[path="support/npu_profile.rs"] mod npu_profile;
type Result<T> = std::result::Result<T,Box<dyn StdError>>;

struct Weights {
    ln1w:HbmTensor<bf16,Chip,m![Rv]>,ln1b:HbmTensor<bf16,Chip,m![Rv]>,
    dense:HbmTensor<bf16,Chip,m![Mv,Rv]>,dense_bias:HbmTensor<bf16,Chip,m![Mv]>,
    ln2w:HbmTensor<bf16,Chip,m![Mv]>,ln2b:HbmTensor<bf16,Chip,m![Mv]>,
    ln3w:HbmTensor<bf16,Chip,m![Mv]>,ln3b:HbmTensor<bf16,Chip,m![Mv]>,
    project:HbmTensor<bf16,Chip,m![H,Ov]>,position:Vec<bf16>,
}
async fn load_bf16<E:M>(d:&mut Device,t:&safetensors::SafeTensors<'_>,name:&str)->Result<HbmTensor<bf16,Chip,E>> {
    let view=t.tensor(name)?;
    let host:HostTensor<bf16,E>=HostTensor::from_safetensors(&view).map_err(|e|format!("{name}: {e}"))?;
    Ok(host.to_hbm(&mut d.pdma).await?)
}
async fn upload<E:M>(d:&mut Device,v:&[bf16])->Result<HbmTensor<bf16,Chip,E>> {
    assert_eq!(v.len(),E::SIZE);
    Ok(HostTensor::<bf16,E>::from_vec(v.iter().copied()).to_hbm(&mut d.pdma).await?)
}
async fn buffer<E:M>(d:&mut Device,poison:bool)->Result<HbmTensor<bf16,Chip,E>> {
    upload(d,&vec![bf16::from_bits(if poison {0x7fc1}else{0});E::SIZE]).await
}
async fn read<E:M>(d:&mut Device,x:&HbmTensor<bf16,Chip,E>)->Result<Vec<bf16>> {
    let h:HostTensor<bf16,E>=x.to_host(&mut d.pdma).await?;Ok(h.into_vec())
}
fn position(w:&Weights,p:(usize,usize))->Vec<bf16> {
    let c=p.0.min(Pv::SIZE-1);let r=p.1.min(Pv::SIZE-1);
    (0..Mv::SIZE).map(|k|bf16::from_f32(w.position[(c*2)*Mv::SIZE+k].to_f32()+w.position[(r*2+1)*Mv::SIZE+k].to_f32())).collect()
}
fn positions(w:&Weights,p:&[(usize,usize)],active:usize)->Vec<bf16> {
    p.iter().enumerate().flat_map(|(i,&p)|if i<active{position(w,p)}else{vec![bf16::from_bits(0);Mv::SIZE]}).collect()
}
struct ScalarBuffers {
    x:HbmTensor<bf16,Chip,m![Rv]>,pos:HbmTensor<bf16,Chip,m![Mv]>,
    patch:HbmTensor<bf16,Chip,m![Mv]>,normed:Option<HbmTensor<bf16,Chip,m![Mv]>>,out:HbmTensor<bf16,Chip,m![H]>,
}
impl ScalarBuffers {
    async fn new(d:&mut Device,x:&[bf16],p:&[bf16],poison:bool)->Result<Self>{Ok(Self{
        x:upload(d,x).await?,pos:upload(d,p).await?,patch:buffer(d,poison).await?,normed:Some(buffer(d,poison).await?),out:buffer(d,poison).await?,
    })}
    async fn run(&mut self,d:&mut Device,w:&Weights)->Result<()> {
        launch(scalar::patch_embed,(d,&self.x,&w.ln1w,&w.ln1b,&w.dense,&w.dense_bias,&w.ln2w,&w.ln2b,&mut self.patch)).await?;
        let mut normed=self.normed.take().unwrap();
        launch(scalar::add_position_and_norm,(d,&self.patch,&self.pos,&w.ln3w,&w.ln3b,&mut normed)).await?;
        let normed:HbmTensor<bf16,Chip,m![Ov]>=unsafe{normed.reshape()};
        launch(scalar::project_to_text_embedding,(d,&normed,&w.project,&mut self.out)).await?;
        self.normed=Some(unsafe{normed.reshape()});Ok(())
    }
}
struct BatchBuffers {
    x:HbmTensor<bf16,Chip,m![B,Rv]>,pos:HbmTensor<bf16,Chip,m![B,Mv]>,
    patch:HbmTensor<bf16,Chip,m![B,Mv]>,normed:Option<HbmTensor<bf16,Chip,m![B,Mv]>>,out:HbmTensor<bf16,Chip,m![B,H]>,
}
impl BatchBuffers {
    async fn new(d:&mut Device,x:&[bf16],p:&[bf16],poison:bool)->Result<Self>{Ok(Self{
        x:upload(d,x).await?,pos:upload(d,p).await?,patch:buffer(d,poison).await?,normed:Some(buffer(d,poison).await?),out:buffer(d,poison).await?,
    })}
    async fn run(&mut self,d:&mut Device,w:&Weights)->Result<()> {
        launch(batch::patch_embed,(d,&self.x,&w.ln1w,&w.ln1b,&w.dense,&w.dense_bias,&w.ln2w,&w.ln2b,&mut self.patch)).await?;
        let mut normed=self.normed.take().unwrap();
        launch(batch::add_position_and_norm,(d,&self.patch,&self.pos,&w.ln3w,&w.ln3b,&mut normed)).await?;
        let normed:HbmTensor<bf16,Chip,m![B,Ov]>=unsafe{normed.reshape()};
        launch(batch::project_to_text_embedding,(d,&normed,&w.project,&mut self.out)).await?;
        self.normed=Some(unsafe{normed.reshape()});Ok(())
    }
    async fn fused(&mut self,d:&mut Device,w:&Weights)->Result<()> {
        launch(batch::encode_patches,(d,&self.x,&self.pos,&w.ln1w,&w.ln1b,&w.dense,&w.dense_bias,&w.ln2w,&w.ln2b,&w.ln3w,&w.ln3b,&w.project,&mut self.out)).await?;Ok(())
    }
}
fn exact(case:usize,name:&str,a:&[bf16],b:&[bf16]) {
    assert_eq!(a.len(),b.len());let mut diff=0;let mut nonfinite=0;
    for(i,(a,b))in a.iter().zip(b).enumerate(){if !a.to_f32().is_finite()||!b.to_f32().is_finite(){nonfinite+=1;}
        if a.to_bits()!=b.to_bits(){if diff<8{eprintln!("MEDIA_DIFF case={case} name={name} i={i} expected={:04x} actual={:04x}",a.to_bits(),b.to_bits());}diff+=1;}}
    println!("MEDIA_EXACT case={case} name={name} values={} bitdiff={diff} nonfinite={nonfinite}",a.len());assert_eq!(diff,0);assert_eq!(nonfinite,0);
}
// BF16 products are exact in F32. Each K768 sum is bounded by the
// standard gamma_768 forward-error bound, then every original BF16 rounding
// and balanced addition is propagated monotonically. This is an independent
// mathematical oracle, not an assertion that RNGD uses an F64 accumulator.
fn sampled_projection(case:usize,name:&str,x:&[bf16],w:&[bf16],bias:Option<&[bf16]>,out:&[bf16],kdim:usize,ndim:usize) {
    fn add(a:f32,b:f32)->f32{bf16::from_f32(a+b).to_f32()}
    fn tree(p:&[f32])->f32 {
        let four=add(add(p[0],p[1]),add(p[2],p[3]));
        if p.len()==5{add(four,p[4])}else{add(add(four,add(add(p[4],p[5]),add(p[6],p[7]))),p[8])}
    }
    let gamma=(768.0*2f64.powi(-24))/(1.0-768.0*2f64.powi(-24));
    let mut bitdiff=0;let mut failures=0;let mut max_abs=0f32;let mut max_interval=0f32;
    for n in 0..256 {
        let row=n%B::SIZE;let col=if n<128{[0,119,120,767,768,1919,1920,3839][n/B::SIZE]}else{word(n+case*999)as usize%ndim};
        let mut nearest=Vec::new();let mut low=Vec::new();let mut high=Vec::new();
        for part in 0..kdim/768 {let mut dot=0f64;let mut absolute=0f64;
            for k in part*768..(part+1)*768{let v=x[row*kdim+k].to_f32()as f64*w[col*kdim+k].to_f32()as f64;dot+=v;absolute+=v.abs();}
            let error=gamma*absolute+1e-36;
            nearest.push(bf16::from_f32(dot as f32).to_f32());low.push(bf16::from_f32((dot-error)as f32).to_f32());high.push(bf16::from_f32((dot+error)as f32).to_f32());
        }
        let mut want=tree(&nearest);let mut lo=tree(&low);let mut hi=tree(&high);
        if let Some(bias)=bias{want=add(want,bias[col].to_f32());lo=add(lo,bias[col].to_f32());hi=add(hi,bias[col].to_f32());}
        let got=out[row*ndim+col].to_f32();if got.to_bits()!=want.to_bits(){bitdiff+=1;}max_abs=max_abs.max((got-want).abs());max_interval=max_interval.max(hi-lo);
        if !got.is_finite()||got<lo||got>hi{if failures<8{eprintln!("MEDIA_F64_DIFF case={case} name={name} row={row} col={col} got={got} nearest={want} bound=[{lo},{hi}]");}failures+=1;}
    }
    println!("MEDIA_F64 case={case} name={name} samples=256 nearest_bitdiff={bitdiff} interval_failures={failures} max_abs={max_abs} max_interval_width={max_interval} Kpartial=768 gamma={gamma}");assert_eq!(failures,0);
}
fn word(i:usize)->u64 {let mut x=(i as u64).wrapping_add(0x13579bdf);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn base_pixels()->Vec<bf16>{(0..B::SIZE*Rv::SIZE).map(|i|{
    let r=i/Rv::SIZE;let k=i%Rv::SIZE;bf16::from_f32(match r%4{
        0=>0.,1=>(r as f32-8.)*0.125,2=>(word(i)>>40)as f32/16777216.*6.-3.,
        _=>if k==r*317%Rv::SIZE{64.}else if k%2==0{-0.03125}else{0.03125},
    })}).collect()}
async fn inclusive(d:&mut Device,w:&Weights,pixels:&[bf16],loc:&[(usize,usize)],variant:usize)->Result<()> {
    if variant==0 {let mut retained=Vec::with_capacity(B::SIZE);for row in 0..B::SIZE{
        let pos=position(w,loc[row]);let mut b=ScalarBuffers::new(d,&pixels[row*Rv::SIZE..(row+1)*Rv::SIZE],&pos,false).await?;
        b.run(d,w).await?;retained.push(b.out);
    }std::hint::black_box(retained);}else{
        let pos=positions(w,loc,B::SIZE);
        if variant==1 {let mut b=BatchBuffers::new(d,pixels,&pos,false).await?;b.run(d,w).await?;std::hint::black_box(b.out);}
        else {
            let x=upload::<m![B,Rv]>(d,pixels).await?;let pos=upload::<m![B,Mv]>(d,&pos).await?;
            let mut out=buffer::<m![B,H]>(d,false).await?;
            launch(batch::encode_patches,(d,&x,&pos,&w.ln1w,&w.ln1b,&w.dense,&w.dense_bias,&w.ln2w,&w.ln2b,&w.ln3w,&w.ln3b,&w.project,&mut out)).await?;
            std::hint::black_box(out);
        }
    }Ok(())
}
async fn row_copy_gate(d:&mut Device)->Result<()> {
    use furiosa_opt_gemma4::ops_media_rows;
    let values=|rows:usize,seed:usize|->Vec<bf16>{(0..rows*H::SIZE).map(|i|{
        let v=word(i+seed);bf16::from_bits(((v as u16)&0x8000)|(0x2000+(v%0x5000)as u16))
    }).collect()};
    let source16=values(B::SIZE,113);let source128=values(P128::SIZE,937);
    let source16_hbm=upload::<m![B,H]>(d,&source16).await?;
    let source128_hbm=upload::<m![P128,H]>(d,&source128).await?;
    for variant in 0..2 {
        let mut expected=values(P128::SIZE,1217+variant*999);
        let mut target=upload::<m![P128,H]>(d,&expected).await?;
        let (rows,source)=if variant==0{(B::SIZE,&source16)}else{(P128::SIZE,&source128)};
        for sr in [0,7,rows-1] {for tr in [0,63,127] {
            let so:HbmTensor<i32,Chip,m![1]>=HostTensor::<i32,m![1]>::from_vec([(sr*H::SIZE*2)as i32]).to_hbm(&mut d.pdma).await?;
            let to:HbmTensor<i32,Chip,m![1]>=HostTensor::<i32,m![1]>::from_vec([(tr*H::SIZE*2)as i32]).to_hbm(&mut d.pdma).await?;
            if variant==0{launch(ops_media_rows::insert_b16_in_p128,(d,&source16_hbm,&so,&to,&mut target)).await?;}
            else{launch(ops_media_rows::insert_p128_in_p128,(d,&source128_hbm,&so,&to,&mut target)).await?;}
            expected[tr*H::SIZE..(tr+1)*H::SIZE].copy_from_slice(&source[sr*H::SIZE..(sr+1)*H::SIZE]);
            let actual=read(d,&target).await?;
            exact(900+variant,&format!("indexed_copy_source{sr}_target{tr}_all_rows"),&expected,&actual);
        }}
    }
    exact(902,"copy_source16_unchanged",&source16,&read(d,&source16_hbm).await?);
    exact(903,"copy_source128_unchanged",&source128,&read(d,&source128_hbm).await?);
    println!("MEDIA_ROW_COPY_PASS variants=2 copies=18 full_destination_checked_each=true");Ok(())
}
#[tokio::main]
async fn main()->Result<()> {
    let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone())?;
    let mut d=Device::new(batch::encode_patches.topology())?;
    row_copy_gate(&mut d).await?;
    let path=load::model_dir()?.join("model.safetensors");let f=File::open(&path)?;let mmap=unsafe{memmap2::Mmap::map(&f)?};let t=safetensors::SafeTensors::deserialize(&mmap)?;
    macro_rules! weight {($name:expr)=>{load_bf16(&mut d,&t,$name).await?};}
    let position_host:HostTensor<bf16,m![Pv,Dummy2,Mv]>=HostTensor::from_safetensors(&t.tensor("model.vision_embedder.pos_embedding")?)?;
    let dense_cpu:HostTensor<bf16,m![Mv,Rv]>=HostTensor::from_safetensors(&t.tensor("model.vision_embedder.patch_dense.weight")?)?;
    let dense_cpu=dense_cpu.into_vec();
    let dense_bias_cpu:HostTensor<bf16,m![Mv]>=HostTensor::from_safetensors(&t.tensor("model.vision_embedder.patch_dense.bias")?)?;
    let dense_bias_cpu=dense_bias_cpu.into_vec();
    let project_cpu:HostTensor<bf16,m![H,Ov]>=HostTensor::from_safetensors(&t.tensor("model.embed_vision.embedding_projection.weight")?)?;
    let project_cpu=project_cpu.into_vec();
    let w=Weights{
        ln1w:weight!("model.vision_embedder.patch_ln1.weight"),ln1b:weight!("model.vision_embedder.patch_ln1.bias"),
        dense:weight!("model.vision_embedder.patch_dense.weight"),dense_bias:weight!("model.vision_embedder.patch_dense.bias"),
        ln2w:weight!("model.vision_embedder.patch_ln2.weight"),ln2b:weight!("model.vision_embedder.patch_ln2.bias"),
        ln3w:weight!("model.vision_embedder.pos_norm.weight"),ln3b:weight!("model.vision_embedder.pos_norm.bias"),
        project:weight!("model.embed_vision.embedding_projection.weight"),position:position_host.into_vec(),
    };
    println!("MEDIA_MODEL weights=provided_vision_only path={} B={} Rv={} Mv={} H={}",path.display(),B::SIZE,Rv::SIZE,Mv::SIZE,H::SIZE);
    let base=base_pixels();let base_loc:Vec<_>=(0..B::SIZE).map(|r|if r==15{(Pv::SIZE+7,Pv::SIZE+13)}else{(r*73,r*61)}).collect();
    let mut first:Option<(Vec<bf16>,Vec<bf16>,Vec<bf16>)>=None;
    for (case,active) in [16,16,7,1].into_iter().enumerate(){
        let order:Vec<_>=(0..B::SIZE).map(|r|if case==1{(r*5+3)%16}else{r}).collect();
        let mut pixels=Vec::new();let mut loc=Vec::new();for (row,&source) in order.iter().enumerate(){
            pixels.extend(if row<active{base[source*Rv::SIZE..(source+1)*Rv::SIZE].to_vec()}else{vec![bf16::from_bits(0);Rv::SIZE]});loc.push(base_loc[source]);}
        let pos=positions(&w,&loc,active);
        let mut fused=BatchBuffers::new(&mut d,&pixels,&pos,true).await?;fused.fused(&mut d,&w).await?;
        let fused_out=read(&mut d,&fused.out).await?;
        let mut batched=BatchBuffers::new(&mut d,&pixels,&pos,true).await?;batched.run(&mut d,&w).await?;
        let bp=read(&mut d,&batched.patch).await?;let bn=read(&mut d,batched.normed.as_ref().unwrap()).await?;let bo=read(&mut d,&batched.out).await?;
        let mut rows=Vec::new();let mut sp=Vec::new();let mut sn=Vec::new();let mut so=Vec::new();
        for row in 0..B::SIZE {
            let mut b=ScalarBuffers::new(&mut d,&pixels[row*Rv::SIZE..(row+1)*Rv::SIZE],&pos[row*Mv::SIZE..(row+1)*Mv::SIZE],true).await?;
            b.run(&mut d,&w).await?;sp.extend(read(&mut d,&b.patch).await?);sn.extend(read(&mut d,b.normed.as_ref().unwrap()).await?);so.extend(read(&mut d,&b.out).await?);rows.push(b);
        }
        // Observe the exact normalized inputs separately; no CPU approximation
        // of device division/sqrt is substituted into a projection oracle.
        let mut patch_normal=buffer::<m![B,Rv]>(&mut d,true).await?;
        launch(batch::normalized_patch,(&mut d,&batched.x,&w.ln1w,&w.ln1b,&mut patch_normal)).await?;
        let pn=read(&mut d,&patch_normal).await?;
        let mut dense_observed=buffer::<m![B,Mv]>(&mut d,true).await?;
        launch(batch::patch_dense_before_ln2,(&mut d,&patch_normal,&w.dense,&w.dense_bias,&mut dense_observed)).await?;
        let dense_observed=read(&mut d,&dense_observed).await?;
        let text_input=upload::<m![B,Ov]>(&mut d,&bn).await?;
        let mut text_normal=buffer::<m![B,Ov]>(&mut d,true).await?;
        launch(batch::text_rms_input,(&mut d,&text_input,&mut text_normal)).await?;
        let tn=read(&mut d,&text_normal).await?;
        let mut scalar_pn=Vec::new();let mut scalar_tn=Vec::new();
        for (row,b) in rows.iter().enumerate(){
            let mut pn=buffer::<m![Rv]>(&mut d,true).await?;
            launch(scalar::normalized_patch,(&mut d,&b.x,&w.ln1w,&w.ln1b,&mut pn)).await?;
            scalar_pn.extend(read(&mut d,&pn).await?);
            let input=upload::<m![Ov]>(&mut d,&sn[row*Ov::SIZE..(row+1)*Ov::SIZE]).await?;
            let mut tn=buffer::<m![Ov]>(&mut d,true).await?;
            launch(scalar::text_rms_input,(&mut d,&input,&mut tn)).await?;
            scalar_tn.extend(read(&mut d,&tn).await?);
        }
        exact(case,"patch_norm_only",&scalar_pn,&pn);exact(case,"text_rms_only",&scalar_tn,&tn);
        sampled_projection(case,"patch_dense",&pn,&dense_cpu,Some(&dense_bias_cpu),&dense_observed,Rv::SIZE,Mv::SIZE);
        sampled_projection(case,"text_dense",&tn,&project_cpu,None,&bo,Ov::SIZE,H::SIZE);
        exact(case,"patch_embed",&sp,&bp);exact(case,"position_norm",&sn,&bn);exact(case,"text_projection",&so,&bo);exact(case,"fused_final",&so,&fused_out);
        if case==0{first=Some((sp.clone(),sn.clone(),so.clone()));}
        if case==1 {let original=first.as_ref().unwrap();for(name,all,got,width)in[("permutation_patch",&original.0,&bp,Mv::SIZE),("permutation_norm",&original.1,&bn,Mv::SIZE),("permutation_output",&original.2,&bo,H::SIZE)] {
            let expected:Vec<_>=order.iter().flat_map(|&r|all[r*width..(r+1)*width].iter().copied()).collect();exact(case,name,&expected,got);
        }}
        println!("MEDIA_CASE_PASS case={case} active={active} all_padded_rows_checked=true");
        if case==0 {
            for variant in [0,1,2,2,1,0] {let start=Instant::now();for _ in 0..5 {
                match variant {0=>for b in &mut rows{b.run(&mut d,&w).await?;},1=>batched.run(&mut d,&w).await?,_=>fused.fused(&mut d,&w).await?,}
            }println!("MEDIA_KERNEL variant={variant} ms={:.9} scope=16_patches_completed_no_upload_allocation_download_scatter",start.elapsed().as_secs_f64()*200.);}
            for variant in [0,1,2,2,1,0] {let start=Instant::now();for _ in 0..3 {inclusive(&mut d,&w,&pixels,&loc,variant).await?;}
                println!("MEDIA_INCLUSIVE variant={variant} ms={:.9} scope=16_patches_position_gather_upload_allocations_completed_kernels_no_download_scatter",start.elapsed().as_secs_f64()*1000./3.);}
            if std::env::var_os("FURIOSA_OPT_PROFILE").is_some(){
                println!("PROFILE_MEDIA variant=scalar_one_patch");collector.enable(true);rows[2].run(&mut d,&w).await?;collector.enable(false);
                println!("PROFILE_MEDIA variant=batch16_three_launch");collector.enable(true);batched.run(&mut d,&w).await?;collector.enable(false);
                println!("PROFILE_MEDIA variant=batch16_fused");collector.enable(true);fused.fused(&mut d,&w).await?;collector.enable(false);
            }
        }
    }
    println!("PASS media vision stage/full/permutation/padding exact with provided weights");Ok(())
}

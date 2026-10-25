//! Complete four-launch oracle and timings with supplied model weights.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::{B,H},ops_norm_batch as norm,
    ops_native_fp8_b16_staged_fused as ffn,ops_b16_ffn_epilogue as fused};
use super::load::{self,Layer,MlpWeights};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
struct Scratch {
    projected:HbmTensor<bf16,Chip,m![B,H]>,
    residual:HbmTensor<bf16,Chip,m![B,H]>,
    norm:HbmTensor<bf16,Chip,m![B,H]>,
    qinput:HbmTensor<f8e4m3,Chip,ffn::Input>,
    xscale:HbmTensor<f32,Chip,m![B]>,
    hidden:HbmTensor<bf16,Chip,ffn::Hidden>,
    qhidden:HbmTensor<f8e4m3,Chip,ffn::QuantHidden>,
    hscale:HbmTensor<f32,Chip,m![B]>,
}
impl Scratch {
    async fn new(d:&mut Device,seed:usize,permuted:bool)->Result<Self> {
        let values=|kind:usize| (0..B::SIZE*H::SIZE).map(move |i| {
            let row=if permuted {(i/H::SIZE*5+3)%B::SIZE} else {i/H::SIZE};
            let j=row*H::SIZE+i%H::SIZE;
            let x=((j*73+seed*131+kind*19)%1009) as f32/504.0-1.0;
            bf16::from_f32(x*2f32.powi((row%7) as i32-3))
        });
        macro_rules! poison {($d:ty,$m:ty,$v:expr)=>{
            HostTensor::<$d,$m>::from_vec((0..<$m>::SIZE).map(|_|$v)).to_hbm(&mut d.pdma).await?
        };}
        Ok(Self {
            projected:HostTensor::<bf16,m![B,H]>::from_vec(values(0)).to_hbm(&mut d.pdma).await?,
            residual:HostTensor::<bf16,m![B,H]>::from_vec(values(1)).to_hbm(&mut d.pdma).await?,
            norm:poison!(bf16,m![B,H],bf16::from_bits(0x7fc1)),
            qinput:poison!(f8e4m3,ffn::Input,f8e4m3::from_f32(f32::NAN)),
            xscale:poison!(f32,m![B],f32::NAN),
            hidden:poison!(bf16,ffn::Hidden,bf16::from_bits(0x7fc1)),
            qhidden:poison!(f8e4m3,ffn::QuantHidden,f8e4m3::from_f32(f32::NAN)),
            hscale:poison!(f32,m![B],f32::NAN),
        })
    }
    async fn reset_inputs(&mut self,d:&mut Device,seed:usize,permuted:bool)->Result<()> {
        // Input allocation/upload is outside each measured complete kernel chain.
        for kind in 0..2 {
            let values=(0..B::SIZE*H::SIZE).map(|i| {
                let row=if permuted {(i/H::SIZE*5+3)%B::SIZE} else {i/H::SIZE};
                let j=row*H::SIZE+i%H::SIZE;
                let x=((j*73+seed*131+kind*19)%1009) as f32/504.0-1.0;
                bf16::from_f32(x*2f32.powi((row%7) as i32-3))
            });
            let x=HostTensor::<bf16,m![B,H]>::from_vec(values).to_hbm(&mut d.pdma).await?;
            if kind==0 {self.projected=x;}else{self.residual=x;}
        }
        Ok(())
    }
    async fn snapshots(&self,d:&mut Device)->Result<Vec<Vec<u8>>> {
        macro_rules! get {($field:ident,$m:ty)=>{
            self.$field.to_host::<$m>(&mut d.pdma).await?.into_inner().into_buf()
        };}
        Ok(vec![get!(projected,m![B,H]),get!(residual,m![B,H]),get!(norm,m![B,H]),
            get!(qinput,ffn::Input),get!(xscale,m![B]),get!(hidden,ffn::Hidden),
            get!(qhidden,ffn::QuantHidden),get!(hscale,m![B])])
    }
}
async fn chain(d:&mut Device,s:&mut Scratch,w:&MlpWeights,
    post_attn:&HbmTensor<bf16,Chip,m![H]>,pre:&HbmTensor<bf16,Chip,m![H]>,
    post:&HbmTensor<bf16,Chip,m![H]>,gate:&HbmTensor<bf16,Chip,m![1#8]>,candidate:bool)->Result<()> {
    let n=&w.native_fp8;
    if candidate {
        launch(fused::post_attention_ffn,(d,&mut s.projected,&mut s.residual,
            post_attn,pre,post,gate,&mut s.norm,&n.up,&n.gate,&n.down,
            &n.up_dequant,&n.gate_dequant,&n.down_dequant,
            &w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
            &mut s.qinput,&mut s.xscale,&mut s.hidden,&mut s.qhidden,&mut s.hscale)).await?;
    } else {
        launch(norm::post_attention_b16,(d,&s.projected,post_attn,&mut s.residual)).await?;
        launch(norm::rmsnorm_b16,(d,&s.residual,pre,&mut s.norm)).await?;
        launch(ffn::batch_ffn,(d,&s.norm,&n.up,&n.gate,&n.down,
            &n.up_dequant,&n.gate_dequant,&n.down_dequant,
            &w.up_global_scale,&w.gate_global_scale,&w.down_global_scale,
            &mut s.qinput,&mut s.xscale,&mut s.hidden,&mut s.qhidden,&mut s.hscale,
            &mut s.projected)).await?;
        launch(norm::post_feedforward_b16,(d,&s.projected,post,gate,&mut s.residual)).await?;
    }
    Ok(())
}
pub async fn run()->Result<()> {
    let mut d=Device::new(fused::post_attention_ffn.topology())?;
    let weights=load::load_weights(&mut d).await?;
    let signed=HostTensor::<bf16,m![1#8]>::from_vec([bf16::from_f32(-0.75);8]).to_hbm(&mut d.pdma).await?;
    let mut total=0usize;
    for (case,index) in [0usize,5,47].into_iter().enumerate() {
        let (w,pa,pre,post,lg)=match &weights.layers[index] {
            Layer::Sliding(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar),
            Layer::Full(l)=>(&l.mlp,&l.post_attention_norm,&l.pre_feedforward_norm,&l.post_feedforward_norm,&l.layer_scalar),
        };
        let lg=if case==2 {&signed}else{lg};
        let permuted=case==1;
        let mut a=Scratch::new(&mut d,case+1,permuted).await?;
        let mut b=Scratch::new(&mut d,case+1,permuted).await?;
        chain(&mut d,&mut b,w,pa,pre,post,lg,true).await?;
        let candidate=b.snapshots(&mut d).await?;
        chain(&mut d,&mut a,w,pa,pre,post,lg,false).await?;
        let control=a.snapshots(&mut d).await?;
        for (stage,(a,b)) in control.iter().zip(&candidate).enumerate() {
            assert_eq!(a.len(),b.len());
            let diff=a.iter().zip(b).filter(|(a,b)|a!=b).count();total+=a.len();
            println!("B16_EPILOGUE_EXACT case={case} layer={index} stage={stage} bytes={} byte_diff={diff}",a.len());
            assert_eq!(diff,0,"stage{stage} changed");
            if matches!(stage,0|1|2|5) {assert!(b.chunks_exact(2).all(|v|u16::from_le_bytes([v[0],v[1]])&0x7f80!=0x7f80));}
        }
        for (round,candidate) in [false,true,true,false].into_iter().enumerate() {
            let s=if candidate {&mut b}else{&mut a};let mut total_ms=0.0;
            for _ in 0..8 {
                s.reset_inputs(&mut d,case+1,permuted).await?;
                let start=std::time::Instant::now();chain(&mut d,s,w,pa,pre,post,lg,candidate).await?;
                total_ms+=start.elapsed().as_secs_f64()*1000.0;
            }
            println!("B16_EPILOGUE_TIME case={case} round={round} candidate={candidate} complete_chain_ms={:.6} samples=8 excludes_upload_download=true",total_ms/8.0);
        }
    }
    println!("PASS exact B16 post-attention+FFN complete chain bytes={total}; model and HTTP pending");
    Ok(())
}

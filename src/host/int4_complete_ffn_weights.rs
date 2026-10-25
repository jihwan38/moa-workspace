//! Diagnostic-only layer0 loading through the unchanged production MLP loader.
//! Keeps original packed tensors and native copies alive, as serving does.
use furiosa_opt_std::prelude::*;
use crate::{Chip,axes::H,ops_ffn_projection_clusters::{JoinedWeight,JoinedScale}};
use super::load::{self,MlpWeights};
pub type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;

pub struct ProductionWeights {
    mlp:MlpWeights,
    pub joined:HbmTensor<f8e4m3,Chip,JoinedWeight>,
    pub joined_scale:HbmTensor<f32,Chip,JoinedScale>,
    pub post_attention:HbmTensor<bf16,Chip,m![H]>,
    pub pre_ffn:HbmTensor<bf16,Chip,m![H]>,
    pub post_ffn:HbmTensor<bf16,Chip,m![H]>,
    pub layer_gate:HbmTensor<bf16,Chip,m![1#8]>,
}
impl ProductionWeights {
    pub async fn load_layer0(d:&mut Device,t:&safetensors::SafeTensors<'_>)->Result<Self> {
        async fn norm(d:&mut Device,t:&safetensors::SafeTensors<'_>,name:&str)->Result<HbmTensor<bf16,Chip,m![H]>> {
            let v=t.tensor(&format!("model.language_model.layers.0.{name}"))?;
            let h:HostTensor<bf16,m![H]>=HostTensor::from_safetensors(&v)?;
            Ok(h.to_hbm(&mut d.pdma).await?)
        }
        let mlp=load::load_mlp(d,t,"model.language_model.layers.0.mlp").await?;
        let (joined,joined_scale)=launch(crate::ops_ffn_projection_join::join,
            (&mut *d,&mlp.native_fp8.up,&mlp.native_fp8.gate,&mlp.native_fp8.up_dequant,&mlp.native_fp8.gate_dequant)).await?;
        let v=t.tensor("model.language_model.layers.0.layer_scalar")?;
        let h:HostTensor<bf16,m![1]>=HostTensor::from_safetensors(&v)?;
        let layer_gate=HostTensor::<bf16,m![1#8]>::from_vec(vec![h.into_vec()[0];8]).to_hbm(&mut d.pdma).await?;
        Ok(Self{mlp,joined,joined_scale,layer_gate,
            post_attention:norm(d,t,"post_attention_layernorm.weight").await?,
            pre_ffn:norm(d,t,"pre_feedforward_layernorm.weight").await?,
            post_ffn:norm(d,t,"post_feedforward_layernorm.weight").await?})
    }
    pub fn down(&self)->&HbmTensor<f8e4m3,Chip,crate::ops_native_fp8_blocked::DownWeight>{&self.mlp.native_fp8.down}
    pub fn down_scale(&self)->&HbmTensor<f32,Chip,m![H]>{&self.mlp.native_fp8.down_dequant}
    pub fn ug(&self)->&HbmTensor<f32,Chip,m![1]>{&self.mlp.up_global_scale}
    pub fn gg(&self)->&HbmTensor<f32,Chip,m![1]>{&self.mlp.gate_global_scale}
    pub fn dg(&self)->&HbmTensor<f32,Chip,m![1]>{&self.mlp.down_global_scale}

    /// Full retained-byte hashes before and after all calls; not an allocator measurement.
    pub async fn fingerprints(&self,d:&mut Device)->Result<Vec<(String,usize,u64)>> {
        async fn one<D:MaterializableScalar,E:M>(d:&mut Device,n:&str,x:&HbmTensor<D,Chip,E>)->Result<(String,usize,u64)> {
            let b=x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf();
            let h=b.iter().fold(0xcbf29ce484222325u64,|a,&v|(a^v as u64).wrapping_mul(0x100000001b3));
            Ok((n.into(),b.len(),h))
        }
        let mut out=Vec::new();
        macro_rules! add {($name:literal,$x:expr)=>{out.push(one(d,$name,$x).await?);};}
        add!("joined",&self.joined);add!("joined_scale",&self.joined_scale);
        add!("up_native",&self.mlp.native_fp8.up);add!("gate_native",&self.mlp.native_fp8.gate);add!("down_native",self.down());
        add!("up_dequant",&self.mlp.native_fp8.up_dequant);add!("gate_dequant",&self.mlp.native_fp8.gate_dequant);add!("down_dequant",self.down_scale());
        add!("up_packed",&self.mlp.up_weight_packed);add!("gate_packed",&self.mlp.gate_weight_packed);add!("down_packed",&self.mlp.down_weight_packed);
        add!("up_blockscale",&self.mlp.up_weight_scale);add!("gate_blockscale",&self.mlp.gate_weight_scale);add!("down_blockscale",&self.mlp.down_weight_scale);
        add!("ug",self.ug());add!("gg",self.gg());add!("dg",self.dg());
        add!("post_attention",&self.post_attention);add!("pre_ffn",&self.pre_ffn);add!("post_ffn",&self.post_ffn);add!("layer_gate",&self.layer_gate);
        Ok(out)
    }
}

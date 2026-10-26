//! Parameter-free gamma compensation diagnostic; unchanged device functions.
//! Fail-stop BF16 inverse/product proofs precede quantized drift interpretation.
#![allow(dead_code)]
use furiosa_opt_gemma4::{Chip,axes::{P128,P512,H,L},host::{load,int4_complete_ffn_weights::ProductionWeights},
    ops_batch_layouts::{P512H960,P512L960},ops_ffn_projection_clusters as prod,
    ops_int4_grouped_activation_projection as candidate,ops_int4_grouped_activation_reference as reference,
    ops_int4_ffn_bridges as bridge,ops_int4_grouped_activation_store as fused,
    ops_norm_blocked_residual as tail,i4_grouped_activation_raw_adapter as raw};
use furiosa_opt_std::prelude::*;
use std::{fs::File,time::Instant};
#[path="support/npu_profile.rs"] mod npu_profile;
#[path="support/int4_grouped_activation_fixture.rs"] mod cpu;
#[path="support/int4_grouped_complete_fixture.rs"] mod fixture;
use cpu::{Result,InputCpu,WeightCpu,HIDDEN,EXPANDED,TOKENS};
#[path="support/int4_channel_balance.rs"] mod balance;
use fused::{Mantissa,Record};
type Sliding = furiosa_opt_gemma4::ops_output_p128_h64_blocked::Output;
async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,b:Vec<u8>)->Result<HbmTensor<D,Chip,E>> {
    assert_eq!(b.len()*8,E::SIZE*D::BITS);
    Ok(HostTensor::<D,E>::from_buf(b).to_hbm(&mut d.pdma).await?)
}
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Result<Vec<u8>> {
    Ok(x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf())
}
fn floats(v:&[f32])->Vec<u8> {v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn joined(a:&[u8],b:&[u8])->Vec<u8> {let mut v=Vec::with_capacity(a.len()+b.len());v.extend_from_slice(a);v.extend_from_slice(b);v}
fn compare_joined(got:&[u8],a:&[u8],b:&[u8],label:&str) {
    assert_eq!(got.len(),a.len()+b.len());assert_eq!(&got[..a.len()],a,"immutable UP plane {label}");assert_eq!(&got[a.len()..],b,"immutable GATE plane {label}");
}
async fn poison<E:M>(d:&mut Device,payload:u16)->Result<HbmTensor<bf16,Chip,E>> {
    Ok(HostTensor::<bf16,E>::from_vec(std::iter::repeat_n(bf16::from_bits(payload),E::SIZE)).to_hbm(&mut d.pdma).await?)
}
struct InputGpu<E:M,S:M> {packed:HbmTensor<i4,Chip,E>,dense:HbmTensor<f8e4m3,Chip,E>,scales:HbmTensor<f32,Chip,S>}
impl<E:M,S:M> InputGpu<E,S> {
    async fn new(d:&mut Device,x:&InputCpu)->Result<Self> {
        Ok(Self{packed:upload(d,x.packed.clone()).await?,dense:upload(d,x.dense.clone()).await?,scales:upload(d,floats(&x.scales)).await?})
    }
    async fn immutable(&self,d:&mut Device,x:&InputCpu)->Result<usize> {
        assert_eq!(read(d,&self.packed).await?,x.packed,"I4 input immutable");
        assert_eq!(read(d,&self.dense).await?,x.dense,"E4 input immutable");
        assert_eq!(read(d,&self.scales).await?,floats(&x.scales),"input scales immutable");
        Ok(x.packed.len()+x.dense.len()+x.scales.len()*4)
    }
}
struct WeightsGpu {
    joined_i4:HbmTensor<i4,Chip,candidate::JoinedWeight>,joined_e4:HbmTensor<f8e4m3,Chip,reference::JoinedWeight>,
    joined_scales:HbmTensor<f32,Chip,candidate::JoinedBlockScale>,
    down_i4:HbmTensor<i4,Chip,candidate::DownWeight>,down_e4:HbmTensor<f8e4m3,Chip,candidate::DownWeight>,
    down_scales:HbmTensor<f32,Chip,candidate::DownBlockScale>,global:HbmTensor<f32,Chip,m![1]>,
}
impl WeightsGpu {
    async fn new(d:&mut Device,up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu)->Result<Self> {
        assert_eq!((up.rows,up.cols),(EXPANDED,HIDDEN));assert_eq!((gate.rows,gate.cols),(EXPANDED,HIDDEN));assert_eq!((down.rows,down.cols),(HIDDEN,EXPANDED));
        Ok(Self{
            joined_i4:upload(d,joined(&up.packed,&gate.packed)).await?,
            joined_e4:upload(d,joined(&up.dense,&gate.dense)).await?,
            joined_scales:upload(d,joined(&floats(&up.scales),&floats(&gate.scales))).await?,
            down_i4:upload(d,down.packed.clone()).await?,down_e4:upload(d,down.dense.clone()).await?,
            down_scales:upload(d,floats(&down.scales)).await?,
            global:upload(d,down.global.to_le_bytes().to_vec()).await?,
        })
    }
    async fn immutable(&self,d:&mut Device,up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu)->Result<usize> {
        compare_joined(&read(d,&self.joined_i4).await?,&up.packed,&gate.packed,"I4");
        compare_joined(&read(d,&self.joined_e4).await?,&up.dense,&gate.dense,"E4");
        compare_joined(&read(d,&self.joined_scales).await?,&floats(&up.scales),&floats(&gate.scales),"scales");
        assert_eq!(read(d,&self.down_i4).await?,down.packed,"DOWN I4 immutable");
        assert_eq!(read(d,&self.down_e4).await?,down.dense,"DOWN E4 immutable");
        assert_eq!(read(d,&self.down_scales).await?,floats(&down.scales),"DOWN scales immutable");
        assert_eq!(read(d,&self.global).await?,down.global.to_le_bytes(),"DOWN global immutable");
        Ok(up.packed.len()+gate.packed.len()+down.packed.len()+up.dense.len()+gate.dense.len()+down.dense.len()+
            (up.scales.len()+gate.scales.len()+down.scales.len()+1)*4)
    }
}

struct Inputs {sliding:[HbmTensor<bf16,Chip,Sliding>;4],canonical:Vec<u8>,wire:[Vec<u8>;4]}
impl Inputs {
    async fn new(d:&mut Device,permuted:bool)->Result<Self>{
        let canonical=fixture::input(permuted,false);let wire=fixture::sliding(&canonical);
        let mut v=Vec::new();for b in &wire{v.push(upload(d,b.clone()).await?);}
        Ok(Self{sliding:v.try_into().ok().unwrap(),canonical,wire})
    }
    async fn immutable(&self,d:&mut Device)->Result<usize>{let mut n=0;for(i,x)in self.sliding.iter().enumerate(){assert_eq!(read(d,x).await?,self.wire[i]);n+=self.wire[i].len();}Ok(n)}
}
struct Quantizer {
    table:HbmTensor<i32,Chip,m![Mantissa,Record]>,
    hshift:HbmTensor<i32,Chip,m![H%8]>,lshift:HbmTensor<i32,Chip,m![L%8]>,table_bytes:Vec<u8>,shift_bytes:Vec<u8>,
}
impl Quantizer {
    async fn new(d:&mut Device)->Result<Self>{
        let mut table_bytes=Vec::new();for m in 128u32..256{let u=7*((1u32<<20).div_ceil(m));
            for lane in 0..8u32{let b=if lane==0{(u as f32).to_bits()}else{0x5a3c0000|(m<<4)|lane};table_bytes.extend(b.to_le_bytes());}}
        let shift_bytes:Vec<u8>=(0..8i32).flat_map(|i|(4*i).to_le_bytes()).collect();
        Ok(Self{table:upload(d,table_bytes.clone()).await?,hshift:upload(d,shift_bytes.clone()).await?,
            lshift:upload(d,shift_bytes.clone()).await?,table_bytes,shift_bytes})
    }
    async fn immutable(&self,d:&mut Device)->Result<usize>{
        assert_eq!(read(d,&self.table).await?,self.table_bytes);assert_eq!(read(d,&self.hshift).await?,self.shift_bytes);assert_eq!(read(d,&self.lshift).await?,self.shift_bytes);
        Ok(self.table_bytes.len()+2*self.shift_bytes.len())
    }
}
struct CandidateScratch {
    residual:HbmTensor<bf16,Chip,tail::Residual>,prefix:HbmTensor<bf16,Chip,bridge::InputBf16>,
    xp:HbmTensor<i32,Chip,raw::PackedInput>,xs:HbmTensor<f32,Chip,candidate::InputScale>,xo:HbmTensor<i32,Chip,candidate::InputScale>,
    joined:HbmTensor<bf16,Chip,candidate::JoinedOutput>,hidden:HbmTensor<bf16,Chip,bridge::HiddenBf16>,
    hp:HbmTensor<i32,Chip,raw::PackedHidden>,hs:HbmTensor<f32,Chip,candidate::HiddenScale>,ho:HbmTensor<i32,Chip,candidate::HiddenScale>,down:HbmTensor<bf16,Chip,candidate::Output>,
}
#[derive(Clone)]
struct Snapshot {residual:Vec<u8>,prefix:Vec<u8>,xp:Vec<u8>,xs:Vec<u8>,xo:Vec<u8>,joined:Vec<u8>,hidden:Vec<u8>,hp:Vec<u8>,hs:Vec<u8>,ho:Vec<u8>,down:Vec<u8>}
impl Snapshot {
    fn fields(&self)->[(&str,&[u8]);11]{[("residual",&self.residual),("prefix",&self.prefix),("xp",&self.xp),("xs",&self.xs),("xo",&self.xo),("joined",&self.joined),("hidden",&self.hidden),("hp",&self.hp),("hs",&self.hs),("ho",&self.ho),("down",&self.down)]}
    fn equal(&self,other:&Self,label:&str)->usize{let mut n=0;for((a,x),(b,y))in self.fields().into_iter().zip(other.fields()){assert_eq!(a,b);assert_eq!(x,y,"{label} {a}");n+=x.len();}n}
    fn permutation(&self,base:&Self)->usize{
        let mut n=0;n+=fixture::compare_rows(&fixture::canonical_residual(&self.residual),&fixture::canonical_residual(&base.residual),HIDDEN,2,"residual");
        n+=fixture::compare_rows(&self.prefix,&base.prefix,HIDDEN,2,"prefix");n+=fixture::compare_rows(&self.hidden,&base.hidden,EXPANDED,2,"hidden");
        n+=cpu::compare_permutation(&self.joined,&base.joined,EXPANDED,2);n+=cpu::compare_permutation(&self.down,&base.down,HIDDEN,1);
        n+=fixture::compare_group_rows(&self.xs,&base.xs,HIDDEN,"inputscale");n+=fixture::compare_group_rows(&self.hs,&base.hs,EXPANDED,"hiddenscale");
        n+=fixture::compare_group_rows(&self.xo,&base.xo,HIDDEN,"inputoffset");n+=fixture::compare_group_rows(&self.ho,&base.ho,EXPANDED,"hiddenoffset");
        // Re-quantization independently checks every packed nibble and its row mapping.
        let x=fixture::quantize(&self.prefix,HIDDEN);let bx=fixture::quantize(&base.prefix,HIDDEN);
        let h=fixture::quantize(&self.hidden,EXPANDED);let bh=fixture::quantize(&base.hidden,EXPANDED);
        assert_eq!(self.xp,x.packed);assert_eq!(base.xp,bx.packed);assert_eq!(self.hp,h.packed);assert_eq!(base.hp,bh.packed);
        for p in 0..512{assert_eq!(&x.codes[p*HIDDEN..(p+1)*HIDDEN],&bx.codes[cpu::perm(p)*HIDDEN..(cpu::perm(p)+1)*HIDDEN]);
            assert_eq!(&h.codes[p*EXPANDED..(p+1)*EXPANDED],&bh.codes[cpu::perm(p)*EXPANDED..(cpu::perm(p)+1)*EXPANDED]);}
        n+512*(HIDDEN+EXPANDED)
    }
}
impl CandidateScratch {
    async fn new(d:&mut Device,r:&[u8])->Result<Self>{Ok(Self{
        residual:upload(d,r.to_vec()).await?,prefix:poison(d,0x7fc1).await?,xp:upload(d,vec![0x5a;512*HIDDEN/2]).await?,xs:upload(d,floats(&vec![f32::NAN;512*HIDDEN/960])).await?,xo:upload(d,vec![0xa7;512*HIDDEN/960*4]).await?,
        joined:poison(d,0x7fc2).await?,hidden:poison(d,0x7fc3).await?,hp:upload(d,vec![0xa5;512*EXPANDED/2]).await?,hs:upload(d,floats(&vec![f32::NAN;512*EXPANDED/960])).await?,ho:upload(d,vec![0xb9;512*EXPANDED/960*4]).await?,down:poison(d,0x7fc4).await?})}
    async fn reset(&mut self,d:&mut Device,r:&[u8])->Result<()>{self.residual=upload(d,r.to_vec()).await?;Ok(())}
    async fn snapshot(&self,d:&mut Device)->Result<Snapshot>{Ok(Snapshot{residual:read(d,&self.residual).await?,prefix:read(d,&self.prefix).await?,xp:read(d,&self.xp).await?,xs:read(d,&self.xs).await?,xo:read(d,&self.xo).await?,joined:read(d,&self.joined).await?,hidden:read(d,&self.hidden).await?,hp:read(d,&self.hp).await?,hs:read(d,&self.hs).await?,ho:read(d,&self.ho).await?,down:read(d,&self.down).await?})}
}
struct ProductionScratch {
    residual:HbmTensor<bf16,Chip,tail::Residual>,qi:HbmTensor<f8e4m3,Chip,P512H960>,xs:HbmTensor<f32,Chip,m![P512]>,
    joined:HbmTensor<bf16,Chip,prod::JoinedOutput>,qh:HbmTensor<f8e4m3,Chip,P512L960>,hs:HbmTensor<f32,Chip,m![P512]>,
    down:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_native_fp8_p512_downblocked::Output>,
}
impl ProductionScratch {
    async fn new(d:&mut Device,r:&[u8])->Result<Self>{Ok(Self{residual:upload(d,r.to_vec()).await?,qi:upload(d,vec![0x7f;512*HIDDEN]).await?,xs:upload(d,floats(&vec![f32::NAN;512])).await?,
        joined:poison(d,0x7fc5).await?,qh:upload(d,vec![0x7f;512*EXPANDED]).await?,hs:upload(d,floats(&vec![f32::NAN;512])).await?,down:poison(d,0x7fc6).await?})}
    async fn reset(&mut self,d:&mut Device,r:&[u8])->Result<()>{self.residual=upload(d,r.to_vec()).await?;Ok(())}
    async fn snapshot(&self,d:&mut Device)->Result<Vec<Vec<u8>>>{Ok(vec![read(d,&self.residual).await?,read(d,&self.qi).await?,read(d,&self.xs).await?,read(d,&self.joined).await?,read(d,&self.qh).await?,read(d,&self.hs).await?,read(d,&self.down).await?])}
}
async fn candidate_prefix(d:&mut Device,p:&ProductionWeights,pre:&HbmTensor<bf16,Chip,m![H]>,x:&Inputs,s:&mut CandidateScratch)->Result<()>{
    launch(bridge::prefix_sliding_bf16,(d,&x.sliding[0],&x.sliding[1],&x.sliding[2],&x.sliding[3],&p.post_attention,pre,&mut s.residual,&mut s.prefix)).await?;Ok(())
}
async fn candidate_rest(d:&mut Device,p:&ProductionWeights,w:&WeightsGpu,q:&Quantizer,s:&mut CandidateScratch)->Result<()>{
    launch(fused::quantize_pack_input,(&mut *d,&s.prefix,&q.table,&q.hshift,&mut s.xp,&mut s.xs,&mut s.xo)).await?;
    raw::project_upgate(d,&s.xp,&s.xs,&w.joined_i4,&w.joined_scales,&mut s.joined).await?;
    launch(bridge::geglu_bf16,(&mut *d,&s.joined,p.ug(),p.gg(),&mut s.hidden)).await?;
    launch(fused::quantize_pack_hidden,(&mut *d,&s.hidden,&q.table,&q.lshift,&mut s.hp,&mut s.hs,&mut s.ho)).await?;
    raw::project_down(d,&s.hp,&s.hs,&w.down_i4,&w.down_scales,&w.global,&mut s.down).await?;
    launch(tail::post_feedforward,(&mut *d,&s.down,&p.post_ffn,&p.layer_gate,&mut s.residual)).await?;Ok(())
}
async fn production_chain(d:&mut Device,p:&ProductionWeights,x:&Inputs,s:&mut ProductionScratch)->Result<()>{
    launch(furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned,
        (d,(&x.sliding[0],&x.sliding[1],&x.sliding[2],&x.sliding[3]),(&p.post_attention,&p.pre_ffn,&p.post_ffn,&p.layer_gate),
         &mut s.residual,&p.joined,&p.joined_scale,p.down(),p.down_scale(),p.ug(),p.gg(),p.dg(),
         &mut s.qi,&mut s.xs,&mut s.joined,&mut s.qh,&mut s.hs,&mut s.down)).await?;Ok(())
}

/// Separate original P128 norm paths, CPU quotient, E4 integer-code dots,
/// original P128 GeGLU and original P128 tail. All copies/downloads are untimed.
async fn intended_step(d:&mut Device,p:&ProductionWeights,pre:&HbmTensor<bf16,Chip,m![H]>,w:&WeightsGpu,x:&Inputs,residual:&[u8],got:&Snapshot,mid_got:&[u8],up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu,round:usize,step:usize)->Result<(Vec<u8>,usize,usize,usize)>{
    let mut prefix=Vec::new();let mut mid=Vec::new();
    for quarter in 0..4 {
        let begin=quarter*128*HIDDEN*2;let end=begin+128*HIDDEN*2;
        let projected:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,x.canonical[begin..end].to_vec()).await?;
        let mut r:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,residual[begin..end].to_vec()).await?;
        let mut n=poison::<m![P128,H]>(d,0x7fc7).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::post_attention_p128,(&mut *d,&projected,&p.post_attention,&mut r)).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::rmsnorm_p128,(&mut *d,&r,pre,&mut n)).await?;
        mid.extend(read(d,&r).await?);prefix.extend(read(d,&n).await?);
        assert_eq!(read(d,&projected).await?,x.canonical[begin..end],"immutable original prefix input");
    }
    let mut exact=cpu::compare_bits(&got.prefix,&prefix,"independent prefix canonical BF16");
    exact+=cpu::compare_bits(mid_got,&fixture::block_residual(&mid),"independent prefix persisted residual");
    let qi=fixture::quantize(&prefix,HIDDEN);assert_eq!(got.xp,qi.packed);assert_eq!(got.xs,floats(&qi.scales));assert_eq!(got.xo,fixture::offset_bytes(&prefix,HIDDEN));
    fixture::quantization_metrics(&prefix,&qi,round,step,"prefix");
    let xg=InputGpu::<P512H960,candidate::InputScale>::new(d,&qi).await?;
    let mut typed=poison::<candidate::JoinedOutput>(d,0x7fc8).await?;
    let mut dots=poison::<reference::JoinedOutput>(d,0x7fc9).await?;
    // Normal typed launches embed exactly the images used by the checked raw binding.
    launch(candidate::project_upgate,(&mut *d,&xg.packed,&xg.scales,&w.joined_i4,&w.joined_scales,&mut typed)).await?;
    launch(reference::project_upgate,(&mut *d,&xg.dense,&xg.scales,&w.joined_e4,&w.joined_scales,&mut dots)).await?;
    let jb=read(d,&dots).await?;exact+=cpu::compare_bits(&got.joined,&jb,"intended E4-code UP/GATE");
    exact+=cpu::compare_bits(&read(d,&typed).await?,&jb,"typed/raw UP/GATE image");
    let(mut sampled,mut immutable)=(0,0);
    for(name,plane,weight)in[("UP",0,up),("GATE",1,gate)]{let(n,radius)=cpu::sampled_bound(&got.joined,plane,&qi,weight,false);sampled+=n;
        println!("I4_GROUPED_COMPLETE_F64 round={round} step={step} projection={name} samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");}
    let upcan=fixture::canonical_projection(&jb,EXPANDED,0);let gatecan=fixture::canonical_projection(&jb,EXPANDED,1);
    let mut hidden=Vec::with_capacity(512*EXPANDED*2);
    for quarter in 0..4 {let begin=quarter*128*EXPANDED*2;let end=begin+128*EXPANDED*2;
        let ub=fixture::reference_column_tile(&upcan[begin..end]);let gb=fixture::reference_column_tile(&gatecan[begin..end]);
        let u:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_int4_complete_reference::Operand>=upload(d,ub.clone()).await?;
        let g:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_int4_complete_reference::Operand>=upload(d,gb.clone()).await?;
        let mut h=poison::<furiosa_opt_gemma4::ops_int4_complete_reference::Operand>(d,0x7fca).await?;
        launch(furiosa_opt_gemma4::ops_int4_complete_reference::geglu_p128,(&mut *d,&u,&g,p.ug(),p.gg(),&mut h)).await?;
        hidden.extend(fixture::reference_canonical_tile(&read(d,&h).await?));assert_eq!(read(d,&u).await?,ub);assert_eq!(read(d,&g).await?,gb);
    }
    exact+=cpu::compare_bits(&got.hidden,&hidden,"independent original P128 GeGLU BF16");
    let qh=fixture::quantize(&hidden,EXPANDED);assert_eq!(got.hp,qh.packed);assert_eq!(got.hs,floats(&qh.scales));assert_eq!(got.ho,fixture::offset_bytes(&hidden,EXPANDED));
    fixture::quantization_metrics(&hidden,&qh,round,step,"hidden");
    let hg=InputGpu::<P512L960,candidate::HiddenScale>::new(d,&qh).await?;
    let mut typed_down=poison::<candidate::Output>(d,0x7fcb).await?;let mut ref_down=poison::<candidate::Output>(d,0x7fcc).await?;
    launch(candidate::project_down,(&mut *d,&hg.packed,&hg.scales,&w.down_i4,&w.down_scales,&w.global,&mut typed_down)).await?;
    launch(reference::project_down,(&mut *d,&hg.dense,&hg.scales,&w.down_e4,&w.down_scales,&w.global,&mut ref_down)).await?;
    let db=read(d,&ref_down).await?;exact+=cpu::compare_bits(&got.down,&db,"intended E4-code DOWN");exact+=cpu::compare_bits(&read(d,&typed_down).await?,&db,"typed/raw DOWN image");
    let(n,radius)=cpu::sampled_bound(&got.down,0,&qh,down,true);sampled+=n;
    println!("I4_GROUPED_COMPLETE_F64 round={round} step={step} projection=DOWN samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");
    let downcan=fixture::canonical_projection(&db,HIDDEN,0);let mut final_residual=Vec::new();
    for quarter in 0..4{let begin=quarter*128*HIDDEN*2;let end=begin+128*HIDDEN*2;
        let y:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,downcan[begin..end].to_vec()).await?;
        let mut r:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,mid[begin..end].to_vec()).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::post_feedforward_p128,(&mut *d,&y,&p.post_ffn,&p.layer_gate,&mut r)).await?;
        final_residual.extend(read(d,&r).await?);assert_eq!(read(d,&y).await?,downcan[begin..end]);
    }
    exact+=cpu::compare_bits(&got.residual,&fixture::block_residual(&final_residual),"independent complete residual/layer scalar");
    immutable+=xg.immutable(d,&qi).await?+hg.immutable(d,&qh).await?;
    assert_eq!(read(d,&dots).await?,jb);assert_eq!(read(d,&ref_down).await?,db);
    Ok((final_residual,exact,sampled,immutable))
}

#[tokio::main]
async fn main()->Result<()>{
    if std::env::args().len()!=1{return Err("fixed provided layer0/two row orders/two matched-production feedback steps; no tuning arguments".into());}
    cpu::validate_integer_encodings();cpu::check_output_layout(EXPANDED);cpu::check_output_layout(HIDDEN);
    let file=File::open(load::model_dir()?.join("model.safetensors"))?;
    let mmap=unsafe{memmap2::Mmap::map(&file)?};let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
    let transform=balance::Balance::load(&tensors)?;
    let startup=Instant::now();
    let up=WeightCpu::load(&tensors,"up_proj")?;let gate=WeightCpu::load(&tensors,"gate_proj")?;let down=WeightCpu::load(&tensors,"down_proj")?;
    let balanced_up=balance::weights(&tensors,"up_proj",&up,&transform)?;
    let balanced_gate=balance::weights(&tensors,"gate_proj",&gate,&transform)?;
    println!("I4_BALANCE_STARTUP cpu_decode_inverse_audit_encode_ms={:.6} includes_original_and_balanced_and_metrics=true not_serving_startup=true",startup.elapsed().as_secs_f64()*1e3);
    let mut d=Device::new(candidate::project_upgate.topology())?;
    let p=ProductionWeights::load_layer0(&mut d,&tensors).await?;
    assert_eq!(read(&mut d,&p.pre_ffn).await?,transform.original,"provided original gamma loader");
    let gamma_prime:HbmTensor<bf16,Chip,m![H]>=upload(&mut d,transform.prime.clone()).await?;
    assert_eq!(read(&mut d,p.ug()).await?,up.global.to_le_bytes());assert_eq!(read(&mut d,p.gg()).await?,gate.global.to_le_bytes());assert_eq!(read(&mut d,p.dg()).await?,down.global.to_le_bytes());
    let w=WeightsGpu::new(&mut d,&up,&gate,&down).await?;
    // Separate diagnostic owners, including an unchanged second DOWN copy.
    // This is temporary layer0 storage, not a48-layer memory-fit claim.
    let bw=WeightsGpu::new(&mut d,&balanced_up,&balanced_gate,&down).await?;
    let q=Quantizer::new(&mut d).await?;
    let production_before=p.fingerprints(&mut d).await?;
    println!("I4_BALANCE_OWNERS gamma_prime_bytes={} additional_balanced_packed_bytes={} additional_balanced_dense_reference_bytes={} additional_balanced_scale_bytes={} unchanged_DOWN_copied=true no_model_capacity_claim=true",HIDDEN*2,3*HIDDEN*EXPANDED/2,3*HIDDEN*EXPANDED,3*HIDDEN*EXPANDED/960*4+4);
    let(mut exact,mut samples,mut permutation,mut immutable,mut prefix_values,mut products,mut reuse)=(0usize,0usize,0usize,0usize,0usize,0usize,0usize);
    let mut baseline=Vec::new();let mut base_unbalanced=Vec::new();
    for round in 0..2 {
        let x=Inputs::new(&mut d,round==1).await?;
        let mut incoming=fixture::input(round==1,true);
        for step in 0..2 {
            let rb=fixture::block_residual(&incoming);
            let mut balanced=CandidateScratch::new(&mut d,&rb).await?;
            let mut unbalanced=CandidateScratch::new(&mut d,&rb).await?;
            let mut production=ProductionScratch::new(&mut d,&rb).await?;
            candidate_prefix(&mut d,&p,&gamma_prime,&x,&mut balanced).await?;
            let bp=read(&mut d,&balanced.prefix).await?;let bm=read(&mut d,&balanced.residual).await?;
            candidate_prefix(&mut d,&p,&p.pre_ffn,&x,&mut unbalanced).await?;
            let op=read(&mut d,&unbalanced.prefix).await?;let om=read(&mut d,&unbalanced.residual).await?;
            assert_eq!(bm,om,"gamma cannot alter persisted post-attention residual");
            prefix_values+=transform.prefix(&op,&bp,round,step)?;
            products+=balance::products(&tensors,&transform,&op,&bp,round,step)?;
            // No changed-code projection/drift is interpreted until all prefix
            // inverse and unquantized sampled product prerequisites above pass.
            candidate_rest(&mut d,&p,&bw,&q,&mut balanced).await?;
            let got=balanced.snapshot(&mut d).await?;
            candidate_rest(&mut d,&p,&w,&q,&mut unbalanced).await?;
            let old=unbalanced.snapshot(&mut d).await?;
            println!("I4_BALANCE_INTENDED_BEGIN arm=balanced round={round} step={step}");
            let(_,n,ns,ni)=intended_step(&mut d,&p,&gamma_prime,&bw,&x,&incoming,&got,&bm,&balanced_up,&balanced_gate,&down,round,step).await?;
            exact+=n;samples+=ns;immutable+=ni;
            println!("I4_BALANCE_INTENDED_BEGIN arm=unbalanced round={round} step={step}");
            let(_,n,ns,ni)=intended_step(&mut d,&p,&p.pre_ffn,&w,&x,&incoming,&old,&om,&up,&gate,&down,round,step).await?;
            exact+=n;samples+=ns;immutable+=ni;
            transform.activation_metrics(&op,&bp,&fixture::quantize(&op,HIDDEN),&fixture::quantize(&bp,HIDDEN),round,step);
            production_chain(&mut d,&p,&x,&mut production).await?;let current=production.snapshot(&mut d).await?;
            let mut production_hidden=poison::<bridge::HiddenBf16>(&mut d,0x7fda).await?;
            launch(bridge::geglu_bf16,(&mut d,&production.joined,p.ug(),p.gg(),&mut production_hidden)).await?;
            let ph=read(&mut d,&production_hidden).await?;
            for(label,bal,unbal,prod)in[("joined",&got.joined,&old.joined,&current[3]),("hidden",&got.hidden,&old.hidden,&ph),("down",&got.down,&old.down,&current[6]),("residual",&got.residual,&old.residual,&current[0])]{
                balance::drift(bal,prod,&format!("balanced_vs_q020_{label}"),round,step);
                balance::drift(unbal,prod,&format!("unbalanced_vs_q020_{label}"),round,step);
                balance::drift(bal,unbal,&format!("balanced_vs_unbalanced_{label}"),round,step);
            }
            if round==0{baseline.push(got.clone());base_unbalanced.push(old.clone());}
            else{permutation+=got.permutation(&baseline[step])+old.permutation(&base_unbalanced[step]);}
            balanced.reset(&mut d,&rb).await?;
            candidate_prefix(&mut d,&p,&gamma_prime,&x,&mut balanced).await?;candidate_rest(&mut d,&p,&bw,&q,&mut balanced).await?;
            reuse+=balanced.snapshot(&mut d).await?.equal(&got,"balanced full reuse");
            reuse+=unbalanced.snapshot(&mut d).await?.equal(&old,"unbalanced scratch unchanged by balanced replay");
            for(a,b)in production.snapshot(&mut d).await?.iter().zip(&current){assert_eq!(a,b,"production scratch unchanged by balanced replay");reuse+=a.len();}
            assert_eq!(read(&mut d,&production_hidden).await?,ph,"independent production GeGLU output unchanged");reuse+=ph.len();
            // Common incoming state for the next step is the original-production
            // residual, never one candidate's state supplied only to its control.
            incoming=fixture::canonical_residual(&current[0]);
            immutable+=x.immutable(&mut d).await?;
            println!("I4_BALANCE_STEP_PASS round={round} step={step} matched_incoming=true original_production_feedback=true inverse_and_unquantized_products_passed=true intended_code_parity=true no_task_quality_claim=true");
        }
        immutable+=q.immutable(&mut d).await?+w.immutable(&mut d,&up,&gate,&down).await?+bw.immutable(&mut d,&balanced_up,&balanced_gate,&down).await?;
        assert_eq!(read(&mut d,&gamma_prime).await?,transform.prime,"gammaPrime immutable");immutable+=transform.prime.len();
        transform.source_immutable(&tensors)?;for v in[&up,&gate,&down,&balanced_up,&balanced_gate]{v.verify_source(&tensors)?;}
    }
    assert_eq!(p.fingerprints(&mut d).await?,production_before,"all original production owners immutable");
    assert_eq!(prefix_values,4*TOKENS*HIDDEN);assert_eq!(products,4*2*cpu::SAMPLES*HIDDEN);
    assert_eq!(samples,12_288);assert_eq!(exact,393_216_000);
    println!("PASS_I4_CHANNEL_BALANCE_DIAGNOSTIC layer=0 prefix_inverse_values={prefix_values} exact_unquantized_products={products} intended_bf16_values={exact} sampled_quantized_outputs={samples} permutation_values={permutation} immutable_bytes_compared={immutable} reuse_bytes={reuse} no_timing_gate=true no_task_quality_or_global_bitexactness_claim=true");
    Ok(())
}

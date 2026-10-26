//! Disconnected fixed-input layer0 attribution. Original complete-probe helpers
//! are retained; exact current-E4 anchors precede factorial interpretation.
use furiosa_opt_gemma4::{Chip,axes::{P128,P512,H,L},host::{load,int4_complete_ffn_weights::ProductionWeights},
    ops_batch_layouts::{P512H960,P512L960},ops_ffn_projection_clusters as prod,
    ops_int4_group960_projection as candidate,ops_int4_group960_reference as reference,
    ops_int4_ffn_bridges as bridge,ops_int4_rational_fused_pack as fused,
    ops_norm_blocked_residual as tail,i4_group960_raw_adapter as raw};
use furiosa_opt_std::prelude::*;
use std::{fs::File,time::Instant};
#[path="support/npu_profile.rs"] mod npu_profile;
#[path="support/int4_group960_fixture.rs"] mod cpu;
#[path="support/int4_complete_fixture.rs"] mod fixture;
use cpu::{Result,InputCpu,WeightCpu,HIDDEN,EXPANDED,TOKENS};
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
struct InputGpu<E:M> {packed:HbmTensor<i4,Chip,E>,dense:HbmTensor<f8e4m3,Chip,E>,scales:HbmTensor<f32,Chip,m![P512]>}
impl<E:M> InputGpu<E> {
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
    joined_scales:HbmTensor<f32,Chip,candidate::JoinedBlockScale>,unit_up:HbmTensor<f32,Chip,reference::JoinedScale>,
    down_i4:HbmTensor<i4,Chip,candidate::DownWeight>,down_e4:HbmTensor<f8e4m3,Chip,candidate::DownWeight>,
    down_scales:HbmTensor<f32,Chip,candidate::DownBlockScale>,unit_down:HbmTensor<f32,Chip,m![H]>,global:HbmTensor<f32,Chip,m![1]>,
}
impl WeightsGpu {
    async fn new(d:&mut Device,up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu)->Result<Self> {
        assert_eq!((up.rows,up.cols),(EXPANDED,HIDDEN));assert_eq!((gate.rows,gate.cols),(EXPANDED,HIDDEN));assert_eq!((down.rows,down.cols),(HIDDEN,EXPANDED));
        Ok(Self{
            joined_i4:upload(d,joined(&up.packed,&gate.packed)).await?,
            joined_e4:upload(d,joined(&up.dense,&gate.dense)).await?,
            joined_scales:upload(d,joined(&floats(&up.scales),&floats(&gate.scales))).await?,
            unit_up:HostTensor::<f32,reference::JoinedScale>::from_vec(std::iter::repeat_n(1.0,2*EXPANDED)).to_hbm(&mut d.pdma).await?,
            down_i4:upload(d,down.packed.clone()).await?,down_e4:upload(d,down.dense.clone()).await?,
            down_scales:upload(d,floats(&down.scales)).await?,
            unit_down:HostTensor::<f32,m![H]>::from_vec(std::iter::repeat_n(1.0,HIDDEN)).to_hbm(&mut d.pdma).await?,
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
        assert_eq!(read(d,&self.unit_up).await?,floats(&vec![1.0;2*EXPANDED]),"UP/GATE reference units");
        assert_eq!(read(d,&self.unit_down).await?,floats(&vec![1.0;HIDDEN]),"DOWN reference units");
        assert_eq!(read(d,&self.global).await?,down.global.to_le_bytes(),"DOWN global immutable");
        Ok(up.packed.len()+gate.packed.len()+down.packed.len()+up.dense.len()+gate.dense.len()+down.dense.len()+
            (up.scales.len()+gate.scales.len()+down.scales.len()+2*EXPANDED+HIDDEN+1)*4)
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
    xp:HbmTensor<i32,Chip,raw::PackedInput>,xs:HbmTensor<f32,Chip,m![P512]>,
    joined:HbmTensor<bf16,Chip,candidate::JoinedOutput>,hidden:HbmTensor<bf16,Chip,bridge::HiddenBf16>,
    hp:HbmTensor<i32,Chip,raw::PackedHidden>,hs:HbmTensor<f32,Chip,m![P512]>,down:HbmTensor<bf16,Chip,candidate::Output>,
}
#[derive(Clone)]
struct Snapshot {residual:Vec<u8>,prefix:Vec<u8>,xp:Vec<u8>,xs:Vec<u8>,joined:Vec<u8>,hidden:Vec<u8>,hp:Vec<u8>,hs:Vec<u8>,down:Vec<u8>}
impl Snapshot {
    fn fields(&self)->[(&str,&[u8]);9]{[("residual",&self.residual),("prefix",&self.prefix),("xp",&self.xp),("xs",&self.xs),("joined",&self.joined),("hidden",&self.hidden),("hp",&self.hp),("hs",&self.hs),("down",&self.down)]}
    fn equal(&self,other:&Self,label:&str)->usize{let mut n=0;for((a,x),(b,y))in self.fields().into_iter().zip(other.fields()){assert_eq!(a,b);assert_eq!(x,y,"{label} {a}");n+=x.len();}n}
    fn permutation(&self,base:&Self)->usize{
        let mut n=0;n+=fixture::compare_rows(&fixture::canonical_residual(&self.residual),&fixture::canonical_residual(&base.residual),HIDDEN,2,"residual");
        n+=fixture::compare_rows(&self.prefix,&base.prefix,HIDDEN,2,"prefix");n+=fixture::compare_rows(&self.hidden,&base.hidden,EXPANDED,2,"hidden");
        n+=cpu::compare_permutation(&self.joined,&base.joined,EXPANDED,2);n+=cpu::compare_permutation(&self.down,&base.down,HIDDEN,1);
        n+=fixture::compare_rows(&self.xs,&base.xs,1,4,"inputscale");n+=fixture::compare_rows(&self.hs,&base.hs,1,4,"hiddenscale");
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
        residual:upload(d,r.to_vec()).await?,prefix:poison(d,0x7fc1).await?,xp:upload(d,vec![0x5a;512*HIDDEN/2]).await?,xs:upload(d,floats(&vec![f32::NAN;512])).await?,
        joined:poison(d,0x7fc2).await?,hidden:poison(d,0x7fc3).await?,hp:upload(d,vec![0xa5;512*EXPANDED/2]).await?,hs:upload(d,floats(&vec![f32::NAN;512])).await?,down:poison(d,0x7fc4).await?})}
    async fn reset(&mut self,d:&mut Device,r:&[u8])->Result<()>{self.residual=upload(d,r.to_vec()).await?;Ok(())}
    async fn snapshot(&self,d:&mut Device)->Result<Snapshot>{Ok(Snapshot{residual:read(d,&self.residual).await?,prefix:read(d,&self.prefix).await?,xp:read(d,&self.xp).await?,xs:read(d,&self.xs).await?,joined:read(d,&self.joined).await?,hidden:read(d,&self.hidden).await?,hp:read(d,&self.hp).await?,hs:read(d,&self.hs).await?,down:read(d,&self.down).await?})}
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
async fn candidate_prefix(d:&mut Device,p:&ProductionWeights,x:&Inputs,s:&mut CandidateScratch)->Result<()>{
    launch(bridge::prefix_sliding_bf16,(d,&x.sliding[0],&x.sliding[1],&x.sliding[2],&x.sliding[3],&p.post_attention,&p.pre_ffn,&mut s.residual,&mut s.prefix)).await?;Ok(())
}
async fn candidate_rest(d:&mut Device,p:&ProductionWeights,w:&WeightsGpu,q:&Quantizer,s:&mut CandidateScratch)->Result<()>{
    launch(fused::quantize_pack_input_aligned,(&mut *d,&s.prefix,&q.table,&q.hshift,&mut s.xp,&mut s.xs)).await?;
    raw::project_upgate(d,&s.xp,&s.xs,&w.joined_i4,&w.joined_scales,&mut s.joined).await?;
    launch(bridge::geglu_bf16,(&mut *d,&s.joined,p.ug(),p.gg(),&mut s.hidden)).await?;
    launch(fused::quantize_pack_hidden,(&mut *d,&s.hidden,&q.table,&q.lshift,&mut s.hp,&mut s.hs)).await?;
    raw::project_down(d,&s.hp,&s.hs,&w.down_i4,&w.down_scales,&w.global,&mut s.down).await?;
    launch(tail::post_feedforward,(&mut *d,&s.down,&p.post_ffn,&p.layer_gate,&mut s.residual)).await?;Ok(())
}
async fn candidate_chain(d:&mut Device,p:&ProductionWeights,w:&WeightsGpu,q:&Quantizer,x:&Inputs,s:&mut CandidateScratch)->Result<()>{candidate_prefix(d,p,x,s).await?;candidate_rest(d,p,w,q,s).await}
async fn production_chain(d:&mut Device,p:&ProductionWeights,x:&Inputs,s:&mut ProductionScratch)->Result<()>{
    launch(furiosa_opt_gemma4::ops_group_ffn_projection_storegroup::sliding_projection_owned,
        (d,(&x.sliding[0],&x.sliding[1],&x.sliding[2],&x.sliding[3]),(&p.post_attention,&p.pre_ffn,&p.post_ffn,&p.layer_gate),
         &mut s.residual,&p.joined,&p.joined_scale,p.down(),p.down_scale(),p.ug(),p.gg(),p.dg(),
         &mut s.qi,&mut s.xs,&mut s.joined,&mut s.qh,&mut s.hs,&mut s.down)).await?;Ok(())
}

/// Separate original P128 norm paths, CPU quotient, E4 integer-code dots,
/// original P128 GeGLU and original P128 tail. All copies/downloads are untimed.
async fn intended_step(d:&mut Device,p:&ProductionWeights,w:&WeightsGpu,x:&Inputs,residual:&[u8],got:&Snapshot,mid_got:&[u8],up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu,round:usize,step:usize)->Result<(Vec<u8>,usize,usize,usize)>{
    let mut prefix=Vec::new();let mut mid=Vec::new();
    for quarter in 0..4 {
        let begin=quarter*128*HIDDEN*2;let end=begin+128*HIDDEN*2;
        let projected:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,x.canonical[begin..end].to_vec()).await?;
        let mut r:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,residual[begin..end].to_vec()).await?;
        let mut n=poison::<m![P128,H]>(d,0x7fc7).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::post_attention_p128,(&mut *d,&projected,&p.post_attention,&mut r)).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::rmsnorm_p128,(&mut *d,&r,&p.pre_ffn,&mut n)).await?;
        mid.extend(read(d,&r).await?);prefix.extend(read(d,&n).await?);
        assert_eq!(read(d,&projected).await?,x.canonical[begin..end],"immutable original prefix input");
    }
    let mut exact=cpu::compare_bits(&got.prefix,&prefix,"independent prefix canonical BF16");
    exact+=cpu::compare_bits(mid_got,&fixture::block_residual(&mid),"independent prefix persisted residual");
    let qi=fixture::quantize(&prefix,HIDDEN);assert_eq!(got.xp,qi.packed);assert_eq!(got.xs,floats(&qi.scales));
    let xg=InputGpu::<P512H960>::new(d,&qi).await?;
    let mut typed=poison::<candidate::JoinedOutput>(d,0x7fc8).await?;
    let mut dots=poison::<reference::JoinedOutput>(d,0x7fc9).await?;
    // Normal typed launches embed exactly the images used by the checked raw binding.
    launch(candidate::project_upgate,(&mut *d,&xg.packed,&xg.scales,&w.joined_i4,&w.joined_scales,&mut typed)).await?;
    launch(reference::project_upgate,(&mut *d,&xg.dense,&xg.scales,&w.joined_e4,&w.joined_scales,&w.unit_up,&mut dots)).await?;
    let jb=read(d,&dots).await?;exact+=cpu::compare_bits(&got.joined,&jb,"intended E4-code UP/GATE");
    exact+=cpu::compare_bits(&read(d,&typed).await?,&jb,"typed/raw UP/GATE image");
    let(mut sampled,mut immutable)=(0,0);
    for(name,plane,weight)in[("UP",0,up),("GATE",1,gate)]{let(n,radius)=cpu::sampled_bound(&got.joined,plane,&qi,weight,false);sampled+=n;
        println!("I4_COMPLETE_F64 round={round} step={step} projection={name} samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");}
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
    let qh=fixture::quantize(&hidden,EXPANDED);assert_eq!(got.hp,qh.packed);assert_eq!(got.hs,floats(&qh.scales));
    let hg=InputGpu::<P512L960>::new(d,&qh).await?;
    let mut typed_down=poison::<candidate::Output>(d,0x7fcb).await?;let mut ref_down=poison::<candidate::Output>(d,0x7fcc).await?;
    launch(candidate::project_down,(&mut *d,&hg.packed,&hg.scales,&w.down_i4,&w.down_scales,&w.global,&mut typed_down)).await?;
    launch(reference::project_down,(&mut *d,&hg.dense,&hg.scales,&w.down_e4,&w.down_scales,&w.unit_down,&w.global,&mut ref_down)).await?;
    let db=read(d,&ref_down).await?;exact+=cpu::compare_bits(&got.down,&db,"intended E4-code DOWN");exact+=cpu::compare_bits(&read(d,&typed_down).await?,&db,"typed/raw DOWN image");
    let(n,radius)=cpu::sampled_bound(&got.down,0,&qh,down,true);sampled+=n;
    println!("I4_COMPLETE_F64 round={round} step={step} projection=DOWN samples={n} max_preBF16_radius={radius:.12e} oracle=i64_local_F64_envelope");
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


#[path="support/int4_drift_metrics.rs"] mod metrics;

async fn original_hidden(d:&mut Device,p:&ProductionWeights,j:&[u8])->Result<Vec<u8>>{
    let u=fixture::canonical_projection(j,EXPANDED,0);let g=fixture::canonical_projection(j,EXPANDED,1);let mut out=Vec::new();
    for q in 0..4{let a=q*128*EXPANDED*2;let b=a+128*EXPANDED*2;let ub=fixture::reference_column_tile(&u[a..b]);let gb=fixture::reference_column_tile(&g[a..b]);
        let ug:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_int4_complete_reference::Operand>=upload(d,ub.clone()).await?;
        let gg:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_int4_complete_reference::Operand>=upload(d,gb.clone()).await?;
        let mut y=poison(d,0x7fd0).await?;
        launch(furiosa_opt_gemma4::ops_int4_complete_reference::geglu_p128,(&mut *d,&ug,&gg,p.ug(),p.gg(),&mut y)).await?;
        out.extend(fixture::reference_canonical_tile(&read(d,&y).await?));assert_eq!(read(d,&ug).await?,ub);assert_eq!(read(d,&gg).await?,gb);
    }Ok(out)
}
async fn original_tail(d:&mut Device,p:&ProductionWeights,down:&[u8],mid:&[u8])->Result<Vec<u8>>{
    let dc=fixture::canonical_projection(down,HIDDEN,0);let mut out=Vec::new();
    for q in 0..4{let a=q*128*HIDDEN*2;let b=a+128*HIDDEN*2;
        let y:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,dc[a..b].to_vec()).await?;let mut r:HbmTensor<bf16,Chip,m![P128,H]>=upload(d,mid[a..b].to_vec()).await?;
        launch(furiosa_opt_gemma4::ops_norm_wide::post_feedforward_p128,(&mut *d,&y,&p.post_ffn,&p.layer_gate,&mut r)).await?;
        out.extend(read(d,&r).await?);assert_eq!(read(d,&y).await?,dc[a..b]);
    }Ok(fixture::block_residual(&out))
}
async fn timed_candidate(d:&mut Device,p:&ProductionWeights,w:&WeightsGpu,q:&Quantizer,x:&Inputs,s:&mut CandidateScratch)->Result<([f64;7],f64)>{
    let total=Instant::now();let mut times=[0.;7];
    let t=Instant::now();candidate_prefix(d,p,x,s).await?;times[0]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();launch(fused::quantize_pack_input_aligned,(&mut *d,&s.prefix,&q.table,&q.hshift,&mut s.xp,&mut s.xs)).await?;times[1]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();raw::project_upgate(d,&s.xp,&s.xs,&w.joined_i4,&w.joined_scales,&mut s.joined).await?;times[2]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();launch(bridge::geglu_bf16,(&mut *d,&s.joined,p.ug(),p.gg(),&mut s.hidden)).await?;times[3]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();launch(fused::quantize_pack_hidden,(&mut *d,&s.hidden,&q.table,&q.lshift,&mut s.hp,&mut s.hs)).await?;times[4]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();raw::project_down(d,&s.hp,&s.hs,&w.down_i4,&w.down_scales,&w.global,&mut s.down).await?;times[5]=t.elapsed().as_secs_f64()*1e6;
    let t=Instant::now();launch(tail::post_feedforward,(&mut *d,&s.down,&p.post_ffn,&p.layer_gate,&mut s.residual)).await?;times[6]=t.elapsed().as_secs_f64()*1e6;
    Ok((times,total.elapsed().as_secs_f64()*1e6))
}

#[tokio::main]
async fn main()->Result<()>{
    if std::env::args().len()!=1{return Err("fixed layer0, two matched row orders, no tuning or fixture arguments".into());}
    cpu::validate_integer_encodings();cpu::check_output_layout(EXPANDED);cpu::check_output_layout(HIDDEN);
    let file=File::open(load::model_dir()?.join("model.safetensors"))?;
    // Same read-only supplied mapping as the original complete diagnostic.
    let mmap=unsafe{memmap2::Mmap::map(&file)?};let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
    let up=WeightCpu::load(&tensors,"up_proj")?;let gate=WeightCpu::load(&tensors,"gate_proj")?;let down=WeightCpu::load(&tensors,"down_proj")?;
    let mut d=Device::new(candidate::project_upgate.topology())?;let p=ProductionWeights::load_layer0(&mut d,&tensors).await?;
    assert_eq!(read(&mut d,p.ug()).await?,up.global.to_le_bytes());assert_eq!(read(&mut d,p.gg()).await?,gate.global.to_le_bytes());assert_eq!(read(&mut d,p.dg()).await?,down.global.to_le_bytes());
    let w=WeightsGpu::new(&mut d,&up,&gate,&down).await?;let q=Quantizer::new(&mut d).await?;let before=p.fingerprints(&mut d).await?;
    let native=read(&mut d,&p.joined).await?;let ns=read(&mut d,&p.joined_scale).await?;
    metrics::weights(&tensors,"up_proj",&up,&native[..HIDDEN*EXPANDED],&ns[..EXPANDED*4])?;
    metrics::weights(&tensors,"gate_proj",&gate,&native[HIDDEN*EXPANDED..],&ns[EXPANDED*4..])?;
    metrics::weights(&tensors,"down_proj",&down,&read(&mut d,p.down()).await?,&read(&mut d,p.down_scale()).await?)?;
    let ub=vec![1f32;2*EXPANDED*(HIDDEN/960)];let db=vec![1f32;HIDDEN*(EXPANDED/960)];
    let up_ones:HbmTensor<f32,Chip,reference::JoinedBlockScale>=upload(&mut d,floats(&ub)).await?;
    let down_ones:HbmTensor<f32,Chip,reference::DownBlockScale>=upload(&mut d,floats(&db)).await?;
    let mut base:Option<(Snapshot,Vec<Vec<u8>>,Vec<Vec<u8>>,Vec<Vec<u8>>,Vec<Vec<u8>>,Vec<Vec<u8>>)>=None;
    let(mut exact,mut sampled,mut immutable,mut reuse,mut perm)=(0usize,0usize,0usize,0usize,0usize);
    for round in 0..2{
        let x=Inputs::new(&mut d,round==1).await?;let r=fixture::input(round==1,true);let rb=fixture::block_residual(&r);
        let mut c=CandidateScratch::new(&mut d,&rb).await?;let mut production=ProductionScratch::new(&mut d,&rb).await?;
        candidate_prefix(&mut d,&p,&x,&mut c).await?;let mid=read(&mut d,&c.residual).await?;
        candidate_rest(&mut d,&p,&w,&q,&mut c).await?;let got=c.snapshot(&mut d).await?;
        let(_,n,s,im)=intended_step(&mut d,&p,&w,&x,&r,&got,&mid,&up,&gate,&down,round,0).await?;exact+=n;sampled+=s;immutable+=im;
        production_chain(&mut d,&p,&x,&mut production).await?;let old=production.snapshot(&mut d).await?;
        for(name,a,b)in[("joined",&got.joined,&old[3]),("down",&got.down,&old[6]),("final_residual",&got.residual,&old[0])]{fixture::drift(a,b,name,round,0);}
        let qi=fixture::quantize(&got.prefix,HIDDEN);let xg=InputGpu::<P512H960>::new(&mut d,&qi).await?;
        let mut eq:HbmTensor<f8e4m3,Chip,P512H960>=upload(&mut d,vec![0x7f;512*HIDDEN]).await?;let mut es=upload(&mut d,floats(&vec![f32::NAN;512])).await?;
        launch(furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_hidden,(&mut d,&c.prefix,&mut eq,&mut es)).await?;
        assert_eq!(read(&mut d,&eq).await?,old[1],"same BF16 prefix -> current E4 codes");assert_eq!(read(&mut d,&es).await?,old[2],"same BF16 prefix -> current E4 scales");
        println!("I4_ATTR_ANCHOR round={round} prefix_E4_codes={} prefix_scale_bits=512 passed=true",512*HIDDEN);
        metrics::activations("prefix",round,&got.prefix,&old[1],&old[2],&qi);
        let(mut joins,mut hidden)=(Vec::new(),Vec::new());
        for(arm,(ia,iw))in[(false,false),(true,false),(false,true),(true,true)].into_iter().enumerate(){
            let a=if ia{&xg.dense}else{&production.qi};let s=if ia{&xg.scales}else{&production.xs};
            let weight=if iw{&w.joined_e4}else{&p.joined};let bs=if iw{&w.joined_scales}else{&up_ones};let rs=if iw{&w.unit_up}else{&p.joined_scale};
            let mut out=poison::<reference::JoinedOutput>(&mut d,0x7fd1).await?;
            launch(reference::project_upgate,(&mut d,a,s,weight,bs,rs,&mut out)).await?;let bytes=read(&mut d,&out).await?;
            if arm==0{exact+=cpu::compare_bits(&bytes,&old[3],"required EE current production UP/GATE anchor; no relaxed tolerance");}
            if arm==3{exact+=cpu::compare_bits(&bytes,&got.joined,"II reproduces independently validated candidate joined");}
            for(plane,name)in[(0,"UP"),(1,"GATE")]{let aa=fixture::canonical_projection(&bytes,EXPANDED,plane);let bb=fixture::canonical_projection(&old[3],EXPANDED,plane);fixture::drift(&aa,&bb,&format!("factor_upgate_{arm}_{name}"),round,0);}
            let h=original_hidden(&mut d,&p,&bytes).await?;if arm==3{exact+=cpu::compare_bits(&h,&got.hidden,"II original GeGLU reproduces candidate");}
            // Separate destination poison on repeat: no dependence on previous output.
            let mut again=poison::<reference::JoinedOutput>(&mut d,0x7fd2).await?;launch(reference::project_upgate,(&mut d,a,s,weight,bs,rs,&mut again)).await?;
            assert_eq!(read(&mut d,&again).await?,bytes,"UP/GATE arm replay");assert_eq!(read(&mut d,&out).await?,bytes);reuse+=bytes.len()*2;
            joins.push(bytes);hidden.push(h);
        }
        metrics::interaction("UP_GATE",round,&joins);metrics::interaction("GeGLU",round,&hidden);
        for i in 0..4{fixture::drift(&hidden[i],&hidden[0],&format!("factor_geglu_{i}"),round,0);}
        // Fixed DOWN input is actual production GeGLU, never the A4 candidate hidden.
        let fixed_hidden=original_hidden(&mut d,&p,&old[3]).await?;assert_eq!(fixed_hidden,hidden[0]);
        let mut bridge_hidden=poison::<bridge::HiddenBf16>(&mut d,0x7fd3).await?;
        launch(bridge::geglu_bf16,(&mut d,&production.joined,p.ug(),p.gg(),&mut bridge_hidden)).await?;
        exact+=cpu::compare_bits(&read(&mut d,&bridge_hidden).await?,&fixed_hidden,"production joined -> original/bridge GeGLU");
        let hb=metrics::canonical_hidden_to_blocked(&fixed_hidden);let hb_gpu:HbmTensor<bf16,Chip,furiosa_opt_gemma4::ops_batch_layouts::P512L240>=upload(&mut d,hb.clone()).await?;
        let mut hq:HbmTensor<f8e4m3,Chip,P512L960>=upload(&mut d,vec![0x7f;512*EXPANDED]).await?;let mut hs=upload(&mut d,floats(&vec![f32::NAN;512])).await?;
        launch(furiosa_opt_gemma4::ops_native_fp8_p512_blocks::quantize_ffn,(&mut d,&hb_gpu,&mut hq,&mut hs)).await?;
        assert_eq!(read(&mut d,&hq).await?,old[4],"production GeGLU -> current hidden E4 codes");assert_eq!(read(&mut d,&hs).await?,old[5],"production GeGLU -> current hidden scales");
        println!("I4_ATTR_ANCHOR round={round} hidden_E4_codes={} hidden_scale_bits=512 passed=true fixed_hidden=production_GeGLU",512*EXPANDED);
        let hi=fixture::quantize(&fixed_hidden,EXPANDED);let hg=InputGpu::<P512L960>::new(&mut d,&hi).await?;metrics::activations("production_hidden",round,&fixed_hidden,&old[4],&old[5],&hi);
        let(mut downs,mut tails)=(Vec::new(),Vec::new());let midcan=fixture::canonical_residual(&mid);
        for(arm,(ia,iw))in[(false,false),(true,false),(false,true),(true,true)].into_iter().enumerate(){
            let a=if ia{&hg.dense}else{&production.qh};let s=if ia{&hg.scales}else{&production.hs};
            let weight=if iw{&w.down_e4}else{p.down()};let bs=if iw{&w.down_scales}else{&down_ones};let rs=if iw{&w.unit_down}else{p.down_scale()};
            let mut out=poison::<candidate::Output>(&mut d,0x7fd4).await?;launch(reference::project_down,(&mut d,a,s,weight,bs,rs,p.dg(),&mut out)).await?;let bytes=read(&mut d,&out).await?;
            if arm==0{exact+=cpu::compare_bits(&bytes,&old[6],"required EE current production DOWN anchor");}
            if arm==3{let mut i4=poison::<candidate::Output>(&mut d,0x7fd5).await?;launch(candidate::project_down,(&mut d,&hg.packed,&hg.scales,&w.down_i4,&w.down_scales,p.dg(),&mut i4)).await?;exact+=cpu::compare_bits(&read(&mut d,&i4).await?,&bytes,"fixed-hidden II I4/E4 code parity");let(n,_)=cpu::sampled_bound(&bytes,0,&hi,&down,true);sampled+=n;}
            fixture::drift(&bytes,&old[6],&format!("factor_down_{arm}"),round,0);let tail=original_tail(&mut d,&p,&bytes,&midcan).await?;
            if arm==0{exact+=cpu::compare_bits(&tail,&old[0],"EE current production tail/residual/layer scalar");}
            fixture::drift(&tail,&old[0],&format!("factor_tail_{arm}"),round,0);
            let mut again=poison::<candidate::Output>(&mut d,0x7fd6).await?;launch(reference::project_down,(&mut d,a,s,weight,bs,rs,p.dg(),&mut again)).await?;
            assert_eq!(read(&mut d,&again).await?,bytes);assert_eq!(read(&mut d,&out).await?,bytes);reuse+=bytes.len()*2;downs.push(bytes);tails.push(tail);
        }
        metrics::interaction("DOWN_fixed_production_hidden",round,&downs);metrics::interaction("tail_fixed_midpoint",round,&tails);
        if let Some((bc,bo,bj,bh,bd,bt))=&base{
            perm+=got.permutation(bc);for i in 0..4{perm+=cpu::compare_permutation(&joins[i],&bj[i],EXPANDED,2);perm+=fixture::compare_rows(&hidden[i],&bh[i],EXPANDED,2,"factor hidden");perm+=cpu::compare_permutation(&downs[i],&bd[i],HIDDEN,1);perm+=fixture::compare_rows(&fixture::canonical_residual(&tails[i]),&fixture::canonical_residual(&bt[i]),HIDDEN,2,"factor tail");}
            perm+=cpu::compare_permutation(&old[3],&bo[3],EXPANDED,2);perm+=cpu::compare_permutation(&old[6],&bo[6],HIDDEN,1);
        }else{base=Some((got.clone(),old.clone(),joins,hidden,downs,tails));}
        // Four restored-input runs per arm, ABBA blocks with two runs each.
        // No intermediate downloads/validation in either timed execution path.
        for(block,is_candidate)in[true,false,false,true].into_iter().enumerate(){for repeat in 0..2{
            if is_candidate{c.reset(&mut d,&rb).await?;let(stages,total)=timed_candidate(&mut d,&p,&w,&q,&x,&mut c).await?;let sum: f64=stages.iter().sum();assert!(total>=sum);
                println!("I4_ATTR_WALL {}",serde_json::json!({"round":round,"block":block,"repeat":repeat,"candidate":true,"stages_us":stages,"sum_us":sum,"total_us":total,"clock_gaps_us":total-sum,"scope":"completed awaits including raw binding; clock-instrumented; not device per-op trace"}));
            }else{production.reset(&mut d,&rb).await?;let t=Instant::now();production_chain(&mut d,&p,&x,&mut production).await?;println!("I4_ATTR_WALL {}",serde_json::json!({"round":round,"block":block,"repeat":repeat,"candidate":false,"total_us":t.elapsed().as_secs_f64()*1e6,"scope":"unchanged one-call current production; no promotion gate"}));}
        }}
        reuse+=c.snapshot(&mut d).await?.equal(&got,"candidate after phase timings");for(a,b)in production.snapshot(&mut d).await?.iter().zip(&old){assert_eq!(a,b);reuse+=a.len();}
        immutable+=xg.immutable(&mut d,&qi).await?+hg.immutable(&mut d,&hi).await?+x.immutable(&mut d).await?+q.immutable(&mut d).await?+w.immutable(&mut d,&up,&gate,&down).await?;
        assert_eq!(read(&mut d,&eq).await?,old[1]);assert_eq!(read(&mut d,&es).await?,old[2]);assert_eq!(read(&mut d,&hq).await?,old[4]);assert_eq!(read(&mut d,&hs).await?,old[5]);assert_eq!(read(&mut d,&hb_gpu).await?,hb);
        assert_eq!(read(&mut d,&up_ones).await?,floats(&ub));assert_eq!(read(&mut d,&down_ones).await?,floats(&db));
        up.verify_source(&tensors)?;gate.verify_source(&tensors)?;down.verify_source(&tensors)?;
    }
    assert_eq!(p.fingerprints(&mut d).await?,before,"all original production owners immutable");
    println!("PASS_I4_ATTRIBUTION exact_bf16_comparisons={exact} integer_F64_samples={sampled} permutation_values={perm} direct_immutable_bytes={immutable} replay_bytes={reuse} original_EE_anchors_required=true rows=512 orders=2 layer=0 synthetic=true no_model_quality_or_speed_claim=true");
    Ok(())
}

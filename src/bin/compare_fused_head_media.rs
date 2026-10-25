//! Integrated media and exact head-attention fusion oracle.
//! Wide-only cases compare scalar/batched embeddings; grouped cases use the same
//! batched media on both sides and explicitly compare original/fused attention.
//! Original control is selected before any transformer layer executes.
use furiosa_opt_gemma4::{LAYERS,axes::{H,P128,Rv,Aa},host::{audio::AudioFrames,image::ImagePatches,
    generate::{SoftToken,read_logits},load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
struct Media {image:ImagePatches,audio:AudioFrames}
fn word(i:usize)->u64 {let mut x=(i as u64).wrapping_add(0x873915b);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn fixture(owner:usize)->Media {
    let pixels=(0..33).map(|row|(0..Rv::SIZE).map(|k|bf16::from_f32(match row%5{
        0=>0.,1=>(owner+row)as f32/40.,2=>if k==row*137%Rv::SIZE{1.}else{0.},
        _=>(word(owner*100003+row*Rv::SIZE+k)>>40)as f32/16777216.,
    })).collect()).collect();
    let positions=(0..33).map(|row|if row==32{(1127,1133)}else{(row*31+owner,row*29+owner*3)}).collect();
    let frames=(0..129).map(|row|(0..Aa::SIZE).map(|k|bf16::from_f32(match row%4{
        0=>0.,1=>if k%2==0{0.25}else{-0.25},2=>if k==row%Aa::SIZE{1.}else{0.},
        _=>(word(owner*300007+row*Aa::SIZE+k)>>40)as f32/16777216.*2.-1.,
    })).collect()).collect();
    Media{image:ImagePatches{pixels,positions},audio:AudioFrames{frames}}
}
fn text(owner:usize,pos:usize,logits:bool)->Input<'static>{Input{token:2+(owner*733+pos*37)%8000,soft:None,logits,image:None,audio:None}}
fn prompt<'a>(media:&'a Media,owner:usize,start:usize,count:usize,mode:usize,permuted:bool)->Vec<Input<'a>>{
    (0..count).map(|i|{
        let source=if permuted{(i*29+17)%count}else{i};
        let kind=match mode{0=>source%3,1=>1,2=>2,_=>0};
        let soft=match kind{1=>Some(SoftToken::Image((source*13+owner*7)%33)),2=>Some(SoftToken::Audio((source*37+owner*11)%129)),_=>None};
        Input{token:2+(owner*733+(start+i)*37)%8000,soft,logits:i+1==count,image:Some(&media.image),audio:Some(&media.audio)}
    }).collect()
}
fn exact_f32(label:&str,a:&[f32],b:&[f32]){
    assert_eq!(a.len(),b.len());let mut diff=0;let mut nonfinite=0;let mut max_abs=0f32;
    for(i,(&a,&b))in a.iter().zip(b).enumerate(){nonfinite+=usize::from(!a.is_finite()||!b.is_finite());max_abs=max_abs.max((a-b).abs());
        if a.to_bits()!=b.to_bits(){if diff<6{eprintln!("MEDIA_MODEL_DIFF label={label} i={i} a={a} b={b}");}diff+=1;}}
    println!("MEDIA_MODEL_EXACT label={label} values={} bitdiff={diff} nonfinite={nonfinite} max_abs={max_abs}",a.len());assert_eq!(nonfinite,0);assert_eq!(diff,0);
}
fn exact_bf16(label:&str,a:&[bf16],b:&[bf16]){
    let a:Vec<_>=a.iter().map(|v|v.to_f32()).collect();let b:Vec<_>=b.iter().map(|v|v.to_f32()).collect();exact_f32(label,&a,&b);
}
async fn prefix(d:&mut Device,m:&mut Model,ws:&mut Workspace,blocks:&[Blocks],slots:&[usize],owner:usize,end:usize)->Result<()> {
    let mut p=0;while p<end{
        let n=(end-p).min(128).min(512-p%512);let rows:Vec<_>=(p..p+n).map(|q|text(owner,q,false)).collect();
        for &slot in slots{m.forward_sequences_native_qkv(d,&[Sequence{inputs:&rows,start_position:p,blocks:&blocks[slot]}],ws).await?;}
        p+=n;
    }Ok(())
}
async fn snapshot(d:&mut Device,m:&mut Model,ws:&mut Workspace,seq:&Sequence<'_>,label:&str)->Result<()> {
    let (scalar,batch)=m.compare_media_prepare(d,seq,ws).await?;
    assert_eq!(scalar.len(),P128::SIZE*H::SIZE);assert_eq!(batch.len(),scalar.len());
    exact_bf16(&format!("{label}_embedding_all128"),&scalar,&batch);Ok(())
}
async fn single(d:&mut Device,m:&mut Model,ws:&mut Workspace,seq:&Sequence<'_>,candidate:bool,label:&str)->Result<Vec<f32>> {
    let t=Instant::now();
    if seq.inputs.len()==512{
        if candidate{m.start_prefill512_fused_head_attention(d,seq,ws).await?;}
        else{m.start_prefill512_original_head_attention(d,seq,ws).await?;}
    } else{m.start_prefill_with_media_batch(d,seq,ws,candidate).await?;}
    let prepare=t.elapsed().as_secs_f64()*1000.;
    assert_eq!(m.active_prefill_rings(),vec![seq.blocks.sliding]);
    if seq.inputs.len()<=P128::SIZE{assert!(m.compare_media_prepare(d,seq,ws).await.is_err(),"snapshot must reject active preparation");}
    let mut layers=0.;let mut final_logits=None;
    for layer in 0..LAYERS {let t=Instant::now();let out=m.advance_prefill(d,ws).await?;layers+=t.elapsed().as_secs_f64()*1000.;
        if let Some(out)=out{assert_eq!(layer+1,LAYERS);final_logits=Some(read_logits(d,out).await?);}else{assert!(layer+1<LAYERS);}}
    assert!(m.active_prefill_rings().is_empty());
    println!("MEDIA_MODEL_TIMING label={label} candidate={candidate} count={} prepare_ms={prepare:.6} layer_finish_ms={layers:.6} total_compute_ms={:.6} excludes_logit_download=true",seq.inputs.len(),prepare+layers);
    final_logits.ok_or_else(||"single prompt returned no final logits".into())
}
async fn decode_pair(d:&mut Device,m:&mut Model,ws:&mut Workspace,blocks:&[Blocks],a:usize,b:usize,owner:usize,start:usize,count:usize,label:&str)->Result<()> {
    for step in 0..count {let rows=[text(owner,start+step,true)];
        let out=m.forward_sequences(d,&[Sequence{inputs:&rows,start_position:start+step,blocks:&blocks[a]}],ws).await?;
        let expected=read_logits(d,&out[0]).await?;
        let out=m.forward_sequences(d,&[Sequence{inputs:&rows,start_position:start+step,blocks:&blocks[b]}],ws).await?;
        exact_f32(&format!("{label}_decode{step}"),&expected,&read_logits(d,&out[0]).await?);
    }Ok(())
}
async fn group(d:&mut Device,m:&mut Model,ws:&mut Workspace,seq:&[Sequence<'_>],candidate:bool,pair:bool,blocks:&[Blocks],interleave:bool,label:&str)->Result<Vec<Vec<f32>>> {
    let t=Instant::now();
    if pair {
        if candidate{m.start_prefill_pair512_fused_head_attention(d,seq,ws).await?;}
        else{m.start_prefill_pair512_original_head_attention(d,seq,ws).await?;}
    } else {
        if candidate{m.start_prefill_group_fused_head_attention(d,seq,ws).await?;}
        else{m.start_prefill_group_original_head_attention(d,seq,ws).await?;}
    }
    let prepare=t.elapsed().as_secs_f64()*1000.;
    let expected_rings:Vec<_>=seq.iter().map(|s|s.blocks.sliding).collect();assert_eq!(m.active_prefill_rings(),expected_rings);
    if seq[0].inputs.len()<=P128::SIZE{assert!(m.compare_media_prepare(d,&seq[0],ws).await.is_err());}
    let mut layer_ms=0.;let mut result=None;
    for layer in 0..LAYERS {
        let t=Instant::now();let out=if pair{m.advance_prefill_pair512(d,ws).await?}else{m.advance_prefill_group(d,ws).await?};layer_ms+=t.elapsed().as_secs_f64()*1000.;
        if let Some(out)=out{assert_eq!(layer+1,LAYERS);assert_eq!(out.len(),seq.len());let mut values=Vec::new();for x in out{values.push(read_logits(d,x).await?);}result=Some(values);}else{assert!(layer+1<LAYERS);}
        if interleave {
            let rows=[text(77,layer,true)];
            let out=m.forward_sequences_r64(d,&[Sequence{inputs:&rows,start_position:layer,blocks:&blocks[28]}],ws).await?;
            let expected=read_logits(d,&out[0]).await?;
            let out=m.forward_sequences_r64(d,&[Sequence{inputs:&rows,start_position:layer,blocks:&blocks[29]}],ws).await?;
            exact_f32(&format!("{label}_interleave{layer}"),&expected,&read_logits(d,&out[0]).await?);
        }
    }
    assert!(m.active_prefill_rings().is_empty());
    println!("MEDIA_MODEL_TIMING label={label} candidate={candidate} owners={} prepare_ms={prepare:.6} layer_finish_ms={layer_ms:.6} total_compute_ms={:.6} excludes_interleave_and_logit_download=true",seq.len(),prepare+layer_ms);
    result.ok_or_else(||"group returned no final logits".into())
}
#[tokio::main]
async fn main()->Result<()> {
    let mut d=Device::new(ops::embed_token.topology())?;
    let weights=load::load_weights(&mut d).await?;
    let mut ws=Workspace::new_canonical64(&mut d,&weights,1536,64).await?;
    let mut m=Model::new(&mut d,weights,&mut ws).await?;m.prepare_group_scratch(&mut d).await?;
    // One set of model weights/pools. Every paired history has disjoint rings
    // and full pages, including reversed physical page ordering.
    let blocks:Vec<_>=(0..64).map(|ring|Blocks{sliding:ring,full:(0..3).map(|p|255-ring*3-p).collect()}).collect();
    let media:Vec<_>=(0..8).map(fixture).collect();
    for (case,(start,count,mode,perm)) in [(0,17,0,false),(385,127,1,false),(384,128,2,false),(0,128,0,true)].into_iter().enumerate(){
        prefix(&mut d,&mut m,&mut ws,&blocks,&[case,32+case],case,start).await?;
        let rows=prompt(&media[case],case,start,count,mode,perm);
        let a=Sequence{inputs:&rows,start_position:start,blocks:&blocks[case]};let b=Sequence{inputs:&rows,start_position:start,blocks:&blocks[32+case]};
        snapshot(&mut d,&mut m,&mut ws,&b,&format!("single{case}")).await?;
        let candidate=single(&mut d,&mut m,&mut ws,&b,true,&format!("single{case}_candidate")).await?;
        let control=single(&mut d,&mut m,&mut ws,&a,false,&format!("single{case}_control")).await?;
        exact_f32(&format!("single{case}_final"),&control,&candidate);
        decode_pair(&mut d,&mut m,&mut ws,&blocks,case,32+case,case,start+count,3,&format!("single{case}_post")).await?;
    }
    // Reset old long histories with a different short mixed prompt. Absolute
    // positions restart at0; inactive old cache data must remain invisible.
    let reset=prompt(&media[6],91,0,17,0,true);
    let b=Sequence{inputs:&reset,start_position:0,blocks:&blocks[33]};let a=Sequence{inputs:&reset,start_position:0,blocks:&blocks[1]};
    snapshot(&mut d,&mut m,&mut ws,&b,"reset").await?;
    let candidate=single(&mut d,&mut m,&mut ws,&b,true,"reset_candidate").await?;
    let control=single(&mut d,&mut m,&mut ws,&a,false,"reset_control").await?;exact_f32("reset_final",&control,&candidate);
    decode_pair(&mut d,&mut m,&mut ws,&blocks,1,33,91,17,3,"reset_post").await?;

    let rows=prompt(&media[4],4,0,512,0,true);
    let b=Sequence{inputs:&rows,start_position:0,blocks:&blocks[40]};let a=Sequence{inputs:&rows,start_position:0,blocks:&blocks[8]};
    let candidate=single(&mut d,&mut m,&mut ws,&b,true,"same_owner512_candidate").await?;
    let control=single(&mut d,&mut m,&mut ws,&a,false,"same_owner512_control").await?;exact_f32("same_owner512_final",&control,&candidate);
    decode_pair(&mut d,&mut m,&mut ws,&blocks,8,40,4,512,3,"same_owner512_post").await?;

    let starts=[0,384,512,896];let modes=[3,1,2,0];
    for r in 0..4{prefix(&mut d,&mut m,&mut ws,&blocks,&[20+r,24+r],r+1,starts[r]).await?;}
    let prompts:Vec<_>=(0..4).map(|r|prompt(&media[r+1],r+1,starts[r],128,modes[r],true)).collect();
    for r in 0..4{snapshot(&mut d,&mut m,&mut ws,&Sequence{inputs:&prompts[r],start_position:starts[r],blocks:&blocks[20+r]},&format!("group_owner{r}")).await?;}
    let candidate_seq:Vec<_>=(0..4).map(|r|Sequence{inputs:&prompts[r],start_position:starts[r],blocks:&blocks[20+r]}).collect();
    let control_seq:Vec<_>=(0..4).map(|r|Sequence{inputs:&prompts[r],start_position:starts[r],blocks:&blocks[24+r]}).collect();
    let candidate=group(&mut d,&mut m,&mut ws,&candidate_seq,true,false,&blocks,true,"group_candidate").await?;
    let control=group(&mut d,&mut m,&mut ws,&control_seq,false,false,&blocks,false,"group_control").await?;
    for r in 0..4{exact_f32(&format!("group_owner{r}_final"),&control[r],&candidate[r]);decode_pair(&mut d,&mut m,&mut ws,&blocks,20+r,24+r,r+1,starts[r]+128,3,&format!("group_owner{r}_post")).await?;}

    // A two-owner256+256 pack exercises media preparation through another
    // constructor while preserving chronological quarters within each owner.
    let pair_rows=[prompt(&media[5],5,0,256,1,true),prompt(&media[7],7,0,256,2,true)];
    let candidate_seq:Vec<_>=(0..2).map(|r|Sequence{inputs:&pair_rows[r],start_position:0,blocks:&blocks[41+r]}).collect();
    let control_seq:Vec<_>=(0..2).map(|r|Sequence{inputs:&pair_rows[r],start_position:0,blocks:&blocks[9+r]}).collect();
    let candidate=group(&mut d,&mut m,&mut ws,&candidate_seq,true,true,&blocks,false,"pair_candidate").await?;
    let control=group(&mut d,&mut m,&mut ws,&control_seq,false,true,&blocks,false,"pair_control").await?;
    for r in 0..2{exact_f32(&format!("pair_owner{r}_final"),&control[r],&candidate[r]);decode_pair(&mut d,&mut m,&mut ws,&blocks,9+r,41+r,if r==0{5}else{7},256,3,&format!("pair_owner{r}_post")).await?;}
    println!("PASS integrated exactmedia andhead-attention fusion: embeddings,17/127/128,reset,Prompt512,four-owner group,256+256 pair,48 R64 interleaves and teacher-forced post-decodes");Ok(())
}

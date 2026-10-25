//! Exact four real quarters from three owners against current packed controls; no Worker activation.
use furiosa_opt_gemma4::{LAYERS,host::{generate::read_logits,load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(owner:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(owner*733+p*37)%8000,soft:None,logits,image:None,audio:None}}
fn exact(label:&str,a:&[f32],b:&[f32]){
    assert_eq!(a.len(),b.len());let mut diff=0;let mut nonfinite=0;
    for (a,b) in a.iter().zip(b){diff+=usize::from(a.to_bits()!=b.to_bits());nonfinite+=usize::from(!a.is_finite()||!b.is_finite());}
    println!("THREE512_EXACT label={label} values={} bitdiff={diff} nonfinite={nonfinite}",a.len());assert_eq!(nonfinite,0);assert_eq!(diff,0);
}
async fn fill(d:&mut Device,m:&mut Model,w:&mut Workspace,block:&Blocks,owner:usize,start:usize)->Result<()> {
    let mut p=0;while p<start {
        let count=if start-p>=512 {512}else{128};let rows:Vec<_>=(p..p+count).map(|p|input(owner,p,false)).collect();
        let seq=Sequence{inputs:&rows,start_position:p,blocks:block};
        if count==512 {m.start_prefill512(d,&seq,w).await?;for layer in 0..LAYERS {assert_eq!(m.advance_prefill(d,w).await?.is_some(),layer+1==LAYERS);}}
        else {m.forward_sequences_native_qkv(d,&[seq],w).await?;}
        p+=count;
    }Ok(())
}
async fn control(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],starts:&[usize],rows:&[Vec<Input<'static>>])->Result<(Vec<Vec<f32>>,f64)> {
    let mut outputs=Vec::new();let mut ms=0.0;
    for owner in 0..rows.len(){for (quarter,chunk) in rows[owner].chunks_exact(128).enumerate(){
        let seq=Sequence{inputs:chunk,start_position:starts[owner]+128*quarter,blocks:&blocks[owner]};
        let now=std::time::Instant::now();let result=m.forward_sequences_native_qkv(d,&[seq],w).await?;ms+=now.elapsed().as_secs_f64()*1000.;
        if chunk.last().unwrap().logits {outputs.push(read_logits(d,&result[0]).await?);}
    }}assert_eq!(outputs.len(),rows.len());Ok((outputs,ms))
}
// Two strong packed controls: actual two-Tail256 work or a Tail384 followed
// by the doubled owner's remaining P128. Every real owner keeps chronological KV.
async fn control_packed(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],starts:&[usize],rows:&[Vec<Input<'static>>],width:usize)->Result<(Vec<Vec<f32>>,f64)> {
    assert_eq!(rows.len(),3);let double=rows.iter().position(|r|r.len()==256).unwrap();
    let groups=if width==2 {
        vec![vec![(0,0),(1,0)],if double==2 {vec![(2,0),(2,1)]}else{vec![(2,0),(double,1)]}]
    }else{assert_eq!(width,3);vec![vec![(0,0),(1,0),(2,0)],vec![(double,1)]]};
    let mut outputs:Vec<Option<Vec<f32>>>=vec![None;3];let mut ms=0.0;
    for group in groups {
        let mut owners=Vec::new();let mut sequences=Vec::new();
        if group.len()==2 && group[0].0==group[1].0 {
            let owner=group[0].0;assert_eq!(group,vec![(owner,0),(owner,1)]);
            owners.push(owner);sequences.push(Sequence{inputs:&rows[owner],start_position:starts[owner],blocks:&blocks[owner]});
        }else{for &(owner,quarter) in &group {
            owners.push(owner);sequences.push(Sequence{inputs:&rows[owner][quarter*128..(quarter+1)*128],start_position:starts[owner]+quarter*128,blocks:&blocks[owner]});
        }}
        if group.len()==1 {
            let now=std::time::Instant::now();let result=m.forward_sequences_native_qkv(d,&sequences,w).await?;
            ms+=now.elapsed().as_secs_f64()*1000.;
            assert!(sequences[0].inputs.last().unwrap().logits);
            outputs[owners[0]]=Some(read_logits(d,&result[0]).await?);continue;
        }
        let now=std::time::Instant::now();
        if group.len()==3 {m.start_prefill_tail384_with_padding(d,&sequences,w,bf16::from_bits(0x7fc1)).await?;}
        else if sequences.len()==1 {m.start_prefill256(d,&sequences[0],w).await?;}
        else {m.start_prefill_pair256(d,&sequences,w).await?;}
        ms+=now.elapsed().as_secs_f64()*1000.;
        for layer in 0..LAYERS {
            let now=std::time::Instant::now();
            let result=if group.len()==3 {m.advance_prefill_tail384(d,w).await?}else{m.advance_prefill_tail256(d,w).await?};
            ms+=now.elapsed().as_secs_f64()*1000.;
            if let Some(result)=result {assert_eq!(layer+1,LAYERS);assert_eq!(result.len(),owners.len());
                for (row,&owner) in owners.iter().enumerate(){if sequences[row].inputs.last().unwrap().logits {
                    assert!(outputs[owner].is_none());outputs[owner]=Some(read_logits(d,&result[row]).await?);
                }}
            }else{assert!(layer+1<LAYERS);}
        }
    }
    Ok((outputs.into_iter().map(|row|row.expect("every actual owner has one final output")).collect(),ms))
}
async fn candidate(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],starts:&[usize],rows:&[Vec<Input<'static>>],expected:&[Vec<f32>],interleave:bool,label:&str)->Result<f64> {
    let sequences:Vec<_>=(0..rows.len()).map(|owner|Sequence{inputs:&rows[owner],start_position:starts[owner],blocks:&blocks[owner+3]}).collect();
    let now=std::time::Instant::now();
    m.start_prefill_three_owner512(d,&sequences,w).await?;
    let prepare_ms=now.elapsed().as_secs_f64()*1000.;let mut layer_ms=0.0;let mut max_quantum_ms=0f64;
    assert_eq!(m.active_prefill_rings(),(0..rows.len()).map(|r|blocks[r+3].sliding).collect::<Vec<_>>());
    for owner in 0..rows.len(){
        let one=[input(owner,starts[owner],true)];
        assert!(m.forward_sequences(d,&[Sequence{inputs:&one,start_position:starts[owner],blocks:&blocks[owner+3]}],w).await.is_err());
        let alias=Blocks{sliding:blocks[6].sliding,full:blocks[owner+3].full.clone()};
        assert!(m.forward_sequences(d,&[Sequence{inputs:&one,start_position:starts[owner],blocks:&alias}],w).await.is_err());
    }
    assert!(m.start_prefill_three_owner512(d,&sequences,w).await.is_err());
    assert!(m.advance_prefill_group(d,w).await.is_err());
    assert!(m.advance_prefill_pair512(d,w).await.is_err());
    assert!(m.advance_prefill_tail256(d,w).await.is_err());
    assert!(m.advance_prefill_tail384(d,w).await.is_err());
    let other:Vec<_>=(0..128).map(|p|input(7,p,false)).collect();
    assert!(m.forward_sequences(d,&[Sequence{inputs:&other,start_position:0,blocks:&blocks[6]}],w).await.is_err());
    for layer in 0..LAYERS {
        let now=std::time::Instant::now();let output=m.advance_prefill_three_owner512(d,w).await?;
        let elapsed=now.elapsed().as_secs_f64()*1000.;layer_ms+=elapsed;max_quantum_ms=max_quantum_ms.max(elapsed);
        if let Some(output)=output {assert_eq!(layer+1,LAYERS);assert_eq!(output.len(),rows.len());
            for owner in 0..rows.len(){exact(&format!("{label}_owner{owner}"),&expected[owner],&read_logits(d,&output[owner]).await?);}
        } else {assert!(layer+1<LAYERS);}
        if interleave {
            let one=[input(70,layer,true)];let mut reference=Vec::new();
            for slot in [6,7] {
                let out=m.forward_sequences_r64(d,&[Sequence{inputs:&one,start_position:layer,blocks:&blocks[slot]}],w).await?;
                let got=read_logits(d,&out[0]).await?;
                if slot==6{reference=got;}else{exact(&format!("{label}_R64_layer{layer}"),&reference,&got);}
            }
        }
    }
    assert!(m.active_prefill_rings().is_empty());assert!(m.active_prefill_ring().is_none());
    println!("THREE512_TIME label={label} prepare_ms={prepare_ms:.6} layers_finish_ms={layer_ms:.6} total_ms={:.6} max_quantum_ms={max_quantum_ms:.6}",prepare_ms+layer_ms);
    Ok(prepare_ms+layer_ms)
}
#[tokio::main]
async fn main()->Result<()> {
    let selected=match std::env::args().nth(1).as_deref(){None|Some("all")=>None,
        Some(value)=>{let n=value.parse::<usize>()?;if n>9{return Err("case must be0..9 or all".into());}Some(n)}};
    let mut d=Device::new(ops::embed_token.topology())?;let weights=load::load_weights(&mut d).await?;
    let capacity=2560;let pages=Workspace::blocks_for(capacity);
    let mut w=Workspace::new_canonical64(&mut d,&weights,capacity,64).await?;
    let mut m=Model::new(&mut d,weights,&mut w).await?;m.prepare_group_scratch(&mut d).await?;
    let rings=[3,27,61,52,7,49,11,41,23,59];
    let blocks:Vec<_>=rings.into_iter().enumerate().map(|(owner,sliding)|Blocks{sliding,full:(0..pages).map(|p|255-owner*pages-p).collect()}).collect();
    for slot in [8,9]{fill(&mut d,&mut m,&mut w,&blocks[slot],90,128).await?;}
    let cases=[(0,[0,0,0]),(1,[0,0,0]),(2,[0,0,0]),
        (0,[896,256,1920]),(1,[256,896,1920]),(2,[256,896,1920]),
        (0,[1792,1920,1920]),(1,[1920,1920,1920]),(2,[1920,1920,1920]),(0,[1920,1920,1920])];
    for (case,(double,starts)) in cases.into_iter().enumerate(){
        if selected.is_some_and(|selected|selected!=case){continue;}
        for owner in 0..3{for slot in [owner,owner+3]{fill(&mut d,&mut m,&mut w,&blocks[slot],owner,starts[owner]).await?;}}
        let rows:Vec<Vec<_>>=(0..3).map(|owner|{let n=if owner==double{256}else{128};(starts[owner]..starts[owner]+n).map(|p|input(owner,p,p+1==starts[owner]+n)).collect()}).collect();
        let label=format!("case{case}_double{double}_start{}",starts[double]);
        let (expected,original_ms)=control(&mut d,&mut m,&mut w,&blocks,&starts,&rows).await?;
        let mut packed_ms=[0f64;2];
        for width in [2,3]{
            for owner in 0..3{fill(&mut d,&mut m,&mut w,&blocks[owner],owner,starts[owner]).await?;}
            let (outputs,ms)=control_packed(&mut d,&mut m,&mut w,&blocks,&starts,&rows,width).await?;packed_ms[width-2]=ms;
            for owner in 0..3{exact(&format!("{label}_control{width}_owner{owner}"),&expected[owner],&outputs[owner]);}
        }
        m.begin_stage_profile();let candidate_ms=candidate(&mut d,&mut m,&mut w,&blocks,&starts,&rows,&expected,case==6,&label).await?;
        for s in m.take_stage_profile(){println!("THREE512_STAGE label={label} stage={} calls={} ms={:.6}",s.stage,s.calls,s.milliseconds);}
        for owner in 0..3{let checked=m.compare_canonical_kv(&mut d,&w,&blocks[owner],&blocks[owner+3]).await?;println!("THREE512_CACHE label={label} owner={owner} bf16_values={checked} bitdiff=0");}
        let checked=m.compare_canonical_kv(&mut d,&w,&blocks[8],&blocks[9]).await?;println!("THREE512_SENTINEL label={label} bf16_values={checked} bitdiff=0");
        for step in 0..4{
            let tokens:Vec<_>=(0..3).map(|owner|[input(owner,starts[owner]+rows[owner].len()+step,true)]).collect();let mut reference=Vec::new();
            for offset in [0,3]{let seq:Vec<_>=(0..3).map(|owner|Sequence{inputs:&tokens[owner],start_position:starts[owner]+rows[owner].len()+step,blocks:&blocks[owner+offset]}).collect();
                let out=m.forward_sequences(&mut d,&seq,&mut w).await?;
                for owner in 0..3{let got=read_logits(&mut d,&out[owner]).await?;if offset==0{reference.push(got);}else{exact(&format!("{label}_post{step}_owner{owner}"),&reference[owner],&got);}}
            }
        }
        println!("THREE512_CASE_PASS label={label} fourP128_ms={original_ms:.6} two_tail256_ms={:.6} tail384_plusP128_ms={:.6} candidate_inclusive_ms={candidate_ms:.6} excludes_reads_interleaves_cache=true",packed_ms[0],packed_ms[1]);
    }
    let order=[blocks[2].clone(),blocks[1].clone(),blocks[0].clone(),blocks[5].clone(),blocks[4].clone(),blocks[3].clone(),blocks[6].clone(),blocks[7].clone(),blocks[8].clone(),blocks[9].clone()];
    let rows:Vec<Vec<_>>=(0..3).map(|owner|{let n=if owner==1{256}else{128};(0..n).map(|p|input(owner+20,p,p+1==n)).collect()}).collect();
    let starts=[0,0,0];let (expected,original_ms)=control(&mut d,&mut m,&mut w,&order,&starts,&rows).await?;
    let mut packed_ms=[0f64;2];for width in [2,3]{let (outputs,ms)=control_packed(&mut d,&mut m,&mut w,&order,&starts,&rows,width).await?;packed_ms[width-2]=ms;
        for owner in 0..3{exact(&format!("reset_control{width}_owner{owner}"),&expected[owner],&outputs[owner]);}}
    let candidate_ms=candidate(&mut d,&mut m,&mut w,&order,&starts,&rows,&expected,false,"reset_reverse").await?;
    for owner in 0..3{let checked=m.compare_canonical_kv(&mut d,&w,&order[owner],&order[owner+3]).await?;println!("THREE512_RESET_CACHE owner={owner} bf16_values={checked} bitdiff=0");}
    println!("PASS ThreeOwner512 all_vocab/cache/ownership/interleaves exact; reset_fourP128_ms={original_ms:.6} reset_two_tail256_ms={:.6} reset_tail384_plusP128_ms={:.6} reset_candidate_ms={candidate_ms:.6}; no Worker activation or HTTP claim",packed_ms[0],packed_ms[1]);Ok(())
}

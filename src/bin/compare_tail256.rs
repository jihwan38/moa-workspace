//! Exact two-live-quarter tails against two current P128 calls; no Worker activation.
use furiosa_opt_gemma4::{LAYERS,host::{generate::read_logits,load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}},ops};
use furiosa_opt_std::prelude::*;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn input(owner:usize,p:usize,logits:bool)->Input<'static>{Input{token:2+(owner*733+p*37)%8000,soft:None,logits,image:None,audio:None}}
fn exact(label:&str,a:&[f32],b:&[f32]){
    assert_eq!(a.len(),b.len());let mut diff=0;let mut nonfinite=0;
    for (a,b) in a.iter().zip(b){diff+=usize::from(a.to_bits()!=b.to_bits());nonfinite+=usize::from(!a.is_finite()||!b.is_finite());}
    println!("TAIL256_EXACT label={label} values={} bitdiff={diff} nonfinite={nonfinite}",a.len());assert_eq!(nonfinite,0);assert_eq!(diff,0);
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
async fn candidate(d:&mut Device,m:&mut Model,w:&mut Workspace,blocks:&[Blocks],starts:&[usize],rows:&[Vec<Input<'static>>],expected:&[Vec<f32>],padding:bf16,interleave:bool,label:&str)->Result<f64> {
    let sequences:Vec<_>=(0..rows.len()).map(|owner|Sequence{inputs:&rows[owner],start_position:starts[owner],blocks:&blocks[owner+2]}).collect();
    let now=std::time::Instant::now();
    if padding.to_bits()!=0 {m.start_prefill_tail256_with_padding(d,&sequences,w,padding).await?;}
    else if rows.len()==1 {m.start_prefill256(d,&sequences[0],w).await?;}
    else {m.start_prefill_pair256(d,&sequences,w).await?;}
    let prepare_ms=now.elapsed().as_secs_f64()*1000.;let mut layer_ms=0.0;let mut max_quantum_ms=0f64;
    assert_eq!(m.active_prefill_rings(),(0..rows.len()).map(|r|blocks[r+2].sliding).collect::<Vec<_>>());
    for owner in 0..rows.len(){
        let one=[input(owner,starts[owner],true)];
        assert!(m.forward_sequences(d,&[Sequence{inputs:&one,start_position:starts[owner],blocks:&blocks[owner+2]}],w).await.is_err());
        let alias=Blocks{sliding:blocks[4].sliding,full:blocks[owner+2].full.clone()};
        assert!(m.forward_sequences(d,&[Sequence{inputs:&one,start_position:starts[owner],blocks:&alias}],w).await.is_err());
    }
    assert!(m.start_prefill_tail256_with_padding(d,&sequences,w,padding).await.is_err());
    assert!(m.advance_prefill_group(d,w).await.is_err());
    assert!(m.advance_prefill_pair512(d,w).await.is_err());
    let other:Vec<_>=(0..128).map(|p|input(7,p,false)).collect();
    assert!(m.forward_sequences(d,&[Sequence{inputs:&other,start_position:0,blocks:&blocks[4]}],w).await.is_err());
    for layer in 0..LAYERS {
        let now=std::time::Instant::now();let output=m.advance_prefill_tail256(d,w).await?;
        let elapsed=now.elapsed().as_secs_f64()*1000.;layer_ms+=elapsed;max_quantum_ms=max_quantum_ms.max(elapsed);
        if let Some(output)=output {assert_eq!(layer+1,LAYERS);assert_eq!(output.len(),rows.len());
            for owner in 0..rows.len(){exact(&format!("{label}_owner{owner}"),&expected[owner],&read_logits(d,&output[owner]).await?);}
        } else {assert!(layer+1<LAYERS);}
        if interleave {
            let one=[input(70,layer,true)];let mut reference=Vec::new();
            for slot in [4,5] {
                let out=m.forward_sequences_r64(d,&[Sequence{inputs:&one,start_position:layer,blocks:&blocks[slot]}],w).await?;
                let got=read_logits(d,&out[0]).await?;
                if slot==4{reference=got;}else{exact(&format!("{label}_R64_layer{layer}"),&reference,&got);}
            }
        }
    }
    assert!(m.active_prefill_rings().is_empty());assert!(m.active_prefill_ring().is_none());
    println!("TAIL256_TIME label={label} prepare_ms={prepare_ms:.6} layers_finish_ms={layer_ms:.6} total_ms={:.6} max_quantum_ms={max_quantum_ms:.6} padding_bits={:04x}",prepare_ms+layer_ms,padding.to_bits());
    Ok(prepare_ms+layer_ms)
}
#[tokio::main]
async fn main()->Result<()> {
    // Optional case0..8 bounds the expensive cache/interleave audit; default all.
    let selected=match std::env::args().nth(1).as_deref() {
        None|Some("all")=>None,
        Some(value)=>{let n=value.parse::<usize>()?;if n>8{return Err("case must be0..8 or all".into());}Some(n)},
    };
    let mut d=Device::new(ops::embed_token.topology())?;let weights=load::load_weights(&mut d).await?;
    let capacity=2304;let pages=Workspace::blocks_for(capacity);
    let mut w=Workspace::new_canonical64(&mut d,&weights,capacity,64).await?;
    let mut m=Model::new(&mut d,weights,&mut w).await?;m.prepare_group_scratch(&mut d).await?;
    let rings=[3,27,61,52,7,49,11,41];
    let blocks:Vec<_>=rings.into_iter().enumerate().map(|(owner,sliding)|Blocks{sliding,full:(0..pages).map(|p|255-owner*pages-p).collect()}).collect();
    for slot in [6,7]{fill(&mut d,&mut m,&mut w,&blocks[slot],90,128).await?;}
    let cases=[(1,[0,0]),(2,[0,0]),(1,[256,0]),(2,[256,896]),(1,[896,0]),(2,[896,256]),(1,[1792,0]),(1,[1920,0]),(2,[1920,1920])];
    for (case,(owners,starts)) in cases.into_iter().enumerate(){
        if selected.is_some_and(|selected|selected!=case){continue;}
        for owner in 0..owners {for slot in [owner,owner+2]{fill(&mut d,&mut m,&mut w,&blocks[slot],owner,starts[owner]).await?;}}
        let rows:Vec<Vec<_>>=(0..owners).map(|owner|{let count=if owners==1{256}else{128};(starts[owner]..starts[owner]+count).map(|p|input(owner,p,p+1==starts[owner]+count)).collect()}).collect();
        let label=format!("case{case}_owners{owners}_start{}",starts[0]);
        m.begin_stage_profile();let (expected,control_ms)=control(&mut d,&mut m,&mut w,&blocks,&starts[..owners],&rows).await?;
        for s in m.take_stage_profile(){println!("TAIL256_CONTROL_STAGE label={label} stage={} calls={} ms={:.6}",s.stage,s.calls,s.milliseconds);}
        let padding=bf16::from_bits(match case%3 {0=>0,1=>0x3f40,_=>0x7fc1});
        m.begin_stage_profile();let candidate_ms=candidate(&mut d,&mut m,&mut w,&blocks,&starts[..owners],&rows,&expected,padding,case==6,&label).await?;
        for s in m.take_stage_profile(){println!("TAIL256_STAGE label={label} stage={} calls={} ms={:.6}",s.stage,s.calls,s.milliseconds);}
        for owner in 0..owners {
            let checked=m.compare_canonical_kv(&mut d,&w,&blocks[owner],&blocks[owner+2]).await?;
            println!("TAIL256_CACHE label={label} owner={owner} bf16_values={checked} bitdiff=0 all_layers=true entire_rings_and_allocated_pages=true");
        }
        let checked=m.compare_canonical_kv(&mut d,&w,&blocks[6],&blocks[7]).await?;
        println!("TAIL256_SENTINEL_CACHE label={label} bf16_values={checked} bitdiff=0");
        for step in 0..4 {
            let tokens:Vec<_>=(0..owners).map(|owner|[input(owner,starts[owner]+rows[owner].len()+step,true)]).collect();let mut reference=Vec::new();
            for offset in [0,2] {
                let seq:Vec<_>=(0..owners).map(|owner|Sequence{inputs:&tokens[owner],start_position:starts[owner]+rows[owner].len()+step,blocks:&blocks[owner+offset]}).collect();
                let out=m.forward_sequences(&mut d,&seq,&mut w).await?;
                for owner in 0..owners {let got=read_logits(&mut d,&out[owner]).await?;if offset==0{reference.push(got);}else{exact(&format!("{label}_post_step{step}_owner{owner}"),&reference[owner],&got);}}
            }
        }
        println!("TAIL256_CASE_PASS label={label} control_twoP128_ms={control_ms:.6} candidate_inclusive_ms={candidate_ms:.6} timing_excludes_downloads_interleaves_cache_checks=true");
    }
    // Reset previously long histories and reverse actual owner order. Padding
    // cannot inherit prior four-quarter scratch or any request's cache metadata.
    let rows:Vec<Vec<_>>=(0..2).map(|owner|(0..128).map(|p|input(owner+20,p,p==127)).collect()).collect();
    let order=[blocks[1].clone(),blocks[0].clone(),blocks[3].clone(),blocks[2].clone(),blocks[4].clone(),blocks[5].clone(),blocks[6].clone(),blocks[7].clone()];
    let (expected,control_ms)=control(&mut d,&mut m,&mut w,&order,&[0,0],&rows).await?;
    let candidate_ms=candidate(&mut d,&mut m,&mut w,&order,&[0,0],&rows,&expected,bf16::from_bits(0x7fc1),false,"reset_reverse").await?;
    for owner in 0..2{let checked=m.compare_canonical_kv(&mut d,&w,&order[owner],&order[owner+2]).await?;println!("TAIL256_RESET_CACHE owner={owner} bf16_values={checked} bitdiff=0");}
    println!("PASS Tail256 all_vocab/cache/padding/ownership/interleaves exact; reset_control_ms={control_ms:.6} reset_candidate_ms={candidate_ms:.6}; no Worker activation or HTTP claim");Ok(())
}

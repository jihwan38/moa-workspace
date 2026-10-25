//! Matched same-arithmetic retained B16 outputs and exact CPU sampling.
use furiosa_opt_gemma4::{LAYERS, host::{generate::read_logits,
    greedy::GreedySelector, load, model::{ForwardOutput, Input, Model, Sequence, Logits},
    runtime::{Blocks, Workspace}, sampling::{sample, SamplingConfig}}, ops};
use furiosa_opt_std::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use std::{collections::HashSet, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn input(row:usize,position:usize,logits:bool)->Input<'static>{
    Input{token:2+(row*313+position*37)%8000,soft:None,logits,image:None,audio:None}
}
async fn read_rows(d:&mut Device,rows:&[Logits],wanted:&[bool])->Result<Vec<Vec<f32>>>{
    let mut out=Vec::new();
    for (row,&wanted) in rows.iter().zip(wanted){out.push(if wanted{read_logits(d,row).await?}else{Vec::new()});}
    Ok(out)
}
fn exact(actual:&[f32],expected:&[f32],label:&str){
    assert_eq!(actual.len(),expected.len(),"{label}");
    let mismatch=actual.iter().zip(expected).enumerate().find(|(_, (a,b))|a.to_bits()!=b.to_bits());
    assert!(mismatch.is_none(),"{label} first_mismatch={mismatch:?}");
}
async fn verify_batch(d:&mut Device,selector:&mut GreedySelector,output:ForwardOutput<'_>,
    expected:&[Vec<f32>],width:usize,step:usize)->Result<()> {
    let ForwardOutput::Batch16(logits,live)=output else{return Err("expected pure B16 retained output".into());};
    assert_eq!(live,width);
    let mut bans:Vec<HashSet<usize>>=(0..width).map(|row|if row%2==0{
        [258882,258883].into_iter().collect()
    }else{[258882,258883,1,50,100,101,106].into_iter().collect()}).collect();
    let policies:Vec<_>=expected.iter().zip(&bans).map(|(x,b)|(!x.is_empty()).then_some(b)).collect();
    let begin=Instant::now();let selected=selector.select_b16(d,logits,&policies).await?;
    let selection_ms=begin.elapsed().as_secs_f64()*1000.0;
    for row in 0..width {
        if expected[row].is_empty(){assert!(selected[row].is_none());continue;}
        let mut rng=StdRng::seed_from_u64(814+row as u64);
        let want=sample(&expected[row],&SamplingConfig::DEFAULT,&bans[row],&mut rng);
        let values=selector.read_fallback_b16(d,logits,row).await?;
        exact(&values,&expected[row],"retained/extracted BF16 row");
        let got=selected[row].unwrap_or_else(||sample(&values,&SamplingConfig::DEFAULT,&bans[row],&mut rng));
        assert_eq!(got,want,"width={width} row={row} step={step}");
        bans[row].insert(want);
    }
    // In the same batch, stochastic rows bypass device selection and retain the
    // original values. This also verifies that None policies do not leak tokens.
    let policies:Vec<_>=expected.iter().zip(&bans).enumerate().map(|(row,(x,b))|
        (!x.is_empty()&&row%3!=0).then_some(b)).collect();
    let selected=selector.select_b16(d,logits,&policies).await?;
    for row in 0..width {
        if expected[row].is_empty(){assert!(selected[row].is_none());continue;}
        if row%3==0 {
            assert!(selected[row].is_none());
            let config=SamplingConfig{temperature:0.7,top_k:33,top_p:0.87};
            let mut control_rng=StdRng::seed_from_u64(991+row as u64);
            let mut candidate_rng=StdRng::seed_from_u64(991+row as u64);
            let values=selector.read_fallback_b16(d,logits,row).await?;
            assert_eq!(sample(&values,&config,&bans[row],&mut candidate_rng),
                sample(&expected[row],&config,&bans[row],&mut control_rng));
        } else {
            let mut rng=StdRng::seed_from_u64(313+row as u64);
            let want=sample(&expected[row],&SamplingConfig::DEFAULT,&bans[row],&mut rng);
            let got=match selected[row]{Some(t)=>t,None=>sample(
                &selector.read_fallback_b16(d,logits,row).await?,&SamplingConfig::DEFAULT,&bans[row],&mut rng)};
            assert_eq!(got,want,"suppressed maximum row={row}");
        }
    }
    println!("B16_MODEL_EXACT width={width} step={step} requested={} normal_suppressed_stochastic=true select_ms={selection_ms:.3}",expected.iter().filter(|x|!x.is_empty()).count());
    Ok(())
}
#[tokio::main]
async fn main()->Result<()> {
    let mut d=Device::new(ops::embed_token.topology())?;
    let weights=load::load_weights(&mut d).await?;
    let mut ws=Workspace::new_canonical64(&mut d,&weights,512,64).await?;
    let mut model=Model::new(&mut d,weights,&mut ws).await?;
    let mut selector=GreedySelector::new(&mut d).await?;
    let blocks:Vec<_>=(0..64).map(|r|Blocks{sliding:(r*13+31)%64,full:vec![255-r]}).collect();
    for width in [2,8,9,16,1] {for step in 0..3 {
        let wanted:Vec<_>=(0..width).map(|r|!(step==1&&r+1==width&&width>1)).collect();
        let rows:Vec<_>=(0..width).map(|r|[input(r,step,wanted[r])]).collect();
        let control:Vec<_>=(0..width).map(|r|Sequence{inputs:&rows[r],start_position:step,blocks:&blocks[r]}).collect();
        let control_output=model.forward_sequences(&mut d,&control,&mut ws).await?;
        let expected=read_rows(&mut d,control_output,&wanted).await?;
        let candidate:Vec<_>=(0..width).map(|r|Sequence{inputs:&rows[r],start_position:step,blocks:&blocks[r+32]}).collect();
        let output=model.forward_outputs(&mut d,&candidate,&mut ws).await?;
        if width==1 {
            let ForwardOutput::Rows(outputs)=output else{return Err("C1 must keep scalar rows".into());};
            exact(&read_logits(&mut d,&outputs[0]).await?,&expected[0],"C1 control");
            println!("C1_ROWS_EXACT step={step}");
        }else{verify_batch(&mut d,&mut selector,output,&expected,width,step).await?;}
    }}
    // Mixed prompt/decode must materialize rows before a prompt can reuse scratch.
    let rows:Vec<Vec<_>>=(0..3).map(|r|(0..if r==1{3}else{1}).map(|p|input(r+21,p,p+1==if r==1{3}else{1})).collect()).collect();
    let control:Vec<_>=(0..3).map(|r|Sequence{inputs:&rows[r],start_position:0,blocks:&blocks[r]}).collect();
    let control_output=model.forward_sequences(&mut d,&control,&mut ws).await?;
    let expected=read_rows(&mut d,control_output,&[true;3]).await?;
    let candidate:Vec<_>=(0..3).map(|r|Sequence{inputs:&rows[r],start_position:0,blocks:&blocks[r+32]}).collect();
    let ForwardOutput::Rows(output)=model.forward_outputs(&mut d,&candidate,&mut ws).await? else{return Err("mixed prompt/decode must keep rows".into());};
    let actual=read_rows(&mut d,output,&[true;3]).await?;
    for row in 0..3{exact(&actual[row],&expected[row],"mixed scratch preservation");}
    println!("MIXED_ROWS_EXACT singleton_prompt_singleton=true");
    // No requested logits: never expose old retained vocabulary data.
    let rows=[[input(1,0,false)],[input(2,0,false)]];
    let sequences:Vec<_>=(0..2).map(|r|Sequence{inputs:&rows[r],start_position:0,blocks:&blocks[r+32]}).collect();
    assert!(matches!(model.forward_outputs(&mut d,&sequences,&mut ws).await?,ForwardOutput::Rows(_)));
    // A paused narrow prompt has a separate hidden checkpoint; B16 retained
    // vocabulary output cannot alias it or admit its cache allocation.
    let prompt:Vec<_>=(0..3).map(|p|input(59,p,p==2)).collect();
    let control=[Sequence{inputs:&prompt,start_position:0,blocks:&blocks[62]}];
    let control_output=model.forward_sequences(&mut d,&control,&mut ws).await?;
    let prompt_expected=read_rows(&mut d,control_output,&[true]).await?;
    model.start_prefill(&mut d,&Sequence{inputs:&prompt,start_position:0,blocks:&blocks[63]},&mut ws).await?;
    assert!(model.advance_prefill(&mut d,&mut ws).await?.is_none());
    let rows=[[input(44,0,true)],[input(45,0,true)]];
    let control:Vec<_>=(0..2).map(|r|Sequence{inputs:&rows[r],start_position:0,blocks:&blocks[r]}).collect();
    let control_output=model.forward_sequences(&mut d,&control,&mut ws).await?;
    let expected=read_rows(&mut d,control_output,&[true;2]).await?;
    let candidate:Vec<_>=(0..2).map(|r|Sequence{inputs:&rows[r],start_position:0,blocks:&blocks[r+32]}).collect();
    let output=model.forward_outputs(&mut d,&candidate,&mut ws).await?;
    verify_batch(&mut d,&mut selector,output,&expected,2,99).await?;
    let alias=[Sequence{inputs:&rows[0],start_position:0,blocks:&blocks[63]}];
    assert!(model.forward_outputs(&mut d,&alias,&mut ws).await.is_err());
    for layer in 1..LAYERS {
        let output=model.advance_prefill(&mut d,&mut ws).await?;
        if layer+1==LAYERS {exact(&read_logits(&mut d,output.ok_or("missing prompt output")?).await?,&prompt_expected[0],"paused narrow prompt");}
        else{assert!(output.is_none());}
    }
    println!("PASS B16 retained outputs widths2/8/9/16, C1, unrequested logits, mixed prompts, active narrow prompt, alias rejection, exact greedy/stochastic CPU semantics");
    Ok(())
}

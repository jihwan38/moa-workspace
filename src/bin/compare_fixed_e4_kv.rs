//! Fixed-scale E4 KV numerical surrogate. Both replay arms execute the same
//! compile-time precision from the beginning of every history; no mode switch.
use furiosa_opt_gemma4::{LAYERS,axes::W,ops,ops_fixed_e4_quality,
 host::{b16_metadata_diagnostic::{collect_retained,logits_bits},
 fixed_e4_model_diagnostic::{exact,verify_physical32_identity},
 greedy::GreedySelector,load,model::{Input,Model,Sequence},runtime::{Blocks,Workspace}}};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
type Rows=Vec<Vec<u16>>;
fn input(owner: usize, p: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (owner * 313 + p * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn blocks(owner: usize) -> Blocks {
    Blocks {
        sliding: (owner * 13 + 7) % 32,
        full: (0..4).map(|p| 255 - owner * 4 - p).collect(),
    }
}
#[cfg(backend = "npu")]
fn memory(d: &Device, label: &str) -> Result<()> {
    let m = d.memory()?;
    println!(
        "FIXED_E4_MEMORY label={label} capacity={} available={} largest={}",
        m.capacity, m.available, m.largest
    );
    Ok(())
}
#[cfg(not(backend = "npu"))]
fn memory(_: &Device, _: &str) -> Result<()> {
    Ok(())
}

async fn fill(
    d: &mut Device,
    m: &mut Model,
    w: &mut Workspace,
    allocation: &[Blocks],
    prefix: usize,
) -> Result<()> {
    let begin = Instant::now();
    for (owner, block) in allocation.iter().enumerate() {
        for p in (0..prefix).step_by(512) {
            let rows: Vec<_> = (p..p + 512).map(|p| input(owner, p, false)).collect();
            m.start_prefill512(
                d,
                &Sequence {
                    inputs: &rows,
                    start_position: p,
                    blocks: block,
                },
                w,
            )
            .await?;
            for layer in 0..LAYERS {
                assert_eq!(
                    m.advance_prefill(d, w).await?.is_some(),
                    layer + 1 == LAYERS
                );
            }
        }
    }
    println!(
        "FIXED_E4_FILL owners={} actual_tokens={} elapsed_ms={:.6}",
        allocation.len(),
        allocation.len() * prefix,
        begin.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn chosen(step: usize) -> Vec<usize> {
    match step {
        0 => vec![0],
        1 | 2 => vec![1, 2],
        3 => (0..8).rev().collect(),
        4 | 7 => (0..16).map(|i| (i * 5 + 3) % 16).collect(),
        5 => (8..16).rev().collect(),
        6 => vec![0, 2],
        _ => (0..8).collect(),
    }
}

async fn model_case(d:&mut Device,m:&mut Model,w:&mut Workspace,selector:&mut GreedySelector,prefix:usize)->Result<()> {
    let allocations:Vec<Vec<_>>=(0..2).map(|half|(0..16).map(|r|blocks(half*16+r)).collect()).collect();
    let mut expected:Vec<Rows>=Vec::new();let mut prompt_reference=Vec::new();let mut checked=0usize;
    for pass in 0..2 {
        let allocation=&allocations[pass];fill(d,m,w,allocation,prefix).await?;
        let prompt:Vec<_>=(prefix..prefix+128).map(|p|input(15,p,p==prefix+127)).collect();
        m.start_prefill(d,&Sequence{inputs:&prompt,start_position:prefix,blocks:&allocation[15]},w).await?;
        let mut positions=vec![prefix;16];
        for step in 0..10 {
            let owners=chosen(step);
            if step==6 {for &r in &owners {positions[r]=0;}}
            let rows:Vec<_>=owners.iter().map(|&r|[input(r,positions[r],true)]).collect();
            let seq:Vec<_>=owners.iter().enumerate().map(|(i,&r)|Sequence{inputs:&rows[i],start_position:positions[r],blocks:&allocation[r]}).collect();
            // Same fallback and re-entry in both arms, preserving arithmetic.
            m.set_b16_epilogue_fused(step!=8);m.set_b16_prefix_fused(step!=8);m.set_b16_output_hsplit(step!=8);
            m.begin_stage_profile();let begin=Instant::now();
            let output=m.forward_outputs(d,&seq,w).await?;let model_ms=begin.elapsed().as_secs_f64()*1000.;
            let found=collect_retained(d,selector,output,owners.len(),if step==3{2}else{0}).await?;
            for (i,bits) in found.iter().enumerate() {
                assert_eq!(bits.len(),W::SIZE);
                if pass==1 {exact(&format!("decode_prefix{prefix}_step{step}_owner{}",owners[i]),&expected[step][i],bits);checked+=bits.len();}
            }
            if pass==0{expected.push(found);}
            println!("FIXED_E4_STEP pass={pass} step={step} live={} model_ms={model_ms:.6} surrogate_timing_not_compression=true",owners.len());
            for s in m.take_stage_profile(){println!("FIXED_E4_STAGE pass={pass} step={step} stage={} calls={} total_ms={:.6}",s.stage,s.calls,s.milliseconds);}
            for r in owners{positions[r]+=1;}
            if step<4 {for quantum in 0..12 {
                if let Some(out)=m.advance_prefill(d,w).await? {
                    assert_eq!((step,quantum),(3,11));let bits=logits_bits(d,out).await?;
                    if pass==0{prompt_reference=bits;}else{exact(&format!("active_p128_prefix{prefix}"),&prompt_reference,&bits);checked+=bits.len();}
                    positions[15]+=128;
                }else{assert_ne!((step,quantum),(3,11));}
            }}
        }
        m.set_b16_epilogue_fused(true);m.set_b16_prefix_fused(true);m.set_b16_output_hsplit(true);
    }
    let mut cache=0usize;for r in 0..16{cache+=m.compare_canonical_kv(d,w,&allocations[0][r],&allocations[1][r]).await?;}
    println!("PASS FIXED_E4_MODEL enabled={} prefix={prefix} replay_exact_logits={checked} replay_cache_bf16_values={cache} own_output_sampling=true quality_gate=false",ops_fixed_e4_quality::ENABLED);Ok(())
}
async fn r32_case(d:&mut Device,m:&mut Model,w:&mut Workspace,selector:&mut GreedySelector,prefix:usize)->Result<()> {
    let allocation:Vec<_>=(0..17).map(blocks).collect();let mut expected:Vec<Rows>=Vec::new();let mut checked=0;
    for pass in 0..2 {
        fill(d,m,w,&allocation,prefix).await?;let mut positions=vec![prefix;17];
        for step in 0..4 {
            let owners:Vec<_>=match step {0=>vec![0],3=>(0..8).rev().collect(),_=>(0..17).map(|i|(i*5+step)%17).collect()};
            let rows:Vec<_>=owners.iter().map(|&r|[input(r,positions[r],true)]).collect();
            let seq:Vec<_>=owners.iter().enumerate().map(|(i,&r)|Sequence{inputs:&rows[i],start_position:positions[r],blocks:&allocation[r]}).collect();
            let output=m.forward_outputs(d,&seq,w).await?;
            let found=collect_retained(d,selector,output,owners.len(),0).await?;
            if pass==0{expected.push(found);}else{for(i,bits)in found.iter().enumerate(){exact(&format!("r32_prefix{prefix}_step{step}_owner{}",owners[i]),&expected[step][i],bits);checked+=bits.len();}}
            for r in owners{positions[r]+=1;}
        }
    }
    println!("PASS FIXED_E4_R32 enabled={} prefix={prefix} replay_exact_logits={checked} same_physical_recomputed_histories=true no_cross_program_exact_claim=true",ops_fixed_e4_quality::ENABLED);Ok(())
}
async fn partial_case(d:&mut Device,m:&mut Model,w:&mut Workspace,prefix:usize)->Result<()> {
    let allocation=[blocks(0),blocks(16)];let mut expected=Vec::new();let mut checked=0;
    for pass in 0..2 {
        // Complete native128 histories include old-ring wrap. Logical owner0
        // input is independent of its different physical slot/page mapping.
        for p in (0..prefix).step_by(128) {
            let rows:Vec<_>=(p..p+128).map(|p|input(0,p,false)).collect();
            m.forward_sequences_native_qkv(d,&[Sequence{inputs:&rows,start_position:p,blocks:&allocation[pass]}],w).await?;
        }
        let mut p=prefix;
        for (step,n) in [17,31,16,1].into_iter().enumerate() {
            let rows:Vec<_>=(p..p+n).map(|p|input(0,p,false)).collect();
            // Only last requested row; count17/31 uses tiled Wide,16 B16,
            // and1 scalar. Report all-vocabulary finite output each time.
            let mut rows=rows;for x in &mut rows{x.logits=false;}rows.last_mut().unwrap().logits=true;
            let seq=[Sequence{inputs:&rows,start_position:p,blocks:&allocation[pass]}];
            let out=if n>16{m.forward_sequences_wide(d,&seq,w).await?}else{m.forward_sequences(d,&seq,w).await?};
            let bits=logits_bits(d,&out[0]).await?;
            if pass==0{expected.push(bits);}else{exact(&format!("partial_prefix{prefix}_step{step}_count{n}"),&expected[step],&bits);checked+=bits.len();}
            p+=n;
        }
        // Force the explicit ordinary BF16-QKV P128 producer at aligned0 after
        // reuse; both passes follow the same route and future history is masked.
        let rows:Vec<_>=(0..128).map(|p|input(0,p,p==127)).collect();
        let out=m.forward_sequences_bf16_qkv(d,&[Sequence{inputs:&rows,start_position:0,blocks:&allocation[pass]}],w).await?;
        let bits=logits_bits(d,&out[0]).await?;
        if pass==0{expected.push(bits);}else{exact("partial_reset_original_p128",&expected[4],&bits);checked+=bits.len();}
    }
    let cache=m.compare_canonical_kv(d,w,&allocation[0],&allocation[1]).await?;
    println!("PASS FIXED_E4_PARTIAL enabled={} prefix={prefix} replay_exact_logits={checked} replay_cache_bf16_values={cache}",ops_fixed_e4_quality::ENABLED);Ok(())
}
#[tokio::main]
async fn main()->Result<()> {
    let mode=std::env::args().nth(1).unwrap_or_else(||"model".into());
    let prefix=std::env::args().nth(2).unwrap_or_else(||"512".into()).parse::<usize>()?;
    assert!(["model","r32","partial"].contains(&mode.as_str()));
    assert!(if mode=="partial"{[0,128,896,1024].contains(&prefix)}else{[512,1024].contains(&prefix)});
    let now=Instant::now();let mut d=Device::new(ops::embed_token.topology())?;let weights=load::load_weights(&mut d).await?;
    let mut w=Workspace::new_canonical64(&mut d,&weights,2048,32).await?;verify_physical32_identity(&w,2048)?;
    let mut m=Model::new(&mut d,weights,&mut w).await?;m.prepare_group_scratch(&mut d).await?;m.prepare_projection_owned_ffn(&mut d).await?;
    let mut selector=GreedySelector::new(&mut d).await?;memory(&d,"ready_original_bf16_storage")?;
    match mode.as_str(){"model"=>model_case(&mut d,&mut m,&mut w,&mut selector,prefix).await?,"r32"=>r32_case(&mut d,&mut m,&mut w,&mut selector,prefix).await?,_=>partial_case(&mut d,&mut m,&mut w,prefix).await?}
    memory(&d,"completed")?;
    println!("FIXED_E4_FINISHED enabled={} mode={mode} prefix={prefix} elapsed_s={:.6} quality_gate=false surrogate_timing_not_compression=true",ops_fixed_e4_quality::ENABLED,now.elapsed().as_secs_f64());Ok(())
}

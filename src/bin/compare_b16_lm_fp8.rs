//! Default-off approximate B16 LM: actual model histories, exact KV/fallback gates, memory.
//! Teacher forcing is identical in both passes; selection changes never alter later input.
use furiosa_opt_gemma4::host::sampling::{SamplingConfig, sample};
use furiosa_opt_gemma4::host::{b16_metadata_diagnostic::collect_retained, greedy::GreedySelector};
use furiosa_opt_gemma4::{
    axes::W,
    host::{
        load,
        model::{Input, Logits, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use std::{collections::HashSet, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn input(request: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (request * 313 + position * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
async fn raw(d: &mut Device, x: &Logits) -> Result<Vec<u16>> {
    let h = x
        .to_host::<m![W]>(&mut d.pdma)
        .await?
        .into_inner()
        .into_buf();
    let values: Vec<_> = h
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect();
    assert_eq!(values.len(), W::SIZE);
    assert!(
        values.iter().all(|x| x & 0x7f80 != 0x7f80),
        "nonfinite logits"
    );
    Ok(values)
}
fn selected(step: usize) -> Vec<usize> {
    match step {
        0 => vec![0],
        1 => vec![1, 16],
        2 => (0..8).map(|i| (i * 7 + 2) % 17).collect(),
        3 => (0..9).map(|i| (i * 3 + 4) % 17).collect(),
        4 => (0..16).rev().collect(),
        5 => (0..17).collect(),
        6 => vec![2, 16],
        7 => (0..14).map(|i| (i * 5 + 2) % 17).collect(),
        _ => (0..8).rev().collect(),
    }
}
fn check(pass:usize,step:usize,request:usize,a:&[u16],b:&[u16],strict:bool) {
    assert_eq!(a.len(),b.len());let mut differences=0;let mut max_abs=0f64;let mut e2=0f64;let mut ref2=0f64;
    for (&a,&b) in a.iter().zip(b) {
        let x=f32::from_bits((a as u32)<<16)as f64;let y=f32::from_bits((b as u32)<<16)as f64;
        assert!(x.is_finite()&&y.is_finite());differences+=usize::from(a!=b);
        let e=(x-y).abs();max_abs=max_abs.max(e);e2+=e*e;ref2+=x*x;
    }
    println!("B16_LM_FP8_LOGITS pass={pass} step={step} request={request} values={} strict={strict} differences={differences} max_abs={max_abs} relative_l2={}",a.len(),(e2/ref2.max(1e-300)).sqrt());
    if strict {assert_eq!(differences,0,"non-B16 LM route must remain exact");}
}
#[cfg(backend="npu")]
fn memory(d:&Device,stage:&str)->Result<usize> {
    let m=d.memory()?;println!("B16_LM_FP8_MEMORY stage={stage} capacity={} available={} largest={} used={}",m.capacity,m.available,m.largest,m.capacity-m.available);Ok(m.available)
}
#[cfg(not(backend="npu"))]
fn memory(_d:&Device,_stage:&str)->Result<usize>{Err("NPU only".into())}
#[tokio::main]
async fn main() -> Result<()> {
    let prefix = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "512".into())
        .parse::<usize>()?;
    assert!([512, 1024].contains(&prefix));
    let started = Instant::now();
    let mut d = Device::new(ops::embed_token.topology())?;
    memory(&d,"opened")?;
    let weights = load::load_weights(&mut d).await?;
    memory(&d,"weights")?;
    let mut ws = Workspace::new_canonical64(&mut d, &weights, 2048, 64).await?;
    memory(&d,"canonical64_pool")?;
    let mut model = Model::new(&mut d, weights, &mut ws).await?;
    model.prepare_group_scratch(&mut d).await?;
    model.prepare_projection_owned_ffn(&mut d).await?;
    let mut selector = GreedySelector::new(&mut d).await?;
    let guard_inputs=[input(0,0,true)];
    let guard_blocks=Blocks { sliding:0,full:vec![0,1,2,3] };
    assert!(model.forward_outputs_b16_lm_fp8(&mut d,&[Sequence {
        inputs:&guard_inputs,start_position:0,blocks:&guard_blocks,
    }],&mut ws).await.is_err(),"missing shadow must reject before model work");
    println!("B16_LM_FP8_MISSING_SHADOW rejected_before_forward=true");
    let before=memory(&d,"current_model_joined_weights_and_scratch")?;
    let start=Instant::now();model.prepare_b16_lm_fp8(&mut d).await?;
    let startup_ms=start.elapsed().as_secs_f64()*1000.0;
    let after=memory(&d,"with_fp8_lm_shadow")?;
    println!("B16_LM_FP8_STARTUP ms={startup_ms} allocator_delta={} original_embedding_retained=true",before-after);
    assert!(before-after<=1_100_000_000,"shadow capacity cap");
    model.prepare_b16_lm_fp8(&mut d).await?;
    assert_eq!(memory(&d,"prepare_idempotent")?,after,"no duplicate shadow allocation");
    let mut expected: Vec<Vec<Vec<u16>>> = Vec::new();
    let mut checked = 0usize;
    let mut reservations: Vec<Vec<Blocks>> = Vec::new();
    let mut prompt_expected = Vec::new();
    let prompt_blocks = [
        Blocks {
            sliding: 50,
            full: vec![80, 81, 82, 83],
        },
        Blocks {
            sliding: 51,
            full: vec![84, 85, 86, 87],
        },
    ];
    for pass in 0..2 {
        // Both passes use current Automatic compute and teacher-forced tokens; only B16 LM differs.
        let blocks: Vec<_> = (0..17)
            .map(|r| {
                let slot = if pass == 0 { r } else { 32 + r };
                Blocks {
                    sliding: slot,
                    full: (0..4)
                        .map(|p| {
                            if pass == 0 {
                                r * 4 + p
                            } else {
                                255 - (r * 4 + p)
                            }
                        })
                        .collect(),
                }
            })
            .collect();
        reservations.push(blocks.clone());
        let fill = Instant::now();
        for group in 0..4 {
            for pos in (0..prefix).step_by(128) {
                let ins: Vec<Vec<_>> = (0..4)
                    .map(|i| {
                        (pos..pos + 128)
                            .map(|p| input(group * 4 + i, p, false))
                            .collect()
                    })
                    .collect();
                let seq: Vec<_> = (0..4)
                    .map(|i| Sequence {
                        inputs: &ins[i],
                        start_position: pos,
                        blocks: &blocks[group * 4 + i],
                    })
                    .collect();
                model.start_prefill_group(&mut d, &seq, &mut ws).await?;
                while model
                    .advance_prefill_group(&mut d, &mut ws)
                    .await?
                    .is_none()
                {}
            }
        }
        for pos in (0..prefix).step_by(128) {
            let ins: Vec<_> = (pos..pos + 128).map(|p| input(16, p, false)).collect();
            model
                .forward_sequences(
                    &mut d,
                    &[Sequence {
                        inputs: &ins,
                        start_position: pos,
                        blocks: &blocks[16],
                    }],
                    &mut ws,
                )
                .await?;
        }
        println!(
            "B16_LM_FP8_FILL pass={pass} actual_history_tokens={} ms={:.3}",
            17 * prefix,
            fill.elapsed().as_secs_f64() * 1000.0
        );
        let active = &prompt_blocks[pass];
        for pos in (0..896).step_by(128) {
            let ins: Vec<_> = (pos..pos + 128).map(|p| input(99, p, false)).collect();
            model
                .forward_sequences(
                    &mut d,
                    &[Sequence {
                        inputs: &ins,
                        start_position: pos,
                        blocks: active,
                    }],
                    &mut ws,
                )
                .await?;
        }
        let prompt: Vec<_> = (896..1024).map(|p| input(99, p, p == 1023)).collect();
        model
            .start_prefill(
                &mut d,
                &Sequence {
                    inputs: &prompt,
                    start_position: 896,
                    blocks: active,
                },
                &mut ws,
            )
            .await?;
        let mut positions = [prefix; 17];
        for step in 0..10 {
            let chosen = selected(step);
            if step == 6 {
                for &r in &chosen {
                    positions[r] = 0;
                }
            }
            // Interleave a short prompt after decode to exercise Prefill's retained canonical raw buffers.
            if step == 8 {
                let request = 2;
                let ins: Vec<_> = (positions[request]..positions[request] + 31)
                    .map(|p| input(request, p, false))
                    .collect();
                for (chunk, inputs) in ins.chunks(16).enumerate() {
                    model
                        .forward_sequences(
                            &mut d,
                            &[Sequence {
                                inputs,
                                start_position: positions[request] + chunk * 16,
                                blocks: &blocks[request],
                            }],
                            &mut ws,
                        )
                        .await?;
                }
                positions[request] += 31;
            }
            let ins: Vec<_> = chosen
                .iter()
                .map(|&r| [input(r, positions[r], true)])
                .collect();
            let seq: Vec<_> = chosen
                .iter()
                .enumerate()
                .map(|(i, &r)| Sequence {
                    inputs: &ins[i],
                    start_position: positions[r],
                    blocks: &blocks[r],
                })
                .collect();
            model.set_b16_epilogue_fused(step != 8); // Explicit original four-launch FFN fallback, then re-entry.
            model.set_b16_output_hsplit(step != 8);
            model.set_decode_blocked_output(step != 8); // This B16 step tests original FFN epilogue/O/canonical-QKV fallback, then step9 re-enters fused blocked mode.
            model.begin_stage_profile();
            let start = Instant::now();
            let out = if pass == 1 {
                model
                    .forward_outputs_b16_lm_fp8(&mut d, &seq, &mut ws)
                    .await?
            } else {
                model.forward_outputs_original_b16_lm(&mut d, &seq, &mut ws).await?
            };
            let model_ms = start.elapsed().as_secs_f64() * 1000.0;
            let found = collect_retained(
                &mut d,
                &mut selector,
                out,
                chosen.len(),
                if step == 2 { 2 } else { 0 },
            )
            .await?;
            for (i, values) in found.iter().enumerate() {
                if pass == 1 {
                    check(pass, step, chosen[i], &expected[step][i], values,chosen.len()==1||chosen.len()>16);
                    sampling_check(step, chosen[i], &expected[step][i], values,chosen.len()==1||chosen.len()>16);
                    checked += values.len();
                }
            }
            if pass == 0 {
                expected.push(found);
            }
            println!(
                "B16_LM_FP8_STEP pass={pass} native_lm={} step={step} live={} model_ms={model_ms:.3} accuracy=reported_per_row",
                pass == 1,
                chosen.len()
            );
            for stage in model.take_stage_profile() {
                println!(
                    "B16_LM_FP8_STAGE pass={pass} step={step} stage={} calls={} total_ms={:.6}",
                    stage.stage, stage.calls, stage.milliseconds
                );
            }
            for r in chosen {
                positions[r] += 1;
            }
            if step < 8 {
                for quantum in 0..6 {
                    if let Some(out) = model.advance_prefill(&mut d, &mut ws).await? {
                        assert_eq!((step, quantum), (7, 5));
                        let values = raw(&mut d, out).await?;
                        if pass == 0 {
                            prompt_expected = values;
                        } else {
                            check(pass, 99, 99, &prompt_expected, &values,true);
                            checked += values.len();
                        }
                    } else {
                        assert_ne!((step, quantum), (7, 5));
                    }
                }
            }
        }
        memory(&d,if pass==0 {"warm_control_complete"}else{"warm_candidate_complete"})?;
        model.set_b16_epilogue_fused(true);
        model.set_decode_blocked_output(true);
        model.set_b16_output_hsplit(true);
    }
    for owner in 0..17 {
        let values = model
            .compare_canonical_kv(
                &mut d,
                &ws,
                &reservations[0][owner],
                &reservations[1][owner],
            )
            .await?;
        println!("B16_LM_FP8_CACHE owner={owner} bf16_values={values} bitdiff=0");
    }
    let prompt_values = model
        .compare_canonical_kv(&mut d, &ws, &prompt_blocks[0], &prompt_blocks[1])
        .await?;
    println!("B16_LM_FP8_PROMPT_CACHE bf16_values={prompt_values} bitdiff=0");
    println!(
        "PASS B16 FP8 LM host isolation checked={checked} BF16logits;approximation reported for B16 widths2/8/9/14/16, scalar1/R32-17 and prompt exact;real512/1024 histories,physical remap,position0 reset,48 active-P128 quanta,narrow reuse,all17+prompt cache exact;device greedy matches each own logits;original-vs-FP8 sampling changes reported;elapsed_s={:.3};no task-quality or HTTP claim",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn sampling_check(step:usize,request:usize,a:&[u16],b:&[u16],strict:bool) {
    let a:Vec<f32>=a.iter().map(|&x|furiosa_opt_gemma4::host_logit_bytes::bf16_bits_to_f32(x)).collect();
    let b:Vec<f32>=b.iter().map(|&x|furiosa_opt_gemma4::host_logit_bytes::bf16_bits_to_f32(x)).collect();
    let empty=HashSet::new();let standard=[258882,258883].into_iter().collect();
    let winner=sample(&a,&SamplingConfig::DEFAULT,&empty,&mut StdRng::seed_from_u64(17));
    let masked=[winner,0,1,106,107,W::SIZE+1].into_iter().collect();
    for (policy,bans) in [&empty,&standard,&masked].into_iter().enumerate() {
        for config in [SamplingConfig::DEFAULT,SamplingConfig{temperature:0.8,top_k:37,top_p:0.9}] {
            let seed=step as u64*1009+request as u64;
            let x=sample(&a,&config,bans,&mut StdRng::seed_from_u64(seed));
            let y=sample(&b,&config,bans,&mut StdRng::seed_from_u64(seed));
            println!("B16_LM_FP8_SELECTION step={step} request={request} policy={policy} temperature={} control={x} candidate={y} same={} strict={strict}",config.temperature,x==y);
            if strict {assert_eq!(x,y,"unchanged route selection");}
        }
    }
}

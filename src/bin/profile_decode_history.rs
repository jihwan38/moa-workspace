//! Current serving decoder with independently computed real prompt histories.
//! Diagnostic compute/select costs only; these are not HTTP throughput scores.
use furiosa_opt_gemma4::{LAYERS, axes::*, host::{generate::read_logits, greedy::GreedySelector,
    load, model::{ForwardOutput, Input, Model, Sequence}, runtime::{Blocks, Workspace},
    sampling::{sample, SamplingConfig}}, ops};
use furiosa_opt_std::prelude::*;
use std::{collections::HashSet, time::Instant};

fn input(slot: usize, position: usize, logits: bool) -> Input<'static> {
    Input { token: 2 + (slot * 313 + position * 37) % 8000,
        soft: None, logits, image: None, audio: None }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let width: usize = args.get(1).map_or("64", String::as_str).parse()?;
    let context: usize = args.get(2).map_or("512", String::as_str).parse()?;
    let steps: usize = args.get(3).map_or("8", String::as_str).parse()?;
    assert!((1..=64).contains(&width) && context % 512 == 0 && context <= 1536);
    assert!((1..=64).contains(&steps));
    let pages = (context + steps).div_ceil(512);
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let mut ws = Workspace::new_canonical64(&mut device, &weights, 2048, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut ws).await?;
    let mut selector = GreedySelector::new(&mut device).await?;
    let blocks: Vec<_> = (0..width).map(|slot| {
        let ring = (slot * 13 + 31) % 64;
        Blocks { sliding: ring, full: (0..pages).map(|p| 255 - ring * pages - p).collect() }
    }).collect();
    let prepare = Instant::now();
    for slot in 0..width {
        for start in (0..context).step_by(512) {
            let rows: Vec<_> = (start..start + 512).map(|p| input(slot, p, false)).collect();
            let seq = Sequence { inputs: &rows, start_position: start, blocks: &blocks[slot] };
            model.start_prefill512(&mut device, &seq, &mut ws).await?;
            for layer in 0..LAYERS {
                let complete = model.advance_prefill(&mut device, &mut ws).await?.is_some();
                assert_eq!(complete, layer + 1 == LAYERS);
            }
        }
    }
    println!("HISTORY width={width} context={context} independently_computed_tokens={} prepare_ms={:.3}",
        width * context, prepare.elapsed().as_secs_f64() * 1000.0);
    let banned: HashSet<usize> = [258882, 258883].into_iter().collect();
    let bans: Vec<_> = (0..width).map(|_| Some(&banned)).collect();
    let mut rng = rand::rng();
    let mut tokens: Vec<_> = (0..width).map(|r| input(r, context, true).token).collect();
    for step in 0..steps {
        let rows: Vec<_> = tokens.iter().map(|&token| [Input { token, soft: None, logits: true,
            image: None, audio: None }]).collect();
        let sequences: Vec<_> = (0..width).map(|r| Sequence {
            inputs: &rows[r], start_position: context + step, blocks: &blocks[r] }).collect();
        model.begin_stage_profile();
        let begin = Instant::now();
        let output = model.forward_outputs(&mut device, &sequences, &mut ws).await?;
        let model_ms = begin.elapsed().as_secs_f64() * 1000.0;
        let select_begin = Instant::now();
        tokens = match output {
            ForwardOutput::Blocked64(output, live) => {
                assert_eq!(live, width);
                let selected = selector.select(&mut device, output, &bans).await?;
                let mut result = Vec::with_capacity(live);
                for (row, token) in selected.into_iter().enumerate() {
                    result.push(match token {
                        Some(t) => t,
                        None => sample(&selector.read_fallback(&mut device, output, row).await?,
                            &SamplingConfig::DEFAULT, &banned, &mut rng),
                    });
                }
                result
            }
            ForwardOutput::Rows(outputs) => {
                let mut result = Vec::with_capacity(outputs.len());
                for output in outputs {
                    result.push(sample(&read_logits(&mut device, output).await?,
                        &SamplingConfig::DEFAULT, &banned, &mut rng));
                }
                result
            }
        };
        let select_ms = select_begin.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(tokens.len(), width);
        assert!(tokens.iter().all(|&t| t < W::SIZE));
        println!("DECODE width={width} context={context} step={step} model_ms={model_ms:.3} select_ms={select_ms:.3} total_ms={:.3}", model_ms + select_ms);
        if step + 1 == steps {
            for stage in model.take_stage_profile() {
                println!("DECODE_STAGE width={width} context={context} name={} calls={} ms={:.6}", stage.stage, stage.calls, stage.milliseconds);
            }
        }
    }
    Ok(())
}

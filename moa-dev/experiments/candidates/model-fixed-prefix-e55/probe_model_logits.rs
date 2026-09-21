//! Fixed-prefix model diagnostic driven by the host: executes NPU kernels, with no sampling or accuracy acceptance threshold.
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use furiosa_opt_gemma4::axes::{E, W};
use furiosa_opt_gemma4::host::generate::read_logits;
use furiosa_opt_gemma4::host::load;
use furiosa_opt_gemma4::host::runtime::Workspace;
use furiosa_opt_gemma4::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SCHEMA: &str = "moa-fixed-prefix-logits-v1";
const PROMPT: &str = "Hi";
const VOCABULARY: usize = 262_144;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    schema: String,
    prompt: String,
    thinking: bool,
    add_generation_prompt: bool,
    element_count: usize,
    token_ids: Vec<usize>,
    logits: Vec<f32>,
}

enum Mode {
    Emit(PathBuf),
    Compare(PathBuf),
}

fn parse_args() -> Result<Mode> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next().ok_or("usage: probe_model_logits (--emit|--compare) reference.json")?;
    let path = PathBuf::from(args.next().ok_or("a reference path is required")?);
    if args.next().is_some() {
        return Err("unexpected extra arguments".into());
    }
    if mode == "--emit" {
        Ok(Mode::Emit(path))
    } else if mode == "--compare" {
        Ok(Mode::Compare(path))
    } else {
        Err("mode must be --emit or --compare".into())
    }
}

fn validate_logits(logits: &[f32], label: &str) -> Result<()> {
    if logits.len() != VOCABULARY {
        return Err(format!("{label}: {} elements, expected {VOCABULARY}", logits.len()).into());
    }
    if let Some((index, value)) = logits.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(format!("{label}: nonfinite logit at index {index}: {value}").into());
    }
    Ok(())
}

fn validate_reference(reference: &Reference) -> Result<()> {
    if reference.schema != SCHEMA || reference.prompt != PROMPT || reference.thinking
        || !reference.add_generation_prompt || reference.element_count != VOCABULARY
    {
        return Err("reference metadata does not match the fixed Hi/chat/thinking=false diagnostic".into());
    }
    validate_tokens(&reference.token_ids)?;
    validate_logits(&reference.logits, "reference")
}

fn validate_tokens(ids: &[usize]) -> Result<()> {
    if ids.is_empty() || ids.len() > E::SIZE || ids.iter().any(|&id| id >= VOCABULARY) {
        return Err("token IDs must be nonempty, within vocabulary and runtime context".into());
    }
    Ok(())
}

// Lowest token ID wins a tie. This only summarizes logits; no selected ID is decoded.
fn top_summary(logits: &[f32]) -> serde_json::Value {
    let mut top1 = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[top1] { top1 = index; }
    }
    let mut top2 = if top1 == 0 { 1 } else { 0 };
    for index in 0..logits.len() {
        if index != top1 && logits[index] > logits[top2] { top2 = index; }
    }
    json!({
        "top1_id": top1, "top1_logit": logits[top1],
        "top2_id": top2, "top2_logit": logits[top2],
        "top1_margin": f64::from(logits[top1]) - f64::from(logits[top2]),
        "top1_tie_count": logits.iter().filter(|&&v| v == logits[top1]).count(),
        "tie_rule": "lowest_token_id"
    })
}

fn compare_logits(reference: &[f32], actual: &[f32]) -> serde_json::Value {
    let thresholds = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0];
    let mut within = [0usize; 10];
    let mut bit_equal = 0usize;
    let mut sum_abs = 0.0f64;
    let mut sum_squared = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_index = 0usize;
    for (index, (&expected, &observed)) in reference.iter().zip(actual).enumerate() {
        let difference = f64::from(observed) - f64::from(expected);
        let error = difference.abs();
        sum_abs += error;
        sum_squared += difference * difference;
        if error > max_abs { max_abs = error; max_index = index; }
        bit_equal += usize::from(expected.to_bits() == observed.to_bits());
        for (count, &threshold) in within.iter_mut().zip(&thresholds) {
            *count += usize::from(error <= threshold);
        }
    }
    let count = reference.len();
    let counts: Vec<_> = thresholds.iter().zip(within).map(|(&threshold, within)| {
        json!({"absolute_error_at_most": threshold, "count": within, "count_above": count - within})
    }).collect();
    json!({
        "elements_compared": count, "max_abs": max_abs, "mean_abs": sum_abs / count as f64,
        "rmse": (sum_squared / count as f64).sqrt(), "bit_equal_count": bit_equal,
        "max_abs_index": max_index, "reference_at_max_abs": reference[max_index],
        "actual_at_max_abs": actual[max_index],
        "reference_top": top_summary(reference), "actual_top": top_summary(actual),
        "diagnostic_absolute_error_counts": counts,
        "accuracy_acceptance_threshold": null, "accuracy_pass": null
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = parse_args()?;
    if W::SIZE != VOCABULARY { return Err("compiled vocabulary is not 262144".into()); }
    // Reject invalid references before touching the device. Never derive a prefix
    // from a previous argmax, and never compare a truncated zip of unequal arrays.
    let reference = match &mode {
        Mode::Compare(path) => {
            let reference: Reference = serde_json::from_reader(BufReader::new(File::open(path)?))?;
            validate_reference(&reference)?;
            Some(reference)
        }
        Mode::Emit(path) => {
            if path.exists() { return Err("refusing to overwrite an existing reference file".into()); }
            None
        }
    };
    let started = Instant::now();
    eprintln!("MODEL_FIXED_PREFIX_PROGRESS {}", json!({"phase": "before_device_new", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    let mut device = Device::new(furiosa_opt_gemma4::ops::embed_token.topology())?;
    eprintln!("MODEL_FIXED_PREFIX_PROGRESS {}", json!({"phase": "after_device_new", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    eprintln!("MODEL_FIXED_PREFIX_PROGRESS {}", json!({"phase": "before_load_model", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    let model = load::load_model(&mut device).await?;
    eprintln!("MODEL_FIXED_PREFIX_PROGRESS {}", json!({"phase": "after_load_model", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    let load_seconds = started.elapsed().as_secs_f64();
    let tokenizer = Tokenizer::new(&load::tokenizer_path(&model))?;
    let messages = vec![json!({"role": "user", "content": PROMPT})];
    let token_ids = tokenizer.encode_chat(&messages, false)?;
    validate_tokens(&token_ids)?;
    if let Some(reference) = &reference {
        if token_ids != reference.token_ids {
            return Err(format!("fixed-prefix token IDs differ: reference {:?}, actual {:?}", reference.token_ids, token_ids).into());
        }
    }
    eprintln!("MODEL_FIXED_PREFIX_READY tokens={} load_seconds={load_seconds:.6}", token_ids.len());
    let workspace_started = Instant::now();
    let mut workspace = Workspace::new(&mut device, &model).await?;
    let mut decode = workspace.begin_decode(&mut device).await?;
    let workspace_seconds = workspace_started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let last = token_ids.len() - 1;
    for (pos, &token) in token_ids.iter().enumerate() {
        decode.run(&mut device, &mut workspace, token, &model, pos, pos == last).await?;
    }
    let logits = read_logits(&mut device, &workspace.logits).await?;
    let decode_and_read_seconds = decode_started.elapsed().as_secs_f64();
    validate_logits(&logits, "actual")?;
    let timings = json!({"load_seconds": load_seconds, "workspace_seconds": workspace_seconds,
        "decode_and_read_seconds": decode_and_read_seconds, "total_seconds": started.elapsed().as_secs_f64()});
    match mode {
        Mode::Emit(path) => {
            let output = Reference { schema: SCHEMA.to_owned(), prompt: PROMPT.to_owned(), thinking: false,
                add_generation_prompt: true, element_count: VOCABULARY, token_ids: token_ids.clone(), logits };
            let file = OpenOptions::new().write(true).create_new(true).open(path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &output)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            println!("{}", json!({"event": "reference_written", "schema": SCHEMA,
                "token_ids": token_ids, "element_count": VOCABULARY, "nonfinite_count": 0,
                "top": top_summary(&output.logits), "timings": timings,
                "accuracy_acceptance_threshold": null, "accuracy_pass": null}));
        }
        Mode::Compare(_) => {
            let reference = reference.as_ref().ok_or("missing loaded reference")?;
            println!("{}", json!({"event": "logits_compared", "schema": SCHEMA,
                "token_ids": token_ids, "reference_token_ids_match": true,
                "element_count": VOCABULARY, "nonfinite_count": 0,
                "comparison": compare_logits(&reference.logits, &logits), "timings": timings}));
        }
    }
    Ok(())
}

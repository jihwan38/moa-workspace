//! Host-driven four-prefix NPU diagnostic; no sampling or accuracy acceptance threshold.
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
const SCHEMA: &str = "moa-model-prefix-suite-v1";
const VOCABULARY: usize = 262_144;
const MAX_TOTAL_TOKENS: usize = 128;
const PROMPTS: [&str; 4] = [
    "Hi",
    "What is 2 + 2?",
    "대한민국의 수도는?",
    "Complete the Python expression: len([1, 2, 3])",
];

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceCase {
    case_index: usize,
    prompt: String,
    token_ids: Vec<usize>,
    logits: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceSuite {
    schema: String,
    thinking: bool,
    add_generation_prompt: bool,
    case_count: usize,
    element_count_per_case: usize,
    max_total_tokens: usize,
    total_token_count: usize,
    cases: Vec<ReferenceCase>,
}

enum Mode {
    Emit(PathBuf),
    Compare(PathBuf),
}

fn parse_args() -> Result<Mode> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next().ok_or("usage: probe_model_prefix_suite (--emit|--compare) reference.json")?;
    let path = PathBuf::from(args.next().ok_or("a reference path is required")?);
    if args.next().is_some() { return Err("unexpected extra arguments".into()); }
    if mode == "--emit" { Ok(Mode::Emit(path)) }
    else if mode == "--compare" { Ok(Mode::Compare(path)) }
    else { Err("mode must be --emit or --compare".into()) }
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

fn validate_tokens(ids: &[usize]) -> Result<()> {
    if ids.is_empty() || ids.len() > E::SIZE || ids.iter().any(|&id| id >= VOCABULARY) {
        return Err("token IDs must be nonempty, within vocabulary and runtime context".into());
    }
    Ok(())
}

fn validate_total_tokens(lengths: impl IntoIterator<Item = usize>) -> Result<usize> {
    let mut total = 0usize;
    for length in lengths {
        total = total.checked_add(length).ok_or("total token count overflow")?;
    }
    if total > MAX_TOTAL_TOKENS {
        return Err(format!("suite has {total} prompt tokens; limit is {MAX_TOTAL_TOKENS}; refusing model execution").into());
    }
    Ok(total)
}

fn validate_reference(reference: &ReferenceSuite) -> Result<()> {
    if reference.schema != SCHEMA || reference.thinking || !reference.add_generation_prompt
        || reference.case_count != PROMPTS.len() || reference.cases.len() != PROMPTS.len()
        || reference.element_count_per_case != VOCABULARY || reference.max_total_tokens != MAX_TOTAL_TOKENS
    {
        return Err("reference metadata does not match the fixed four-prefix suite".into());
    }
    for (index, (case, prompt)) in reference.cases.iter().zip(PROMPTS).enumerate() {
        if case.case_index != index || case.prompt != prompt {
            return Err(format!("reference case {index}: index/prompt/order mismatch").into());
        }
        validate_tokens(&case.token_ids)?;
        validate_logits(&case.logits, &format!("reference case {index}"))?;
    }
    let total = validate_total_tokens(reference.cases.iter().map(|case| case.token_ids.len()))?;
    if total != reference.total_token_count {
        return Err("reference total token count disagrees with its case IDs".into());
    }
    Ok(())
}

fn validate_encoded_tokens(encoded: &[Vec<usize>], reference: Option<&ReferenceSuite>) -> Result<usize> {
    if encoded.len() != PROMPTS.len() { return Err("encoded suite case count mismatch".into()); }
    for (index, ids) in encoded.iter().enumerate() {
        validate_tokens(ids)?;
        if let Some(reference) = reference {
            if reference.cases[index].token_ids != *ids {
                return Err(format!("case {index}: fixed-prefix token IDs differ from reference").into());
            }
        }
    }
    validate_total_tokens(encoded.iter().map(Vec::len))
}

// Lowest token ID wins a tie. Raw logits only: no suppression, decoding or sampling.
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

fn print_summary(value: serde_json::Value) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, &value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = parse_args()?;
    if W::SIZE != VOCABULARY { return Err("compiled vocabulary is not 262144".into()); }
    let reference = match &mode {
        Mode::Compare(path) => {
            let reference: ReferenceSuite = serde_json::from_reader(BufReader::new(File::open(path)?))?;
            validate_reference(&reference)?;
            Some(reference)
        }
        Mode::Emit(path) => {
            if path.exists() { return Err("refusing to overwrite an existing reference file".into()); }
            None
        }
    };
    let started = Instant::now();
    // Public tokenizer APIs allow all guards before Device::new / model loading.
    let tokenizer_path = load::tokenizer_path_in(&load::model_dir()?);
    let tokenizer = Tokenizer::new(&tokenizer_path)?;
    let encoded: Vec<Vec<usize>> = PROMPTS.iter().map(|prompt| {
        tokenizer.encode_chat(&[json!({"role": "user", "content": prompt})], false)
    }).collect::<Result<_>>()?;
    let total_tokens = validate_encoded_tokens(&encoded, reference.as_ref())?;
    print_summary(json!({"event": "suite_inputs_validated", "schema": SCHEMA,
        "case_count": PROMPTS.len(), "prompts": PROMPTS, "token_ids": encoded,
        "total_token_count": total_tokens, "max_total_tokens": MAX_TOTAL_TOKENS,
        "guard_before_device_new": true, "accuracy_acceptance_threshold": null, "accuracy_pass": null}))?;
    let load_started = Instant::now();
    eprintln!("MODEL_PREFIX_SUITE_PROGRESS {}", json!({"phase": "before_device_new", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    let mut device = Device::new(furiosa_opt_gemma4::ops::embed_token.topology())?;
    eprintln!("MODEL_PREFIX_SUITE_PROGRESS {}", json!({"phase": "after_device_new", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    eprintln!("MODEL_PREFIX_SUITE_PROGRESS {}", json!({"phase": "before_load_model", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    let model = load::load_model(&mut device).await?;
    eprintln!("MODEL_PREFIX_SUITE_PROGRESS {}", json!({"phase": "after_load_model", "elapsed_seconds": started.elapsed().as_secs_f64()}));
    if load::tokenizer_path(&model) != tokenizer_path {
        return Err("loader selected a different model directory after token validation".into());
    }
    let load_seconds = load_started.elapsed().as_secs_f64();
    let mut emitted = Vec::with_capacity(PROMPTS.len());
    for (index, (prompt, token_ids)) in PROMPTS.iter().zip(&encoded).enumerate() {
        eprintln!("MODEL_PREFIX_SUITE_PROGRESS {}", json!({"phase": "before_case", "case_index": index,
            "token_count": token_ids.len(), "elapsed_seconds": started.elapsed().as_secs_f64()}));
        // This scope owns a fresh zero-initialized KV cache and decode buffers.
        // Both are dropped before the next case; there is no prefix/cache reuse.
        let (logits, workspace_seconds, decode_and_read_seconds) = {
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
            (logits, workspace_seconds, decode_started.elapsed().as_secs_f64())
        };
        validate_logits(&logits, &format!("actual case {index}"))?;
        let timings = json!({"workspace_seconds": workspace_seconds,
            "decode_and_read_seconds": decode_and_read_seconds,
            "elapsed_suite_seconds": started.elapsed().as_secs_f64()});
        if let Some(reference) = &reference {
            print_summary(json!({"event": "case_logits_compared", "schema": SCHEMA, "case_index": index,
                "prompt": prompt, "token_ids": token_ids, "reference_token_ids_match": true,
                "fresh_workspace": true, "element_count": VOCABULARY, "nonfinite_count": 0,
                "comparison": compare_logits(&reference.cases[index].logits, &logits), "timings": timings}))?;
        } else {
            print_summary(json!({"event": "reference_case_ready", "schema": SCHEMA, "case_index": index,
                "prompt": prompt, "token_ids": token_ids, "fresh_workspace": true,
                "element_count": VOCABULARY, "nonfinite_count": 0,
                "top": top_summary(&logits), "timings": timings,
                "accuracy_acceptance_threshold": null, "accuracy_pass": null}))?;
            emitted.push(ReferenceCase { case_index: index, prompt: (*prompt).to_owned(), token_ids: token_ids.clone(), logits });
        }
    }
    if let Mode::Emit(path) = mode {
        let output = ReferenceSuite { schema: SCHEMA.to_owned(), thinking: false, add_generation_prompt: true,
            case_count: PROMPTS.len(), element_count_per_case: VOCABULARY,
            max_total_tokens: MAX_TOTAL_TOKENS, total_token_count: total_tokens, cases: emitted };
        validate_reference(&output)?;
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &output)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    print_summary(json!({"event": if reference.is_some() { "suite_comparison_complete" } else { "suite_reference_written" },
        "schema": SCHEMA, "completed_cases": PROMPTS.len(), "total_token_count": total_tokens,
        "elements_per_case": VOCABULARY, "total_elements": PROMPTS.len() * VOCABULARY,
        "load_seconds": load_seconds, "total_seconds": started.elapsed().as_secs_f64(),
        "accuracy_acceptance_threshold": null, "accuracy_pass": null}))?;
    Ok(())
}

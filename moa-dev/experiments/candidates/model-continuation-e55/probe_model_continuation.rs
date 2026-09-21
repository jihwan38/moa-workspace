//! Host-driven NPU diagnostic: replay original greedy tokens, compare every full logits array.
//! This is not independent free generation or an accuracy acceptance test.
use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use furiosa_opt_gemma4::axes::{E, W};
use furiosa_opt_gemma4::host::generate::{read_logits, EOA_TOKEN, EOI_TOKEN, EOS_TOKENS};
use furiosa_opt_gemma4::host::load;
use furiosa_opt_gemma4::host::runtime::Workspace;
use furiosa_opt_gemma4::host::sampling::{sample, SamplingConfig};
use furiosa_opt_gemma4::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SCHEMA: &str = "moa-model-continuation-v1";
const VOCABULARY: usize = 262_144;
const MAX_NEW_TOKENS: usize = 8;
const MAX_TOTAL_TOKENS: usize = 128;
const BANNED_TOKENS: [usize; 2] = [258882, 258883];
const EXPECTED_EOS: [usize; 3] = [1, 106, 50];
const REPLAY_POLICY: &str = "reference_tokens_even_if_candidate_diverges_or_stops";
const TERMINATION_POLICY: &str = "reference_original_eos_or_max_new_tokens";
const PROMPTS: [&str; 4] = [
    "Hi",
    "What is 2 + 2?",
    "대한민국의 수도는?",
    "Complete the Python expression: len([1, 2, 3])",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SamplerSettings { temperature: f32, top_k: usize, top_p: f32 }

fn sampler_settings() -> SamplerSettings {
    SamplerSettings { temperature: 1.0, top_k: 1, top_p: 1.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FinishReason { Eos, Length }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceCase {
    case_index: usize,
    prompt: String,
    prompt_token_ids: Vec<usize>,
    checkpoint_count: usize,
    completion_tokens: usize,
    forward_tokens: usize,
    finish_reason: FinishReason,
    reference_selected_tokens: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    thinking: bool,
    add_generation_prompt: bool,
    case_count: usize,
    element_count_per_checkpoint: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
    sampler: SamplerSettings,
    banned_tokens: [usize; 2],
    eos_tokens: [usize; 3],
    termination_policy: String,
    total_prompt_tokens: usize,
    guarded_total_tokens: usize,
    total_checkpoints: usize,
    total_forward_tokens: usize,
    total_elements: usize,
    accuracy_acceptance_threshold: Option<f64>,
    accuracy_pass: Option<bool>,
    cases: Vec<ReferenceCase>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    case_index: usize,
    step: usize,
    position: usize,
    selected_token: usize,
    selected_is_eos: bool,
    element_count: usize,
    logits: Vec<f32>,
}

struct LoadedReference { manifest: Manifest, checkpoints: Vec<Vec<Checkpoint>> }

#[derive(Debug, Serialize)]
struct LegalChoice {
    selected_id: usize,
    selected_logit: f32,
    selected_logit_bits: u32,
    runner_up_id: usize,
    runner_up_logit: f32,
    runner_up_logit_bits: u32,
    margin: f64,
    tie_count: usize,
    rank_order: &'static str,
    tie_rule: &'static str,
}

enum Mode { Emit(PathBuf), Compare(PathBuf) }

impl Mode {
    fn name(&self) -> &'static str {
        match self { Self::Emit(_) => "emit", Self::Compare(_) => "compare" }
    }
    fn directory(&self) -> &Path {
        match self { Self::Emit(path) | Self::Compare(path) => path }
    }
}

fn parse_args() -> Result<Mode> {
    let mut args = std::env::args_os().skip(1);
    let mode = args.next().ok_or("usage: probe_model_continuation (--emit|--compare) DIR")?;
    let path = PathBuf::from(args.next().ok_or("reference directory required")?);
    if args.next().is_some() { return Err("unexpected extra arguments".into()); }
    if mode == "--emit" { Ok(Mode::Emit(path)) }
    else if mode == "--compare" { Ok(Mode::Compare(path)) }
    else { Err("mode must be --emit or --compare".into()) }
}

fn checkpoint_filename(case: usize, step: usize) -> String {
    format!("case-{case}-step-{step}.json")
}

fn is_eos(token: usize) -> bool { EXPECTED_EOS.contains(&token) }
fn is_banned(token: usize) -> bool { BANNED_TOKENS.contains(&token) }

fn validate_logits(logits: &[f32], label: &str) -> Result<()> {
    if logits.len() != VOCABULARY {
        return Err(format!("{label}: {} elements; expected {VOCABULARY}", logits.len()).into());
    }
    if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(format!("{label}: nonfinite logit at {index}").into());
    }
    Ok(())
}

fn validate_tokens(ids: &[usize]) -> Result<()> {
    if ids.is_empty() || ids.iter().any(|&token| token >= VOCABULARY) {
        return Err("prompt token IDs must be nonempty and inside vocabulary".into());
    }
    if ids.len().checked_add(MAX_NEW_TOKENS).ok_or("context count overflow")? > E::SIZE {
        return Err("prompt plus generation exceeds context".into());
    }
    Ok(())
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value).ok_or_else(|| "counter overflow".into())
    })
}

fn token_budget(lengths: impl IntoIterator<Item = usize>) -> Result<(usize, usize)> {
    let mut prompt_total = 0usize;
    let mut guarded_total = 0usize;
    for length in lengths {
        if length == 0 { return Err("empty prompt".into()); }
        let guarded = length.checked_add(MAX_NEW_TOKENS).ok_or("token count overflow")?;
        if guarded > E::SIZE { return Err("prompt plus generation exceeds context".into()); }
        prompt_total = prompt_total.checked_add(length).ok_or("token count overflow")?;
        guarded_total = guarded_total.checked_add(guarded).ok_or("token count overflow")?;
    }
    if guarded_total > MAX_TOTAL_TOKENS {
        return Err(format!("guarded token budget {guarded_total} exceeds {MAX_TOTAL_TOKENS}").into());
    }
    Ok((prompt_total, guarded_total))
}

// Numeric greedy is defined by the actual public sampler's f32::total_cmp order.
// Do not impose the diagnostic raw-top helper's lowest-ID tie convention here.
fn legal_choice(logits: &[f32], selected: usize) -> Result<LegalChoice> {
    validate_logits(logits, "legal choice")?;
    if selected >= VOCABULARY || is_banned(selected) {
        return Err("selected token is outside vocabulary or suppressed".into());
    }
    let mut runner_up: Option<usize> = None;
    let mut tie_count = 0usize;
    for (index, &logit) in logits.iter().enumerate() {
        if is_banned(index) { continue; }
        match logit.total_cmp(&logits[selected]) {
            Ordering::Greater => return Err("selected token is not a legal greedy maximum".into()),
            Ordering::Equal => tie_count += 1,
            Ordering::Less => (),
        }
        if index != selected && runner_up.is_none_or(|old| logit.total_cmp(&logits[old]).is_gt()) {
            runner_up = Some(index);
        }
    }
    let runner_up = runner_up.ok_or("fewer than two legal vocabulary entries")?;
    Ok(LegalChoice {
        selected_id: selected, selected_logit: logits[selected], selected_logit_bits: logits[selected].to_bits(),
        runner_up_id: runner_up, runner_up_logit: logits[runner_up], runner_up_logit_bits: logits[runner_up].to_bits(),
        margin: f64::from(logits[selected]) - f64::from(logits[runner_up]), tie_count,
        rank_order: "f32_total_cmp", tie_rule: "public_sample_unstable_order",
    })
}

fn greedy_choice(logits: &[f32]) -> Result<LegalChoice> {
    validate_logits(logits, "sampler input")?;
    let config = SamplingConfig { temperature: 1.0, top_k: 1, top_p: 1.0 };
    let banned: HashSet<usize> = BANNED_TOKENS.into_iter().collect();
    let selected = sample(logits, &config, &banned, &mut rand::rng());
    legal_choice(logits, selected)
}

fn reference_choice_candidate_rank(actual: &[f32], reference_token: usize) -> usize {
    1 + actual.iter().enumerate().filter(|&(index, value)| {
        !is_banned(index) && value.total_cmp(&actual[reference_token]).is_gt()
    }).count()
}

fn case_counts(prompt_len: usize, selected: &[usize]) -> Result<(usize, usize, FinishReason)> {
    if selected.is_empty() || selected.len() > MAX_NEW_TOKENS {
        return Err("checkpoint count must be 1..=8".into());
    }
    if selected.iter().any(|&token| token >= VOCABULARY || is_banned(token)) {
        return Err("invalid or suppressed reference selected token".into());
    }
    if selected[..selected.len() - 1].iter().copied().any(is_eos) {
        return Err("reference continues after an EOS choice".into());
    }
    let eos = is_eos(*selected.last().ok_or("no selected token")?);
    if !eos && selected.len() != MAX_NEW_TOKENS {
        return Err("non-EOS reference must have all eight checkpoints".into());
    }
    let completion = selected.len() - usize::from(eos);
    let forwards = prompt_len.checked_add(selected.len() - 1).ok_or("forward count overflow")?;
    Ok((completion, forwards, if eos { FinishReason::Eos } else { FinishReason::Length }))
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema != SCHEMA || manifest.thinking || !manifest.add_generation_prompt
        || manifest.case_count != PROMPTS.len() || manifest.cases.len() != PROMPTS.len()
        || manifest.element_count_per_checkpoint != VOCABULARY
        || manifest.max_new_tokens != MAX_NEW_TOKENS || manifest.max_total_tokens != MAX_TOTAL_TOKENS
        || manifest.sampler != sampler_settings() || manifest.banned_tokens != BANNED_TOKENS
        || manifest.eos_tokens != EXPECTED_EOS || manifest.termination_policy != TERMINATION_POLICY
        || manifest.accuracy_acceptance_threshold.is_some() || manifest.accuracy_pass.is_some()
    { return Err("reference manifest metadata mismatch".into()); }
    for (index, case) in manifest.cases.iter().enumerate() {
        if case.case_index != index || case.prompt != PROMPTS[index] {
            return Err("reference case index, prompt or order mismatch".into());
        }
        validate_tokens(&case.prompt_token_ids)?;
        let (completion, forwards, finish) = case_counts(case.prompt_token_ids.len(), &case.reference_selected_tokens)?;
        if case.checkpoint_count != case.reference_selected_tokens.len() || case.completion_tokens != completion
            || case.forward_tokens != forwards || case.finish_reason != finish {
            return Err("reference case counters or termination mismatch".into());
        }
    }
    let (prompt_total, guarded) = token_budget(manifest.cases.iter().map(|case| case.prompt_token_ids.len()))?;
    let checkpoint_total = checked_sum(manifest.cases.iter().map(|case| case.checkpoint_count))?;
    let forward_total = checked_sum(manifest.cases.iter().map(|case| case.forward_tokens))?;
    if manifest.total_prompt_tokens != prompt_total || manifest.guarded_total_tokens != guarded
        || manifest.total_checkpoints != checkpoint_total || manifest.total_forward_tokens != forward_total
        || manifest.total_elements != checkpoint_total.checked_mul(VOCABULARY).ok_or("element count overflow")? {
        return Err("reference manifest totals mismatch".into());
    }
    Ok(())
}

fn validate_checkpoint(checkpoint: &Checkpoint, case: &ReferenceCase, step: usize) -> Result<()> {
    let selected = *case.reference_selected_tokens.get(step).ok_or("unexpected checkpoint")?;
    let position = case.prompt_token_ids.len().checked_sub(1).and_then(|last| last.checked_add(step))
        .ok_or("invalid checkpoint position")?;
    if checkpoint.schema != SCHEMA || checkpoint.case_index != case.case_index || checkpoint.step != step
        || checkpoint.position != position || checkpoint.selected_token != selected
        || checkpoint.selected_is_eos != is_eos(selected) || checkpoint.element_count != VOCABULARY {
        return Err("reference checkpoint metadata mismatch".into());
    }
    legal_choice(&checkpoint.logits, selected)?;
    Ok(())
}

fn read_reference(directory: &Path) -> Result<LoadedReference> {
    if !fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err("reference path must be a real directory".into());
    }
    let manifest_path = directory.join("manifest.json");
    if !fs::symlink_metadata(&manifest_path)?.file_type().is_file() {
        return Err("manifest must be a regular file".into());
    }
    let manifest: Manifest = serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    validate_manifest(&manifest)?;
    let mut expected_files: HashSet<String> = HashSet::from(["manifest.json".to_owned()]);
    for case in &manifest.cases {
        for step in 0..case.checkpoint_count { expected_files.insert(checkpoint_filename(case.case_index, step)); }
    }
    let mut actual_files = HashSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() { return Err("reference contains a nonregular entry".into()); }
        let name = entry.file_name().into_string().map_err(|_| "non-UTF8 reference filename")?;
        actual_files.insert(name);
    }
    if actual_files != expected_files { return Err("reference checkpoint file set mismatch".into()); }
    let mut checkpoints = Vec::with_capacity(PROMPTS.len());
    for case in &manifest.cases {
        let mut loaded = Vec::with_capacity(case.checkpoint_count);
        for step in 0..case.checkpoint_count {
            let file = File::open(directory.join(checkpoint_filename(case.case_index, step)))?;
            let checkpoint: Checkpoint = serde_json::from_reader(BufReader::new(file))?;
            validate_checkpoint(&checkpoint, case, step)?;
            loaded.push(checkpoint);
        }
        checkpoints.push(loaded);
    }
    Ok(LoadedReference { manifest, checkpoints })
}

fn validate_encoded_tokens(encoded: &[Vec<usize>], reference: Option<&LoadedReference>) -> Result<(usize, usize)> {
    if encoded.len() != PROMPTS.len() { return Err("encoded case count mismatch".into()); }
    for (index, ids) in encoded.iter().enumerate() {
        validate_tokens(ids)?;
        if reference.is_some_and(|reference| reference.manifest.cases[index].prompt_token_ids != *ids) {
            return Err(format!("case {index}: encoded prompt IDs differ from reference").into());
        }
    }
    token_budget(encoded.iter().map(Vec::len))
}

// Raw full-vocabulary diagnostic only. Its tie convention is not sampler semantics.
fn top_summary(logits: &[f32]) -> Value {
    let mut top1 = 0;
    for index in 1..logits.len() { if logits[index] > logits[top1] { top1 = index; } }
    let mut top2 = if top1 == 0 { 1 } else { 0 };
    for index in 0..logits.len() {
        if index != top1 && logits[index] > logits[top2] { top2 = index; }
    }
    json!({"top1_id": top1, "top1_logit": logits[top1], "top2_id": top2, "top2_logit": logits[top2],
        "top1_margin": f64::from(logits[top1]) - f64::from(logits[top2]),
        "top1_tie_count": logits.iter().filter(|&&value| value == logits[top1]).count(),
        "tie_rule": "lowest_token_id"})
}

fn compare_logits(reference: &[f32], actual: &[f32]) -> Value {
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
        for (count, &threshold) in within.iter_mut().zip(&thresholds) { *count += usize::from(error <= threshold); }
    }
    let count = reference.len();
    let counts: Vec<_> = thresholds.iter().zip(within).map(|(&threshold, within)| {
        json!({"absolute_error_at_most": threshold, "count": within, "count_above": count - within})
    }).collect();
    json!({"elements_compared": count, "max_abs": max_abs, "mean_abs": sum_abs / count as f64,
        "rmse": (sum_squared / count as f64).sqrt(), "bit_equal_count": bit_equal,
        "max_abs_index": max_index, "reference_at_max_abs": reference[max_index], "actual_at_max_abs": actual[max_index],
        "reference_top": top_summary(reference), "actual_top": top_summary(actual),
        "diagnostic_absolute_error_counts": counts, "accuracy_acceptance_threshold": null, "accuracy_pass": null})
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn print_event(mode: &Mode, event: &str, mut value: Value) -> Result<()> {
    let object = value.as_object_mut().ok_or("event payload must be an object")?;
    object.insert("event".to_owned(), json!(event));
    object.insert("schema".to_owned(), json!(SCHEMA));
    object.insert("mode".to_owned(), json!(mode.name()));
    object.insert("accuracy_acceptance_threshold".to_owned(), Value::Null);
    object.insert("accuracy_pass".to_owned(), Value::Null);
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, &value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn progress(phase: &str, started: &Instant) {
    eprintln!("MODEL_CONTINUATION_PROGRESS {}", json!({"phase": phase, "elapsed_seconds": started.elapsed().as_secs_f64()}));
}

#[tokio::main]
async fn main() -> Result<()> {
    // Includes reference parsing/validation in both the overall clock and a separate duration.
    let started = Instant::now();
    let mode = parse_args()?;
    if W::SIZE != VOCABULARY || [EOI_TOKEN, EOA_TOKEN] != BANNED_TOKENS || EOS_TOKENS != EXPECTED_EOS {
        return Err("compiled vocabulary/suppression/EOS contract changed".into());
    }
    let parse_started = Instant::now();
    let reference = match &mode {
        Mode::Compare(directory) => Some(read_reference(directory)?),
        Mode::Emit(directory) => { fs::create_dir(directory)?; None }
    };
    let reference_parse_seconds = parse_started.elapsed().as_secs_f64();
    let tokenizer_path = load::tokenizer_path_in(&load::model_dir()?);
    let tokenizer = Tokenizer::new(&tokenizer_path)?;
    let encoded: Vec<Vec<usize>> = PROMPTS.iter().map(|prompt| {
        tokenizer.encode_chat(&[json!({"role": "user", "content": prompt})], false)
    }).collect::<Result<_>>()?;
    let (total_prompt_tokens, guarded_total_tokens) = validate_encoded_tokens(&encoded, reference.as_ref())?;
    print_event(&mode, "continuation_inputs_validated", json!({
        "case_count": PROMPTS.len(), "prompts": PROMPTS, "token_ids": encoded,
        "thinking": false, "add_generation_prompt": true,
        "sampler": sampler_settings(), "banned_tokens": BANNED_TOKENS, "eos_tokens": EXPECTED_EOS,
        "termination_policy": TERMINATION_POLICY, "replay_policy": REPLAY_POLICY,
        "total_prompt_tokens": total_prompt_tokens, "guarded_total_tokens": guarded_total_tokens,
        "max_total_tokens": MAX_TOTAL_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
        "guard_before_device_new": true, "reference_all_checkpoints_validated": reference.is_some(),
        "reference_parse_seconds": reference_parse_seconds, "elapsed_seconds": started.elapsed().as_secs_f64()
    }))?;

    let load_started = Instant::now();
    progress("before_device_new", &started);
    let mut device = Device::new(furiosa_opt_gemma4::ops::embed_token.topology())?;
    progress("after_device_new", &started);
    progress("before_load_model", &started);
    let model = load::load_model(&mut device).await?;
    progress("after_load_model", &started);
    if load::tokenizer_path(&model) != tokenizer_path {
        return Err("loader model directory changed after token validation".into());
    }
    let load_seconds = load_started.elapsed().as_secs_f64();
    let mut realized_cases = Vec::with_capacity(PROMPTS.len());
    let mut first_disagreements = Vec::with_capacity(PROMPTS.len());

    for (case_index, prompt_token_ids) in encoded.iter().enumerate() {
        eprintln!("MODEL_CONTINUATION_PROGRESS {}", json!({"phase": "before_case", "case_index": case_index,
            "prompt_tokens": prompt_token_ids.len(), "elapsed_seconds": started.elapsed().as_secs_f64()}));
        let case_started = Instant::now();
        let mut selected_tokens = Vec::with_capacity(MAX_NEW_TOKENS);
        let mut consumed_token_ids = prompt_token_ids.clone();
        let mut first_disagreement: Option<usize> = None;
        let workspace_seconds;
        let prefill_seconds;
        // Fresh cache per case; the same cache and Decode survive all of this case's checkpoints.
        {
            let workspace_started = Instant::now();
            let mut workspace = Workspace::new(&mut device, &model).await?;
            let mut decode = workspace.begin_decode(&mut device).await?;
            workspace_seconds = workspace_started.elapsed().as_secs_f64();
            let prefill_started = Instant::now();
            let last_prompt_position = prompt_token_ids.len() - 1;
            for (position, &token) in prompt_token_ids.iter().enumerate() {
                decode.run(&mut device, &mut workspace, token, &model, position, position == last_prompt_position).await?;
            }
            prefill_seconds = prefill_started.elapsed().as_secs_f64();
            for step in 0..MAX_NEW_TOKENS {
                let checkpoint_started = Instant::now();
                // step > 0 feeds the preceding ORIGINAL choice, even when candidate would stop/diverge.
                if step > 0 {
                    let previous = selected_tokens[step - 1];
                    if is_eos(previous) { return Err("internal error: trying to feed EOS".into()); }
                    let position = prompt_token_ids.len() + step - 1;
                    decode.run(&mut device, &mut workspace, previous, &model, position, true).await?;
                    consumed_token_ids.push(previous);
                }
                let logits = read_logits(&mut device, &workspace.logits).await?;
                let decode_and_read_seconds = checkpoint_started.elapsed().as_secs_f64();
                validate_logits(&logits, "current checkpoint")?;
                let actual_choice = greedy_choice(&logits)?;
                let position = last_prompt_position + step;
                let file_name = checkpoint_filename(case_index, step);
                let selected;
                if let Some(reference) = &reference {
                    let original = reference.checkpoints[case_index].get(step).ok_or("reference ended before replay")?;
                    selected = original.selected_token;
                    let reference_choice = legal_choice(&original.logits, selected)?;
                    if selected != actual_choice.selected_id && first_disagreement.is_none() { first_disagreement = Some(step); }
                    let comparison = compare_logits(&original.logits, &logits);
                    print_event(&mode, "checkpoint_logits_compared", json!({
                        "case_index": case_index, "step": step, "position": position,
                        "checkpoint_file": file_name, "consumed_token_ids": consumed_token_ids,
                        "element_count": VOCABULARY, "nonfinite_count": 0, "fresh_workspace_for_case": true,
                        "reference_selected_token": selected, "reference_selected_is_eos": is_eos(selected),
                        "reference_legal_choice": reference_choice,
                        "candidate_selected_token": actual_choice.selected_id,
                        "candidate_selected_is_eos": is_eos(actual_choice.selected_id),
                        "candidate_legal_choice": actual_choice,
                        "selected_token_match": selected == actual_choice.selected_id,
                        "reference_choice_candidate_rank": reference_choice_candidate_rank(&logits, selected),
                        "first_legal_greedy_disagreement_step": first_disagreement,
                        "replay_policy": REPLAY_POLICY, "comparison": comparison,
                        "timings": {"decode_and_read_seconds": decode_and_read_seconds,
                            "checkpoint_seconds": checkpoint_started.elapsed().as_secs_f64(),
                            "elapsed_suite_seconds": started.elapsed().as_secs_f64()}
                    }))?;
                } else {
                    selected = actual_choice.selected_id;
                    let checkpoint = Checkpoint { schema: SCHEMA.to_owned(), case_index, step, position,
                        selected_token: selected, selected_is_eos: is_eos(selected), element_count: VOCABULARY, logits };
                    write_new_json(&mode.directory().join(&file_name), &checkpoint)?;
                    print_event(&mode, "reference_checkpoint_written", json!({
                        "case_index": case_index, "step": step, "position": position,
                        "checkpoint_file": file_name, "consumed_token_ids": consumed_token_ids,
                        "element_count": VOCABULARY, "nonfinite_count": 0, "fresh_workspace_for_case": true,
                        "reference_selected_token": selected, "reference_selected_is_eos": is_eos(selected),
                        "reference_legal_choice": actual_choice,
                        "timings": {"decode_and_read_seconds": decode_and_read_seconds,
                            "checkpoint_seconds": checkpoint_started.elapsed().as_secs_f64(),
                            "elapsed_suite_seconds": started.elapsed().as_secs_f64()}
                    }))?;
                }
                selected_tokens.push(selected);
                if is_eos(selected) { break; }
            }
        }
        let (completion_tokens, forward_tokens, finish_reason) = case_counts(prompt_token_ids.len(), &selected_tokens)?;
        if consumed_token_ids.len() != forward_tokens { return Err("actual feed history length mismatch".into()); }
        let case = ReferenceCase { case_index, prompt: PROMPTS[case_index].to_owned(), prompt_token_ids: prompt_token_ids.clone(),
            checkpoint_count: selected_tokens.len(), completion_tokens, forward_tokens, finish_reason,
            reference_selected_tokens: selected_tokens };
        if let Some(reference) = &reference {
            let expected = &reference.manifest.cases[case_index];
            if case.reference_selected_tokens != expected.reference_selected_tokens || case.checkpoint_count != expected.checkpoint_count
                || case.finish_reason != expected.finish_reason || case.forward_tokens != expected.forward_tokens {
                return Err("completed candidate replay differs from reference history".into());
            }
        }
        print_event(&mode, if reference.is_some() { "comparison_case_complete" } else { "reference_case_complete" }, json!({
            "case_index": case_index, "prompt": case.prompt, "token_ids": case.prompt_token_ids,
            "checkpoint_count": case.checkpoint_count, "completion_tokens": completion_tokens,
            "forward_tokens": forward_tokens, "finish_reason": finish_reason,
            "reference_selected_tokens": case.reference_selected_tokens, "consumed_token_ids": consumed_token_ids,
            "first_legal_greedy_disagreement_step": first_disagreement,
            "total_elements": case.checkpoint_count * VOCABULARY, "fresh_workspace": true,
            "replay_policy": REPLAY_POLICY,
            "timings": {"workspace_seconds": workspace_seconds, "prefill_seconds": prefill_seconds,
                "case_seconds": case_started.elapsed().as_secs_f64(), "elapsed_suite_seconds": started.elapsed().as_secs_f64()}
        }))?;
        realized_cases.push(case);
        first_disagreements.push(first_disagreement);
    }

    let total_checkpoints = checked_sum(realized_cases.iter().map(|case| case.checkpoint_count))?;
    let total_forward_tokens = checked_sum(realized_cases.iter().map(|case| case.forward_tokens))?;
    let total_elements = total_checkpoints.checked_mul(VOCABULARY).ok_or("element count overflow")?;
    let case_checkpoint_counts: Vec<_> = realized_cases.iter().map(|case| case.checkpoint_count).collect();
    let case_finish_reasons: Vec<_> = realized_cases.iter().map(|case| case.finish_reason).collect();
    if reference.is_none() {
        let manifest = Manifest { schema: SCHEMA.to_owned(), thinking: false, add_generation_prompt: true,
            case_count: PROMPTS.len(), element_count_per_checkpoint: VOCABULARY, max_new_tokens: MAX_NEW_TOKENS,
            max_total_tokens: MAX_TOTAL_TOKENS, sampler: sampler_settings(), banned_tokens: BANNED_TOKENS,
            eos_tokens: EXPECTED_EOS, termination_policy: TERMINATION_POLICY.to_owned(),
            total_prompt_tokens, guarded_total_tokens, total_checkpoints, total_forward_tokens, total_elements,
            accuracy_acceptance_threshold: None, accuracy_pass: None, cases: realized_cases };
        validate_manifest(&manifest)?;
        // Published only after every checkpoint is fully serialized, flushed and synced.
        write_new_json(&mode.directory().join("manifest.json"), &manifest)?;
    }
    print_event(&mode, if reference.is_some() { "continuation_comparison_complete" } else { "continuation_reference_complete" }, json!({
        "completed_cases": PROMPTS.len(), "total_prompt_tokens": total_prompt_tokens,
        "guarded_total_tokens": guarded_total_tokens, "max_total_tokens": MAX_TOTAL_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS, "elements_per_checkpoint": VOCABULARY,
        "total_checkpoints": total_checkpoints, "total_forward_tokens": total_forward_tokens,
        "total_elements": total_elements, "case_checkpoint_counts": case_checkpoint_counts,
        "case_finish_reasons": case_finish_reasons, "first_legal_greedy_disagreement_steps": first_disagreements,
        "termination_policy": TERMINATION_POLICY, "replay_policy": REPLAY_POLICY,
        "manifest_file": "manifest.json", "reference_parse_seconds": reference_parse_seconds,
        "load_seconds": load_seconds, "total_seconds": started.elapsed().as_secs_f64()
    }))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits_for(selected: usize) -> Vec<f32> {
        let mut logits = vec![-1.0; VOCABULARY];
        logits[selected] = 1.0;
        logits
    }

    fn eos_case(index: usize) -> ReferenceCase {
        ReferenceCase { case_index: index, prompt: PROMPTS[index].to_owned(), prompt_token_ids: vec![2, 101],
            checkpoint_count: 1, completion_tokens: 0, forward_tokens: 2, finish_reason: FinishReason::Eos,
            reference_selected_tokens: vec![1] }
    }

    fn manifest_fixture() -> Manifest {
        Manifest { schema: SCHEMA.to_owned(), thinking: false, add_generation_prompt: true, case_count: 4,
            element_count_per_checkpoint: VOCABULARY, max_new_tokens: 8, max_total_tokens: 128,
            sampler: sampler_settings(), banned_tokens: BANNED_TOKENS, eos_tokens: EXPECTED_EOS,
            termination_policy: TERMINATION_POLICY.to_owned(), total_prompt_tokens: 8, guarded_total_tokens: 40,
            total_checkpoints: 4, total_forward_tokens: 8, total_elements: 4 * VOCABULARY,
            accuracy_acceptance_threshold: None, accuracy_pass: None, cases: (0..4).map(eos_case).collect() }
    }

    fn checkpoint_fixture() -> Checkpoint {
        Checkpoint { schema: SCHEMA.to_owned(), case_index: 0, step: 0, position: 1,
            selected_token: 1, selected_is_eos: true, element_count: VOCABULARY, logits: logits_for(1) }
    }

    #[test]
    fn actual_public_sampler_suppresses_the_two_largest_logits() {
        let mut logits = logits_for(123);
        logits[BANNED_TOKENS[0]] = 100.0;
        logits[BANNED_TOKENS[1]] = 99.0;
        let choice = greedy_choice(&logits).unwrap();
        assert_eq!(choice.selected_id, 123);
        assert_eq!(choice.margin, 2.0);
        assert!(legal_choice(&logits, BANNED_TOKENS[0]).is_err());
    }

    #[test]
    fn eos_is_selectable_and_not_suppressed() {
        for token in EXPECTED_EOS { assert_eq!(greedy_choice(&logits_for(token)).unwrap().selected_id, token); }
    }

    #[test]
    fn public_tie_is_not_replaced_by_lowest_id_and_signed_zero_uses_total_cmp() {
        let mut logits = logits_for(7);
        logits[23] = 1.0;
        let choice = greedy_choice(&logits).unwrap();
        assert!([7, 23].contains(&choice.selected_id));
        assert_eq!(choice.tie_count, 2);
        assert_eq!(choice.margin, 0.0);
        assert!(legal_choice(&logits, 7).is_ok());
        assert!(legal_choice(&logits, 23).is_ok());
        logits.fill(-1.0);
        logits[7] = -0.0;
        logits[23] = 0.0;
        let choice = greedy_choice(&logits).unwrap();
        assert_eq!(choice.selected_id, 23);
        assert_eq!(choice.tie_count, 1);
        assert_eq!(choice.runner_up_logit_bits, (-0.0f32).to_bits());
        assert!(legal_choice(&logits, 7).is_err());
    }

    #[test]
    fn eos_and_eighth_choice_are_never_fed() {
        assert_eq!(case_counts(14, &[1]).unwrap(), (0, 14, FinishReason::Eos));
        assert_eq!(case_counts(14, &[7, 8, 106]).unwrap(), (2, 16, FinishReason::Eos));
        assert_eq!(case_counts(14, &[7; 8]).unwrap(), (8, 21, FinishReason::Length));
        assert_eq!(case_counts(14, &[7, 7, 7, 7, 7, 7, 7, 50]).unwrap(), (7, 21, FinishReason::Eos));
        assert!(case_counts(14, &[7; 7]).is_err());
        assert!(case_counts(14, &[1, 7]).is_err());
        assert!(case_counts(14, &[7; 9]).is_err());
        assert!(case_counts(14, &[]).is_err());
    }

    #[test]
    fn conservative_budget_is_114_and_forward_bound_is_110() {
        assert_eq!(token_budget([14, 21, 19, 28]).unwrap(), (82, 114));
        let actual = [14, 21, 19, 28].into_iter().map(|length| case_counts(length, &[7; 8]).unwrap().1).sum::<usize>();
        assert_eq!(actual, 110);
        assert!(token_budget([usize::MAX]).is_err());
        assert!(token_budget([E::SIZE - 7]).is_err());
        assert!(token_budget([121]).is_err());
        assert!(token_budget([0]).is_err());
    }

    #[test]
    fn manifest_rejects_false_completion_counters_and_changed_policy() {
        let mut manifest = manifest_fixture();
        validate_manifest(&manifest).unwrap();
        manifest.cases[0].completion_tokens = 1;
        assert!(validate_manifest(&manifest).is_err());
        manifest.cases[0].completion_tokens = 0;
        manifest.total_forward_tokens += 1;
        assert!(validate_manifest(&manifest).is_err());
        manifest.total_forward_tokens -= 1;
        manifest.sampler.top_k = 64;
        assert!(validate_manifest(&manifest).is_err());
        manifest.sampler = sampler_settings();
        manifest.accuracy_pass = Some(true);
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn every_checkpoint_checks_coordinates_maximality_finiteness_and_length() {
        let case = eos_case(0);
        let mut checkpoint = checkpoint_fixture();
        validate_checkpoint(&checkpoint, &case, 0).unwrap();
        checkpoint.position += 1;
        assert!(validate_checkpoint(&checkpoint, &case, 0).is_err());
        checkpoint.position -= 1;
        checkpoint.logits[7] = 2.0;
        assert!(validate_checkpoint(&checkpoint, &case, 0).is_err());
        checkpoint.logits[7] = f32::NAN;
        assert!(validate_checkpoint(&checkpoint, &case, 0).is_err());
        checkpoint.logits[7] = -1.0;
        checkpoint.logits.pop();
        assert!(validate_checkpoint(&checkpoint, &case, 0).is_err());
    }

    #[test]
    fn changed_tokenization_fails_before_device_creation() {
        let loaded = LoadedReference { manifest: manifest_fixture(), checkpoints: Vec::new() };
        let mut encoded = vec![vec![2, 101]; 4];
        validate_encoded_tokens(&encoded, Some(&loaded)).unwrap();
        encoded[2][1] = 100;
        assert!(validate_encoded_tokens(&encoded, Some(&loaded)).is_err());
    }

    #[test]
    fn candidate_reference_rank_does_not_require_candidate_greedy_agreement() {
        let mut actual = logits_for(50); // Candidate would stop; reference chooses 7 and continues.
        actual[7] = 0.5;
        assert_eq!(greedy_choice(&actual).unwrap().selected_id, 50);
        assert_eq!(reference_choice_candidate_rank(&actual, 7), 2);
        assert_eq!(case_counts(2, &[7; 8]).unwrap().2, FinishReason::Length);
    }

    #[test]
    fn full_array_statistics_keep_signed_zero_bit_difference_and_last_element_error() {
        let mut reference = vec![0.0; VOCABULARY];
        let mut actual = reference.clone();
        reference[0] = -0.0;
        actual[VOCABULARY - 1] = 2.0;
        let comparison = compare_logits(&reference, &actual);
        assert_eq!(comparison["elements_compared"], VOCABULARY);
        assert_eq!(comparison["bit_equal_count"], VOCABULARY - 2);
        assert_eq!(comparison["max_abs_index"], VOCABULARY - 1);
        assert_eq!(comparison["max_abs"], 2.0);
        assert!(comparison["accuracy_pass"].is_null());
    }

    #[test]
    fn reference_unknown_fields_are_rejected() {
        let mut value = serde_json::to_value(manifest_fixture()).unwrap();
        value["pretend_success"] = json!(true);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
    }
}

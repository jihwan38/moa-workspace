//! Real paired KV prefixes: automatic native B16 vs forced R64 mixed attention.
//! This diagnostic owns only distinct slots/pages and uses public Model methods.
use furiosa_opt_gemma4::{
    axes::{B, P128, Tf, W},
    host::{
        load,
        model::{Input, Logits, Model, Sequence},
        runtime::{Blocks, Workspace},
    },
    ops,
};
use furiosa_opt_std::prelude::*;
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const REQUESTS: usize = 17;
const PREFIX: [usize; REQUESTS] = [
    128, 128, 128, 128, 128, 128, 128, 511,
    512, 512, 512, 512, 512, 512, 512, 1023, 0,
];
fn input(request: usize, position: usize, logits: bool) -> Input<'static> {
    Input {
        token: 2 + (request * 313 + position * 37) % 8000,
        soft: None,
        logits,
        image: None,
        audio: None,
    }
}
fn chosen(step: usize) -> Vec<usize> {
    match step {
        0 => (0..REQUESTS).collect(),
        1 => vec![15, 7, 16, 8, 0, 6, 14],
        2 => vec![16, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7],
        _ => unreachable!(),
    }
}
// Keep every automatic reference call in the independent native B16 route.
// In particular17=9+8, never16+1 (the scalar singleton has another arithmetic).
fn reference_ranges(count: usize) -> Vec<std::ops::Range<usize>> {
    match count {
        17 => vec![0..9, 9..17],
        7 => vec![0..7],
        _ => panic!("unexpected diagnostic width"),
    }
}
async fn read_bytes(device: &mut Device, logits: &Logits) -> Result<Vec<u8>> {
    Ok(logits.to_host::<m![W]>(&mut device.pdma).await?.into_inner().into_buf())
}
fn check(step: usize, request: usize, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), W::SIZE * 2);
    assert_eq!(actual.len(), expected.len());
    let mut differences = 0;
    let mut max_abs = 0.0f32;
    for (token, (a, b)) in actual.chunks_exact(2).zip(expected.chunks_exact(2)).enumerate() {
        let a = u16::from_le_bytes([a[0], a[1]]);
        let b = u16::from_le_bytes([b[0], b[1]]);
        assert_ne!(a & 0x7f80, 0x7f80, "nonfinite actual step={step} request={request} token={token}");
        assert_ne!(b & 0x7f80, 0x7f80, "nonfinite reference step={step} request={request} token={token}");
        if a != b {
            if differences < 4 {
                eprintln!("MIXED_MODEL_DIFF step={step} request={request} token={token} actual_bits={a:04x} reference_bits={b:04x}");
            }
            differences += 1;
            max_abs = max_abs.max((bf16::from_bits(a).to_f32() - bf16::from_bits(b).to_f32()).abs());
        }
    }
    println!("MIXED_MODEL_EXACT step={step} request={request} checked={} differences={differences} max_abs={max_abs}", W::SIZE);
    assert_eq!(differences, 0, "real-prefix B16/R64 mixed decode changed logits");
}
#[tokio::main]
async fn main() -> Result<()> {
    let total_start = Instant::now();
    let mut device = Device::new(ops::embed_token.topology())?;
    let weights = load::load_weights(&mut device).await?;
    let capacity = 1152;
    let pages = Workspace::blocks_for(capacity);
    assert_eq!(pages, 3);
    let mut workspace = Workspace::new_canonical64(&mut device, &weights, capacity, 64).await?;
    let mut model = Model::new(&mut device, weights, &mut workspace).await?;
    println!("MIXED_MODEL_STARTUP_MS {:.3}", total_start.elapsed().as_secs_f64() * 1000.0);
    let reference: Vec<_> = (0..REQUESTS).map(|r| Blocks {
        sliding: r,
        full: (0..pages).map(|p| r * pages + p).collect(),
    }).collect();
    let candidate: Vec<_> = (0..REQUESTS).map(|r| Blocks {
        sliding: REQUESTS + r,
        full: (0..pages).map(|p| 255 - (r * pages + p)).collect(),
    }).collect();
    let fill_start = Instant::now();
    for request in 0..REQUESTS {
        let request_start = Instant::now();
        let mut position = 0;
        while position < PREFIX[request] {
            let remain = (PREFIX[request] - position).min(Tf::SIZE - position % Tf::SIZE);
            let count = if remain >= P128::SIZE { P128::SIZE } else { remain.min(B::SIZE) };
            assert!(count > 1, "these prefixes deliberately avoid scalar prompt tails");
            let inputs: Vec<_> = (position..position + count).map(|p| input(request, p, false)).collect();
            let sequences = [
                Sequence { inputs: &inputs, start_position: position, blocks: &reference[request] },
                Sequence { inputs: &inputs, start_position: position, blocks: &candidate[request] },
            ];
            model.forward_sequences(&mut device, &sequences, &mut workspace).await?;
            position += count;
        }
        println!("MIXED_MODEL_PREFIX request={request} tokens={} paired_ms={:.3}", PREFIX[request], request_start.elapsed().as_secs_f64() * 1000.0);
    }
    println!("MIXED_MODEL_FILL paired_tokens={} ms={:.3}", PREFIX.iter().sum::<usize>() * 2, fill_start.elapsed().as_secs_f64() * 1000.0);
    let mut positions = PREFIX;
    let mut checked = 0usize;
    for step in 0..3 {
        let selected = chosen(step);
        let rows: Vec<_> = selected.iter().map(|&r| [input(r, positions[r], true)]).collect();
        let policy = selected.iter().enumerate().fold(255u8, |mask, (row, &r)| {
            if positions[r] >= 512 { mask & !(1 << (row / 8)) } else { mask }
        });
        println!("MIXED_MODEL_STEP step={step} width={} expected_group_policy={policy:02x} requests={selected:?} positions={:?}", selected.len(), selected.iter().map(|&r| positions[r]).collect::<Vec<_>>());
        assert!(policy != 0 && policy != 255, "each step must exercise runtime mixed-group dispatch");
        let mut expected: Vec<Option<Vec<u8>>> = (0..REQUESTS).map(|_| None).collect();
        let mut reference_ms = 0.0;
        for range in reference_ranges(selected.len()) {
            assert!((2..=B::SIZE).contains(&range.len()));
            let sequences: Vec<_> = range.clone().map(|row| {
                let r = selected[row];
                Sequence { inputs: &rows[row], start_position: positions[r], blocks: &reference[r] }
            }).collect();
            let begin = Instant::now();
            let outputs = model.forward_sequences(&mut device, &sequences, &mut workspace).await?;
            reference_ms += begin.elapsed().as_secs_f64() * 1000.0;
            for (row, output) in range.zip(outputs.iter()) {
                expected[selected[row]] = Some(read_bytes(&mut device, output).await?);
            }
        }
        let sequences: Vec<_> = selected.iter().enumerate().map(|(row, &r)| Sequence {
            inputs: &rows[row], start_position: positions[r], blocks: &candidate[r],
        }).collect();
        let begin = Instant::now();
        let outputs = model.forward_sequences_r64(&mut device, &sequences, &mut workspace).await?;
        let candidate_ms = begin.elapsed().as_secs_f64() * 1000.0;
        for (row, output) in outputs.iter().enumerate() {
            let r = selected[row];
            let actual = read_bytes(&mut device, output).await?;
            check(step, r, &actual, expected[r].as_ref().expect("reference computed this request"));
            checked += W::SIZE;
        }
        println!("MIXED_MODEL_TIMING step={step} B16_reference_ms={reference_ms:.3} forced_R64_ms={candidate_ms:.3} scope=model_forward_excluding_host_read");
        for r in selected { positions[r] += 1; }
    }
    assert_eq!(checked, 41 * W::SIZE);
    println!("PASS real-prefix mixed-context B16/R64 all_logits={checked},17->7->17, reorderedrows,511->512,1023->1024; total_ms={:.3}", total_start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

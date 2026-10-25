//! Sequential same-schedule whole-model parity: BF16 roundtrip versus sole E4.
//! Both passes use identical R64/B16/scalar routes, including partitioned LM.
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
struct Record {
    phase: String,
    row: usize,
    bytes: Vec<u8>,
}
const PREFIX: [usize; 17] = [
    128, 128, 128, 128, 128, 128, 128, 511, 512, 512, 512, 512, 512, 512, 1023, 1408, 17,
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
async fn save(
    device: &mut Device,
    records: &mut Vec<Record>,
    phase: &str,
    rows: &[usize],
    outputs: &[Logits],
) -> Result<()> {
    assert_eq!(rows.len(), outputs.len());
    for (&row, out) in rows.iter().zip(outputs) {
        records.push(Record {
            phase: phase.to_string(),
            row,
            bytes: out
                .to_host::<m![W]>(&mut device.pdma)
                .await?
                .into_inner()
                .into_buf(),
        });
    }
    Ok(())
}
async fn pass(device: &mut Device, e4: bool) -> Result<Vec<Record>> {
    let start = Instant::now();
    let weights = load::load_weights(device).await?;
    let mut ws = if e4 {
        Workspace::new_canonical64_e4(device, &weights, 1536, 64).await?
    } else {
        Workspace::new_canonical64(device, &weights, 1536, 64).await?
    };
    let mut model = Model::new(device, weights, &mut ws).await?;
    model.prepare_group_scratch(device).await?;
    let blocks: Vec<_> = (0..64)
        .map(|r| Blocks {
            sliding: (r * 5 + 3) % 64,
            full: (0..3).map(|p| 255 - (r * 3 + p)).collect(),
        })
        .collect();
    println!(
        "E4_STORAGE_STARTUP e4={e4} ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.
    );
    let mut records = Vec::new();
    let mut positions = [0usize; 17];
    // Four independent prompt owners share dense work and retain separate KV.
    let inputs: Vec<Vec<_>> = (0..4)
        .map(|r| (0..128).map(|p| input(r, p, false)).collect())
        .collect();
    let sequences: Vec<_> = (0..4)
        .map(|r| Sequence {
            inputs: &inputs[r],
            start_position: 0,
            blocks: &blocks[r],
        })
        .collect();
    model
        .start_prefill_group(device, &sequences, &mut ws)
        .await?;
    while model
        .advance_prefill_group(device, &mut ws)
        .await?
        .is_none()
    {}
    positions[..4].fill(128);
    for r in 0..17 {
        while positions[r] < PREFIX[r] {
            let pos = positions[r];
            let remaining = PREFIX[r] - pos;
            if (r == 8 && pos == 0) || (r == 15 && pos == 896) {
                let inputs: Vec<_> = (pos..pos + 512).map(|p| input(r, p, false)).collect();
                let sequence = Sequence {
                    inputs: &inputs,
                    start_position: pos,
                    blocks: &blocks[r],
                };
                model.start_prefill512(device, &sequence, &mut ws).await?;
                while model.advance_prefill(device, &mut ws).await?.is_none() {}
                positions[r] += 512;
            } else if r == 16 && pos == 0 {
                let inputs: Vec<_> = (0..17).map(|p| input(r, p, false)).collect();
                model
                    .forward_sequences_wide(
                        device,
                        &[Sequence {
                            inputs: &inputs,
                            start_position: 0,
                            blocks: &blocks[r],
                        }],
                        &mut ws,
                    )
                    .await?;
                positions[r] = 17;
            } else {
                let remain = remaining.min(Tf::SIZE - pos % Tf::SIZE);
                let count = if remain >= P128::SIZE {
                    P128::SIZE
                } else {
                    remain.min(B::SIZE)
                };
                let inputs: Vec<_> = (pos..pos + count).map(|p| input(r, p, false)).collect();
                model
                    .forward_sequences(
                        device,
                        &[Sequence {
                            inputs: &inputs,
                            start_position: pos,
                            blocks: &blocks[r],
                        }],
                        &mut ws,
                    )
                    .await?;
                positions[r] += count;
            }
        }
    }
    println!(
        "E4_STORAGE_FILL e4={e4} tokens={} elapsed_ms={:.3}",
        PREFIX.iter().sum::<usize>(),
        start.elapsed().as_secs_f64() * 1000.
    );
    let phases = [
        ("r64_17", (0..17).collect::<Vec<_>>(), 0),
        ("b16_7", vec![14, 7, 16, 8, 0, 6, 15], 1),
        ("scalar_1", vec![15], 1),
        (
            "r64_regrow",
            vec![16, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7],
            0,
        ),
        ("forced_scalar", vec![7, 14], 2),
    ];
    for (phase, selected, route) in phases {
        let inputs: Vec<_> = selected
            .iter()
            .map(|&r| [input(r, positions[r], true)])
            .collect();
        let sequences: Vec<_> = selected
            .iter()
            .enumerate()
            .map(|(i, &r)| Sequence {
                inputs: &inputs[i],
                start_position: positions[r],
                blocks: &blocks[r],
            })
            .collect();
        let t = Instant::now();
        let outputs = match route {
            0 => {
                model
                    .forward_sequences_r64(device, &sequences, &mut ws)
                    .await?
            }
            1 => model.forward_sequences(device, &sequences, &mut ws).await?,
            _ => {
                model
                    .forward_sequences_serial(device, &sequences, &mut ws)
                    .await?
            }
        };
        let elapsed = t.elapsed().as_secs_f64() * 1000.;
        save(device, &mut records, phase, &selected, outputs).await?;
        println!(
            "E4_STORAGE_PHASE e4={e4} phase={phase} rows={} model_ms={elapsed:.3}",
            selected.len()
        );
        for r in selected {
            positions[r] += 1;
        }
    }
    // A short prompt appended after request-batch and scalar decode uses old E4 history.
    let pos = positions[0];
    let inputs: Vec<_> = (pos..pos + 7).map(|p| input(0, p, p == pos + 6)).collect();
    let outputs = model
        .forward_sequences(
            device,
            &[Sequence {
                inputs: &inputs,
                start_position: pos,
                blocks: &blocks[0],
            }],
            &mut ws,
        )
        .await?;
    save(device, &mut records, "appended_prompt7", &[0], outputs).await?;
    // Reuse every physical slot at position0: exercise live64 fused commits and stale-future masks.
    for pos in 0..2 {
        let inputs: Vec<_> = (0..64).map(|r| [input(r + 80, pos, true)]).collect();
        let sequences: Vec<_> = (0..64)
            .map(|r| Sequence {
                inputs: &inputs[r],
                start_position: pos,
                blocks: &blocks[r],
            })
            .collect();
        let t = Instant::now();
        let outputs = model
            .forward_sequences_r64(device, &sequences, &mut ws)
            .await?;
        let elapsed = t.elapsed().as_secs_f64() * 1000.;
        save(
            device,
            &mut records,
            &format!("reset64_pos{pos}"),
            &(0..64).collect::<Vec<_>>(),
            outputs,
        )
        .await?;
        println!("E4_STORAGE_PHASE e4={e4} reset64_pos={pos} model_ms={elapsed:.3}");
    }
    println!(
        "E4_STORAGE_PASS_READY e4={e4} rows={} elapsed_ms={:.3}",
        records.len(),
        start.elapsed().as_secs_f64() * 1000.
    );
    // Model and Workspace (including all weights/caches) drop before next pass.
    Ok(records)
}
#[tokio::main]
async fn main() -> Result<()> {
    assert!(cfg!(feature = "kv-e4-quality"));
    let start = Instant::now();
    let mut device = Device::new(ops::embed_token.topology())?;
    let reference = pass(&mut device, false).await?;
    let actual = pass(&mut device, true).await?;
    assert_eq!(reference.len(), actual.len());
    let mut checked = 0;
    for (a, b) in actual.iter().zip(&reference) {
        assert_eq!((&a.phase, a.row), (&b.phase, b.row));
        assert_eq!(a.bytes.len(), b.bytes.len());
        let mut mismatches = 0;
        for (i, (a, b)) in a
            .bytes
            .chunks_exact(2)
            .zip(b.bytes.chunks_exact(2))
            .enumerate()
        {
            let av = u16::from_le_bytes([a[0], a[1]]);
            let bv = u16::from_le_bytes([b[0], b[1]]);
            assert_ne!(av & 0x7f80, 0x7f80);
            assert_ne!(bv & 0x7f80, 0x7f80);
            if av != bv {
                if mismatches < 4 {
                    println!(
                        "E4_STORAGE_DIFF phase={} row={} index={i} actual={av:04x} reference={bv:04x}",
                        actual[checked / W::SIZE].phase,
                        actual[checked / W::SIZE].row
                    );
                }
                mismatches += 1;
            }
        }
        println!(
            "E4_STORAGE_EXACT phase={} row={} values={} differences={mismatches}",
            a.phase,
            a.row,
            W::SIZE
        );
        assert_eq!(
            mismatches, 0,
            "same-schedule E4 storage must preserve all reconstructed BF16 logits"
        );
        checked += W::SIZE;
    }
    println!(
        "PASS E4 sole-pool same-schedule model logits_exact={checked}; independentGroup/P128/Prompt512 wrap/partialWide17/B16/R64/scalar/reset64; elapsed={:.3}s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

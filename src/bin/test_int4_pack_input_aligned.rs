//! Integer packing/raw-byte gate and completed-call costs; no quantizer or GEMM.
use furiosa_opt_gemma4::{
    Chip,
    axes::{H, L, P512},
    i4_p512_raw_adapter as raw, ops_int4_pack_staged_store as k,
    ops_int4_pack_p512 as original, ops_int4_pack_input_aligned as aligned,
};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::{Buffers, DeviceSend};
use std::time::Instant;
#[path = "support/npu_profile.rs"]
mod npu_profile;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn signed(n: usize) -> i32 {
    let n = (n & 15) as i32;
    if n >= 8 { n - 16 } else { n }
}
fn code(case: usize, p: usize, c: usize, width: usize) -> i32 {
    match case {
        0 => {
            // Encode all nine row bits in each K960 block, so P/256 cluster
            // swaps and rows sixteen apart cannot evade the byte oracle.
            if c % 960 < 3 {
                return signed(p >> (4 * (c % 960)));
            }
            let mut z = (p as u32).wrapping_mul(0x9e3779b9)
                ^ (c as u32).wrapping_mul(0x85ebca6b)
                ^ 0xc2b2ae35;
            z ^= z >> 16;
            z = z.wrapping_mul(0x7feb352d);
            z ^= z >> 15;
            signed(z as usize)
        }
        1 => -8,
        2 => 7,
        3 => 0,
        4 => code(0, (73 * p + 17) % 512, (37 * c + 19) % width, width),
        _ => panic!("fixed fixture case"),
    }
}
fn canonical_bytes(values: &[i32]) -> Vec<u8> {
    values
        .chunks_exact(2)
        .map(|pair| {
            assert!(pair.iter().all(|v| (-8..=7).contains(v)));
            ((pair[0] as u8) & 15) | (((pair[1] as u8) & 15) << 4)
        })
        .collect()
}
fn blocked_pair_bytes(values: &[i32], width: usize) -> Vec<u8> {
    let mut bytes = vec![0; values.len() / 2];
    for p in 0..512 {
        for c in (0..width).step_by(2) {
            let position =
                (((p / 256 * (width / 960) + c / 960) * 256 + p % 256) * 960 + c % 960) / 2;
            let pair = &values[p * width + c..p * width + c + 2];
            bytes[position] = ((pair[0] as u8) & 15) | (((pair[1] as u8) & 15) << 4);
        }
    }
    bytes
}
fn independently_indexed_words(values: &[i32], width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() / 2);
    for word_index in 0..values.len() / 8 {
        let block_row = word_index / 120;
        let row_in_half = block_row % 256;
        let group = (block_row / 256) % (width / 960);
        let half = block_row / (256 * (width / 960));
        let row = half * 256 + row_in_half;
        let col = group * 960 + (word_index % 120) * 8;
        let mut word = 0u32;
        for lane in 0..8 {
            word = word
                .checked_add(((values[row * width + col + lane] as u32) & 15) << (4 * lane))
                .unwrap();
        }
        bytes.extend(word.to_le_bytes());
    }
    bytes
}
async fn upload<D: MaterializableScalar, E: M>(
    d: &mut Device,
    bytes: Vec<u8>,
) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(bytes)
        .to_hbm(&mut d.pdma)
        .await
        .unwrap()
}
async fn read<D: MaterializableScalar, E: M>(d: &mut Device, t: &HbmTensor<D, Chip, E>) -> Vec<u8> {
    t.to_host::<E>(&mut d.pdma)
        .await
        .unwrap()
        .into_inner()
        .into_buf()
}

fn reject_bad_owners<P: M, O: M>(
    packed: &HbmTensor<i32, Chip, P>,
    output: &HbmTensor<i4, Chip, O>,
    width: usize,
) -> Result<()> {
    let mut valid = Buffers::new();
    DeviceSend::bind(packed, &mut valid)?;
    DeviceSend::bind(output, &mut valid)?;
    raw::validate_bound(&valid, width)?;
    let other_width = if width == H::SIZE { L::SIZE } else { H::SIZE };
    assert!(raw::validate_bound(&valid, other_width).is_err());
    let mut one = Buffers::new();
    DeviceSend::bind(packed, &mut one)?;
    assert!(raw::validate_bound(&one, width).is_err());
    DeviceSend::bind(packed, &mut one)?;
    assert!(raw::validate_bound(&one, width).is_err(), "aliased owners rejected");
    // All three failures are read-only validation; no Function::run occurs.
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let collector = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(collector.clone())?;
    let mut d = Device::new(original::pack_input.topology())?;
    let mut total_codes = 0usize;
    let mut total_packed_bytes = 0usize;
    let mut owner_rejection_checks = 0usize;
    let mut original_packed_bytes = 0usize;
    let mut input_speed_gates: Vec<(usize, f64, f64, bool)> = Vec::new();
    macro_rules! check_shape {($axis:ty,$packed:ty,$i4:ty,$pack:expr,$baseline:expr,$reader:expr,$raw:expr)=>{{
        let width=<$axis>::SIZE;
        for case in 0..5 {
            let values:Vec<i32>=(0..512*width).map(|i|code(case,i/width,i%width,width)).collect();
            let canonical=canonical_bytes(&values);
            let blocked=blocked_pair_bytes(&values,width);
            assert_eq!(blocked,independently_indexed_words(&values,width),"independent word/nibble/layout oracle");
            // Check every physical signed nibble, including the legacy I4K960
            // UP contract and the same L960 DOWN ordering.
            for p in 0..512 { for c in 0..width {
                let nibble=((p/256*(width/960)+c/960)*256+p%256)*960+c%960;
                let actual=(blocked[nibble/2]>>(4*(nibble%2)))&15;
                assert_eq!(signed(actual as usize),values[p*width+c]);
            }}
            if case==0 {
                let mut seen=[false;16];
                for word in blocked.chunks_exact(4) { seen[(word[3]>>4) as usize]=true; }
                assert!(seen.iter().all(|x|*x),"all sixteen top nibbles");
                for p in 0..512 { for block in 0..width/960 {
                    let tag=(0..3).fold(0usize,|n,j|n|(((values[p*width+block*960+j] as usize)&15)<<(4*j)));
                    assert_eq!(tag,p,"all nine row bits survive in every K960 block");
                }}
            }
            if case==4 { for p in 0..512 { for c in 0..width {
                assert_eq!(values[p*width+c],code(0,(73*p+17)%512,(37*c+19)%width,width));
            }}}
            let input:HbmTensor<i32,Chip,m![P512,$axis]>=HostTensor::<i32,m![P512,$axis]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await?;
            let shifts:HbmTensor<i32,Chip,m![$axis%8]>=HostTensor::<i32,m![$axis%8]>::from_vec((0..8).map(|i|4*i)).to_hbm(&mut d.pdma).await?;
            let mut packed:HbmTensor<i32,Chip,$packed>=HostTensor::<i32,$packed>::from_vec(std::iter::repeat_n(0x13579bdfi32,512*width/8)).to_hbm(&mut d.pdma).await?;
            let mut baseline_packed:HbmTensor<i32,Chip,$packed>=HostTensor::<i32,$packed>::from_vec(std::iter::repeat_n(0x2468ace0i32,512*width/8)).to_hbm(&mut d.pdma).await?;
            let typed:HbmTensor<i4,Chip,$i4>=upload(&mut d,blocked.clone()).await;
            let mut candidate:HbmTensor<i4,Chip,m![P512,$axis]>=upload(&mut d,vec![0xad;512*width/2]).await;
            let mut control:HbmTensor<i4,Chip,m![P512,$axis]>=upload(&mut d,vec![0x52;512*width/2]).await;
            if case==0 {
                reject_bad_owners(&packed,&candidate,width)?;
                owner_rejection_checks+=3;
                assert_eq!(read(&mut d,&candidate).await,vec![0xad;512*width/2],"rejected binding changed output");
            }
            launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;
            assert_eq!(read(&mut d,&packed).await,blocked,"whole blocked packed owner");
            launch($baseline,(&mut d,&input,&shifts,&mut baseline_packed)).await?;
            assert_eq!(read(&mut d,&baseline_packed).await,blocked,"independent original pack owner");
            original_packed_bytes+=blocked.len();
            // Candidate first, with independently initialized output. The later
            // normal typed launch embeds the identical reader image.
            $raw(&mut d,&packed,&mut candidate).await?;
            assert_eq!(read(&mut d,&candidate).await,canonical,"raw binding/canonical byte oracle");
            launch($reader,(&mut d,&typed,&mut control)).await?;
            assert_eq!(read(&mut d,&control).await,canonical,"typed blocked reader oracle");
            assert_eq!(read(&mut d,&packed).await,blocked,"packed owner immutable after await");
            assert_eq!(read(&mut d,&typed).await,blocked,"typed I4 source immutable");
            let input_bytes:Vec<u8>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
            assert_eq!(read(&mut d,&input).await,input_bytes,"ordinary I32 input immutable");
            let shift_bytes:Vec<u8>=(0..8i32).flat_map(|i|(4*i).to_le_bytes()).collect();
            assert_eq!(read(&mut d,&shifts).await,shift_bytes,"shift table immutable");
            if case==0 || case==4 {
                let mut timing=[Vec::new(),Vec::new()];
                for mode in [0,1,1,0] {
                    let start=Instant::now();
                    for _ in 0..16 {
                        launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;
                        if mode==1 {$raw(&mut d,&packed,&mut candidate).await?;}
                    }
                    timing[mode].push(start.elapsed().as_secs_f64()*1e6/16.0);
                }
                let mut paired=[Vec::new(),Vec::new()];
                for mode in [0,1,1,0] {
                    let start=Instant::now();
                    for _ in 0..16 {
                        if mode==0 {launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;}
                        else {launch($baseline,(&mut d,&input,&shifts,&mut baseline_packed)).await?;}
                    }
                    paired[mode].push(start.elapsed().as_secs_f64()*1e6/16.0);
                }
                println!("PACK_ALIGNED_ABBA width={width} case={case} samples=16 candidate_us={:?} original_us={:?} input_same_function_control=false all_real_DMA_charged=true",paired[0],paired[1]);
                if width==H::SIZE {
                    let candidate_worst=paired[0].iter().copied().fold(f64::NEG_INFINITY,f64::max);
                    let original_best=paired[1].iter().copied().fold(f64::INFINITY,f64::min);
                    let passed=candidate_worst<=0.8*original_best;
                    println!("PACK_ALIGNED_H_GATE case={case} candidate_worst_us={candidate_worst} original_best_us={original_best} required_gain=0.20 pass={passed}");
                    input_speed_gates.push((case,candidate_worst,original_best,passed));
                }
                assert_eq!(read(&mut d,&baseline_packed).await,blocked,"original packed bytes after all timed reuse");
                assert_eq!(read(&mut d,&packed).await,blocked,"packed bytes after timed reuse");
                assert_eq!(read(&mut d,&candidate).await,canonical,"raw output after timed reuse");
                assert_eq!(read(&mut d,&control).await,canonical,"independent typed output unchanged");
                assert_eq!(read(&mut d,&input).await,input_bytes,"input immutable after timed reuse");
                assert_eq!(read(&mut d,&shifts).await,shift_bytes,"shifts immutable after timed reuse");
                assert_eq!(read(&mut d,&typed).await,blocked,"typed source immutable after timed reuse");
                println!("PACK_P512_TIMING width={width} case={case} repetitions=16 pack_us={:?} pack_plus_raw_copy_us={:?} complete_awaited_calls=true excludes_quantizer_gemm_setup_readback=true",timing[0],timing[1]);
            }
            assert_eq!(read(&mut d,&baseline_packed).await,blocked,"original output preserved through all later calls");
            if case==0 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
                println!("PACK_P512_PROFILE width={width} mode=pack begin");
                collector.enable(true);
                launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;
                collector.enable(false);
                println!("PACK_P512_PROFILE width={width} mode=pack end");
                println!("PACK_P512_PROFILE width={width} mode=pack_plus_raw_copy begin");
                collector.enable(true);
                launch($pack,(&mut d,&input,&shifts,&mut packed)).await?;
                $raw(&mut d,&packed,&mut candidate).await?;
                collector.enable(false);
                println!("PACK_P512_PROFILE width={width} mode=pack_plus_raw_copy end");
                assert_eq!(read(&mut d,&packed).await,blocked,"packed bytes after profiling");
                assert_eq!(read(&mut d,&candidate).await,canonical,"raw bytes after profiling");
                assert_eq!(read(&mut d,&input).await,input_bytes,"input immutable after profiling");
                assert_eq!(read(&mut d,&shifts).await,shift_bytes,"shifts immutable after profiling");
                assert_eq!(read(&mut d,&typed).await,blocked,"typed source immutable after profiling");
            }
            println!("PACK_P512_CASE width={width} case={case} codes={} packed_bytes={} whole_owner_exact=true typed_raw_canonical_exact=true",values.len(),blocked.len());
            total_codes+=values.len();total_packed_bytes+=blocked.len();
        }
    }};}
    check_shape!(
        H,
        k::PackedInput,
        k::I4Input,
        aligned::pack_input,
        original::pack_input,
        k::read_input,
        raw::read_input
    );
    check_shape!(
        L,
        k::PackedHidden,
        k::I4Hidden,
        k::pack_hidden,
        original::pack_hidden,
        k::read_hidden,
        raw::read_hidden
    );
    assert_eq!(total_codes, 49_152_000);
    assert_eq!(total_packed_bytes, 24_576_000);
    assert_eq!(owner_rejection_checks, 6);
    assert_eq!(original_packed_bytes, 24_576_000);
    println!(
        "PASS_PACK_P512 codes={total_codes} packed_bytes={total_packed_bytes} owner_rejection_checks={owner_rejection_checks} scope=integer_packing_and_raw_blocked_layout_only no_quantizer_gemm_model_gain_claim=true"
    );
    println!("PASS_PACK_ALIGNED original_packed_bytes={original_packed_bytes} aligned_input_executed=true hidden_staged_executed=true failed_four_row_input_executed=false complete_FFN_or_model_claim=false");
    // Finish every byte/ownership/post-reuse gate before judging measured speed.
    assert_eq!(input_speed_gates.iter().map(|g|g.0).collect::<Vec<_>>(),vec![0,4]);
    let performance_pass=input_speed_gates.iter().all(|g|g.3);
    println!("PACK_ALIGNED_PERFORMANCE required_each_H_case_gain=0.20 pass={performance_pass} correctness_complete=true");
    assert!(performance_pass,"H complete pack did not meet the frozen per-case20percent gate");
    Ok(())
}

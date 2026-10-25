//! Strict complete quantize+pack gate; no raw-I4 consumer, GEMM, FFN or model.
use furiosa_opt_gemma4::{
    Chip, axes::{H, L, P512}, ops_int4_rational_fused_pack as fused,
    ops_int4_rational_lut_p512_compact as qh, ops_int4_rational_lut_hidden as ql,
    ops_int4_pack_input_aligned as aligned,
};
use fused::{Mantissa, Record};
use qh::{Mantissa as HMantissa, Record as HRecord};
use ql::{Mantissa as LMantissa, Record as LRecord};
use furiosa_opt_std::prelude::*;
#[path = "support/npu_profile.rs"] mod npu_profile;
#[path = "support/rational_a4_oracle.rs"] mod oracle;
#[path = "support/rational_a4_p512_fixture.rs"] mod input_fixture;
#[path = "support/rational_a4_hidden_fixture.rs"] mod hidden_fixture;

async fn raw<D: MaterializableScalar, E: M>(d: &mut Device, x: &HbmTensor<D, Chip, E>) -> Vec<u8> {
    x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()
}

fn encode_pairs(codes: &[i32], width: usize) -> Vec<u8> {
    assert_eq!(codes.len(), 512 * width);
    assert!(codes.iter().all(|q| (-7..=7).contains(q)));
    let mut result = vec![0; codes.len() / 2];
    for p in 0..512 {
        for c in (0..width).step_by(2) {
            let i = (((p / 256 * (width / 960) + c / 960) * 256 + p % 256) * 960 + c % 960) / 2;
            result[i] = (codes[p * width + c] as u8 & 15)
                | ((codes[p * width + c + 1] as u8 & 15) << 4);
        }
    }
    result
}

fn encode_words(codes: &[i32], width: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(codes.len() / 2);
    for physical_word in 0..codes.len() / 8 {
        let block_row = physical_word / 120;
        let row = (block_row / (256 * (width / 960))) * 256 + block_row % 256;
        let col = ((block_row / 256) % (width / 960)) * 960 + (physical_word % 120) * 8;
        let mut word = 0u32;
        for nibble in 0..8 {
            word = word.checked_add(((codes[row * width + col + nibble] as u32) & 15) << (4 * nibble)).unwrap();
        }
        result.extend(word.to_le_bytes());
    }
    result
}

fn decode_blocked(bytes: &[u8], width: usize) -> Vec<i32> {
    assert_eq!(bytes.len(), 512 * width / 2);
    (0..512 * width).map(|i| {
        let (p, c) = (i / width, i % width);
        let n = ((p / 256 * (width / 960) + c / 960) * 256 + p % 256) * 960 + c % 960;
        let q = ((bytes[n / 2] >> (4 * (n % 2))) & 15) as i32;
        if q >= 8 { q - 16 } else { q }
    }).collect()
}

#[derive(Default)]
struct Counts {
    codes: usize, packed_bytes: usize, scales: usize,
    permutation_codes: usize, permutation_scales: usize,
    post_reuse_bytes: usize, immutable_bytes: usize, calls: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if !["input", "hidden", "all"].contains(&mode.as_str()) { return Err("expected input, hidden or all".into()); }
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    let mut d = Device::new(fused::quantize_pack_input_aligned.topology())?;
    let values: Vec<i32> = (128u32..256).flat_map(|m| {
        let u = 7 * ((1u32 << 20).div_ceil(m));
        std::array::from_fn::<_, 8, _>(|lane| if lane == 0 { (u as f32).to_bits() as i32 }
            else { (0x5a3c0000u32 | (m << 4) | lane as u32) as i32 })
    }).collect();
    let table_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let table = HostTensor::<i32, m![Mantissa, Record]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await?;
    assert_eq!(raw(&mut d, &table).await, table_bytes);
    let mut counts = Counts::default();
    let mut gates = Vec::new();
    macro_rules! run_width {($axis:ty,$packed:ty,$fixture:ident,$qm:ty,$qr:ty,$quant:expr,$pack:expr,$candidate:expr)=>{{
        let width = <$axis>::SIZE;
        assert_eq!(width, $fixture::WIDTH);
        let original_table = HostTensor::<i32,m![$qm,$qr]>::from_vec(values.clone()).to_hbm(&mut d.pdma).await?;
        assert_eq!(raw(&mut d,&original_table).await,table_bytes);
        let shifts = HostTensor::<i32,m![$axis%8]>::from_vec((0..8i32).map(|i|4*i)).to_hbm(&mut d.pdma).await?;
        let shift_bytes:Vec<u8> = (0..8i32).flat_map(|i|(4*i).to_le_bytes()).collect();
        let mut saved:Vec<(Vec<i32>,Vec<u32>)> = Vec::new();
        let before = (counts.codes,counts.scales,counts.packed_bytes,counts.post_reuse_bytes,counts.immutable_bytes,counts.calls);
        for case in 0..$fixture::CASES {
            let bits = $fixture::fixture(case);
            let input_bytes:Vec<u8> = bits.iter().flat_map(|v|v.to_le_bytes()).collect();
            let mut expected = Vec::with_capacity(bits.len());
            let mut expected_scales = Vec::with_capacity(512);
            for row in bits.chunks_exact(width) {
                let(q,s) = oracle::reference(row); expected.extend(q); expected_scales.push(s.to_bits());
            }
            assert_eq!(expected_scales.len(),512);
            if case==0 {assert_eq!(expected_scales.iter().copied().collect::<std::collections::BTreeSet<_>>().len(),512);}
            let expected_packed = encode_pairs(&expected,width);
            assert_eq!(expected_packed,encode_words(&expected,width),"two independent blocked encoders");
            let expected_code_bytes:Vec<u8> = expected.iter().flat_map(|v|v.to_le_bytes()).collect();
            let expected_scale_bytes:Vec<u8> = expected_scales.iter().flat_map(|v|v.to_le_bytes()).collect();
            let input = HostTensor::<bf16,m![P512,$axis]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
            let mut candidate = HostTensor::<i32,$packed>::from_vec(std::iter::repeat_n(0x13579bdfi32,512*width/8)).to_hbm(&mut d.pdma).await?;
            let mut control = HostTensor::<i32,$packed>::from_vec(std::iter::repeat_n(0x2468ace0i32,512*width/8)).to_hbm(&mut d.pdma).await?;
            let mut codes = HostTensor::<i32,m![P512,$axis]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32,512*width)).to_hbm(&mut d.pdma).await?;
            let mut cs = HostTensor::<f32,m![P512]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001),512)).to_hbm(&mut d.pdma).await?;
            let mut os = HostTensor::<f32,m![P512]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc20002),512)).to_hbm(&mut d.pdma).await?;
            launch($candidate,(&mut d,&input,&table,&shifts,&mut candidate,&mut cs)).await?;
            let candidate_bytes = raw(&mut d,&candidate).await;
            let candidate_scales = raw(&mut d,&cs).await;
            assert_eq!(candidate_bytes,expected_packed,"all candidate packed bytes versus CPU");
            assert_eq!(candidate_scales,expected_scale_bytes,"all512 candidate scale bits");
            let candidate_codes = decode_blocked(&candidate_bytes,width);
            assert_eq!(candidate_codes,expected,"every signed candidate nibble");
            launch($quant,(&mut d,&input,&original_table,&mut codes,&mut os)).await?;
            launch($pack,(&mut d,&codes,&shifts,&mut control)).await?;
            let control_bytes = raw(&mut d,&control).await;
            let control_scales = raw(&mut d,&os).await;
            let control_codes = raw(&mut d,&codes).await;
            assert_eq!(control_codes,expected_code_bytes,"full original rational I32 codes versus CPU");
            assert_eq!(control_bytes,expected_packed,"full separate packed bytes versus CPU");
            assert_eq!(control_scales,expected_scale_bytes,"all512 original scale bits");
            counts.calls+=3;
            if case>=2 {
                let source = &saved[case-2];
                for p in 0..512 {
                    let sp = $fixture::source_coordinate(case,p,0).0;
                    assert_eq!(expected_scales[p],source.1[sp]);
                    for c in 0..width {
                        let sc = $fixture::source_coordinate(case,p,c).1;
                        assert_eq!(candidate_codes[p*width+c],source.0[sp*width+sc]);
                    }
                }
                counts.permutation_codes+=expected.len(); counts.permutation_scales+=512;
            } else {saved.push((candidate_codes,expected_scales));}
            let mut times = [Vec::new(),Vec::new()];
            for arm in [0usize,1,1,0] {
                let begin = std::time::Instant::now();
                for _ in 0..20 {
                    if arm==0 {launch($candidate,(&mut d,&input,&table,&shifts,&mut candidate,&mut cs)).await?;}
                    else {
                        launch($quant,(&mut d,&input,&original_table,&mut codes,&mut os)).await?;
                        launch($pack,(&mut d,&codes,&shifts,&mut control)).await?;
                    }
                }
                times[arm].push(begin.elapsed().as_secs_f64()*50000.0);
                counts.calls+=20*(arm+1);
            }
            println!("RATIONAL_FUSED_ABBA width={width} case={case} samples=20 candidate_us={:?} current_separate_us={:?} scope=complete_quantize_pack_no_consumer_GEMM_FFN_model",times[0],times[1]);
            let worst = times[0].iter().copied().fold(f64::NEG_INFINITY,f64::max);
            let best = times[1].iter().copied().fold(f64::INFINITY,f64::min);
            gates.push((width,case,worst,best,worst<=0.8*best));
            if case==3 {
                println!("PROFILE_RATIONAL_FUSED width={width} arm=candidate");profile.enable(true);
                launch($candidate,(&mut d,&input,&table,&shifts,&mut candidate,&mut cs)).await?;
                profile.enable(false);
                println!("PROFILE_RATIONAL_FUSED width={width} arm=current_separate");profile.enable(true);
                launch($quant,(&mut d,&input,&original_table,&mut codes,&mut os)).await?;
                launch($pack,(&mut d,&codes,&shifts,&mut control)).await?;
                profile.enable(false);counts.calls+=3;
            }
            assert_eq!(raw(&mut d,&candidate).await,candidate_bytes,"candidate whole packed owner after reuse");
            assert_eq!(raw(&mut d,&cs).await,candidate_scales,"candidate scales after reuse");
            assert_eq!(raw(&mut d,&control).await,control_bytes,"control packed owner after reuse");
            assert_eq!(raw(&mut d,&os).await,control_scales,"control scales after reuse");
            assert_eq!(raw(&mut d,&codes).await,control_codes,"control canonical codes after reuse");
            assert_eq!(raw(&mut d,&input).await,input_bytes,"entire BF16 input immutable");
            assert_eq!(raw(&mut d,&table).await,table_bytes,"entire candidate table immutable");
            assert_eq!(raw(&mut d,&original_table).await,table_bytes,"entire control table immutable");
            assert_eq!(raw(&mut d,&shifts).await,shift_bytes,"all shift bytes immutable");
            let post = candidate_bytes.len()+candidate_scales.len()+control_bytes.len()+control_scales.len()+control_codes.len();
            counts.codes+=expected.len();counts.packed_bytes+=expected_packed.len();counts.scales+=512;
            counts.post_reuse_bytes+=post;counts.immutable_bytes+=input_bytes.len()+2*table_bytes.len()+shift_bytes.len();
            println!("RATIONAL_FUSED_EXACT width={width} case={case} codes={} candidate_packed_bytes={} control_packed_bytes={} each_scales=512 post_reuse_output_bytes={post}",expected.len(),candidate_bytes.len(),control_bytes.len());
        }
        assert_eq!(counts.codes-before.0,4*512*width);
        assert_eq!(counts.scales-before.1,2048);
        assert_eq!(counts.packed_bytes-before.2,4*512*width/2);
        assert_eq!(counts.post_reuse_bytes-before.3,4*(5*512*width+4096));
        assert_eq!(counts.immutable_bytes-before.4,4*(2*512*width+8192+32));
        assert_eq!(counts.calls-before.5,495);
    }};}
    if mode!="hidden" {run_width!(H,fused::PackedInput,input_fixture,HMantissa,HRecord,qh::quantize_row,aligned::pack_input,fused::quantize_pack_input_aligned);}
    if mode!="input" {run_width!(L,fused::PackedHidden,hidden_fixture,LMantissa,LRecord,ql::quantize_row,fused::pack_hidden_control,fused::quantize_pack_hidden);}
    if mode=="all" {
        assert_eq!((counts.codes,counts.scales,counts.packed_bytes),(39_321_600,4096,19_660_800));
        assert_eq!((counts.permutation_codes,counts.permutation_scales),(19_660_800,2048));
        assert_eq!((counts.post_reuse_bytes,counts.immutable_bytes,counts.calls),(196_640_768,78_708_992,990));
    }
    println!("PASS_RATIONAL_FUSED_CORRECTNESS mode={mode} candidate_codes={} control_canonical_codes={} candidate_packed_bytes={} control_packed_bytes={} each_scales={} permutation_codes={} permutation_scales={} post_reuse_output_bytes={} immutable_bytes={} kernel_calls={} no_consumer_GEMM_FFN_model_claim",counts.codes,counts.codes,counts.packed_bytes,counts.packed_bytes,counts.scales,counts.permutation_codes,counts.permutation_scales,counts.post_reuse_bytes,counts.immutable_bytes,counts.calls);
    for &(width,case,candidate,control,passed) in &gates {println!("RATIONAL_FUSED_SPEED_GATE width={width} case={case} candidate_worst_us={candidate} control_best_us={control} required_gain=0.20 pass={passed}");}
    let passed = gates.iter().all(|g|g.4);
    println!("RATIONAL_FUSED_PERFORMANCE pass={passed} correctness_complete=true");
    assert!(passed,"fused full quantize+pack did not meet each-case20percent gate");
    Ok(())
}

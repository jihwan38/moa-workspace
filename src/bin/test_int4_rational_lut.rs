//! Strict exact-rational A4 foundation: complete quantization, no pack/model claim.
use furiosa_opt_gemma4::{
    Chip, axes::{B, H, P512}, ops_int4_rational_lut as op,
    ops_int4_rational_lut_p512 as original,
    ops_int4_rational_lut_p512_compact as compact,
};
use furiosa_opt_std::prelude::*;
#[path = "support/npu_profile.rs"] mod npu_profile;
#[path = "support/rational_a4_oracle.rs"] mod oracle;
#[path = "support/rational_a4_p512_fixture.rs"] mod p512;

async fn raw<D: MaterializableScalar, E: M>(d: &mut Device, x: &HbmTensor<D, Chip, E>) -> Vec<u8> {
    x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()
}

fn table_values() -> Vec<i32> {
    (128u32..256).flat_map(|m| {
        let u = 7 * ((1u32 << 20).div_ceil(m));
        std::array::from_fn::<_, 8, _>(|lane| {
            if lane == 0 { (u as f32).to_bits() as i32 }
            else { (0x5a3c0000u32 | (m << 4) | lane as u32) as i32 }
        })
    }).collect()
}

fn expected(bits: &[u16]) -> (Vec<i32>, Vec<f32>) {
    let mut codes = Vec::with_capacity(bits.len());
    let mut scales = Vec::with_capacity(bits.len() / oracle::WIDTH);
    for row in bits.chunks_exact(oracle::WIDTH) {
        let (q, s) = oracle::reference(row);
        codes.extend(q);
        scales.push(s);
    }
    (codes, scales)
}

fn verify(tag: &str, case: usize, qr: Vec<u8>, sr: Vec<u8>, expected: &[i32], scales: &[f32]) -> (Vec<i32>, Vec<u32>) {
    assert_eq!(qr.len(), expected.len() * 4);
    assert_eq!(sr.len(), scales.len() * 4);
    let got: Vec<i32> = qr.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let scale_bits: Vec<u32> = sr.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut differences = 0;
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        if a != b {
            if differences < 8 {
                println!("RATIONAL_LUT_A4_CODE_DIFF mode={tag} case={case} row={} col={} actual={a} expected={b}", i / oracle::WIDTH, i % oracle::WIDTH);
            }
            differences += 1;
        }
    }
    let scale_diff = scale_bits.iter().zip(scales).filter(|(a, b)| **a != b.to_bits()).count();
    println!("RATIONAL_LUT_A4_EXACT mode={tag} case={case} codes={} code_diff={differences} scales={} scale_diff={scale_diff}", expected.len(), scales.len());
    assert_eq!(differences, 0, "exact rational code gate");
    assert_eq!(scale_diff, 0, "separate rounded multiplication scale gate");
    (got, scale_bits)
}

async fn b16(d: &mut Device, profile: &npu_profile::Collector) -> Result<(), Box<dyn std::error::Error>> {
    let values = table_values();
    let table_bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table = HostTensor::<i32, m![op::Mantissa, op::Record]>::from_vec(values).to_hbm(&mut d.pdma).await?;
    assert_eq!(raw(d, &table).await, table_bytes, "all4096 table bytes initialized");
    let mut total_codes = 0usize;
    let mut total_scales = 0usize;
    let mut immutable_bytes = 0usize;
    for case in 0..oracle::CASES {
        let bits = oracle::fixture(case);
        let input_bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (expect, scales) = expected(&bits);
        let x = HostTensor::<bf16, m![B, H]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
        let mut q = HostTensor::<i32, m![B, H]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32, oracle::ROWS * oracle::WIDTH)).to_hbm(&mut d.pdma).await?;
        let mut s = HostTensor::<f32, m![B]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001), oracle::ROWS)).to_hbm(&mut d.pdma).await?;
        launch(op::quantize_row, (&mut *d, &x, &table, &mut q, &mut s)).await?;
        let qr = raw(d, &q).await;
        let sr = raw(d, &s).await;
        verify("b16", case, qr, sr, &expect, &scales);
        total_codes += expect.len(); total_scales += scales.len();
        for repeat in 0..2 {
            let start = std::time::Instant::now();
            for _ in 0..20 { launch(op::quantize_row, (&mut *d, &x, &table, &mut q, &mut s)).await?; }
            println!("RATIONAL_LUT_A4_TIME mode=b16 case={case} repeat={repeat} samples=20 us={:.6} scope=whole_I32_codes_and_scale_no_pack_H15360_or_model", start.elapsed().as_secs_f64() * 50000.0);
        }
        if case == 12 {
            println!("PROFILE_RATIONAL_LUT_A4_B16"); profile.enable(true);
            launch(op::quantize_row, (&mut *d, &x, &table, &mut q, &mut s)).await?;
            profile.enable(false);
        }
        assert_eq!(raw(d, &x).await, input_bytes, "input immutable after all timed/profile calls");
        assert_eq!(raw(d, &table).await, table_bytes, "full table immutable after all calls");
        immutable_bytes += input_bytes.len() + table_bytes.len();
    }
    assert_eq!((total_codes, total_scales, immutable_bytes), (798720, 208, 1650688));
    println!("PASS_RATIONAL_LUT_A4_B16 codes={total_codes} scales={total_scales} immutable_bytes={immutable_bytes} contract=finite_BF16_FTZ_exact_rational_ties_away no_pack_hidden_quant_or_model_claim");
    Ok(())
}

async fn full512(d: &mut Device, profile: &npu_profile::Collector) -> Result<(), Box<dyn std::error::Error>> {
    let values = table_values();
    let table_bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let candidate_table = HostTensor::<i32, m![compact::Mantissa, compact::Record]>::from_vec(values.iter().copied()).to_hbm(&mut d.pdma).await?;
    let original_table = HostTensor::<i32, m![original::Mantissa, original::Record]>::from_vec(values).to_hbm(&mut d.pdma).await?;
    assert_eq!(raw(d, &candidate_table).await, table_bytes);
    assert_eq!(raw(d, &original_table).await, table_bytes);
    let mut saved: Vec<(Vec<i32>, Vec<u32>)> = Vec::new();
    let mut codes = 0usize;
    let mut scale_count = 0usize;
    let mut immutable_bytes = 0usize;
    let mut permutation_codes = 0usize;
    let mut permutation_scales = 0usize;
    for case in 0..p512::CASES {
        let bits = p512::fixture(case);
        let input_bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (expect, scales) = expected(&bits);
        let x = HostTensor::<bf16, m![P512, H]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
        let mut cq = HostTensor::<i32, m![P512, H]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32, p512::ROWS * p512::WIDTH)).to_hbm(&mut d.pdma).await?;
        let mut cs = HostTensor::<f32, m![P512]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001), p512::ROWS)).to_hbm(&mut d.pdma).await?;
        let mut oq = HostTensor::<i32, m![P512, H]>::from_vec(std::iter::repeat_n(0x216b45d3i32, p512::ROWS * p512::WIDTH)).to_hbm(&mut d.pdma).await?;
        let mut os = HostTensor::<f32, m![P512]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc20002), p512::ROWS)).to_hbm(&mut d.pdma).await?;
        // Candidate is consumed before the separately poisoned original runs.
        launch(compact::quantize_row, (&mut *d, &x, &candidate_table, &mut cq, &mut cs)).await?;
        let qr = raw(d, &cq).await;
        let sr = raw(d, &cs).await;
        let candidate = verify("p512_compact", case, qr, sr, &expect, &scales);
        launch(original::quantize_row, (&mut *d, &x, &original_table, &mut oq, &mut os)).await?;
        let qr = raw(d, &oq).await;
        let sr = raw(d, &os).await;
        let old = verify("p512_original", case, qr, sr, &expect, &scales);
        assert_eq!(candidate, old, "all code/scale bits original versus compact");
        println!("RATIONAL_LUT_A4_PAIRED case={case} codes={} scales={}", expect.len(), scales.len());
        if case >= 2 {
            let source = &saved[case - 2];
            for row in 0..p512::ROWS {
                let source_row = p512::source_coordinate(case, row, 0).0;
                assert_eq!(candidate.1[row], source.1[source_row], "all512 scale permutation");
                for col in 0..p512::WIDTH {
                    let (_, source_col) = p512::source_coordinate(case, row, col);
                    assert_eq!(candidate.0[row * p512::WIDTH + col], source.0[source_row * p512::WIDTH + source_col], "full device code permutation");
                }
            }
            permutation_codes += expect.len(); permutation_scales += scales.len();
        }
        saved.push(candidate);
        codes += expect.len(); scale_count += scales.len();
        for (block, which) in ["original", "compact", "compact", "original"].into_iter().enumerate() {
            let start = std::time::Instant::now();
            for _ in 0..20 {
                if which == "original" { launch(original::quantize_row, (&mut *d, &x, &original_table, &mut oq, &mut os)).await?; }
                else { launch(compact::quantize_row, (&mut *d, &x, &candidate_table, &mut cq, &mut cs)).await?; }
            }
            println!("RATIONAL_LUT_A4_TIME mode=p512 variant={which} case={case} block={block} samples=20 us={:.6} scope=whole_I32_codes_and_scale_no_pack_H15360_or_model", start.elapsed().as_secs_f64() * 50000.0);
        }
        if case == 3 {
            println!("PROFILE_RATIONAL_LUT_A4_P512_ORIGINAL"); profile.enable(true);
            launch(original::quantize_row, (&mut *d, &x, &original_table, &mut oq, &mut os)).await?;
            profile.enable(false);
            println!("PROFILE_RATIONAL_LUT_A4_P512_COMPACT"); profile.enable(true);
            launch(compact::quantize_row, (&mut *d, &x, &candidate_table, &mut cq, &mut cs)).await?;
            profile.enable(false);
        }
        assert_eq!(raw(d, &x).await, input_bytes, "input immutable after all calls");
        assert_eq!(raw(d, &candidate_table).await, table_bytes, "candidate table immutable after all calls");
        assert_eq!(raw(d, &original_table).await, table_bytes, "original table immutable after all calls");
        immutable_bytes += input_bytes.len() + 2 * table_bytes.len();
    }
    assert_eq!((codes, scale_count, immutable_bytes), (7864320, 2048, 15761408));
    assert_eq!((permutation_codes, permutation_scales), (3932160, 1024));
    println!("PASS_RATIONAL_LUT_A4_P512 candidate_codes={codes} original_codes={codes} candidate_scales={scale_count} original_scales={scale_count} paired_codes={codes} paired_scales={scale_count} permutation_codes={permutation_codes} permutation_scales={permutation_scales} immutable_bytes={immutable_bytes} no_pack_hidden_quant_or_model_claim");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if !["b16", "p512", "all"].contains(&mode.as_str()) { return Err("expected b16, p512 or all".into()); }
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    let mut d = Device::new(op::quantize_row.topology())?;
    if mode != "p512" { b16(&mut d, &profile).await?; }
    if mode != "b16" { full512(&mut d, &profile).await?; }
    println!("PASS_RATIONAL_LUT_A4_REQUEST mode={mode} no_model_quality_or_complete_FFN_claim");
    Ok(())

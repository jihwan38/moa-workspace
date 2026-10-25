//! Strict512×15360 rational quantizer foundation; no packing, FFN or model claim.
use furiosa_opt_gemma4::{Chip, axes::{P512, L}, ops_int4_rational_lut_hidden as op};
use op::{Mantissa, Record};
use furiosa_opt_std::prelude::*;
#[path = "support/npu_profile.rs"] mod npu_profile;
#[path = "support/rational_a4_oracle.rs"] mod oracle;
#[path = "support/rational_a4_hidden_fixture.rs"] mod fixture;

async fn raw<D: MaterializableScalar, E: M>(d: &mut Device, x: &HbmTensor<D, Chip, E>) -> Vec<u8> {
    x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile = npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    let mut d = Device::new(op::quantize_row.topology())?;
    let values: Vec<i32> = (128u32..256).flat_map(|m| {
        let u = 7 * ((1u32 << 20).div_ceil(m));
        std::array::from_fn::<_, 8, _>(|lane| if lane == 0 { (u as f32).to_bits() as i32 }
            else { (0x5a3c0000u32 | (m << 4) | lane as u32) as i32 })
    }).collect();
    let table_bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let table = HostTensor::<i32, m![Mantissa, Record]>::from_vec(values).to_hbm(&mut d.pdma).await?;
    assert_eq!(raw(&mut d, &table).await, table_bytes, "all4096 mathematical/sentinel table bytes initialized");
    let mut saved: Vec<(Vec<i32>, Vec<u32>)> = Vec::new();
    let (mut total_codes, mut total_scales, mut immutable_bytes) = (0usize, 0usize, 0usize);
    let (mut permutation_codes, mut permutation_scales) = (0usize, 0usize);
    let mut post_reuse_output_bytes = 0usize;
    for case in 0..fixture::CASES {
        let bits = fixture::fixture(case);
        assert_eq!(bits.len(), fixture::ROWS * fixture::WIDTH);
        let input_bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut expected = Vec::with_capacity(bits.len());
        let mut scales = Vec::with_capacity(fixture::ROWS);
        // The retained quotient/remainder oracle consumes each full15360 row.
        for row in bits.chunks_exact(fixture::WIDTH) {
            let (q, s) = oracle::reference(row); expected.extend(q); scales.push(s);
        }
        assert_eq!(scales.len(), fixture::ROWS);
        let x = HostTensor::<bf16, m![P512, L]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
        let mut q = HostTensor::<i32, m![P512, L]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32, fixture::ROWS * fixture::WIDTH)).to_hbm(&mut d.pdma).await?;
        let mut scale = HostTensor::<f32, m![P512]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001), fixture::ROWS)).to_hbm(&mut d.pdma).await?;
        launch(op::quantize_row, (&mut d, &x, &table, &mut q, &mut scale)).await?;
        let code_bytes = raw(&mut d, &q).await;
        let scale_bytes = raw(&mut d, &scale).await;
        assert_eq!(code_bytes.len(), fixture::ROWS * fixture::WIDTH * 4);
        assert_eq!(scale_bytes.len(), fixture::ROWS * 4);
        let codes: Vec<i32> = code_bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        let scale_bits: Vec<u32> = scale_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut differences = 0usize;
        for (i, (&actual, &reference)) in codes.iter().zip(&expected).enumerate() {
            if actual != reference {
                if differences < 8 { println!("RATIONAL_LUT_HIDDEN_DIFF case={case} row={} col={} actual={actual} expected={reference}", i / fixture::WIDTH, i % fixture::WIDTH); }
                differences += 1;
            }
        }
        let scale_diff = scale_bits.iter().zip(&scales).filter(|(a, b)| **a != b.to_bits()).count();
        println!("RATIONAL_LUT_HIDDEN_EXACT case={case} rows={} width={} codes={} code_diff={differences} scales={} scale_diff={scale_diff}", fixture::ROWS, fixture::WIDTH, expected.len(), scales.len());
        assert_eq!(differences, 0, "all full15360-width exact rational codes");
        assert_eq!(scale_diff, 0, "all512 reconstruction scale bits");
        if case >= 2 {
            let source = &saved[case - 2];
            for row in 0..fixture::ROWS {
                let sr = fixture::source_coordinate(case, row, 0).0;
                assert_eq!(scale_bits[row], source.1[sr]);
                for col in 0..fixture::WIDTH {
                    let sc = fixture::source_coordinate(case, row, col).1;
                    assert_eq!(codes[row * fixture::WIDTH + col], source.0[sr * fixture::WIDTH + sc]);
                }
            }
            permutation_codes += expected.len(); permutation_scales += scales.len();
        } else { saved.push((codes, scale_bits)); }
        total_codes += expected.len(); total_scales += scales.len();
        for repeat in 0..2 {
            let start = std::time::Instant::now();
            for _ in 0..20 { launch(op::quantize_row, (&mut d, &x, &table, &mut q, &mut scale)).await?; }
            println!("RATIONAL_LUT_HIDDEN_TIME case={case} repeat={repeat} samples=20 us={:.6} scope=whole_P512_L15360_I32_codes_and_scales_no_pack_consumer_FFN_model", start.elapsed().as_secs_f64() * 50000.0);
        }
        if case == 3 {
            println!("PROFILE_RATIONAL_LUT_HIDDEN"); profile.enable(true);
            launch(op::quantize_row, (&mut d, &x, &table, &mut q, &mut scale)).await?;
            profile.enable(false);
        }
        // The first bytes already passed the independent CPU code/scale oracle.
        // Recheck the whole output after every timed/profile call has completed.
        assert_eq!(raw(&mut d, &q).await, code_bytes, "all code bytes stable after repeated use");
        assert_eq!(raw(&mut d, &scale).await, scale_bytes, "all scale bytes stable after repeated use");
        let checked_output_bytes = code_bytes.len() + scale_bytes.len();
        post_reuse_output_bytes += checked_output_bytes;
        println!("RATIONAL_LUT_HIDDEN_POST_REUSE case={case} output_bytes={checked_output_bytes}");
        assert_eq!(raw(&mut d, &x).await, input_bytes, "entire15MiB input immutable after all calls");
        assert_eq!(raw(&mut d, &table).await, table_bytes, "entire table immutable after all calls");
        immutable_bytes += input_bytes.len() + table_bytes.len();
    }
    assert_eq!((total_codes, total_scales), (31457280, 2048));
    assert_eq!((permutation_codes, permutation_scales), (15728640, 1024));
    assert_eq!(immutable_bytes, 62930944);
    assert_eq!(post_reuse_output_bytes, 125837312);
    println!("PASS_RATIONAL_LUT_HIDDEN codes={total_codes} scales={total_scales} permutation_codes={permutation_codes} permutation_scales={permutation_scales} immutable_bytes={immutable_bytes} post_reuse_output_bytes={post_reuse_output_bytes} rows=512 width=15360 no_pack_consumer_FFN_or_model_claim");
    Ok(())
}

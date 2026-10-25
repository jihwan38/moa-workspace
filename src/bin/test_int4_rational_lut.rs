//! Strict exact-rational A4 code/scale foundation, not a model-quality test.
use furiosa_opt_gemma4::{Chip, axes::{B,H}, ops_int4_rational_lut as op};
use op::{Mantissa, Record};
use furiosa_opt_std::prelude::*;
#[path="support/npu_profile.rs"] mod npu_profile;
#[path="support/rational_a4_oracle.rs"] mod oracle;

async fn raw<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8> {
    x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()
}

#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
    let profile=npu_profile::Collector::default();
    tracing::subscriber::set_global_default(profile.clone())?;
    let mut d=Device::new(op::quantize_row.topology())?;
    let table_values:Vec<i32>=(128u32..256).flat_map(|m| {
        let u=7*((1u32<<20).div_ceil(m));
        [(u as f32).to_bits() as i32,0,0,0,0,0,0,0]
    }).collect();
    let odds=HostTensor::<i32,m![Mantissa,Record]>::from_vec(table_values).to_hbm(&mut d.pdma).await?;
    let odd_bytes=raw(&mut d,&odds).await;
    let mut total_codes=0usize;
    let mut total_scales=0usize;
    for case in 0..oracle::CASES {
        let bits=oracle::fixture(case);
        let input_bytes:Vec<u8>=bits.iter().flat_map(|x|x.to_le_bytes()).collect();
        let mut expected=Vec::new();
        let mut scales=Vec::new();
        for row in bits.chunks_exact(oracle::WIDTH) {
            let(q,s)=oracle::reference(row);expected.extend(q);scales.push(s);
        }
        let x=HostTensor::<bf16,m![B,H]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
        let mut q=HostTensor::<i32,m![B,H]>::from_vec(std::iter::repeat_n(0x5a3c17e9i32,oracle::ROWS*oracle::WIDTH)).to_hbm(&mut d.pdma).await?;
        let mut s=HostTensor::<f32,m![B]>::from_vec(std::iter::repeat_n(f32::from_bits(0x7fc10001),oracle::ROWS)).to_hbm(&mut d.pdma).await?;
        launch(op::quantize_row,(&mut d,&x,&odds,&mut q,&mut s)).await?;
        let qr=raw(&mut d,&q).await;
        let sr=raw(&mut d,&s).await;
        assert_eq!(qr.len(),expected.len()*4);
        assert_eq!(sr.len(),oracle::ROWS*4);
        let got:Vec<i32>=qr.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut differences=0;
        for(i,(&a,&b))in got.iter().zip(&expected).enumerate() {
            if a!=b {
                if differences<8 {println!("RATIONAL_LUT_A4_CODE_DIFF case={case} row={} col={} actual={a} expected={b}",i/oracle::WIDTH,i%oracle::WIDTH);}
                differences+=1;
            }
        }
        let scale_diff=sr.chunks_exact(4).zip(&scales).filter(|(b,s)|u32::from_le_bytes((*b).try_into().unwrap())!=s.to_bits()).count();
        println!("RATIONAL_LUT_A4_EXACT case={case} codes={} code_diff={differences} scales={} scale_diff={scale_diff}",expected.len(),scales.len());
        assert_eq!(differences,0,"exact rational code gate");
        assert_eq!(scale_diff,0,"separate rounded multiplication scale gate");
        assert_eq!(raw(&mut d,&x).await,input_bytes,"immutable input");
        assert_eq!(raw(&mut d,&odds).await,odd_bytes,"immutable generated reciprocal table");
        total_codes+=expected.len();total_scales+=scales.len();
        for repeat in 0..2 {
            let start=std::time::Instant::now();
            for _ in 0..20 {launch(op::quantize_row,(&mut d,&x,&odds,&mut q,&mut s)).await?;}
            println!("RATIONAL_LUT_A4_TIME case={case} repeat={repeat} samples=20 us={:.6} scope=whole_I32_codes_and_scale_no_pack_or_consumer",start.elapsed().as_secs_f64()*50000.0);
        }
        if case==12 {
            println!("PROFILE_RATIONAL_LUT_A4");profile.enable(true);
            launch(op::quantize_row,(&mut d,&x,&odds,&mut q,&mut s)).await?;
            profile.enable(false);
        }
    }
    println!("PASS_RATIONAL_LUT_A4 codes={total_codes} scales={total_scales} contract=finite_BF16_FTZ_exact_rational_ties_away no_model_quality_or_complete_FFN_claim");
    Ok(())
}

use furiosa_opt_gemma4::{axes::*, host::fixed_e4_codec as cpu, ops_fixed_e4_roundtrip_probe as ops};
use furiosa_opt_std::prelude::*;
use std::collections::BTreeSet;

fn codes(n: usize, salt: usize, poison: bool) -> Vec<u8> {
    (0..n).map(|i| {
        let c = ((i % 256 * 73 + i / 256 * 17 + i / 2048 * 29 + salt) % 256) as u8;
        if !poison && c & 127 == 127 { c & 128 } else { c }
    }).collect()
}

fn exact(label: &str, actual: &[u8], expected: &[u8]) -> usize {
    assert_eq!(actual.len(), expected.len());
    let different = actual.iter().zip(expected).filter(|(a,b)| a != b).count();
    assert_eq!(different, 0, "{label}: first={:?}", actual.iter().zip(expected).position(|(a,b)|a!=b));
    actual.len()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = Device::new(ops::b16.topology())?;
    let mut checked = 0usize;
    macro_rules! gate {
        ($axis:ident, $kernel:path, $salt:expr) => {{
            type Shape = m![$axis,Ns,Ds];
            let input = codes(Shape::SIZE, $salt, false);
            let distinct: BTreeSet<_> = input.iter().copied().collect();
            assert_eq!(distinct.len(), 254);
            for c in [0,128,126,254] { assert!(distinct.contains(&c)); }
            let expected = cpu::decode_bytes(&input);
            assert_eq!(cpu::encode_bytes(&expected, &cpu::table()), input);
            let x = HostTensor::<f8e4m3,Shape>::from_buf(input.clone()).to_hbm(&mut device.pdma).await?;
            let mut y = HostTensor::<bf16,Shape>::from_buf([0xc1,0x7f].repeat(Shape::SIZE)).to_hbm(&mut device.pdma).await?;
            let mut z = HostTensor::<f8e4m3,Shape>::from_buf(vec![127;Shape::SIZE]).to_hbm(&mut device.pdma).await?;
            launch($kernel, (&mut device,&x,&mut y,&mut z)).await?;
            checked += exact("decoded", &y.to_host::<Shape>(&mut device.pdma).await?.into_inner().into_buf(), &expected);
            checked += exact("recoded raw E4", &z.to_host::<Shape>(&mut device.pdma).await?.into_inner().into_buf(), &input);
            checked += exact("input immutable", &x.to_host::<Shape>(&mut device.pdma).await?.into_inner().into_buf(), &input);
            println!("STORAGE_CODEC_CASE_PASS shape={} finite_codes=254 values={} checked_bytes={checked}", stringify!($axis),Shape::SIZE);
        }};
    }
    gate!(B, ops::b16, 3);
    gate!(R64, ops::r64, 17);
    gate!(P128, ops::p128, 91);
    type Shape = m![B,Ns,Ds];
    let input = codes(Shape::SIZE, 5, true);
    let x = HostTensor::<f8e4m3,Shape>::from_buf(input.clone()).to_hbm(&mut device.pdma).await?;
    let mut y = HostTensor::<bf16,Shape>::from_buf(vec![0;Shape::SIZE*2]).to_hbm(&mut device.pdma).await?;
    launch(ops::read_poison, (&mut device,&x,&mut y)).await?;
    let decoded = y.to_host::<Shape>(&mut device.pdma).await?.into_inner().into_buf();
    let mut poison_bits = BTreeSet::new();
    let mut poison_count = 0usize;
    for (c, pair) in input.iter().zip(decoded.chunks_exact(2)) {
        let bits = u16::from_le_bytes([pair[0],pair[1]]);
        if c & 127 == 127 {
            assert!(f32::from_bits((bits as u32)<<16).is_nan());
            poison_bits.insert((*c,bits));
            poison_count += 1;
        } else { assert_eq!(bits,cpu::decode(*c)); }
    }
    assert_eq!(poison_count,Shape::SIZE/128);
    checked += decoded.len();
    checked += exact("poison input immutable",&x.to_host::<Shape>(&mut device.pdma).await?.into_inner().into_buf(),&input);
    println!("STORAGE_CODEC_POISON_PASS values={} nan_coordinates={poison_count} nan_raw_to_bf16={poison_bits:?} finite_neighbors_exact=true reader_only=true",Shape::SIZE);
    println!("PASS FIXED_E4_STORAGE_CODEC finite_raw_idempotence=true signed_zero=true saturation448=true checked_bytes={checked} no_model_or_performance_claim=true");
    Ok(())
}

//! Standalone diagnostic SOURCE only until explicit device-build authorization.
use furiosa_opt_gemma4::{Chip, axes::{H,L,P512},ops_int4_grouped_activation_store as op};
use furiosa_opt_std::prelude::*;
use op::{Mantissa,Record};
#[path="support/int4_grouped_activation_fixture.rs"]mod fixture;
#[path="support/npu_profile.rs"]mod npu_profile;
async fn raw<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Vec<u8>{x.to_host::<E>(&mut d.pdma).await.unwrap().into_inner().into_buf()}
fn bytes_i32(xs:&[i32])->Vec<u8>{xs.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn u32s(bytes:&[u8])->Vec<u32>{bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect()}
fn verify(q:Vec<u8>,s:Vec<u8>,o:Vec<u8>,expected:&fixture::Expected)->fixture::Expected{
    assert_eq!(q,expected.packed,"all packed bytes; independent low/high-nibble oracle");
    let scales=u32s(&s);assert_eq!(scales,expected.scales,"all group scales RN32 bits");
    let offsets:Vec<i32>=u32s(&o).into_iter().map(|x|x as i32).collect();assert_eq!(offsets,expected.offsets,"all relative offsets");
    fixture::Expected{packed:q,scales,offsets}
}
#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
    let profile=npu_profile::Collector::default();tracing::subscriber::set_global_default(profile.clone())?;
    let mut d=Device::new(op::quantize_pack_input.topology())?;
    let values=fixture::table();let table_bytes=bytes_i32(&values);
    let table=HostTensor::<i32,m![Mantissa,Record]>::from_vec(values).to_hbm(&mut d.pdma).await?;
    assert_eq!(raw(&mut d,&table).await,table_bytes);
    let mut codes_count=0usize;let mut packed_bytes=0usize;let mut scale_count=0usize;let mut offset_count=0usize;
    macro_rules! run_width {($K:ident,$Packed:ty,$Scales:ty,$kernel:path)=>{{
        let width=$K::SIZE;let groups=width/960;
        let shifts_v:Vec<i32>=(0..8).map(|x|4*x).collect();let shift_bytes=bytes_i32(&shifts_v);
        let shifts=HostTensor::<i32,m![$K%8]>::from_vec(shifts_v).to_hbm(&mut d.pdma).await?;
        let mut output=HostTensor::<i32,$Packed>::from_vec(vec![0x35a719e2;512*width/8]).to_hbm(&mut d.pdma).await?;
        let mut scales=HostTensor::<f32,$Scales>::from_vec(vec![f32::from_bits(0x7fc12345);512*groups]).to_hbm(&mut d.pdma).await?;
        let mut offsets=HostTensor::<i32,$Scales>::from_vec(vec![-1234567;512*groups]).to_hbm(&mut d.pdma).await?;
        let mut saved=Vec::new();
        for case in 0..fixture::CASES {
            let bits=fixture::fixture(case,width);let expected=fixture::expected(&bits,width);
            let input_bytes:Vec<u8>=bits.iter().flat_map(|b|b.to_le_bytes()).collect();
            let input=HostTensor::<bf16,m![P512,$K]>::from_vec(bits.into_iter().map(bf16::from_bits)).to_hbm(&mut d.pdma).await?;
            // Distinct mutable owners; reuse across cases must overwrite every byte.
            launch($kernel,(&mut d,&input,&table,&shifts,&mut output,&mut scales,&mut offsets)).await?;
            let got=verify(raw(&mut d,&output).await,raw(&mut d,&scales).await,raw(&mut d,&offsets).await,&expected);
            if case>=3 {fixture::verify_permutation(case,width,&got,&saved[case-3]);}
            saved.push(got);
            for repeat in 0..2 {
                let start=std::time::Instant::now();
                for _ in 0..10 {launch($kernel,(&mut d,&input,&table,&shifts,&mut output,&mut scales,&mut offsets)).await?;}
                println!("GROUPED_A4_PACK_TIME width={width} case={case} repeat={repeat} samples=10 us={:.6} scope=complete_group_quant_offset_write_read_pack_scale_no_GEMM",start.elapsed().as_secs_f64()*100000.0);
            }
            if case==0 {println!("PROFILE_GROUPED_A4_PACK width={width}");profile.enable(true);launch($kernel,(&mut d,&input,&table,&shifts,&mut output,&mut scales,&mut offsets)).await?;profile.enable(false);}
            verify(raw(&mut d,&output).await,raw(&mut d,&scales).await,raw(&mut d,&offsets).await,&expected);
            assert_eq!(raw(&mut d,&input).await,input_bytes,"entire input immutable after timing/reuse");
            assert_eq!(raw(&mut d,&table).await,table_bytes,"entire padded immutable LUT");
            assert_eq!(raw(&mut d,&shifts).await,shift_bytes,"all shifts immutable");
            println!("GROUPED_A4_PACK_EXACT width={width} case={case} codes={} packed_bytes={} scales={} offsets={} postcheck=true",512*width,512*width/2,512*groups,512*groups);
            codes_count+=512*width;packed_bytes+=512*width/2;scale_count+=512*groups;offset_count+=512*groups;
        }
    }};}
    run_width!(H,op::PackedInput,op::InputScales,op::quantize_pack_input);
    run_width!(L,op::PackedHidden,op::HiddenScales,op::quantize_pack_hidden);
    assert_eq!((codes_count,packed_bytes,scale_count,offset_count),(49_152_000,24_576_000,51_200,51_200));
    println!("PASS_GROUPED_A4_PACK codes={codes_count} packed_bytes={packed_bytes} group_scales={scale_count} offsets={offset_count} no_GEMM_complete_FFN_model_quality_claim");
    Ok(())
}

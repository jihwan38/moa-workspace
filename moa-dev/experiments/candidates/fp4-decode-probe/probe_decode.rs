use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip,decode_probe};
use decode_probe::{DecodeK,DecodeR,DecodeS};
use std::io::Write;

// Independent explicit e2m1 value specification encoded as exact BF16 bits.
// Includes both signed zeros; no FP4 SDK conversion is used to form the oracle.
const ORACLE: [u16;16] = [0x0000,0x3f00,0x3f80,0x3fc0,0x4000,0x4040,0x4080,0x40c0,
                         0x8000,0xbf00,0xbf80,0xbfc0,0xc000,0xc040,0xc080,0xc0c0];
const NAMES: [&str;3] = ["constant_code_rows","all_ordered_nibble_pairs","packet_staggered"];

struct Case { id:usize, codes:Vec<u8>, packed:Vec<u8>, expected:Vec<u16> }
fn make_case(id:usize)->Case {
    let mut codes=Vec::with_capacity(DecodeR::SIZE*DecodeK::SIZE);
    for row in 0..DecodeR::SIZE {
        for k in 0..DecodeK::SIZE {
            let pair=k/2;
            let code=match id {
                0 => row%16,
                1 => if k%2==0 { (pair%16+row)%16 } else { (pair/16+3*row)%16 },
                2 => (k%32 + 11*(k/32) + 7*row)%16,
                _ => unreachable!(),
            };
            codes.push(code as u8);
        }
    }
    let packed:Vec<u8>=codes.chunks_exact(2).map(|p|p[0]|p[1]<<4).collect();
    let expected=codes.iter().map(|&code|ORACLE[code as usize]).collect();
    let mut code_mask=0u32;let mut nibble_masks=[0u32;2];let mut lane_masks=[0u32;64];let mut pairs=[false;256];
    for (i,&code) in codes.iter().enumerate() {
        assert_eq!((packed[i/2] >> (4*(i%2))) & 15,code);
        code_mask|=1<<code;nibble_masks[i%2]|=1<<code;lane_masks[i%64]|=1<<code;
        pairs[packed[i/2] as usize]=true;
        // Cross-check SDK host decoding against the independent table, including signed zero.
        assert_eq!(bf16::from_f32(f4e2m1::from_bits(code).to_f32()).to_bits(),ORACLE[code as usize]);
    }
    assert_eq!(code_mask,0xffff);assert_eq!(nibble_masks,[0xffff;2]);
    if id==1 { assert!(pairs.iter().all(|&b|b));assert!(lane_masks.iter().all(|&m|m==0xffff)); }
    println!("CPU_CASE case={id} name={} elements={} packed_bytes={} codes=16 nibble_phases=2 ordered_pairs={} all_codes_at_each_mod64_position={}",NAMES[id],codes.len(),packed.len(),pairs.iter().filter(|&&b|b).count(),lane_masks.iter().all(|&m|m==0xffff));
    Case{id,codes,packed,expected}
}
fn mark(mode:&str,id:usize,stage:&str) {
    println!("DECODE_PROBE mode={mode} case={id} {stage}");std::io::stdout().flush().unwrap();
}
fn check(mode:&str,c:&Case,actual:&[bf16])->usize {
    let local=c.expected.len();assert_eq!(actual.len(),DecodeS::SIZE*local);
    let mut failures=0usize;let mut code_fail=[0usize;16];let mut nibble_fail=[0usize;2];let mut lane_fail=[0usize;64];
    let mut nonfinite=0usize;let mut signed_zero_only=0usize;let mut samples=[[None;16];2];
    for (index,&value) in actual.iter().enumerate() {
        let i=index%local;let code=c.codes[i] as usize;let got=value.to_bits();let want=c.expected[i];
        if samples[i%2][code].is_none() {samples[i%2][code]=Some(got);}
        if got==want {continue;}
        if failures<32 {
            let neighbor=c.codes[i^1];
            println!("DECODE_MISMATCH mode={mode} case={} slice={} row={} k={} nibble={} code=0x{code:x} neighbor=0x{neighbor:x} packed=0x{:02x} mod64={} expected=0x{want:04x} actual=0x{got:04x}",c.id,index/local,i/DecodeK::SIZE,i%DecodeK::SIZE,i%2,c.packed[i/2],i%64);
        }
        failures+=1;code_fail[code]+=1;nibble_fail[i%2]+=1;lane_fail[i%64]+=1;
        if got&0x7f80==0x7f80 {nonfinite+=1;}
        if got&0x7fff==0 && want&0x7fff==0 {signed_zero_only+=1;}
    }
    for (phase,values) in samples.iter().enumerate() {
        let hex:Vec<_>=values.iter().map(|v|v.map_or("missing".into(),|v|format!("0x{v:04x}"))).collect();
        println!("DECODE_SAMPLES mode={mode} case={} nibble={phase} by_code={hex:?}",c.id);
    }
    println!("DECODE_COUNTS mode={mode} case={} by_code={code_fail:?} by_nibble={nibble_fail:?} by_mod64={lane_fail:?}",c.id);
    println!("DECODE_RESULT mode={mode} case={} checked={} mismatches={failures} nonfinite={nonfinite} signed_zero_only={signed_zero_only} status={}",c.id,actual.len(),if failures==0 {"PASS"} else {"FAIL"});
    failures
}

#[tokio::main]
async fn main() {
    let cases:Vec<_>=(0..3).map(make_case).collect();
    if std::env::args().any(|a|a=="--oracle-only") {
        println!("CPU_ORACLE_PASS variants=2 cases=3; no Device created, no NPU launch");return;
    }
    let mut device=Device::new(decode_probe::legacy_decode.topology()).expect("create NPU Device");
    let mut total_failures=0usize;
    for c in cases {
        let input:HbmTensor<f4e2m1,Chip,m![DecodeR,DecodeK]>=HostTensor::<f4e2m1,m![DecodeR,DecodeK]>::from_buf(c.packed.clone())
            .to_hbm(&mut device.pdma).await.unwrap();
        let copied:HostTensor<f4e2m1,m![DecodeR,DecodeK]>=input.to_host(&mut device.pdma).await.unwrap();
        assert_eq!(copied.into_vec().into_iter().map(f4e2m1::to_bits).collect::<Vec<_>>(),c.codes,"packed HBM input round-trip");
        println!("INPUT_ROUNDTRIP case={} elements={} status=PASS",c.id,c.codes.len());
        let mut legacy:HbmTensor<bf16,Chip,m![DecodeS,DecodeR,DecodeK]>=HostTensor::<bf16,m![DecodeS,DecodeR,DecodeK]>::from_vec(vec![bf16::from_bits(0x7fc1);<m![DecodeS,DecodeR,DecodeK]>::SIZE])
            .to_hbm(&mut device.pdma).await.unwrap();
        mark("legacy",c.id,"LAUNCH_BEGIN");
        launch(decode_probe::legacy_decode,(&mut device,&input,&mut legacy)).await.expect("legacy decode launch");
        mark("legacy",c.id,"LAUNCH_RETURNED");
        let a:HostTensor<bf16,m![DecodeS,DecodeR,DecodeK]>=legacy.to_host(&mut device.pdma).await.unwrap();
        total_failures+=check("legacy",&c,&a.into_vec());
        let mut fused:HbmTensor<bf16,Chip,m![DecodeS,DecodeR,DecodeK]>=HostTensor::<bf16,m![DecodeS,DecodeR,DecodeK]>::from_vec(vec![bf16::from_bits(0x7fc1);<m![DecodeS,DecodeR,DecodeK]>::SIZE])
            .to_hbm(&mut device.pdma).await.unwrap();
        mark("fused",c.id,"LAUNCH_BEGIN");
        launch(decode_probe::fused_decode,(&mut device,&input,&mut fused)).await.expect("fused decode launch");
        mark("fused",c.id,"LAUNCH_RETURNED");
        let b:HostTensor<bf16,m![DecodeS,DecodeR,DecodeK]>=fused.to_host(&mut device.pdma).await.unwrap();
        total_failures+=check("fused",&c,&b.into_vec());
    }
    println!("FP4_DECODE_PROBE variants=2 cases=3 total_mismatches={total_failures} status={}",if total_failures==0 {"PASS"} else {"FAIL"});
    if total_failures!=0 {std::process::exit(1);}
}

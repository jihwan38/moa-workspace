use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::{H, L}, stream_probe};

const FP4: [f64; 16] = [0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.];
const EDGES: [usize; 19] = [0,15,16,31,32,63,64,479,480,1919,1920,7679,7680,7695,7696,9599,9600,15358,15359];
const ATOL: f32 = 0.01;
const RTOL: f32 = 1e-2;
fn activation(case: usize, k: usize) -> f64 {
    let c=k/7680; let j=k%7680;
    match case {
        1 => if EDGES.contains(&k) { (1+(k%7)) as f64 / 16. * if c==0 {1.} else {8.} } else {0.},
        2 => match (k/16%2,k%16) {(0,0)=>1.,(0,1)=>15./256.,(1,0)=>-1.,(1,1)=>-14./256.,_=>0.},
        3 => 0.,
        // Large amax values have zero weights; tiny contributions are isolated.
        4 => if j==0 {if c==0 {128.} else {0.5}} else {1./16777216.},
        5 => if j==0 {if c==0 {1.} else {0.5}} else {1./4194304.},
        6 => match (c,j%16) {(0,0)=>1.,(0,1)=>15./256.,(1,0)=>-1.,(1,1)=>-14./256.,_=>0.},
        7 => if j==0 {if c==0 {128.} else {0.5}} else { ((j*3%9) as f64-4.)/64. },
        _ => ((k*7%17) as f64-8.)/32. * if c==0 {1.} else {8.},
    }
}
fn code(case: usize, row: usize, k: usize) -> u8 {
    match case {
        1 => if k==EDGES[row%EDGES.len()] {[1,2,3,4,5,6,7][row/EDGES.len()%7]} else {0},
        2|6 => [1,2,4,6][row%4],
        4|5 => if k%7680==0 {0} else {7},
        7 => if k%7680==0 {0} else {[1,2,3,4,5,6,7][(row+k)%7]},
        _ => {let pair=k/2;let block=k/512;if k%2==0 {((pair%16+((row>>((block%4)*4))&15))%16) as u8} else {((pair/16+13*row+7*block)%16) as u8}},
    }
}
fn scale(case: usize, row: usize, block: usize) -> f64 {
    match case {2|6=>1.,4=>256.,5=>1.,_=>[0.25,0.5,1.,2.][(row*3+row/60+block)%4]}
}
fn fingerprint(bytes: impl IntoIterator<Item=u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325,|h,b|(h^u64::from(b)).wrapping_mul(0x100000001b3))
}
fn exact_f32(value: f64) -> f64 {assert_eq!(value,(value as f32) as f64,"reference intermediate is not F32-exact");value}
fn inputs(case: usize, x: &[bf16]) -> (Vec<u8>,Vec<f8e4m3>,Vec<bf16>) {
    assert_eq!(x.len(),L::SIZE);
    let mut packed=vec![0u8;H::SIZE*L::SIZE/2];
    let mut scales=Vec::with_capacity(H::SIZE*L::SIZE/16);
    let mut expected=Vec::with_capacity(H::SIZE);let mut mask=0u32;
    for row in 0..H::SIZE {
        let mut clusters=[0f64;2];
        for (c,cluster) in clusters.iter_mut().enumerate() {
            let mut columns=[0f64;4];
            for (q,column) in columns.iter_mut().enumerate() {
                for local_block in 0..120 {
                    let block=c*480+q*120+local_block;let factor=scale(case,row,block);
                    let encoded=f8e4m3::from_f32(factor as f32);assert_eq!(encoded.to_f32() as f64,factor);scales.push(encoded);
                    let mut dot=0f64;
                    for j in 0..16 {
                        let k=block*16+j;let code=code(case,row,k);
                        packed[(row*L::SIZE+k)/2]|=code<<(4*(k%2));mask|=1<<code;
                        // Reference uses the actual original BF16 vector, never an E5 reconstruction.
                        dot=exact_f32(dot+FP4[code as usize]*f64::from(x[k].to_f32()));
                    }
                    *column=exact_f32(*column+exact_f32(dot*factor));
                }
                *cluster=exact_f32(*cluster+*column);
            }
        }
        let sum=exact_f32(clusters[0]+clusters[1]);expected.push(bf16::from_f32(sum as f32));
    }
    if case==0||case==3 {assert_eq!(mask,0xffff);}
    if case==4 {assert!(expected.iter().all(|v|v.to_f32()>1.),"tiny contribution must not be hidden by atol");}
    let wp=fingerprint(packed.iter().copied());let ws=fingerprint(scales.iter().flat_map(|x|x.to_f32().to_bits().to_le_bytes()));
    let y=fingerprint(expected.iter().flat_map(|x|x.to_bits().to_le_bytes()));
    println!("E133_ORACLE case={case} rows=3840 width=15360 code_mask={mask:x} packed={wp:016x} scales={ws:016x} expected={y:016x}");
    (packed,scales,expected)
}

// The unchanged standard compare body is inserted verbatim below by the source preparer.
fn compare(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> bool {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: shape mismatch ({} expected vs {} from device)",
        expected.len(),
        actual.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_diff = 0.0f64;
    let mut within = 0usize;

    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if !got.is_finite() {
            println!("[{label:34}] FAIL -- non-finite device output at {index}");
            return false;
        }
        let diff = (want - got).abs();
        if diff <= atol + rtol * want.abs() {
            within += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
        sum_diff += f64::from(diff);
    }

    let count = expected.len();
    let ok = within == count;
    let relative = if expected[max_index].abs() > 1e-12 {
        max_diff / expected[max_index].abs() * 100.0
    } else {
        0.0
    };
    println!(
        "[{label:34}] max|Δ|={max_diff:9.5} ({relative:7.2}% of expected)  mean|Δ|={:9.6}  \
         within tol={:6.2}%  -> {}",
        sum_diff / count as f64,
        within as f32 / count as f32 * 100.0,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

#[tokio::main]
async fn main() {
    let oracle_only=std::env::args().any(|x|x=="--oracle-only");
    let mut device=if oracle_only {None} else {Some(Device::new(stream_probe::down.topology()).unwrap())};
    let mut required_failures=0usize;let mut adversarial_failures=0usize;
    for case in 0..8 {
        let xv:Vec<bf16>=(0..L::SIZE).map(|k|bf16::from_f32(activation(case,k) as f32)).collect();
        for (k,x) in xv.iter().enumerate() {assert_eq!(x.to_f32() as f64,activation(case,k));}
        let hash=fingerprint(xv.iter().flat_map(|x|x.to_bits().to_le_bytes()));
        println!("E133_INPUT case={case} x={hash:016x} count=15360");
        let (wp,ws,expected)=inputs(case,&xv);
        if let Some(d)=device.as_mut() {
            let x:HbmTensor<bf16,Chip,m![L]>=HostTensor::<bf16,m![L]>::from_vec(xv).to_hbm(&mut d.pdma).await.unwrap();
            let w:HbmTensor<f4e2m1,Chip,m![H,L]>=HostTensor::<f4e2m1,m![H,L]>::from_buf(wp).to_hbm(&mut d.pdma).await.unwrap();
            let s:HbmTensor<f8e4m3,Chip,m![H,L/16]>=HostTensor::<f8e4m3,m![H,L/16]>::from_vec(ws).to_hbm(&mut d.pdma).await.unwrap();
            let mut y:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec(vec![bf16::from_bits(0x7fc1);H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
            launch(stream_probe::down,(&mut *d,&x,&w,&s,&mut y)).await.unwrap();
            let y:HostTensor<bf16,m![H]>=y.to_host(&mut d.pdma).await.unwrap();let actual=y.into_vec();
            let want:Vec<f32>=expected.iter().map(|v|v.to_f32()).collect();let got:Vec<f32>=actual.iter().map(|v|v.to_f32()).collect();
            let tolerance_pass=compare(&format!("down boundary case{case}"),&want,&got,ATOL,RTOL);
            let exact_mismatches=got.iter().zip(&want).filter(|(a,e)|!a.is_finite()||a!=e).count();
            let mut max_abs=0f32;for (a,e) in got.iter().zip(&want) {max_abs=max_abs.max((a-e).abs());}
            let role=if case==4 {"adversarial"} else {"mapping"};
            if case==4 {adversarial_failures+=usize::from(!tolerance_pass);required_failures+=got.iter().filter(|x|!x.is_finite()).count();} else {required_failures+=exact_mismatches;}
            println!("E133_CASE case={case} role={role} checked=3840 exact_mismatches={exact_mismatches} tolerance_pass={tolerance_pass} max_abs={max_abs:?} first_actual={:?} first_expected={:?}",got[0],want[0]);
        }
    }
    println!("E133_PROBE_COMPLETE oracle_only={oracle_only} cases=8 checked={} required_failures={required_failures} adversarial_tolerance_failures={adversarial_failures}",if oracle_only {0} else {8*H::SIZE});
    if required_failures!=0 {std::process::exit(1);}
}

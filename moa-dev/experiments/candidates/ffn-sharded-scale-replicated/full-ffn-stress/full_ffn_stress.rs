use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::{Chip, axes::{H, L}, ops};

fn checksum(bytes: impl IntoIterator<Item=u8>) -> u64 {
    bytes.into_iter().fold(0xcbf29ce484222325, |h,b| (h^u64::from(b)).wrapping_mul(0x100000001b3))
}

#[tokio::main]
async fn main() {
    let expected_raw=include_bytes!("expected.bf16.bin");
    assert_eq!(expected_raw.len(),H::SIZE*2);
    let expected:Vec<f32>=expected_raw.chunks_exact(2).map(|b|bf16::from_bits(u16::from_le_bytes([b[0],b[1]])).to_f32()).collect();
    let mut up=vec![0u8;L::SIZE*H::SIZE/2];
    let mut gate=up.clone();
    for row in 0..L::SIZE {up[row*H::SIZE/2]=1;gate[row*H::SIZE/2]=2;}
    for matrix in [&mut up,&mut gate] {
        matrix[..H::SIZE/2].fill(0);
        matrix[..1365].fill(0x77);
        matrix[1365]=6;
    }
    let anchor=7680*H::SIZE/2;
    up[anchor..anchor+H::SIZE/2].fill(0);
    up[anchor..anchor+2].fill(0x77);up[anchor+2]=0x47;
    gate[anchor..anchor+H::SIZE/2].fill(0);
    gate[anchor..anchor+341].fill(0x77);gate[anchor+341]=6;
    let mut local=vec![f8e4m3::from_f32(2.);L::SIZE*H::SIZE/16];
    for row in [0,7680] {local[row*H::SIZE/16..(row+1)*H::SIZE/16].fill(f8e4m3::from_f32(4.));}
    let mut down=vec![0u8;H::SIZE*L::SIZE/2];
    for row in 0..H::SIZE {
        let start=row*L::SIZE/2+if row<1920 {0} else {3840};
        down[start..start+3840].fill(0x77);down[start]=0x70;
    }
    println!("FULL_STRESS_INPUT up={:016x} gate={:016x} down={:016x} local={:016x} expected={:016x}",
        checksum(up.iter().copied()),checksum(gate.iter().copied()),checksum(down.iter().copied()),
        checksum(local.iter().flat_map(|x|x.to_f32().to_bits().to_le_bytes())),checksum(expected_raw.iter().copied()));
    if std::env::args().any(|s|s=="--prepare-only") {println!("FULL_STRESS_PREPARED");return;}
    let mut d=Device::new(ops::decoder_feedforward.topology()).unwrap();
    let mut residual:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec(vec![bf16::from_f32(1.);H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
    let pre:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec(vec![bf16::from_f32(1.);H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
    let post:HbmTensor<bf16,Chip,m![H]>=HostTensor::<bf16,m![H]>::from_vec(vec![bf16::from_f32(1.);H::SIZE]).to_hbm(&mut d.pdma).await.unwrap();
    let up:HbmTensor<f4e2m1,Chip,m![L,H]>=HostTensor::<f4e2m1,m![L,H]>::from_buf(up).to_hbm(&mut d.pdma).await.unwrap();
    let gate:HbmTensor<f4e2m1,Chip,m![L,H]>=HostTensor::<f4e2m1,m![L,H]>::from_buf(gate).to_hbm(&mut d.pdma).await.unwrap();
    let down:HbmTensor<f4e2m1,Chip,m![H,L]>=HostTensor::<f4e2m1,m![H,L]>::from_buf(down).to_hbm(&mut d.pdma).await.unwrap();
    let ups:HbmTensor<f8e4m3,Chip,m![L,H/16]>=HostTensor::<f8e4m3,m![L,H/16]>::from_vec(local.clone()).to_hbm(&mut d.pdma).await.unwrap();
    let gates:HbmTensor<f8e4m3,Chip,m![L,H/16]>=HostTensor::<f8e4m3,m![L,H/16]>::from_vec(local).to_hbm(&mut d.pdma).await.unwrap();
    let downs:HbmTensor<f8e4m3,Chip,m![H,L/16]>=HostTensor::<f8e4m3,m![H,L/16]>::from_vec(vec![f8e4m3::from_f32(15.);H::SIZE*L::SIZE/16]).to_hbm(&mut d.pdma).await.unwrap();
    let us:HbmTensor<f32,Chip,m![1]>=HostTensor::<f32,m![1]>::from_vec(vec![1./2048.]).to_hbm(&mut d.pdma).await.unwrap();
    let gs:HbmTensor<f32,Chip,m![1]>=HostTensor::<f32,m![1]>::from_vec(vec![1./2048.]).to_hbm(&mut d.pdma).await.unwrap();
    let ds:HbmTensor<f32,Chip,m![1]>=HostTensor::<f32,m![1]>::from_vec(vec![1./2048.]).to_hbm(&mut d.pdma).await.unwrap();
    let layer:HbmTensor<bf16,Chip,m![1#8]>=HostTensor::<bf16,m![1#8]>::from_vec(vec![bf16::from_f32(0.375);8]).to_hbm(&mut d.pdma).await.unwrap();
    launch(ops::decoder_feedforward,(&mut d,&mut residual,&pre,&up,&gate,&down,&ups,&gates,&downs,&us,&gs,&ds,&post,&layer)).await.unwrap();
    let actual:HostTensor<bf16,m![H]>=residual.to_host(&mut d.pdma).await.unwrap();
    let actual=actual.into_vec();
    let mut mismatches=0;let mut exact=0;let mut nonfinite=0;let mut max_abs=0f32;
    for (a,e) in actual.iter().zip(&expected) {
        let a=a.to_f32();let delta=(a-e).abs();max_abs=max_abs.max(delta);
        nonfinite+=usize::from(!a.is_finite());
        mismatches+=usize::from(!a.is_finite()||delta>0.01+0.01*e.abs());
        exact+=usize::from(a==*e);
    }
    println!("FULL_STRESS_RESULT checked=3840 exact={exact} mismatches={mismatches} nonfinite={nonfinite} max_abs={max_abs:?} first={:?} second={:?} expected={:?}",actual[0].to_f32(),actual[1920].to_f32(),expected[0]);
    // This diagnostic records the known E133 failure without skipping the other images.
    // The collector separately enforces the control and repaired candidate's acceptance.
    println!("FULL_STRESS_COMPLETE");
}

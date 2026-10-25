//! Diagnostic: no candidate kernel is selected or arithmetic gate relaxed.
use furiosa_opt_gemma4::{Chip,ops_f32_reduce_order as op};
use furiosa_opt_std::prelude::*;
fn hash(mut v:u64)->u64 {v=(v^(v>>30)).wrapping_mul(0xbf58476d1ce4e5b9);v=(v^(v>>27)).wrapping_mul(0x94d049bb133111eb);v^(v>>31)}
fn add(a:f32,b:f32)->f32 { // explicit binary F32 boundary; avoid any possible excess precision.
    f32::from_bits((a+b).to_bits())
}
#[tokio::main]
async fn main() {
    let mut d=Device::new(op::sum_four.topology()).unwrap();
    let permutations:Vec<[usize;4]>=(0..4).flat_map(|a|(0..4).flat_map(move|b|(0..4).flat_map(move|c|(0..4).filter_map(move|e|if a!=b&&a!=c&&a!=e&&b!=c&&b!=e&&c!=e{Some([a,b,c,e])}else{None})))).collect();
    let mut balanced_mismatches=[0usize;3];let mut sequential_mismatches=vec![0usize;24];let mut samples=0usize;
    let mut output=HostTensor::<f32,op::Outputs>::zero().to_hbm(&mut d.pdma).await.unwrap();
    for seed in 0..256usize {
        let mut values=Vec::with_capacity(64*4*8);let mut rows=Vec::new();
        for row in 0..64usize {
            let i=seed*64+row;
            let mut x=[0f32;4];
            for k in 0..4 {let v=hash((i*4+k) as u64+31);let exponent=100+((v>>32)%55) as u32;let bits=((v as u32)&0x807fffff)|(exponent<<23);x[k]=f32::from_bits(bits);}
            if i%4==0 {x[1]=-x[0];}
            if i%4==1 {x[2]=-x[0];}
            if i%4==2 {x[3]=-x[0];}
            if i<24 {let pattern=[16777216.0,1.0,-16777216.0,1.0];let p=permutations[i];x=[pattern[p[0]],pattern[p[1]],pattern[p[2]],pattern[p[3]]];}
            rows.push(x);
            for v in x {values.push(v);values.extend([0.0;7]);}
        }
        let input=HostTensor::<f32,op::Inputs>::from_vec(values.clone()).to_hbm(&mut d.pdma).await.unwrap();
        launch(op::sum_four,(&mut d,&input,&mut output)).await.unwrap();
        let got:HostTensor<f32,op::Outputs>=output.to_host(&mut d.pdma).await.unwrap();let got=got.into_vec();
        for (row,x) in rows.into_iter().enumerate() {
            let actual=got[row*8];assert!(actual.is_finite());
            let candidates=[add(add(x[0],x[1]),add(x[2],x[3])),add(add(x[0],x[2]),add(x[1],x[3])),add(add(x[0],x[3]),add(x[1],x[2]))];
            for (i,v) in candidates.into_iter().enumerate() {balanced_mismatches[i]+=usize::from(actual.to_bits()!=v.to_bits());}
            for (i,p) in permutations.iter().enumerate() {let v=add(add(add(x[p[0]],x[p[1]]),x[p[2]]),x[p[3]]);sequential_mismatches[i]+=usize::from(actual.to_bits()!=v.to_bits());}
            if samples<24 {println!("F32_SUM_OBSERVATION row={samples} inputs={x:?} actual={actual} balanced={candidates:?}");}
            samples+=1;
        }
        let reread:HostTensor<f32,op::Inputs>=input.to_host(&mut d.pdma).await.unwrap();assert!(reread.into_vec().iter().zip(&values).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
    println!("F32_SUM_ORDER samples={samples} balanced_01_23__02_13__03_12={balanced_mismatches:?}");
    for (p,n) in permutations.iter().zip(&sequential_mismatches) {println!("F32_SUM_LEFT_FOLD order={p:?} mismatches={n}");}
    println!("DIAGNOSTIC_COMPLETE finite inputsimmutable; no candidate or wholemodel correctness claim");
}

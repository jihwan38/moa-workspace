//! Complete raw-bit zero check and the actual canonical 80/16 allocation count.
use furiosa_opt_gemma4::{
    Chip, axes::*, ops_pool_views::Rc64, ops_request_attention::Fc, ops_zero_pool as op,
};
use furiosa_opt_std::prelude::*;
async fn check<Mp: M>(d: &mut Device, x: &HbmTensor<bf16, Chip, Mp>) {
    let h: HostTensor<bf16, Mp> = x.to_host(&mut d.pdma).await.unwrap();
    let v = h.into_vec();
    let wrong = v.iter().filter(|x| x.to_bits() != 0).count();
    println!("checked {} BF16 values, nonzero raw bits={wrong}", v.len());
    assert_eq!(wrong, 0);
}
#[tokio::main]
async fn main() {
    let mut d = Device::new(op::zero_sliding_pool.topology()).unwrap();
    let now = std::time::Instant::now();
    let s = launch(op::zero_sliding_pool, &mut d).await.unwrap();
    println!("first_sliding_ms={:.3}", now.elapsed().as_secs_f64() * 1e3);
    check::<m![Rc64, Ts, Ns, Ds]>(&mut d, &s).await;
    let now = std::time::Instant::now();
    let f = launch(op::zero_full_pool, &mut d).await.unwrap();
    println!("first_full_ms={:.3}", now.elapsed().as_secs_f64() * 1e3);
    check::<m![Fc, Tf, Df]>(&mut d, &f).await;
    drop((s, f));
    let now = std::time::Instant::now();
    let mut sliding = Vec::new();
    let mut full = Vec::new();
    for _ in 0..80 {
        sliding.push(launch(op::zero_sliding_pool, &mut d).await.unwrap());
    }
    for _ in 0..16 {
        full.push(launch(op::zero_full_pool, &mut d).await.unwrap());
    }
    println!(
        "canonical_22GiB_alloc_fill_ms={:.3}",
        now.elapsed().as_secs_f64() * 1e3
    );
    check::<m![Rc64, Ts, Ns, Ds]>(&mut d, sliding.last().unwrap()).await;
    check::<m![Fc, Tf, Df]>(&mut d, full.last().unwrap()).await;
    println!("PASS returned pools remain live with all22GiB allocated");
}

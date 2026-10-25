//! Compile/allocation check only. Complete nonzero parity probe is separate.
use furiosa_opt_gemma4::{
    axes::*,
    ops_pool_views::Rc64,
    ops_prefill_head_attn_commit as op,
    ops_qkv_shared_blocked::{KeyOutput, QueryOutput},
    ops_sliding_wide::Pk128,
};
use furiosa_opt_std::prelude::*;
#[tokio::main]
async fn main() {
    if !std::env::args().any(|a| a == "--execute-zero-smoke") {
        println!("Compile-only binary: do not use for correctness or timing evidence.");
        return;
    }
    let mut d = Device::new(op::sliding_fused_p128.topology()).unwrap();
    macro_rules! z {
        ($t:ty,$m:ty) => {
            HostTensor::<$t, $m>::zero()
                .to_hbm(&mut d.pdma)
                .await
                .unwrap()
        };
    }
    let q = z!(bf16, QueryOutput);
    let k = z!(bf16, KeyOutput);
    let v = z!(bf16, KeyOutput);
    let qw = z!(bf16, m![Ds]);
    let kw = z!(bf16, m![Ds]);
    let rope = z!(i32, m![P128]);
    let cos = z!(bf16, m![E, Ds]);
    let sin = z!(bf16, m![E, Ds]);
    let mut pk = z!(bf16, m![Rc64, Ts, Ns, Ds]);
    let mut pv = z!(bf16, m![Rc64, Ts, Ns, Ds]);
    let bases = z!(i32, m![P128 / 64]);
    let old = z!(f32, m![P128, Ts]);
    let chunk = z!(f32, m![P128, Pk128]);
    let offsets = z!(i32, m![P128]);
    let mut ks = z!(bf16, m![P128, Ns, Ds]);
    let mut vs = z!(bf16, m![P128, Ns, Ds]);
    let mut out = z!(bf16, m![P128, Ns, Gs, Ds]);
    let mut qo = z!(bf16, m![P128, Ns, Gs, Ds]);
    launch(
        op::headed_outputs,
        (
            &mut d, &q, &k, &v, &qw, &kw, &rope, &cos, &sin, &mut qo, &mut ks, &mut vs,
        ),
    )
    .await
    .unwrap();
    launch(
        op::sliding_fused_p128,
        (
            &mut d, &q, &k, &v, &qw, &kw, &rope, &cos, &sin, &mut pk, &mut pv, &bases, &old,
            &chunk, &offsets, &mut ks, &mut vs, &mut out,
        ),
    )
    .await
    .unwrap();
    launch(
        op::sliding_fused_first_p128,
        (
            &mut d, &q, &k, &v, &qw, &kw, &rope, &cos, &sin, &mut pk, &mut pv, &bases, &old,
            &chunk, &offsets, &mut ks, &mut vs, &mut out,
        ),
    )
    .await
    .unwrap();
}

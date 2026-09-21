use furiosa_opt_std::prelude::*;
use std::io::Write;

pub struct Case {
    pub id: usize,
    pub x: Vec<f8e4m3>,
    pub weights: Vec<f8e4m3>,
    pub scalar: Vec<f32>,
    pub blocks: Vec<f32>,
}

pub fn cases() -> Vec<Case> {
    (0..3).map(|id| {
        // Every source is an exactly representable finite FP8 multiple of1/8.
        // Products are multiples of1/64; sums below256 are exact in F32.
        let x: Vec<_> = (0..64).map(|k| f8e4m3::from_f32((((k*7+id*5)%31) as i32-15) as f32/8.0)).collect();
        let weights: Vec<_> = (0..4*64).map(|i| {
            let r=i/64; let k=i%64;
            f8e4m3::from_f32((((k*11+r*7+id*3)%29) as i32-14) as f32/8.0)
        }).collect();
        assert!(x.iter().chain(&weights).all(|v| v.to_f32().is_finite()));
        let mut blocks=Vec::new(); let mut scalar=Vec::new();
        for r in 0..4 {
            let mut total=0.0f64;
            for block in 0..4 {
                let mut dot=0.0f64;
                for k in block*16..(block+1)*16 {
                    dot += f64::from(x[k].to_f32()) * f64::from(weights[r*64+k].to_f32());
                }
                assert_eq!(f64::from(dot as f32),dot);
                blocks.push(dot as f32); total+=dot;
            }
            let direct: f64=(0..64).map(|k| f64::from(x[k].to_f32())*f64::from(weights[r*64+k].to_f32())).sum();
            assert_eq!(total,direct); assert_eq!(f64::from(total as f32),total);
            scalar.push(total as f32);
        }
        Case { id,x,weights,scalar,blocks }
    }).collect()
}

pub fn oracle_only(cases: &[Case]) -> bool {
    let only=std::env::args().any(|a|a=="--oracle-only");
    if only {
        for c in cases { println!("case={} scalar={:?} blocks={:?}",c.id,c.scalar,c.blocks); }
        println!("CPU_ORACLE_PASS; no Device was created and no kernel was launched");
    }
    only
}

pub fn mark(label: &str,id:usize,stage:&str) {
    println!("PROBE {label} case={id} {stage}");
    std::io::stdout().flush().unwrap();
}

pub fn compare(label:&str,id:usize,actual:&[f32],one_slice:&[f32]) -> bool {
    assert_eq!(actual.len(),256*one_slice.len());
    let mut failures=0; let mut max_abs=0.0f32;
    for (i,&got) in actual.iter().enumerate() {
        let want=one_slice[i%one_slice.len()];
        if !got.is_finite() || got != want {
            if failures<16 { println!("MISMATCH {label} case={id} slice={} local={} want={want:?} got={got:?} bits={:08x}",i/one_slice.len(),i%one_slice.len(),got.to_bits()); }
            failures+=1;
        }
        max_abs=max_abs.max((got-want).abs());
    }
    println!("RESULT {label} case={id} checked={} failures={failures} max_abs={max_abs} {}",actual.len(),if failures==0 {"PASS"}else{"FAIL"});
    failures==0
}

//! Independent CPU fixture, canonical/blocked mappings and exact A4 wire oracle.
use super::cpu::{self,InputCpu,TOKENS,HIDDEN,EXPANDED};
use furiosa_opt_std::prelude::bf16;
#[path="rational_a4_oracle.rs"] mod rational;

pub fn floats(v:&[f32])->Vec<u8>{v.iter().flat_map(|v|v.to_le_bytes()).collect()}
pub fn bits(b:&[u8])->Vec<u16>{assert_eq!(b.len()%2,0);b.chunks_exact(2).map(|x|u16::from_le_bytes(x.try_into().unwrap())).collect()}
pub fn residual_index(p:usize,h:usize)->usize {
    (((((p/64%2)*32+p/2%32)*8+h/480)*4+p/128)*2+p%2)*480+h%480
}
pub fn block_residual(canonical:&[u8])->Vec<u8>{
    assert_eq!(canonical.len(),TOKENS*HIDDEN*2);let mut out=vec![0;canonical.len()];let mut seen=vec![false;TOKENS*HIDDEN];
    for p in 0..TOKENS {for h in 0..HIDDEN {let at=residual_index(p,h);assert!(!seen[at]);seen[at]=true;out[at*2..at*2+2].copy_from_slice(&canonical[(p*HIDDEN+h)*2..(p*HIDDEN+h)*2+2]);}}
    assert!(seen.into_iter().all(|x|x));out
}
pub fn canonical_residual(blocked:&[u8])->Vec<u8>{
    assert_eq!(blocked.len(),TOKENS*HIDDEN*2);let mut out=Vec::with_capacity(blocked.len());
    for p in 0..TOKENS {for h in 0..HIDDEN {let at=residual_index(p,h);out.extend_from_slice(&blocked[2*at..2*at+2]);}}out
}
pub fn canonical_projection(blocked:&[u8],rows:usize,plane:usize)->Vec<u8>{
    assert!(blocked.len()>=(plane+1)*TOKENS*rows*2);let mut out=Vec::with_capacity(TOKENS*rows*2);
    for p in 0..TOKENS {for h in 0..rows {let at=plane*TOKENS*rows+cpu::output_index(p,h,rows);out.extend_from_slice(&blocked[2*at..2*at+2]);}}out
}
pub fn sliding(canonical:&[u8])->[Vec<u8>;4]{
    assert_eq!(canonical.len(),TOKENS*HIDDEN*2);
    std::array::from_fn(|q|{let mut out=vec![0;128*HIDDEN*2];let mut seen=vec![false;128*HIDDEN];
        for p in 0..128 {for h in 0..HIDDEN {let at=((p/64*60+h/64)*64+p%64)*64+h%64;
            assert!(!seen[at]);seen[at]=true;let source=((q*128+p)*HIDDEN+h)*2;
            out[at*2..at*2+2].copy_from_slice(&canonical[source..source+2]);}}
        assert!(seen.into_iter().all(|x|x));out})
}
fn mix(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
pub fn input(permuted:bool,residual:bool)->Vec<u8>{
    let mut out=Vec::with_capacity(TOKENS*HIDDEN*2);
    for p in 0..TOKENS {let source=if permuted{cpu::perm(p)}else{p};
        for h in 0..HIDDEN {
            let value=if h<9 {let a=0.625f32+(source%128)as f32/256.0;
                if (source>>h)&1==0 {a}else{-a}}
            else if (source+h)%997==0 {0.0}
            else {let n=(mix((source*HIDDEN+h+if residual{19831}else{5437})as u64)%4093)as i32-2046;
                n as f32/(if residual{2111.0}else{1777.0})};
            out.extend(bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    assert_eq!(out.chunks_exact(HIDDEN*2).map(cpu::fingerprint).collect::<std::collections::BTreeSet<_>>().len(),512);out
}

/// Unchanged independent u128 quotient oracle; this never calls the p20 LUT.
pub fn quantize(canonical:&[u8],width:usize)->InputCpu {
    assert!([HIDDEN,EXPANDED].contains(&width));assert_eq!(canonical.len(),TOKENS*width*2);
    let mut codes=Vec::with_capacity(TOKENS*width);let mut scales=Vec::with_capacity(TOKENS);
    for row in canonical.chunks_exact(width*2){let(q,s)=rational::reference(&bits(row));codes.extend(q.into_iter().map(|q|q as i8));scales.push(s);}
    let mut packed=vec![0;TOKENS*width/2];let mut dense=vec![0;TOKENS*width];let mut seen=vec![false;TOKENS*width];
    const E4:[u8;8]=[0,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e];
    for p in 0..TOKENS {for c in 0..width {let q=codes[p*width+c];assert!((-7..=7).contains(&q));
        let at=cpu::input_group(p,c/960,width)*960+c%960;assert!(!seen[at]);seen[at]=true;
        packed[at/2]|=((q as u8)&15)<<(4*(at%2));dense[at]=E4[q.unsigned_abs()as usize]|if q<0{128}else{0};}}
    assert!(seen.into_iter().all(|x|x));
    // Separate inverse-physical eight-field encoder, little-endian I32 words.
    let mut words=Vec::with_capacity(packed.len());
    for w in 0..TOKENS*width/8 {let g=w/120;let p=g/(256*(width/960))*256+g%256;
        let c=(g/256%(width/960))*960+w%120*8;let mut word=0u32;
        for nibble in 0..8 {word=word.checked_add(((codes[p*width+c+nibble]as u32)&15)<<(4*nibble)).unwrap();}
        words.extend(word.to_le_bytes());}
    assert_eq!(packed,words,"independent pair versus word packing");
    InputCpu{cols:width,codes,packed,dense,scales}
}
pub fn compare_rows(got:&[u8],base:&[u8],width:usize,item:usize,label:&str)->usize {
    assert_eq!(got.len(),TOKENS*width*item);assert_eq!(base.len(),got.len());
    for p in 0..TOKENS {assert_eq!(&got[p*width*item..(p+1)*width*item],&base[cpu::perm(p)*width*item..(cpu::perm(p)+1)*width*item],"true row permutation {label} p={p}");}
    TOKENS*width
}
pub fn drift(got:&[u8],reference:&[u8],label:&str,round:usize,step:usize){
    assert_eq!(got.len(),reference.len());let(mut e,mut den,mut max,mut changed)=(0f64,0f64,0f64,0usize);
    for(a,b)in bits(got).into_iter().zip(bits(reference)){let x=bf16::from_bits(a).to_f32()as f64;let y=bf16::from_bits(b).to_f32()as f64;
        assert!(x.is_finite()&&y.is_finite(),"nonfinite drift {label}");e+=(x-y).powi(2);den+=y*y;max=max.max((x-y).abs());changed+=usize::from(a!=b);}
    println!("I4_COMPLETE_DRIFT round={round} step={step} stage={label} bf16_values={} changed={changed} relative_l2={:.9e} maxabs={max:.9e} comparator=current_production_E4 no_model_quality_claim=true",got.len()/2,(e/den.max(1e-300)).sqrt());
}

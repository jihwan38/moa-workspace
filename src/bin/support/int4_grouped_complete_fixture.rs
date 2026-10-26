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
/// Native original-P128 GeGLU wire order [P/64,L/240,L%240,P%64].
/// (p,c) -> (((p/64)*64+c/240)*240+c%240)*64+p%64.
pub fn reference_column_tile(canonical:&[u8])->Vec<u8>{
    assert_eq!(canonical.len(),128*EXPANDED*2);let mut out=vec![0;canonical.len()];let mut seen=vec![false;128*EXPANDED];
    for p in 0..128{for c in 0..EXPANDED{let at=(((p/64)*64+c/240)*240+c%240)*64+p%64;
        assert!(!seen[at]);seen[at]=true;out[2*at..2*at+2].copy_from_slice(&canonical[2*(p*EXPANDED+c)..2*(p*EXPANDED+c)+2]);}}
    assert!(seen.into_iter().all(|x|x));assert_eq!(reference_canonical_tile(&out),canonical,"independent column-wire inverse");out
}
/// Inverse starts from physical indices, independently recovering each axis.
pub fn reference_canonical_tile(column:&[u8])->Vec<u8>{
    assert_eq!(column.len(),128*EXPANDED*2);let mut out=vec![0;column.len()];let mut seen=vec![false;128*EXPANDED];
    for at in 0..128*EXPANDED{let p=(at/(64*240*64))*64+at%64;let c=(at/(64*240)%64)*240+at/64%240;
        assert!(p<128&&c<EXPANDED);let index=p*EXPANDED+c;assert!(!seen[index]);seen[index]=true;
        out[2*index..2*index+2].copy_from_slice(&column[2*at..2*at+2]);}
    assert!(seen.into_iter().all(|x|x));out
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
    let groups=width/960;
    let mut codes=vec![0i8;TOKENS*width];let mut scales=vec![f32::NAN;TOKENS*groups];
    let mut group_seen=vec![false;TOKENS*groups];
    for p in 0..TOKENS {for group in 0..groups {
        let begin=(p*width+group*960)*2;
        let(q,s)=rational::reference(&bits(&canonical[begin..begin+960*2]));
        let at=cpu::input_group(p,group,width);assert!(!group_seen[at]);group_seen[at]=true;
        scales[at]=s;
        for(k,value)in q.into_iter().enumerate(){codes[p*width+group*960+k]=value as i8;}
    }}
    assert!(group_seen.into_iter().all(|v|v));
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
/// Scale/offset wire [P/256,K/960,P%256], independently compared by token/group.
pub fn compare_group_rows(got:&[u8],base:&[u8],width:usize,label:&str)->usize {
    let groups=width/960;assert_eq!(got.len(),TOKENS*groups*4);assert_eq!(got.len(),base.len());
    for p in 0..TOKENS {for g in 0..groups {
        let at=cpu::input_group(p,g,width)*4;let old=cpu::input_group(cpu::perm(p),g,width)*4;
        assert_eq!(&got[at..at+4],&base[old..old+4],"group permutation {label} token={p} group={g}");
    }}TOKENS*groups
}
/// Independent relative-byte address expectation, without the p20 table values.
pub fn offset_bytes(canonical:&[u8],width:usize)->Vec<u8> {
    assert_eq!(canonical.len(),TOKENS*width*2);let groups=width/960;
    let mut out=vec![0u8;TOKENS*groups*4];let mut seen=vec![false;TOKENS*groups];
    for p in 0..TOKENS {for g in 0..groups {
        let start=(p*width+g*960)*2;
        let a=bits(&canonical[start..start+1920]).into_iter().map(|b|{
            let b=b&0x7fff;assert!(b>>7!=255);if b>>7==0{0}else{b}
        }).max().unwrap().max(rational::FLOOR_BITS);
        let offset=i32::from(a&127)*32;assert!((0..=4064).contains(&offset));
        let at=cpu::input_group(p,g,width);assert!(!seen[at]);seen[at]=true;
        out[at*4..at*4+4].copy_from_slice(&offset.to_le_bytes());
    }}assert!(seen.into_iter().all(|v|v));out
}
/// Scalar RN32(code*group-scale) reconstruction; not a dot-product decomposition.
pub fn quantization_metrics(canonical:&[u8],q:&InputCpu,round:usize,step:usize,label:&str){
    let width=q.cols;assert_eq!(canonical.len(),TOKENS*width*2);assert_eq!(q.codes.len(),TOKENS*width);
    let(mut sse,mut den,mut maxabs,mut zero,mut input_zero)=(0f64,0f64,0f64,0usize,0usize);
    for(i,b)in bits(canonical).into_iter().enumerate(){
        let p=i/width;let c=i%width;let value=bf16::from_bits(b).to_f32();assert!(value.is_finite());
        let value=if b&0x7f80==0{0.0}else{value};
        let reconstructed=q.codes[i]as f32*q.scales[cpu::input_group(p,c/960,width)];assert!(reconstructed.is_finite());
        let x=value as f64;let e=reconstructed as f64-x;sse+=e*e;den+=x*x;maxabs=maxabs.max(e.abs());
        zero+=usize::from(q.codes[i]==0);input_zero+=usize::from(value==0.0);
    }
    println!("I4_GROUPED_COMPLETE_ACTIVATION round={round} step={step} stage={label} values={} zero_codes={zero} zero_fraction={:.12} bf16_ftz_zeros={input_zero} scalar_f32_reconstruction_l2={:.12e} maxabs={maxabs:.12e} scope=own_BF16_input_group960_not_output_error_decomposition",
        q.codes.len(),zero as f64/q.codes.len()as f64,(sse/den.max(1e-300)).sqrt());
}
pub fn production_zero_fraction(bytes:&[u8],round:usize,step:usize,label:&str){
    assert!(bytes.iter().all(|b|b&127!=127),"finite current E4 activation {label}");
    let zeros=bytes.iter().filter(|&&b|b&127==0).count();
    println!("I4_GROUPED_COMPLETE_PRODUCTION_ACTIVATION round={round} step={step} stage={label} values={} zero_codes={zeros} zero_fraction={:.12} scope=current_E4_own_inputs_step1_feedback_diverged",bytes.len(),zeros as f64/bytes.len()as f64);
}
pub fn drift(got:&[u8],reference:&[u8],label:&str,round:usize,step:usize){
    assert_eq!(got.len(),reference.len());let(mut e,mut den,mut max,mut changed)=(0f64,0f64,0f64,0usize);
    for(a,b)in bits(got).into_iter().zip(bits(reference)){let x=bf16::from_bits(a).to_f32()as f64;let y=bf16::from_bits(b).to_f32()as f64;
        assert!(x.is_finite()&&y.is_finite(),"nonfinite drift {label}");e+=(x-y).powi(2);den+=y*y;max=max.max((x-y).abs());changed+=usize::from(a!=b);}
    println!("I4_GROUPED_COMPLETE_DRIFT round={round} step={step} stage={label} bf16_values={} changed={changed} relative_l2={:.9e} maxabs={max:.9e} comparator=current_production_E4 matched_inputs={} no_model_quality_claim=true",got.len()/2,(e/den.max(1e-300)).sqrt(),step==0);
}

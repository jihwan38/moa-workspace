//! CPU construction/oracle only. No candidate device helper is imported.
use furiosa_opt_std::prelude::bf16;
use std::collections::BTreeMap;
#[path = "rational_a4_oracle.rs"]
mod rational;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const TOKENS: usize = 512;
pub const HIDDEN: usize = 3840;
pub const EXPANDED: usize = 15360;
pub const SAMPLES: usize = 512;

pub fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}
fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
pub fn perm(p: usize) -> usize { (73 * p + 127) % TOKENS }
pub fn weight_group(row: usize, part: usize, cols: usize) -> usize {
    ((row / 240) * (cols / 960) + part) * 240 + row % 240
}
pub fn input_group(p: usize, part: usize, cols: usize) -> usize {
    ((p / 256) * (cols / 960) + part) * 256 + p % 256
}
pub fn output_index(p: usize, row: usize, rows: usize) -> usize {
    ((p / 256 * (rows / 240) + row / 240) * 256 + p % 256) * 240 + row % 240
}
pub fn sample_row(i: usize, rows: usize) -> usize { (317 * i + 19) % rows }
fn nibble(bytes: &[u8], i: usize) -> i8 {
    let x = ((bytes[i / 2] >> (4 * (i % 2))) & 15) as i8;
    if x >= 8 { x - 16 } else { x }
}
fn e4_decode(code: u8) -> f32 {
    let e = (code >> 3) & 15; let m = code & 7;
    assert!(!(e == 15 && m == 7), "nonfinite supplied E4");
    let v = if e == 0 { m as f32 / 512.0 } else { (8 + m) as f32 * 2f32.powi(e as i32 - 10) };
    if code & 128 == 0 { v } else { -v }
}
fn f4_decode(code: u8) -> f32 {
    let v = [0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0][(code & 7) as usize];
    if code & 8 == 0 { v } else { -v }
}
fn e4_code(q: i8) -> u8 {
    assert!((-7..=7).contains(&q));
    // Exact E4M3 encodings of integers0..7; no unavailable SDK bit accessor.
    let magnitude = [0x00u8,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e][q.unsigned_abs() as usize];
    let b = magnitude | if q < 0 { 0x80 } else { 0 };
    assert_eq!(e4_decode(b), q as f32, "independent dense E4 signed-code check");
    b
}
fn put_codes(packed: &mut [u8], dense: &mut [u8], start: usize, codes: &[i32]) {
    assert_eq!(start % 2, 0); assert_eq!(codes.len() % 2, 0);
    for (j, pair) in codes.chunks_exact(2).enumerate() {
        assert!(pair.iter().all(|x| (-7..=7).contains(x)));
        packed[start/2+j] = ((pair[0] as u8)&15) | (((pair[1] as u8)&15)<<4);
    }
    for (j, &q) in codes.iter().enumerate() {
        dense[start+j] = e4_code(q as i8);
        assert_eq!(nibble(packed,start+j),q as i8,"signed I4 packing inverse");
    }
}

pub struct SampleWeight { pub codes: Vec<i8>, pub scales: Vec<f32> }
pub struct WeightCpu {
    pub rows: usize, pub cols: usize, pub packed: Vec<u8>, pub dense: Vec<u8>,
    pub scales: Vec<f32>, pub samples: BTreeMap<usize,SampleWeight>, pub global: f32,
    pub source: [(String,u64);3],
}
impl WeightCpu {
    pub fn load(tensors: &safetensors::SafeTensors<'_>, projection: &str) -> Result<Self> {
        let (rows,cols)=if projection=="down_proj" {(HIDDEN,EXPANDED)} else {(EXPANDED,HIDDEN)};
        assert!(["up_proj","gate_proj","down_proj"].contains(&projection));
        let prefix=format!("model.language_model.layers.0.mlp.{projection}");
        let names=[format!("{prefix}.weight_packed"),format!("{prefix}.weight_scale"),format!("{prefix}.weight_global_scale")];
        let packed=tensors.tensor(&names[0])?; let scale=tensors.tensor(&names[1])?; let global=tensors.tensor(&names[2])?;
        assert_eq!(packed.dtype(),safetensors::Dtype::U8); assert_eq!(packed.shape(),[rows,cols/2]);
        assert_eq!(packed.data().len(),rows*cols/2);
        assert_eq!(scale.dtype(),safetensors::Dtype::F8_E4M3); assert_eq!(scale.shape(),[rows,cols/16]);
        assert_eq!(scale.data().len(),rows*cols/16);
        assert_eq!(global.dtype(),safetensors::Dtype::F32); assert_eq!(global.data().len(),4);
        let stored_global=f32::from_le_bytes(global.data().try_into()?);
        assert!(stored_global.is_finite()&&stored_global!=0.0);
        let global_value=1.0f32/stored_global;
        assert!(global_value.is_finite()&&global_value!=0.0);
        let source=std::array::from_fn(|i| (names[i].clone(),fingerprint(tensors.tensor(&names[i]).unwrap().data())));
        let mut result=Self{rows,cols,packed:vec![0;rows*cols/2],dense:vec![0;rows*cols],
            scales:vec![f32::NAN;rows*cols/960],samples:BTreeMap::new(),global:global_value,source};
        let selected:BTreeMap<usize,usize>=(0..SAMPLES).map(|i|(sample_row(i,rows),i)).collect();
        assert_eq!(selected.len(),SAMPLES);
        let mut seen=vec![false;rows*cols/960];
        for row in 0..rows {
            let mut sample=if selected.contains_key(&row) {Some(SampleWeight{codes:Vec::with_capacity(cols),scales:Vec::with_capacity(cols/960)})}else{None};
            for part in 0..cols/960 {
                let values:Vec<u16>=(0..960).map(|k| {
                    let at=row*cols+part*960+k;
                    let nib=(packed.data()[at/2]>>(4*(at%2)))&15;
                    let v=bf16::from_f32(f4_decode(nib)*e4_decode(scale.data()[at/16]));
                    assert!(v.to_f32().is_finite()); v.to_bits()
                }).collect();
                let (codes,s)=rational::reference(&values);
                assert!(s.is_normal()&&s>0.0);
                if let Some(v)=sample.as_mut() {v.codes.extend(codes.iter().map(|&x|x as i8));v.scales.push(s);}
                let g=weight_group(row,part,cols);assert!(!seen[g]);seen[g]=true;
                // Inverse physical group coordinates independently recover the source row/part.
                assert_eq!((g/(cols/960)/240)*240+g%240,row);
                assert_eq!(g/240%(cols/960),part);
                result.scales[g]=s;
                put_codes(&mut result.packed,&mut result.dense,g*960,&codes);
            }
            if let Some(v)=sample {result.samples.insert(row,v);}
        }
        assert!(seen.iter().all(|&b|b)); assert_eq!(result.samples.len(),SAMPLES);
        Ok(result)
    }
    pub fn verify_source(&self,tensors:&safetensors::SafeTensors<'_>) -> Result<()> {
        for (name,hash) in &self.source {assert_eq!(fingerprint(tensors.tensor(name)?.data()),*hash,"immutable supplied source {name}");} Ok(())
    }
}

pub struct InputCpu { pub cols:usize, pub codes:Vec<i8>, pub packed:Vec<u8>,pub dense:Vec<u8>,pub scales:Vec<f32> }
impl InputCpu {
    pub fn new(cols:usize,permuted:bool)->Self {
        assert!([HIDDEN,EXPANDED].contains(&cols));
        let mut result=Self{cols,codes:Vec::with_capacity(TOKENS*cols),packed:vec![0;TOKENS*cols/2],dense:vec![0;TOKENS*cols],scales:Vec::with_capacity(TOKENS)};
        let mut seen=vec![false;TOKENS*(cols/960)];
        for p in 0..TOKENS {
            let source=if permuted{perm(p)}else{p};
            let a_bits=(((125+source/128) as u16)<<7)|(source as u16%128);
            let a=bf16::from_bits(a_bits).to_f32();
            let values:Vec<u16>=(0..cols).map(|k| {
                if k<9 {return a_bits|if source&(1<<k)!=0{0x8000}else{0};}
                if k==cols-2{return a_bits;} if k==cols-1{return a_bits|0x8000;}
                if k%997==0{return [0u16,0x8000,1,0x807f][(source+k)%4];}
                let n=(mix((source*cols+k+cols*17) as u64)%2047) as i32-1023;
                bf16::from_f32((n as f32/1024.0)*a).to_bits()
            }).collect();
            let(q,s)=rational::reference(&values);
            let tag=(0..9).fold(0usize,|t,k|{assert_eq!(q[k].abs(),7);t|((q[k]<0)as usize)<<k});assert_eq!(tag,source);
            result.codes.extend(q.iter().map(|&v|v as i8));result.scales.push(s);
            for part in 0..cols/960 {
                let g=input_group(p,part,cols);assert!(!seen[g]);seen[g]=true;
                assert_eq!((g/(cols/960)/256)*256+g%256,p);assert_eq!(g/256%(cols/960),part);
                put_codes(&mut result.packed,&mut result.dense,g*960,&q[part*960..(part+1)*960]);
            }
        }
        assert!(seen.iter().all(|&b|b));
        assert_eq!(result.scales.iter().map(|x|x.to_bits()).collect::<std::collections::BTreeSet<_>>().len(),TOKENS);
        result
    }
}

pub fn bf_at(bytes:&[u8],index:usize)->f32 {
    bf16::from_bits(u16::from_le_bytes(bytes[2*index..2*index+2].try_into().unwrap())).to_f32()
}
// One representable F32 step outwards, even when F64->F32 already rounded outwards.
fn next_f32(x:f32,up:bool)->f32 {
    assert!(x.is_finite());
    if x==0.0 {return f32::from_bits(if up{1}else{0x80000001});}
    let bits=x.to_bits();f32::from_bits(if (x>0.0)==up{bits+1}else{bits-1})
}
fn rounded_interval(lo:f64,hi:f64)->(f32,f32) {
    assert!(lo.is_finite()&&hi.is_finite()&&lo<=hi);
    let a=next_f32(lo as f32,false);let b=next_f32(hi as f32,true);
    let a=bf16::from_f32(a).to_f32();let b=bf16::from_f32(b).to_f32();
    assert!(a.is_finite()&&b.is_finite());(a,b)
}
pub fn sampled_bound(output:&[u8],plane:usize,input:&InputCpu,weight:&WeightCpu,down:bool)->(usize,f64) {
    let n=input.cols/960;assert_eq!(input.cols,weight.cols);assert_eq!(n,if down{16}else{4});
    let u=2f64.powi(-24);let m=(2*n+2)as f64;let gamma=m*u/(1.0-m*u);
    let mut max_radius=0f64;
    for i in 0..SAMPLES {
        let p=i;let row=sample_row(i,weight.rows);let w=&weight.samples[&row];
        let mut center=0f64;let mut abs=0f64;
        for part in 0..n {
            let dot=(0..960).map(|k|i64::from(input.codes[p*input.cols+part*960+k])*i64::from(w.codes[part*960+k])).sum::<i64>();
            assert!(dot.abs()<=47_040);
            // <=16+24 significant bits: product is exact inF64; sum may round.
            let term=dot as f64*w.scales[part] as f64;assert!(term.is_finite());
            center+=term;abs+=term.abs();
        }
        let xs=input.scales[p] as f64;center*=xs;
        let f64_error=64.0*f64::EPSILON*abs*xs.abs();
        let radius=gamma*abs*xs.abs()+m*(f32::MIN_POSITIVE as f64)*(1.0+xs.abs())+f64_error;
        max_radius=max_radius.max(radius);
        let(mut low,mut high)=rounded_interval(center-radius,center+radius);
        if down {
            // Preserve the first BF16 boundary and monotonicity for either sign global.
            let a=bf16::from_f32(low*weight.global).to_f32();let b=bf16::from_f32(high*weight.global).to_f32();
            low=a.min(b);high=a.max(b);assert!(low.is_finite()&&high.is_finite());
        }
        let at=plane*TOKENS*weight.rows+output_index(p,row,weight.rows);let got=bf_at(output,at);
        assert!(got.is_finite()&&got>=low&&got<=high,
            "F64 interval projection_plane={plane} token={p} row={row} got={got} lower={low} upper={high} center={center} radius={radius}");
    }
    (SAMPLES,max_radius)
}
pub fn check_output_layout(rows:usize) {
    let mut seen=vec![false;TOKENS*(rows/240)];
    for p in 0..TOKENS {for block in 0..rows/240 {
        let g=output_index(p,block*240,rows)/240;assert!(!seen[g]);seen[g]=true;
        assert_eq!((g/(rows/240)/256)*256+g%256,p);assert_eq!(g/256%(rows/240),block);
        assert_eq!(output_index(p,block*240+239,rows),g*240+239);
    }}assert!(seen.iter().all(|&b|b));
}
pub fn compare_bits(candidate:&[u8],reference:&[u8],label:&str)->usize {
    assert_eq!(candidate.len(),reference.len());assert_eq!(candidate.len()%2,0);
    let mut differences=0usize;
    for (i,(a,b)) in candidate.chunks_exact(2).zip(reference.chunks_exact(2)).enumerate() {
        let av=u16::from_le_bytes(a.try_into().unwrap());let bv=u16::from_le_bytes(b.try_into().unwrap());
        assert!(bf16::from_bits(av).to_f32().is_finite(),"candidate nonfinite {label} index{i}");
        assert!(bf16::from_bits(bv).to_f32().is_finite(),"reference nonfinite {label} index{i}");
        if av!=bv {if differences<8{println!("I4_GROUP960_DIFF label={label} index={i} actual={av:04x} expected={bv:04x}");}differences+=1;}
    }
    assert_eq!(differences,0,"all-output bit gate {label}");candidate.len()/2
}
pub fn compare_permutation(now:&[u8],base:&[u8],rows:usize,planes:usize)->usize {
    assert_eq!(now.len(),base.len());assert_eq!(now.len(),planes*TOKENS*rows*2);
    for plane in 0..planes {for p in 0..TOKENS {for row in 0..rows {
        let a=2*(plane*TOKENS*rows+output_index(p,row,rows));
        let b=2*(plane*TOKENS*rows+output_index(perm(p),row,rows));
        assert_eq!(&now[a..a+2],&base[b..b+2],"full output row permutation plane{plane} token{p} row{row}");
    }}}now.len()/2
}

//! Pure-std, provided-format-only I4 fitting. The original code/scale pair is
//! candidate zero. SSE is exact real arithmetic on the common 2^-38 grid;
//! it is deliberately not the SSE of RN32(code * scale).
use std::collections::BTreeSet;
use std::sync::OnceLock;

pub const GROUP: usize = 960;
pub const EXTRA_CANDIDATES: usize = 25;
pub const GRID_EXPONENT: i32 = -38;
const RECIP7: f32 = f32::from_bits(0x3e12_4925);
const UNIT: i128 = 1i128 << 38;
pub type Result<T> = std::result::Result<T, &'static str>;

pub struct Fit {
    pub codes: Vec<i8>,
    pub scale: f32,
    /// Zero is the unchanged rational pair; 1..25 denotes j=8..32.
    pub selected: usize,
    pub baseline_sse: i128,
    pub selected_sse: i128,
    pub source_ss: i128,
}

struct Scale {
    value: f32,
    units: i128,
    /// First index in the next bucket; a threshold tie belongs to that bucket.
    cuts: [usize; 7],
    codes: Vec<u8>,
}
struct Maximum {
    original_scale: f32,
    original_codes: Vec<u8>,
    original_squared: Vec<i128>,
    choices: [Scale; EXTRA_CANDIDATES],
}
struct Domain {
    bits: Vec<u16>,
    units: Vec<i128>,
    lookup: Vec<u16>,
    maximum: Vec<Option<Maximum>>,
}
static DOMAIN: OnceLock<Domain> = OnceLock::new();

fn f32_units(x: f32) -> Result<i128> {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 || b >> 23 & 255 == 255 { return Err("invalid grid value"); }
    if b == 0 { return Ok(0); }
    let e = ((b >> 23) & 255) as i32;
    if e == 0 { return Err("subnormal outside nonzero provided grid"); }
    let mantissa = i128::from((b & 0x7f_ffff) | 0x80_0000);
    let shift = e - 127 - 23 - GRID_EXPONENT;
    if shift < 0 || shift > 100 { return Err("value not on supported exact grid"); }
    mantissa.checked_shl(shift as u32).ok_or("grid overflow")
}

fn domain() -> &'static Domain {
    DOMAIN.get_or_init(|| {
        // This format enumeration is independent of the model file. Every
        // finite F4*E4 product has <=6 significant bits, hence exact BF16.
        let levels = [0f32, 0.5, 1., 1.5, 2., 3., 4., 6.];
        let mut all = BTreeSet::new();
        for f in 0..16usize { for e in 0..256u32 {
            if e & 127 == 127 { continue; }
            let exp = (e >> 3) & 15;
            let man = e & 7;
            let magnitude = if exp == 0 { man as f32 / 512. } else {
                (8 + man) as f32 * 2f32.powi(exp as i32 - 10)
            };
            let a = if f & 8 == 0 { levels[f & 7] } else { -levels[f & 7] };
            let b = if e & 128 == 0 { magnitude } else { -magnitude };
            let product = a * b;
            assert!(product.is_finite());
            assert_eq!(product.to_bits() & 0xffff, 0, "provided product must be exact BF16");
            all.insert(((product.to_bits() >> 16) as u16) & 0x7fff);
        }}
        let bits: Vec<u16> = all.into_iter().collect();
        assert_eq!(bits.len(), 238);
        let units: Vec<i128> = bits.iter().map(|&b| f32_units(f32::from_bits(u32::from(b) << 16)).unwrap()).collect();
        assert_eq!(units[0], 0);
        assert_eq!(units[1], UNIT / 1024);
        assert_eq!(*units.last().unwrap(), 2688 * UNIT);
        let mut lookup = vec![u16::MAX; 32768];
        for (i, &b) in bits.iter().enumerate() { lookup[usize::from(b)] = i as u16; }
        let maximum = bits.iter().enumerate().map(|(mi, &b)| {
            if mi == 0 { return None; }
            let a = units[mi];
            let original_scale = f32::from_bits(u32::from(b) << 16) * RECIP7;
            let su = f32_units(original_scale).unwrap();
            let original_codes: Vec<u8> = units.iter().map(|&x| ((14*x+a)/(2*a)).min(7) as u8).collect();
            let original_squared = units.iter().zip(&original_codes).map(|(&x,&q)| {
                let error = x - i128::from(q)*su;
                assert!(error.abs() < 8192*UNIT);
                error*error
            }).collect();
            let choices = std::array::from_fn(|c| {
                let value = original_scale * ((c+8) as f32 / 32.0);
                let su = f32_units(value).unwrap();
                let cuts = std::array::from_fn(|k| units.partition_point(|&x| 2*x < (2*k+1) as i128*su));
                let codes = (0..bits.len()).map(|i| cuts.iter().filter(|&&cut| i >= cut).count() as u8).collect();
                Scale { value, units: su, cuts, codes }
            });
            Some(Maximum { original_scale, original_codes, original_squared, choices })
        }).collect();
        Domain { bits, units, lookup, maximum }
    })
}

/// Initialize immutable tables explicitly when reporting setup versus encoding.
pub fn prepare() { let _ = domain(); }
pub fn provided_magnitudes() -> &'static [u16] { &domain().bits }

fn bucket_sse(count: i128, sum: i128, square: i128, level: i128) -> Result<i128> {
    let cross = (2i128).checked_mul(level).and_then(|x| x.checked_mul(sum)).ok_or("SSE overflow")?;
    let norm = level.checked_mul(level).and_then(|x| x.checked_mul(count)).ok_or("SSE overflow")?;
    let result = square.checked_sub(cross).and_then(|x| x.checked_add(norm)).ok_or("SSE overflow")?;
    if result < 0 { return Err("negative exact SSE"); }
    Ok(result)
}

fn score(scale: &Scale, counts: &[i128], sums: &[i128], squares: &[i128]) -> Result<i128> {
    let mut answer = 0i128;
    let mut begin = 0;
    for q in 0..8 {
        let end = if q == 7 { counts.len()-1 } else { scale.cuts[q] };
        let term = bucket_sse(counts[end]-counts[begin], sums[end]-sums[begin],
            squares[end]-squares[begin], q as i128*scale.units)?;
        answer = answer.checked_add(term).ok_or("SSE overflow")?;
        begin = end;
    }
    if answer >= (1i128<<112) { return Err("group SSE exceeds preregistered bound"); }
    Ok(answer)
}

pub fn fit(values: &[u16], original_codes: &[i8], original_scale: f32) -> Result<Fit> {
    if values.len()!=GROUP || original_codes.len()!=GROUP { return Err("group extent must be960"); }
    let d=domain();
    let mut histogram=vec![0i128;d.bits.len()];
    for &b in values {
        let index=d.lookup[usize::from(b & 0x7fff)];
        if index==u16::MAX { return Err("BF16 value outside provided F4 times finite E4 domain"); }
        histogram[usize::from(index)]+=1;
    }
    let maximum=histogram.iter().rposition(|&n|n!=0).unwrap();
    if maximum==0 {
        let floor_scale=f32::from_bits(15<<23)*RECIP7;
        if original_codes.iter().any(|&q|q!=0) || original_scale.to_bits()!=floor_scale.to_bits() {
            return Err("allzero baseline pair mismatch");
        }
        return Ok(Fit{codes:original_codes.to_vec(),scale:original_scale,selected:0,baseline_sse:0,selected_sse:0,source_ss:0});
    }
    let table=d.maximum[maximum].as_ref().unwrap();
    if original_scale.to_bits()!=table.original_scale.to_bits() { return Err("baseline scale mismatch"); }
    for (&b,&q) in values.iter().zip(original_codes) {
        let index=usize::from(d.lookup[usize::from(b&0x7fff)]);
        let expected=table.original_codes[index] as i8 * if b&0x8000==0 {1} else {-1};
        if q!=expected { return Err("baseline rational code mismatch"); }
    }
    let mut counts=vec![0i128;d.bits.len()+1];
    let mut sums=counts.clone();let mut squares=counts.clone();
    let mut baseline_sse=0i128;
    for i in 0..d.bits.len() {
        let n=histogram[i];let x=d.units[i];
        counts[i+1]=counts[i]+n;sums[i+1]=sums[i]+n*x;squares[i+1]=squares[i]+n*x*x;
        baseline_sse=baseline_sse.checked_add(n.checked_mul(table.original_squared[i]).ok_or("baseline SSE overflow")?).ok_or("baseline SSE overflow")?;
    }
    if baseline_sse >= (1i128<<112) { return Err("baseline exceeds group bound"); }
    let(mut selected,mut selected_sse)=(0,baseline_sse);
    for (i,choice) in table.choices.iter().enumerate() {
        let sse=score(choice,&counts,&sums,&squares)?;
        if sse<selected_sse { selected=i+1;selected_sse=sse; }
    }
    let (codes,scale)=if selected==0 {(original_codes.to_vec(),original_scale)}else{
        let choice=&table.choices[selected-1];
        (values.iter().map(|&b|choice.codes[usize::from(d.lookup[usize::from(b&0x7fff)])] as i8 * if b&0x8000==0 {1} else {-1}).collect(),choice.value)
    };
    Ok(Fit{codes,scale,selected,baseline_sse,selected_sse,source_ss:*squares.last().unwrap()})
}

pub fn pack(codes: &[i8]) -> Result<Vec<u8>> {
    if codes.len()%2!=0 || codes.iter().any(|&q| !(-7..=7).contains(&q)) {return Err("invalid signed codes");}
    Ok(codes.chunks_exact(2).map(|q|((q[0] as u8)&15)|(((q[1] as u8)&15)<<4)).collect())
}

#[cfg(test)]
#[path="rational_a4_oracle.rs"]
mod rational;
#[cfg(test)]
mod tests {
    use super::*;
    fn baseline(values:&[u16])->(Vec<i8>,f32){let(q,s)=rational::reference(values);(q.into_iter().map(|q|q as i8).collect(),s)}
    fn scalar(values:&[u16],codes:&[i8],scale:f32)->i128 {
        let su=f32_units(scale).unwrap();values.iter().zip(codes).map(|(&b,&q)|{
            let x=f32_units(f32::from_bits(u32::from(b&0x7fff)<<16)).unwrap();let e=x-i128::from(q.unsigned_abs())*su;e*e
        }).sum()
    }
    #[test]
    fn finite_domain_and_every_allowed_candidate_code(){
        let d=domain();assert_eq!(d.bits.len(),238);let(mut pairs,mut ties)=(0,0);
        for mi in 1..d.bits.len(){let t=d.maximum[mi].as_ref().unwrap();
            for c in &t.choices {for i in 0..=mi {
                let x=d.units[i];let q=((2*x+c.units)/(2*c.units)).min(7) as u8;
                assert_eq!(c.codes[i],q);ties+=usize::from((2*x)% (2*c.units)==c.units);pairs+=1;
            }}
        }assert_eq!(pairs,711000);assert_eq!(ties,1508);
    }
    #[test]
    fn histogram_scores_equal_independent_per_value_and_permutations(){
        let d=domain();
        for case in 0..40usize {
            let mi=1+(case*73)%237;
            let values:Vec<u16>=(0..GROUP).map(|i|d.bits[if i==0{mi}else{(i*31+case*17)%(mi+1)}]|if i%3==0{0x8000}else{0}).collect();
            let (old,s)=baseline(&values);let got=fit(&values,&old,s).unwrap();
            assert_eq!(got.baseline_sse,scalar(&values,&old,s));
            let(mut best,mut best_sse,mut expected)=(0,got.baseline_sse,old.clone());
            for (j,c) in d.maximum[mi].as_ref().unwrap().choices.iter().enumerate(){
                let codes:Vec<i8>=values.iter().map(|&b|{let x=f32_units(f32::from_bits(u32::from(b&0x7fff)<<16)).unwrap();
                    ((2*x+c.units)/(2*c.units)).min(7) as i8 * if b&0x8000==0{1}else{-1}}).collect();
                let sse=scalar(&values,&codes,c.value);if sse<best_sse{best=j+1;best_sse=sse;expected=codes;}
            }
            assert_eq!((got.selected,got.selected_sse,&got.codes),(best,best_sse,&expected));assert!(got.selected_sse<=got.baseline_sse);
            let perm:Vec<_>=(0..GROUP).map(|i|values[(73*i+127)%GROUP]).collect();
            let (q,s)=baseline(&perm);let pg=fit(&perm,&q,s).unwrap();
            assert_eq!((pg.selected,pg.scale.to_bits(),pg.selected_sse),(got.selected,got.scale.to_bits(),got.selected_sse));
            for i in 0..GROUP{assert_eq!(pg.codes[i],got.codes[(73*i+127)%GROUP]);}
            for (&byte,pair) in pack(&got.codes).unwrap().iter().zip(got.codes.chunks_exact(2)){
                for (lane,&q) in pair.iter().enumerate(){let nib=(byte>>(4*lane))&15;let back=if nib>=8{nib as i8-16}else{nib as i8};assert_eq!(back,q);}
            }
        }
    }
    #[test]
    fn zeros_baseline_rounding_and_rejections(){
        let zero:Vec<_>=(0..GROUP).map(|i|if i%2==0{0}else{0x8000}).collect();let(q,s)=baseline(&zero);let z=fit(&zero,&q,s).unwrap();
        assert_eq!((z.selected,z.selected_sse,z.scale.to_bits()),(0,0,s.to_bits()));assert_eq!(z.codes,q);
        let mut values=vec![0x3f80;GROUP];values[0]=0x4000;let(q,s)=baseline(&values);assert_eq!(q[1],4);
        let su=f32_units(s).unwrap();assert_eq!(((2*UNIT+su)/(2*su)).min(7),3,"original rational code differs from stored-scale nearest");
        assert!(fit(&values[..959],&q,s).is_err());let mut bad=q.clone();bad[1]=3;assert!(fit(&values,&bad,s).is_err());
        assert!(fit(&values,&q,f32::from_bits(s.to_bits()+1)).is_err());
        values[3]=0x7fc0;assert!(fit(&values,&q,s).is_err());values[3]=1;assert!(fit(&values,&q,s).is_err());
        assert!(pack(&[8,0]).is_err());assert!(pack(&[0]).is_err());
    }
}

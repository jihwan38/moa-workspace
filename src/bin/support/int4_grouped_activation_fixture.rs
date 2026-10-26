//! Independent finite-BF16 quotient oracle and physically indexed packed bytes.
#[path = "rational_a4_oracle.rs"] mod rational;
pub const ROWS: usize = 512;
pub const GROUP: usize = 960;
pub const CASES: usize = 5;
fn mix(mut x:u64)->u64 { x^=x>>30;x=x.wrapping_mul(0xbf58476d1ce4e5b9);x^=x>>27;x=x.wrapping_mul(0x94d049bb133111eb);x^(x>>31) }
pub fn source(case:usize,p:usize,g:usize,k:usize,groups:usize)->(usize,usize,usize){
    if case>=3 {((73*p+17)%512,(3*g+1)%groups,(73*k+127)%960)} else {(p,g,k)}
}
pub fn fixture(case:usize,width:usize)->Vec<u16>{
    assert!(matches!(width,3840|15360));assert!(case<CASES);
    let groups=width/960;
    if case>=3 {
        let original=fixture(case-3,width);
        return (0..512*width).map(|i|{let(p,g,k)=source(case,i/width,(i%width)/960,i%960,groups);original[p*width+g*960+k]}).collect();
    }
    let mut out=vec![0u16;512*width];
    for p in 0..512 {for g in 0..groups {
        let id=p*groups+g;
        let ma=128+(mix(id as u64+0x13579bdf)%128) as u16;
        let exponent=if case==1 {[0u16,1,14,15,16,31,63,126,127,128,200,250,253,254][id%14]}else{90+(mix(id as u64+31)%70)as u16};
        let a=(exponent<<7)|(ma-128);
        for k in 0..960 {
            let z=mix((p as u64)<<32 ^ (g as u64)<<20 ^ k as u64 ^ 0xa5c317e9);
            let sign=((z&1) as u16)<<15;
            let delta=((z>>1)%9) as u16;
            let mut x=if exponent==0 {(z as u16)&127}else if delta>=exponent {0}else {((exponent-delta)<<7)|(((z>>9)%128)as u16)};
            if x>a {x=a;}
            x|=sign;
            if case==2 {
                // Exact half thresholds at A=14, x=1,3,...13, both signs.
                let shift=((p*7+g*13)%101)as i32-50;
                let v=match k%16 {0=>14.0f32,1=>0.0,_=>(2*((k%16-2)/2)+1)as f32};
                let v=if k%2==0 {v}else{-v};
                x=((v*2.0f32.powi(shift)).to_bits()>>16)as u16;
            }
            if case!=1 && k<13 { // All9 row bits and4 group bits are explicit sign tags.
                let tag=if k<9 {(p>>k)&1}else{(g>>(k-9))&1};
                let max=if case==2 {((14.0f32*2.0f32.powi(((p*7+g*13)%101)as i32-50)).to_bits()>>16)as u16}else{a};
                x=max|((tag as u16)<<15);
            }
            if case==1 && id%14==0 {x=if k%2==0 {0}else{0x8000};}
            if case==1 && id%14==1 {x=((z as u16)&127)|sign;}
            if case==1 && id%14>1 && (k==959||k==0) {x=a|sign;}
            out[p*width+g*960+k]=x;
        }
    }}
    out
}
pub struct Expected {pub packed:Vec<u8>,pub scales:Vec<u32>,pub offsets:Vec<i32>}
pub fn expected(bits:&[u16],width:usize)->Expected{
    assert_eq!(bits.len(),512*width);let groups=width/960;
    let mut packed=vec![0u8;bits.len()/2];let mut scales=vec![0u32;512*groups];let mut offsets=vec![0i32;512*groups];
    let mut writes=vec![false;512*groups];
    for p in 0..512 {for g in 0..groups {
        let row=&bits[p*width+g*960..p*width+(g+1)*960];
        let(q,s)=rational::reference(row); // Integer quotient/remainder, no LUT multiplication.
        let slot=((p/256)*groups+g)*256+p%256;
        assert!(!writes[slot]);writes[slot]=true;scales[slot]=s.to_bits();
        let a=row.iter().map(|b|{let x=b&0x7fff;assert!(x>>7!=255);if x>>7==0 {0}else{x}}).max().unwrap().max(rational::FLOOR_BITS);
        let normalized_mantissa=128+usize::from(a&127);
        offsets[slot]=((normalized_mantissa-128)*8*std::mem::size_of::<i32>())as i32;
        assert!((0..=4064).contains(&offsets[slot])&&offsets[slot]%32==0);
        for k in (0..960).step_by(2){
            assert!((-7..=7).contains(&q[k])&&(-7..=7).contains(&q[k+1]));
            packed[slot*480+k/2]=(q[k]&15)as u8|(((q[k+1]&15)as u8)<<4);
        }
    }}
    assert!(writes.into_iter().all(|x|x));Expected{packed,scales,offsets}
}
pub fn table()->Vec<i32>{(128u32..256).flat_map(|m|{
    let u=7*((1u32<<20).div_ceil(m));std::array::from_fn::<_,8,_>(|lane|if lane==0 {(u as f32).to_bits()as i32}else{(0x5a3c0000|(m<<4)|lane as u32)as i32})
}).collect()}
pub fn verify_permutation(case:usize,width:usize,got:&Expected,original:&Expected){
    let groups=width/960;
    for p in 0..512 {for g in 0..groups {
        let(sp,sg,_)=source(case,p,g,0,groups);let dst=((p/256)*groups+g)*256+p%256;let src=((sp/256)*groups+sg)*256+sp%256;
        assert_eq!(got.scales[dst],original.scales[src]);assert_eq!(got.offsets[dst],original.offsets[src]);
        for k in 0..960 {let(_,_,sk)=source(case,p,g,k,groups);let a=(got.packed[dst*480+k/2]>>(4*(k%2)))&15;let b=(original.packed[src*480+sk/2]>>(4*(sk%2)))&15;assert_eq!(a,b);}
    }}
}

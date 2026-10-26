//! Host-only reconstruction attribution. No device quantizer/LUT implementation.
use super::{cpu,fixture};
use furiosa_opt_std::prelude::bf16;
use std::collections::BTreeMap;

pub fn f32s(b:&[u8])->Vec<f32>{assert_eq!(b.len()%4,0);b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect()}
fn e4(b:u8)->f32 {
    let e=(b>>3)&15;let m=b&7;assert!(e!=15||m!=7);
    let x=if e==0{m as f32/512.0}else{(8+m)as f32*2f32.powi(e as i32-10)};
    if b&128==0{x}else{-x}
}
fn f4(b:u8)->f32 {let x=[0.,0.5,1.,1.5,2.,3.,4.,6.][(b&7)as usize];if b&8==0{x}else{-x}}
fn ftz(b:u16)->f32 {if b&0x7f80==0{f32::from_bits((b as u32&0x8000)<<16)}else{bf16::from_bits(b).to_f32()}}
fn encode_e4_nearest(x:f32)->u8 {
    assert!(x.is_finite());let sign=if x.is_sign_negative(){128}else{0};let x=x.abs();
    let(mut lo,mut hi)=(0u8,126u8);
    while lo<hi{let mid=lo+(hi-lo)/2;if e4(mid)<x{lo=mid+1;}else{hi=mid;}}
    let top=lo;let low=top.saturating_sub(1);let dl=x-e4(low);let dh=e4(top)-x;
    let out=if dl<dh{low}else if dh<dl{top}else if low&1==0{low}else{top};out|sign
}
#[derive(Default)] struct Error {energy:f64,error:f64,max:f64,zeros:usize,reference_zeros:usize,n:usize}
impl Error {
    fn add(&mut self,x:f32,y:f32){assert!(x.is_finite()&&y.is_finite());let e=x as f64-y as f64;self.energy+=(y as f64).powi(2);self.error+=e*e;self.max=self.max.max(e.abs());self.zeros+=usize::from(x==0.);self.reference_zeros+=usize::from(y==0.);self.n+=1;}
    fn json(&self)->serde_json::Value {serde_json::json!({"values":self.n,"relative_l2":(self.error/self.energy.max(1e-300)).sqrt(),"error_squared":self.error,"reference_squared":self.energy,"maxabs":self.max,"zeros":self.zeros,"zero_fraction":self.zeros as f64/self.n.max(1) as f64,"reference_zeros":self.reference_zeros})}
}
pub fn activations(label:&str,round:usize,bf:&[u8],native:&[u8],ns:&[u8],a4:&cpu::InputCpu) {
    let width=a4.cols;assert_eq!(bf.len(),512*width*2);assert_eq!(native.len(),512*width);let ns=f32s(ns);assert_eq!(ns.len(),512);
    let(mut ee,mut ie)=(Error::default(),Error::default());let mut rows=Vec::new();let mut columns=vec![0f32;width];
    for p in 0..512 {let(mut e,mut i)=(Error::default(),Error::default());let mut amax=0f32;
        for c in 0..width{let y=cpu::bf_at(bf,p*width+c);let at=cpu::input_group(p,c/960,width)*960+c%960;
            let x=e4(native[at])*ns[p];let z=a4.codes[p*width+c]as f32*a4.scales[p];e.add(x,y);i.add(z,y);ee.add(x,y);ie.add(z,y);amax=amax.max(y.abs());columns[c]=columns[c].max(y.abs());}
        rows.push(serde_json::json!({"p":p,"amax":amax,"E4_scale":ns[p],"A4_scale":a4.scales[p],"E4":e.json(),"A4":i.json()}));
    }
    println!("I4_ATTR_ACTIVATION {}",serde_json::json!({"label":label,"round":round,"width":width,"E4":ee.json(),"A4":ie.json(),"rows":rows,"column_absmax":columns,"scope":"all fixed BF16 rows; reconstruction only"}));
}
pub fn weights(t:&safetensors::SafeTensors<'_>,projection:&str,w:&cpu::WeightCpu,native:&[u8],scales:&[u8])->cpu::Result<()> {
    assert_eq!(native.len(),w.rows*w.cols);let scales=f32s(scales);assert_eq!(scales.len(),w.rows);
    let pre=format!("model.language_model.layers.0.mlp.{projection}");let packed=t.tensor(&format!("{pre}.weight_packed"))?;let blocks=t.tensor(&format!("{pre}.weight_scale"))?;
    let(mut ee,mut ie)=(Error::default(),Error::default());let mut rows=Vec::new();let mut columns=vec![[f32::INFINITY,f32::NEG_INFINITY,0.,0.];w.cols];
    // Each distinct exact power2 scale gets an independent finite BF16->E4 table.
    // This is a host oracle acceleration only, never serving calibration.
    let mut tables:BTreeMap<u32,Vec<u8>>=BTreeMap::new();
    for row in 0..w.rows {
        let values:Vec<u16>=(0..w.cols).map(|c|{let at=row*w.cols+c;let nib=(packed.data()[at/2]>>(4*(at%2)))&15;bf16::from_f32(f4(nib)*e4(blocks.data()[at/16])).to_bits()}).collect();
        let a=values.iter().map(|&v|ftz(v).abs()).fold(2f32.powi(-112),f32::max);
        let expected=f32::from_bits(a.to_bits()&0x7f800000)*0.0078125;
        assert_eq!(scales[row].to_bits(),expected.to_bits(),"independent native row-scale {projection} row={row}");
        let table=tables.entry(expected.to_bits()).or_insert_with(||(0..65536).map(|b|{
            if b&0x7f80==0x7f80{0}else{let x=ftz(b as u16)/expected;if x.is_finite(){encode_e4_nearest(x)}else{0}}
        }).collect());
        let(mut er,mut ir)=(Error::default(),Error::default());
        for c in 0..w.cols {let at=cpu::weight_group(row,c/960,w.cols)*960+c%960;let y=bf16::from_bits(values[c]).to_f32();
            assert_eq!(native[at],table[values[c]as usize],"independent native E4 code {projection} row={row} channel={c}");
            let x=e4(native[at])*scales[row];let z=e4(w.dense[at])*w.scales[at/960];
            ee.add(x,y);ie.add(z,y);er.add(x,y);ir.add(z,y);columns[c][0]=columns[c][0].min(y);columns[c][1]=columns[c][1].max(y);columns[c][2]=columns[c][2].max(x.abs());columns[c][3]=columns[c][3].max(z.abs());}
        rows.push(serde_json::json!([row,(er.error/er.energy.max(1e-300)).sqrt(),(ir.error/ir.energy.max(1e-300)).sqrt(),er.zeros,ir.zeros]));
    }
    println!("I4_ATTR_WEIGHT {}",serde_json::json!({"projection":projection,"rows":w.rows,"cols":w.cols,"native_codes_exact":native.len(),"native_scale_bits_exact":scales.len(),"E4":ee.json(),"I4":ie.json(),"row_fields":["row","E4_relative_l2","I4_relative_l2","E4_zeros","I4_zeros"],"row_errors":rows,"column_ranges_provided_min_max_E4absmax_I4absmax":columns,"globals_excluded":true,"scope":"direct provided BF16; no native row scale on I4"}));
    Ok(())
}
pub fn interaction(label:&str,round:usize,arms:&[Vec<u8>]) {
    assert_eq!(arms.len(),4);assert!(arms.iter().all(|a|a.len()==arms[0].len()));let(mut sum,mut max)=(0f64,0f64);
    // Order EE, IE, EI, II. F64 diagnostics do not alter kernel arithmetic.
    for i in 0..arms[0].len()/2{let v:[f64;4]=std::array::from_fn(|a|cpu::bf_at(&arms[a],i)as f64);let z=v[3]-v[1]-v[2]+v[0];assert!(z.is_finite());sum+=z*z;max=max.max(z.abs());}
    println!("I4_ATTR_INTERACTION {}",serde_json::json!({"label":label,"round":round,"values":arms[0].len()/2,"sum_squared":sum,"maxabs":max,"meaning":"II-IE-EI+EE; rounding/nonlinear interaction, not additive L2 attribution"}));
}
pub fn canonical_hidden_to_blocked(b:&[u8])->Vec<u8>{
    assert_eq!(b.len(),512*cpu::EXPANDED*2);let mut out=vec![0;b.len()];let mut seen=vec![false;512*cpu::EXPANDED];
    for p in 0..512{for c in 0..cpu::EXPANDED{let at=cpu::output_index(p,c,cpu::EXPANDED);assert!(!seen[at]);seen[at]=true;out[at*2..at*2+2].copy_from_slice(&b[(p*cpu::EXPANDED+c)*2..(p*cpu::EXPANDED+c+1)*2]);}}
    assert_eq!(fixture::canonical_projection(&out,cpu::EXPANDED,0),b);out
}

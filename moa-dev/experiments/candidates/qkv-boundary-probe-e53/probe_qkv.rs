//! Standalone QKV diagnostics; original device sources and test harness are unchanged.
use furiosa_opt_std::prelude::*;
use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops};
use std::io::Write;

const NAMES: [&str; 2] = ["tiny_lost_after_pre_rms", "larger_tiny_preserved_control"];
const POS: usize = 1763;
const K_SENTINEL: u16 = 0x3e80;
const V_SENTINEL: u16 = 0xbf00;

fn checksum(bytes: &[u8]) -> u32 {
    bytes.chunks(4).enumerate().fold(0u32, |acc,(i,chunk)| {
        let mut word=[0u8;4]; word[..chunk.len()].copy_from_slice(chunk);
        acc.wrapping_add(u32::from_le_bytes(word).wrapping_mul((2*i+1) as u32))
    })
}
fn weight_codes(rows: usize) -> Vec<u8> {
    (0..rows*H::SIZE).map(|i| if i%H::SIZE%480 < 2 {0x38} else {0x77}).collect()
}
fn sentinel(count: usize, bits: u16) -> Vec<u8> {
    (0..count).flat_map(|_| bits.to_le_bytes()).collect()
}
fn raw(f: &safetensors::SafeTensors<'_>, case: usize, name: &str) -> Vec<u8> {
    let t=f.tensor(&format!("case{case}.input.{name}")).unwrap();
    assert_eq!(t.dtype(),safetensors::Dtype::U8); t.data().to_vec()
}
fn check_input(f: &safetensors::SafeTensors<'_>, case: usize, name: &str, bytes: &[u8]) {
    let t=f.tensor(&format!("case{case}.checksum.{name}")).unwrap();
    assert_eq!(t.dtype(),safetensors::Dtype::I64);
    let want=i64::from_le_bytes(t.data().try_into().unwrap()) as u32;
    assert_eq!(checksum(bytes),want,"case {case} {name} payload/generator checksum");
    println!("INPUT_CHECK case={case} tensor={name} bytes={} checksum={want:08x} status=PASS",bytes.len());
}
fn expected(f: &safetensors::SafeTensors<'_>,case:usize,key:&str,n:usize)->Vec<f32> {
    let t=f.tensor(&format!("case{case}.expected.{key}")).unwrap();
    assert_eq!(t.dtype(),safetensors::Dtype::F32);
    let v:Vec<_>=t.data().chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!(v.len(),n); assert!(v.iter().all(|v|v.is_finite())); v
}
fn compare(case:usize,key:&str,got:&[bf16],want:&[f32])->usize {
    assert_eq!(got.len(),want.len());
    let(mut fail,mut nonfinite,mut exact)=(0usize,0usize,0usize);
    let(mut max,mut sum)=(0.0f32,0.0f64);
    let mut first_mismatch: Option<serde_json::Value> = None;
    for(i,(&g,&w))in got.iter().zip(want).enumerate(){
        let v=g.to_f32(); let err=(v-w).abs(); let limit=0.04f32+0.01f32*w.abs();
        if v.to_bits()==w.to_bits(){exact+=1;}
        if v.is_finite(){max=max.max(err);sum+=f64::from(err);}else{nonfinite+=1;}
        if !v.is_finite()||err>limit{
            if first_mismatch.is_none(){first_mismatch=Some(serde_json::json!({"index":i,"got":v,"want":w,"got_bf16":format!("{:04x}",g.to_bits()),"want_bf16":format!("{:04x}",bf16::from_f32(w).to_bits()),"error":err,"limit":limit}));}
            if fail<16 {println!("QKV_MISMATCH case={case} output={key} index={i} got={v:.9e} want={w:.9e} got_bf16={:04x} want_bf16={:04x} error={err:.9e} limit={limit:.9e}",g.to_bits(),bf16::from_f32(w).to_bits());}
            fail+=1;
        }
    }
    println!("QKV_RESULT case={case} name={} output={key} checked={} failures={fail} nonfinite={nonfinite} exact_f32={exact} max_abs_finite={max:.9e} mean_abs_finite={:.9e} status={}",NAMES[case],got.len(),sum/(got.len()-nonfinite).max(1)as f64,if fail==0{"PASS"}else{"FAIL"});
    println!("{}",serde_json::json!({"kind":"qkv_result","case":case,"name":NAMES[case],"output":key,"count":got.len(),"fail":fail,"nonfinite":nonfinite,"exact":exact,"max_error_finite":max,"first_mismatch":first_mismatch,"status":if fail==0{"PASS"}else{"FAIL"}})); fail
}
fn check_cache(case:usize,key:&str,phase:&str,values:&[bf16],bits:u16)->usize{
    assert_eq!(values.len(),Ts::SIZE*Ps::SIZE);
    let start=(POS%Ts::SIZE)*Ps::SIZE;
    let(mut failures,mut checked)=(0usize,0usize);
    for(i,&v)in values.iter().enumerate(){
        if phase=="after"&&(start..start+Ps::SIZE).contains(&i){continue;}
        checked+=1;
        if v.to_bits()!=bits{
            if failures<8{println!("CACHE_MISMATCH case={case} cache={key} phase={phase} index={i} got={:04x} want={bits:04x}",v.to_bits());}
            failures+=1;
        }
    }
    println!("CACHE_CHECK case={case} cache={key} phase={phase} checked={checked} excluded_start={} excluded_end={} failures={failures} status={}",if phase=="after"{start}else{0},if phase=="after"{start+Ps::SIZE}else{0},if failures==0{"PASS"}else{"FAIL"});
    println!("{}",serde_json::json!({"kind":"cache_check","case":case,"cache":key,"phase":phase,"count":checked,"fail":failures,"sentinel_bf16":format!("{bits:04x}"),"excluded_start":if phase=="after"{start}else{0},"excluded_end":if phase=="after"{start+Ps::SIZE}else{0},"status":if failures==0{"PASS"}else{"FAIL"}}));failures
}
async fn upload<D:ScalarBytes+MaterializableScalar,T:M>(d:&mut Device,b:Vec<u8>)->HbmTensor<D,Chip,T>{
    assert_eq!(b.len(),T::SIZE*D::BITS/8);
    HostTensor::<D,T>::from_buf(b).to_hbm(&mut d.pdma).await.unwrap()
}
async fn read<T:M>(d:&mut Device,t:&HbmTensor<bf16,Chip,T>)->Vec<bf16>{
    let h:HostTensor<bf16,T>=t.to_host(&mut d.pdma).await.unwrap();h.into_vec()
}

#[tokio::main]
async fn main(){
    let args:Vec<_>=std::env::args().collect();
    let host_only=args.iter().any(|a|a=="--host-check-only");
    let dump=args.windows(2).find(|a|a[0]=="--dump-generated").map(|a|a[1].clone());
    let path=args.windows(2).find(|a|a[0]=="--fixture").map(|a|a[1].as_str()).unwrap_or("boundary.safetensors");
    let bytes=std::fs::read(path).expect("read immutable oracle payload");
    let fixture=safetensors::SafeTensors::deserialize(&bytes).unwrap();
    if let Some(dir)=&dump{assert!(host_only);std::fs::create_dir_all(dir).unwrap();}
    let mut device=if host_only{None}else{Some(Device::new(ops::sliding_project_qkv.topology()).unwrap())};
    println!("QKV_SUITE_BEGIN cases=2 atol=0.04 rtol=0.01 diagnostic=true official_performance=false host_only={host_only} pos={POS} slot={}",POS%Ts::SIZE);
    let(mut output_failed,mut cache_failed,mut completed)=(0usize,0usize,0usize);
    for case in 0..NAMES.len(){
        let mut data=std::collections::BTreeMap::<String,Vec<u8>>::new();
        for name in ["x","input_rms_weight","q_weight_scale","k_weight_scale","v_weight_scale","q_rms_weight","k_rms_weight","cos_row","sin_row","rope_offset","kv_offset"]{data.insert(name.to_string(),raw(&fixture,case,name));}
        for(name,rows)in[("q_weight",Qs::SIZE),("k_weight",Ps::SIZE),("v_weight",Ps::SIZE)]{data.insert(name.to_string(),weight_codes(rows));}
        for name in ["cos","sin"]{
            let row=&data[&format!("{name}_row")];assert_eq!(row.len(),Ds::SIZE*2);
            let mut full=vec![0u8;E::SIZE*Ds::SIZE*2];
            full[POS*Ds::SIZE*2..(POS+1)*Ds::SIZE*2].copy_from_slice(row);data.insert(name.to_string(),full);
        }
        for(name,count,bits)in[("k_cache_init",Ts::SIZE*Ps::SIZE,K_SENTINEL),("v_cache_init",Ts::SIZE*Ps::SIZE,V_SENTINEL),("q_out_init",Qs::SIZE,0x3fc0)]{data.insert(name.to_string(),sentinel(count,bits));}
        assert_eq!(i32::from_le_bytes(data["rope_offset"].as_slice().try_into().unwrap()),(POS*Ds::SIZE*2)as i32);
        assert_eq!(i32::from_le_bytes(data["kv_offset"].as_slice().try_into().unwrap()),((POS%Ts::SIZE)*Ps::SIZE*2)as i32);
        for(name,bytes)in &data{
            check_input(&fixture,case,name,bytes);
            if let Some(dir)=&dump{std::fs::write(format!("{dir}/case{case}.{name}.bin"),bytes).unwrap();}
        }
        let want_q=expected(&fixture,case,"q",Qs::SIZE);let want_k=expected(&fixture,case,"k",Ps::SIZE);let want_v=expected(&fixture,case,"v",Ps::SIZE);
        if host_only{completed+=1;continue;}
        let d=device.as_mut().unwrap();
        let x:HbmTensor<bf16,Chip,m![H]>=upload(d,data.remove("x").unwrap()).await;
        let input_gamma:HbmTensor<bf16,Chip,m![H]>=upload(d,data.remove("input_rms_weight").unwrap()).await;
        let qw:HbmTensor<f8e4m3,Chip,m![Qs,H]>=upload(d,data.remove("q_weight").unwrap()).await;
        let kw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(d,data.remove("k_weight").unwrap()).await;
        let vw:HbmTensor<f8e4m3,Chip,m![Ps,H]>=upload(d,data.remove("v_weight").unwrap()).await;
        let qs:HbmTensor<bf16,Chip,m![Qs]>=upload(d,data.remove("q_weight_scale").unwrap()).await;
        let ks:HbmTensor<bf16,Chip,m![Ps]>=upload(d,data.remove("k_weight_scale").unwrap()).await;
        let vs:HbmTensor<bf16,Chip,m![Ps]>=upload(d,data.remove("v_weight_scale").unwrap()).await;
        let qg:HbmTensor<bf16,Chip,m![Ds]>=upload(d,data.remove("q_rms_weight").unwrap()).await;
        let kg:HbmTensor<bf16,Chip,m![Ds]>=upload(d,data.remove("k_rms_weight").unwrap()).await;
        let cos:HbmTensor<bf16,Chip,m![E,Ds]>=upload(d,data.remove("cos").unwrap()).await;
        let sin:HbmTensor<bf16,Chip,m![E,Ds]>=upload(d,data.remove("sin").unwrap()).await;
        let ro:HbmTensor<i32,Chip,m![1]>=upload(d,data.remove("rope_offset").unwrap()).await;
        let ko:HbmTensor<i32,Chip,m![1]>=upload(d,data.remove("kv_offset").unwrap()).await;
        let mut kc:HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>=upload(d,data.remove("k_cache_init").unwrap()).await;
        let mut vc:HbmTensor<bf16,Chip,m![Ts,Ns,Ds]>=upload(d,data.remove("v_cache_init").unwrap()).await;
        let mut qo:HbmTensor<bf16,Chip,m![Ns,Gs,Ds]>=upload(d,data.remove("q_out_init").unwrap()).await;
        cache_failed+=check_cache(case,"k","before",&read(d,&kc).await,K_SENTINEL);
        cache_failed+=check_cache(case,"v","before",&read(d,&vc).await,V_SENTINEL);
        println!("QKV_LAUNCH_BEGIN case={case} name={}",NAMES[case]);std::io::stdout().flush().unwrap();
        launch(ops::sliding_project_qkv,(&mut *d,&x,&qw,&kw,&vw,&qs,&ks,&vs,&input_gamma,&qg,&kg,&ko,&ro,&cos,&sin,&mut kc,&mut vc,&mut qo)).await.expect("QKV diagnostic launch");
        let kval=read(d,&kc).await;let vval=read(d,&vc).await;let qval=read(d,&qo).await;
        cache_failed+=check_cache(case,"k","after",&kval,K_SENTINEL);
        cache_failed+=check_cache(case,"v","after",&vval,V_SENTINEL);
        let start=(POS%Ts::SIZE)*Ps::SIZE;
        output_failed+=compare(case,"q",&qval,&want_q);
        output_failed+=compare(case,"k",&kval[start..start+Ps::SIZE],&want_k);
        output_failed+=compare(case,"v",&vval[start..start+Ps::SIZE],&want_v);
        completed+=1;
    }
    println!("QKV_SUITE_RESULT completed={completed} planned=2 total_failed_values={output_failed} cache_failed_values={cache_failed} status={}",if output_failed+cache_failed==0{"PASS"}else{"FAIL"});
    if output_failed+cache_failed!=0{std::process::exit(1);}
}

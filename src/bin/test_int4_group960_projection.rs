//! Layer0 intended-code projection gate only: not complete FFN/model quality.
use furiosa_opt_gemma4::{Chip,axes::{P512,H},host::load,
    ops_batch_layouts::{P512H960,P512L960},
    ops_int4_group960_projection as candidate,ops_int4_group960_reference as reference};
use furiosa_opt_std::prelude::*;
use std::{fs::File,time::Instant};
#[path="support/npu_profile.rs"] mod npu_profile;
#[path="support/int4_group960_fixture.rs"] mod cpu;
use cpu::{Result,InputCpu,WeightCpu,HIDDEN,EXPANDED,TOKENS};

async fn upload<D:MaterializableScalar,E:M>(d:&mut Device,b:Vec<u8>)->Result<HbmTensor<D,Chip,E>> {
    assert_eq!(b.len()*8,E::SIZE*D::BITS);
    Ok(HostTensor::<D,E>::from_buf(b).to_hbm(&mut d.pdma).await?)
}
async fn read<D:MaterializableScalar,E:M>(d:&mut Device,x:&HbmTensor<D,Chip,E>)->Result<Vec<u8>> {
    Ok(x.to_host::<E>(&mut d.pdma).await?.into_inner().into_buf())
}
fn floats(v:&[f32])->Vec<u8> {v.iter().flat_map(|x|x.to_le_bytes()).collect()}
fn joined(a:&[u8],b:&[u8])->Vec<u8> {let mut v=Vec::with_capacity(a.len()+b.len());v.extend_from_slice(a);v.extend_from_slice(b);v}
fn compare_joined(got:&[u8],a:&[u8],b:&[u8],label:&str) {
    assert_eq!(got.len(),a.len()+b.len());assert_eq!(&got[..a.len()],a,"immutable UP plane {label}");assert_eq!(&got[a.len()..],b,"immutable GATE plane {label}");
}
async fn poison<E:M>(d:&mut Device,payload:u16)->Result<HbmTensor<bf16,Chip,E>> {
    Ok(HostTensor::<bf16,E>::from_vec(std::iter::repeat_n(bf16::from_bits(payload),E::SIZE)).to_hbm(&mut d.pdma).await?)
}
struct InputGpu<E:M> {packed:HbmTensor<i4,Chip,E>,dense:HbmTensor<f8e4m3,Chip,E>,scales:HbmTensor<f32,Chip,m![P512]>}
impl<E:M> InputGpu<E> {
    async fn new(d:&mut Device,x:&InputCpu)->Result<Self> {
        Ok(Self{packed:upload(d,x.packed.clone()).await?,dense:upload(d,x.dense.clone()).await?,scales:upload(d,floats(&x.scales)).await?})
    }
    async fn immutable(&self,d:&mut Device,x:&InputCpu)->Result<usize> {
        assert_eq!(read(d,&self.packed).await?,x.packed,"I4 input immutable");
        assert_eq!(read(d,&self.dense).await?,x.dense,"E4 input immutable");
        assert_eq!(read(d,&self.scales).await?,floats(&x.scales),"input scales immutable");
        Ok(x.packed.len()+x.dense.len()+x.scales.len()*4)
    }
}
struct WeightsGpu {
    joined_i4:HbmTensor<i4,Chip,candidate::JoinedWeight>,joined_e4:HbmTensor<f8e4m3,Chip,reference::JoinedWeight>,
    joined_scales:HbmTensor<f32,Chip,candidate::JoinedBlockScale>,unit_up:HbmTensor<f32,Chip,reference::JoinedScale>,
    down_i4:HbmTensor<i4,Chip,candidate::DownWeight>,down_e4:HbmTensor<f8e4m3,Chip,candidate::DownWeight>,
    down_scales:HbmTensor<f32,Chip,candidate::DownBlockScale>,unit_down:HbmTensor<f32,Chip,m![H]>,global:HbmTensor<f32,Chip,m![1]>,
}
impl WeightsGpu {
    async fn new(d:&mut Device,up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu)->Result<Self> {
        assert_eq!((up.rows,up.cols),(EXPANDED,HIDDEN));assert_eq!((gate.rows,gate.cols),(EXPANDED,HIDDEN));assert_eq!((down.rows,down.cols),(HIDDEN,EXPANDED));
        Ok(Self{
            joined_i4:upload(d,joined(&up.packed,&gate.packed)).await?,
            joined_e4:upload(d,joined(&up.dense,&gate.dense)).await?,
            joined_scales:upload(d,joined(&floats(&up.scales),&floats(&gate.scales))).await?,
            unit_up:HostTensor::<f32,reference::JoinedScale>::from_vec(std::iter::repeat_n(1.0,2*EXPANDED)).to_hbm(&mut d.pdma).await?,
            down_i4:upload(d,down.packed.clone()).await?,down_e4:upload(d,down.dense.clone()).await?,
            down_scales:upload(d,floats(&down.scales)).await?,
            unit_down:HostTensor::<f32,m![H]>::from_vec(std::iter::repeat_n(1.0,HIDDEN)).to_hbm(&mut d.pdma).await?,
            global:upload(d,down.global.to_le_bytes().to_vec()).await?,
        })
    }
    async fn immutable(&self,d:&mut Device,up:&WeightCpu,gate:&WeightCpu,down:&WeightCpu)->Result<usize> {
        compare_joined(&read(d,&self.joined_i4).await?,&up.packed,&gate.packed,"I4");
        compare_joined(&read(d,&self.joined_e4).await?,&up.dense,&gate.dense,"E4");
        compare_joined(&read(d,&self.joined_scales).await?,&floats(&up.scales),&floats(&gate.scales),"scales");
        assert_eq!(read(d,&self.down_i4).await?,down.packed,"DOWN I4 immutable");
        assert_eq!(read(d,&self.down_e4).await?,down.dense,"DOWN E4 immutable");
        assert_eq!(read(d,&self.down_scales).await?,floats(&down.scales),"DOWN scales immutable");
        assert_eq!(read(d,&self.unit_up).await?,floats(&vec![1.0;2*EXPANDED]),"UP/GATE reference units");
        assert_eq!(read(d,&self.unit_down).await?,floats(&vec![1.0;HIDDEN]),"DOWN reference units");
        assert_eq!(read(d,&self.global).await?,down.global.to_le_bytes(),"DOWN global immutable");
        Ok(up.packed.len()+gate.packed.len()+down.packed.len()+up.dense.len()+gate.dense.len()+down.dense.len()+
            (up.scales.len()+gate.scales.len()+down.scales.len()+2*EXPANDED+HIDDEN+1)*4)
    }
}
async fn project_up(d:&mut Device,w:&WeightsGpu,x:&InputGpu<P512H960>,out:&mut HbmTensor<bf16,Chip,candidate::JoinedOutput>,is_candidate:bool)->Result<()> {
    if is_candidate {launch(candidate::project_upgate,(d,&x.packed,&x.scales,&w.joined_i4,&w.joined_scales,out)).await?;}
    else {launch(reference::project_upgate,(d,&x.dense,&x.scales,&w.joined_e4,&w.joined_scales,&w.unit_up,out)).await?;}
    Ok(())
}
async fn project_down(d:&mut Device,w:&WeightsGpu,x:&InputGpu<P512L960>,out:&mut HbmTensor<bf16,Chip,candidate::Output>,is_candidate:bool)->Result<()> {
    if is_candidate {launch(candidate::project_down,(d,&x.packed,&x.scales,&w.down_i4,&w.down_scales,&w.global,out)).await?;}
    else {launch(reference::project_down,(d,&x.dense,&x.scales,&w.down_e4,&w.down_scales,&w.unit_down,&w.global,out)).await?;}
    Ok(())
}

#[tokio::main]
async fn main()->Result<()> {
    if std::env::args().len()!=1 {return Err("this preregistered probe accepts no layer/fixture overrides; layer0 and two fixed row orders only".into());}
    let profile=npu_profile::Collector::default();tracing::subscriber::set_global_default(profile.clone())?;
    cpu::check_output_layout(EXPANDED);cpu::check_output_layout(HIDDEN);
    let startup=Instant::now();let path=load::model_dir()?.join("model.safetensors");let file=File::open(&path)?;
    // Read-only file mapping, identical to existing provided-weight diagnostics.
    let mmap=unsafe{memmap2::Mmap::map(&file)?};let tensors=safetensors::SafeTensors::deserialize(&mmap)?;
    println!("I4_GROUP960_SOURCE layer=0 file={} file_bytes={} reference=BF16_provided_F4_times_E4_before_global no_native_E4_row_dequant=true",
        path.display(),file.metadata()?.len());
    let up=WeightCpu::load(&tensors,"up_proj")?;let gate=WeightCpu::load(&tensors,"gate_proj")?;let down=WeightCpu::load(&tensors,"down_proj")?;
    for (name,w) in [("UP",&up),("GATE",&gate),("DOWN",&down)] {
        println!("I4_GROUP960_WEIGHT name={name} values={} scales={} packed_fnv1a64={:016x} dense_fnv1a64={:016x} global_reciprocal_bits={:08x} original_source_fnv1a64={:?}",
            w.rows*w.cols,w.scales.len(),cpu::fingerprint(&w.packed),cpu::fingerprint(&w.dense),w.global.to_bits(),w.source);
    }
    println!("I4_GROUP960_STARTUP cpu_load_reconstruct_quantize_audit_ms={:.6} values={} scale_groups={} code_rule=independent_u128_quotient_remainder",
        startup.elapsed().as_secs_f64()*1e3,3*HIDDEN*EXPANDED,3*HIDDEN*EXPANDED/960);
    let mut d=Device::new(candidate::project_upgate.topology())?;
    let upload_start=Instant::now();let w=WeightsGpu::new(&mut d,&up,&gate,&down).await?;
    println!("I4_GROUP960_STARTUP weight_upload_ms={:.6} estimated_peak_live_device_payload_bytes=351824900 allocator_overhead_excluded=true down_resource_status=HOLD_245684_gt_240000",
        upload_start.elapsed().as_secs_f64()*1e3);
    let mut baseline:Option<(Vec<u8>,Vec<u8>)>=None;
    let(mut checked,mut sampled,mut permuted,mut immutable)=(0usize,0usize,0usize,0usize);
    for round in 0..2 {
        let prep=Instant::now();let xi=InputCpu::new(HIDDEN,round==1);let hi=InputCpu::new(EXPANDED,round==1);
        println!("I4_GROUP960_INPUT round={round} cpu_fixture_quantize_pack_ms={:.6} input_scales=512 hidden_scales=512 all_rows_unique=true",prep.elapsed().as_secs_f64()*1e3);
        let upload_start=Instant::now();let x=InputGpu::<P512H960>::new(&mut d,&xi).await?;let h=InputGpu::<P512L960>::new(&mut d,&hi).await?;
        let mut cu=poison::<candidate::JoinedOutput>(&mut d,0x7fc1).await?;let mut ru=poison::<candidate::JoinedOutput>(&mut d,0x7fc2).await?;
        let mut cd=poison::<candidate::Output>(&mut d,0x7fc3).await?;let mut rd=poison::<candidate::Output>(&mut d,0x7fc4).await?;
        println!("I4_GROUP960_INPUT round={round} input_upload_output_poison_ms={:.6}",upload_start.elapsed().as_secs_f64()*1e3);
        // Candidate completes first. No shared output or scratch with the reference.
        project_up(&mut d,&w,&x,&mut cu,true).await?;project_down(&mut d,&w,&h,&mut cd,true).await?;
        let cb_u=read(&mut d,&cu).await?;let cb_d=read(&mut d,&cd).await?;
        project_up(&mut d,&w,&x,&mut ru,false).await?;project_down(&mut d,&w,&h,&mut rd,false).await?;
        let rb_u=read(&mut d,&ru).await?;let rb_d=read(&mut d,&rd).await?;
        checked+=cpu::compare_bits(&cb_u,&rb_u,"joined_upgate");checked+=cpu::compare_bits(&cb_d,&rb_d,"down");
        for (name,bytes,plane,input,weight,is_down) in [("UP",&cb_u,0,&xi,&up,false),("GATE",&cb_u,1,&xi,&gate,false),("DOWN",&cb_d,0,&hi,&down,true)] {
            let(n,radius)=cpu::sampled_bound(bytes,plane,input,weight,is_down);sampled+=n;
            println!("I4_GROUP960_F64 round={round} projection={name} samples={n} max_preBF16_radius={radius:.12e} local_dot=i64_exact bound=gamma_2Nplus2_plus_FTZ_plus_F64_error");
        }
        if let Some((base_u,base_d))=&baseline {
            permuted+=cpu::compare_permutation(&cb_u,base_u,EXPANDED,2);permuted+=cpu::compare_permutation(&cb_d,base_d,HIDDEN,1);
            // Both independent routes also satisfy the same complete permutation.
            cpu::compare_permutation(&rb_u,base_u,EXPANDED,2);cpu::compare_permutation(&rb_d,base_d,HIDDEN,1);
        }
        // Eight complete awaits/block; CPU setup, downloads and profiling are outside.
        for (block,is_candidate) in [true,false,false,true].into_iter().enumerate() {
            let start=Instant::now();for _ in 0..8 {if is_candidate{project_up(&mut d,&w,&x,&mut cu,true).await?;}else{project_up(&mut d,&w,&x,&mut ru,false).await?;}}
            let up_us=start.elapsed().as_secs_f64()*1e6/8.0;
            let start=Instant::now();for _ in 0..8 {if is_candidate{project_down(&mut d,&w,&h,&mut cd,true).await?;}else{project_down(&mut d,&w,&h,&mut rd,false).await?;}}
            let down_us=start.elapsed().as_secs_f64()*1e6/8.0;
            println!("I4_GROUP960_TIME round={round} block={block} candidate={is_candidate} samples=8 upgate_us={up_us:.6} down_us={down_us:.6} scope=projection_with_all_output_stores excludes=activation_quant_pack_GeGLU_upload_download reference_extra_row_mul=unity");
        }
        if round==0 && std::env::var_os("FURIOSA_OPT_PROFILE").is_some() {
            for is_candidate in [false,true] {
                println!("PROFILE_I4_GROUP960_UPGATE candidate={is_candidate}");profile.enable(true);
                if is_candidate{project_up(&mut d,&w,&x,&mut cu,true).await?;}else{project_up(&mut d,&w,&x,&mut ru,false).await?;}profile.enable(false);
                println!("PROFILE_I4_GROUP960_DOWN candidate={is_candidate}");profile.enable(true);
                if is_candidate{project_down(&mut d,&w,&h,&mut cd,true).await?;}else{project_down(&mut d,&w,&h,&mut rd,false).await?;}profile.enable(false);
            }
        }
        assert_eq!(read(&mut d,&cu).await?,cb_u,"candidate UP/GATE after all reuse");assert_eq!(read(&mut d,&cd).await?,cb_d,"candidate DOWN after all reuse");
        assert_eq!(read(&mut d,&ru).await?,rb_u,"reference UP/GATE after all reuse");assert_eq!(read(&mut d,&rd).await?,rb_d,"reference DOWN after all reuse");
        immutable+=x.immutable(&mut d,&xi).await?+h.immutable(&mut d,&hi).await?;
        immutable+=w.immutable(&mut d,&up,&gate,&down).await?;
        up.verify_source(&tensors)?;gate.verify_source(&tensors)?;down.verify_source(&tensors)?;
        println!("I4_GROUP960_EXACT round={round} output_bf16_values={} all_output_bits=true finite=true device_and_supplied_sources_immutable=true",(cb_u.len()+cb_d.len())/2);
        if round==0 {baseline=Some((cb_u,cb_d));}
    }
    assert_eq!(checked,35_389_440);assert_eq!(sampled,3072);assert_eq!(permuted,17_694_720);
    println!("PASS_I4_GROUP960_PROJECTION layer=0 output_bf16_values={checked} independent_i64_F64_samples={sampled} candidate_permutation_values={permuted} immutable_bytes_compared={immutable} down_resource_HOLD_preserved=true no_full_FFN_or_model_quality_claim=true");
    Ok(())
}

#[path = "support/npu_profile.rs"]
mod npu_profile;
// R64 IO: independent embedding bytes, fourB16 logit controls, sampled F64 dots.
use furiosa_opt_gemma4::{axes::{B,R64,H,W},EMBED_SCALE,LOGIT_SOFTCAP,ops_batch_io,ops_r64_io,ops_r64_io_bandoutput,ops_r64_io_blocked::{self as blocked,BlockedLogits}};
use furiosa_opt_std::prelude::*;
fn word(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn hidden(b:usize,h:usize)->bf16 {bf16::from_f32(((word((b*H::SIZE+h)as u64+37)>>40)as f32/16777216.0-0.5)*2.0)}
fn weight(n:usize,h:usize)->bf16 {bf16::from_f32(((word(((n%257)*H::SIZE+h)as u64+79)>>40)as f32/16777216.0-0.5)*0.0625)}
#[tokio::main(flavor="current_thread")]
async fn main()->Result<(),Box<dyn std::error::Error>> {
 let collector=npu_profile::Collector::default();tracing::subscriber::set_global_default(collector.clone())?;
 let mut device=Device::new(ops_r64_io_bandoutput::logits_batch.topology())?;
 let patterns:Vec<Vec<u8>>=(0..257).map(|n|(0..H::SIZE).flat_map(|h|weight(n,h).to_bits().to_le_bytes()).collect()).collect();
 let mut bytes=vec![0u8;W::SIZE*H::SIZE*2];
 for (n,row) in bytes.chunks_exact_mut(H::SIZE*2).enumerate(){row.copy_from_slice(&patterns[n%257]);}
 let weights=HostTensor::<bf16,m![W,H]>::from_buf(bytes).to_hbm(&mut device.pdma).await?;
 let tokens:Vec<usize>=(0..R64::SIZE).map(|r|match r {0=>0,31=>W::SIZE-1,32=>W::SIZE/2,63=>1,_=>(word(r as u64+419)as usize)%W::SIZE}).collect();
 let offsets=HostTensor::<i32,m![R64]>::from_vec(tokens.iter().map(|n|(n*H::SIZE*2)as i32)).to_hbm(&mut device.pdma).await?;
 let mut embedded=HostTensor::<bf16,m![R64,H]>::zero().to_hbm(&mut device.pdma).await?;
 launch(ops_r64_io::embed_tokens,(&mut device,&weights,&offsets,&mut embedded)).await?;
 let got=embedded.to_host::<m![R64,H]>(&mut device.pdma).await?.into_vec();let mut bad=0;
 for r in 0..R64::SIZE{for h in 0..H::SIZE{bad+=usize::from(got[r*H::SIZE+h].to_bits()!=bf16::from_f32(weight(tokens[r],h).to_f32()*EMBED_SCALE).to_bits());}}
 println!("R64_embedding checked={} bit_differences={bad}",got.len());assert_eq!(bad,0);
 let input=HostTensor::<bf16,m![R64,H]>::from_vec((0..R64::SIZE).flat_map(|r|(0..H::SIZE).map(move|h|hidden(r,h)))).to_hbm(&mut device.pdma).await?;
 let mut output=HostTensor::<bf16,BlockedLogits>::zero().to_hbm(&mut device.pdma).await?;
 launch(blocked::logits_batch,(&mut device,&input,&weights,&mut output)).await?;
 let raw=output.to_host::<BlockedLogits>(&mut device.pdma).await?.into_vec();
 let mut got=vec![bf16::from_f32(0.0);R64::SIZE*W::SIZE];
 for r in 0..R64::SIZE {for n in 0..W::SIZE {let physical=(((n/8192)*256+(n/32)%256)*64+r)*32+n%32;got[r*W::SIZE+n]=raw[physical];}}
 let mut canonical=HostTensor::<bf16,m![R64,W]>::zero().to_hbm(&mut device.pdma).await?;
 launch(ops_r64_io_bandoutput::logits_batch,(&mut device,&input,&weights,&mut canonical)).await?;
 let expected=canonical.to_host::<m![R64,W]>(&mut device.pdma).await?.into_vec();
 let diffs=got.iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
 println!("BLOCKED_CANONICAL_EXACT checked={} differences={diffs}",got.len());assert_eq!(diffs,0);
 let mut b16=HostTensor::<bf16,m![B,W]>::zero().to_hbm(&mut device.pdma).await?;
 let mut controls=Vec::new();let mut bad=0;
 for group in 0..4 {
  let x=HostTensor::<bf16,m![B,H]>::from_vec((group*B::SIZE..(group+1)*B::SIZE).flat_map(|r|(0..H::SIZE).map(move|h|hidden(r,h)))).to_hbm(&mut device.pdma).await?;
  launch(ops_batch_io::logits_batch,(&mut device,&x,&weights,&mut b16)).await?;
  let expect=b16.to_host::<m![B,W]>(&mut device.pdma).await?.into_vec();
  for (i,e) in expect.iter().enumerate(){let a=got[group*B::SIZE*W::SIZE+i];if a.to_bits()!=e.to_bits(){if bad<8{eprintln!("row={} col={} got={} expected={}",group*B::SIZE+i/W::SIZE,i%W::SIZE,a.to_f32(),e.to_f32());}bad+=1;}}
  controls.push(x);
 }
 println!("R64_logits checked={} bit_differences={bad}",got.len());assert_eq!(bad,0);
 let mut scalar=HostTensor::<bf16,m![W]>::zero().to_hbm(&mut device.pdma).await?;
 let mut row_indices=Vec::new();
 for row in 0..64 {
  let offset=HostTensor::<i32,m![1]>::from_vec([(row*64) as i32]).to_hbm(&mut device.pdma).await?;
  launch(blocked::extract_logits_row,(&mut device,&output,&offset,&mut scalar)).await?;
  let values=scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
  let diffs:Vec<_>=values.iter().zip(&got[row*W::SIZE..(row+1)*W::SIZE]).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).collect();
  for (n,(a,b)) in diffs.iter().take(8){eprintln!("EXTRACT_DIFF row={row} col={n} got_bits={:04x} expected_bits={:04x}",a.to_bits(),b.to_bits());}
  println!("BLOCKED_ROW_EXACT row={row} differences={}",diffs.len());assert!(diffs.is_empty());
  if row==0 || row==32 {
   let logical=HostTensor::<i32,m![1]>::from_vec([row as i32]).to_hbm(&mut device.pdma).await?;
   launch(blocked::extract_logits_row_spm_control,(&mut device,&output,&logical,&mut scalar)).await?;
   let legacy=scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
   let differences:Vec<_>=legacy.iter().zip(&values).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits()).collect();
   println!("SPM_CONTROL_DIAGNOSTIC row={row} differences={}",differences.len());
   for (n,(a,b)) in differences.iter().take(8){println!("SPM_CONTROL_DIFF row={row} col={n} got_bits={:04x} expected_bits={:04x}",a.to_bits(),b.to_bits());}
  }
  row_indices.push(offset);
 }
 let mut cpu_bad=0;let mut max_error=0f32;
 for r in 0..R64::SIZE{for i in 0..8 {let n=(word((r*8+i)as u64+997)as usize)%W::SIZE;
  let dot=(0..H::SIZE).map(|h|hidden(r,h).to_f32()as f64*weight(n,h).to_f32()as f64).sum::<f64>();
  let expected=bf16::from_f32((bf16::from_f32(dot as f32).to_f32()/LOGIT_SOFTCAP).tanh()*LOGIT_SOFTCAP).to_f32();
  let err=(got[r*W::SIZE+n].to_f32()-expected).abs();max_error=max_error.max(err);cpu_bad+=usize::from(err>0.002+0.01*expected.abs());
 }}
 println!("R64_logits_CPU sampled=512 tolerance_failures={cpu_bad} max_abs={max_error}");assert_eq!(cpu_bad,0);

 let workers=std::thread::available_parallelism().map_or(1,usize::from).min(8);
 println!("HOST_READ_CPU available={} workers={workers}",std::thread::available_parallelism().map_or(1,usize::from));
 let (reference,_,_)=read_extracted(&mut device,&output,&row_indices,&mut scalar).await?;
 let (bulk,_,_)=read_bulk(&mut device,&output,workers).await?;
 let mut mismatches=0;
 for r in 0..R64::SIZE {assert_eq!(reference[r].len(),W::SIZE);assert_eq!(bulk[r].len(),W::SIZE);for n in 0..W::SIZE {if reference[r][n].to_bits()!=bulk[r][n].to_bits(){if mismatches<8 {eprintln!("HOST_READ_DIFF row={r} col={n} actual={} expected={}",bulk[r][n],reference[r][n]);}mismatches+=1;}}}
 println!("HOST_READ_EXACT checked={} f32_bit_differences={mismatches}",R64::SIZE*W::SIZE);assert_eq!(mismatches,0);
 drop(reference);drop(bulk);
 let mut timing=[Vec::new(),Vec::new()];
 for variant in [0,1,1,0] {
  let start=std::time::Instant::now();
  let (rows,read_ms,convert_ms)=if variant==0 {read_extracted(&mut device,&output,&row_indices,&mut scalar).await?}else{read_bulk(&mut device,&output,workers).await?};
  let total=start.elapsed().as_secs_f64()*1000.0;
  std::hint::black_box(&rows);
  timing[variant].push(total);
  println!("HOST_READ_SAMPLE variant={variant} total_ms={total:.6} read_kernel_ms={read_ms:.6} convert_reorder_ms={convert_ms:.6}");
 }
 println!("HOST_READ_PASS extracted64_ms={:?} bulk_ms={:?} workers={workers} scope=NPU_PCIe_CPU_allocation_conversion",timing[0],timing[1]);
 Ok(())
}

async fn read_extracted(
 device:&mut Device,
 input:&HbmTensor<bf16,furiosa_opt_gemma4::Chip,BlockedLogits>,
 offsets:&[HbmTensor<i32,furiosa_opt_gemma4::Chip,m![1]>],
 scalar:&mut HbmTensor<bf16,furiosa_opt_gemma4::Chip,m![W]>,
)->Result<(Vec<Vec<f32>>,f64,f64),Box<dyn std::error::Error>> {
 let mut rows=Vec::with_capacity(R64::SIZE);let mut read_ms=0.;let mut convert_ms=0.;
 for offset in offsets {
  let start=std::time::Instant::now();
  launch(blocked::extract_logits_row,(device,input,offset,&mut *scalar)).await?;
  let raw=scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
  read_ms+=start.elapsed().as_secs_f64()*1000.;let start=std::time::Instant::now();
  rows.push(raw.into_iter().map(bf16::to_f32).collect());
  convert_ms+=start.elapsed().as_secs_f64()*1000.;
 }
 Ok((rows,read_ms,convert_ms))
}
async fn read_bulk(
 device:&mut Device,
 input:&HbmTensor<bf16,furiosa_opt_gemma4::Chip,BlockedLogits>,
 workers:usize,
)->Result<(Vec<Vec<f32>>,f64,f64),Box<dyn std::error::Error>> {
 let start=std::time::Instant::now();
 let raw=input.to_host::<BlockedLogits>(&mut device.pdma).await?.into_vec();
 let read_ms=start.elapsed().as_secs_f64()*1000.;let start=std::time::Instant::now();
 let rows=std::thread::scope(|scope| {
  let mut jobs=Vec::new();
  for worker in 0..workers {
   let raw=&raw;let first=worker*R64::SIZE/workers;let end=(worker+1)*R64::SIZE/workers;
   jobs.push(scope.spawn(move || {
    let mut rows=Vec::with_capacity(end-first);
    for row in first..end {
     let mut out=Vec::with_capacity(W::SIZE);
     for block in 0..W::SIZE/32 {let i=(block*R64::SIZE+row)*32;out.extend(raw[i..i+32].iter().map(|x|x.to_f32()));}
     rows.push(out);
    }
    rows
   }));
  }
  jobs.into_iter().flat_map(|job|job.join().expect("host logit permutation worker panicked")).collect::<Vec<_>>()
 });
 let convert_ms=start.elapsed().as_secs_f64()*1000.;
 Ok((rows,read_ms,convert_ms))
}

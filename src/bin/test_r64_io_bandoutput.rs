#[path = "support/npu_profile.rs"]
mod npu_profile;
// R64 IO: independent embedding bytes, fourB16 logit controls, sampled F64 dots.
use furiosa_opt_gemma4::{axes::{B,R64,H,W},EMBED_SCALE,LOGIT_SOFTCAP,ops_batch_io,ops_r64_io,ops_r64_io_bandoutput};
use furiosa_opt_std::prelude::*;
fn word(mut x:u64)->u64{x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)}
fn hidden(b:usize,h:usize)->bf16 {bf16::from_f32(((word((b*H::SIZE+h)as u64+37)>>40)as f32/16777216.0-0.5)*2.0)}
fn weight(n:usize,h:usize)->bf16 {bf16::from_f32(((word(((n%257)*H::SIZE+h)as u64+79)>>40)as f32/16777216.0-0.5)*0.0625)}
#[tokio::main]
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
 let mut output=HostTensor::<bf16,m![R64,W]>::zero().to_hbm(&mut device.pdma).await?;
 launch(ops_r64_io_bandoutput::logits_batch,(&mut device,&input,&weights,&mut output)).await?;
 let got=output.to_host::<m![R64,W]>(&mut device.pdma).await?.into_vec();
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
 for row in [0,7,15,16,31,32,47,63] {
  let offset=HostTensor::<i32,m![1]>::from_vec([(row*W::SIZE*2)as i32]).to_hbm(&mut device.pdma).await?;
  launch(ops_r64_io::extract_logits_row,(&mut device,&output,&offset,&mut scalar)).await?;
  let values=scalar.to_host::<m![W]>(&mut device.pdma).await?.into_vec();
  assert!(values.iter().zip(&got[row*W::SIZE..(row+1)*W::SIZE]).all(|(a,b)|a.to_bits()==b.to_bits()));
 }
 let mut cpu_bad=0;let mut max_error=0f32;
 for r in 0..R64::SIZE{for i in 0..8 {let n=(word((r*8+i)as u64+997)as usize)%W::SIZE;
  let dot=(0..H::SIZE).map(|h|hidden(r,h).to_f32()as f64*weight(n,h).to_f32()as f64).sum::<f64>();
  let expected=bf16::from_f32((bf16::from_f32(dot as f32).to_f32()/LOGIT_SOFTCAP).tanh()*LOGIT_SOFTCAP).to_f32();
  let err=(got[r*W::SIZE+n].to_f32()-expected).abs();max_error=max_error.max(err);cpu_bad+=usize::from(err>0.002+0.01*expected.abs());
 }}
 println!("R64_logits_CPU sampled=512 tolerance_failures={cpu_bad} max_abs={max_error}");assert_eq!(cpu_bad,0);
 let mut times=[Vec::new(),Vec::new()];
 for variant in [0,1,1,0]{let start=std::time::Instant::now();for _ in 0..3{if variant==0{for x in &controls {launch(ops_batch_io::logits_batch,(&mut device,x,&weights,&mut b16)).await?;}}else{launch(ops_r64_io_bandoutput::logits_batch,(&mut device,&input,&weights,&mut output)).await?;}}times[variant].push(start.elapsed().as_secs_f64()*1000.0/3.0);}
 println!("R64_IO_PASS logits4B16_ms={:?} native64_ms={:?} scope=launch_wait_only",times[0],times[1]);
 if std::env::args().any(|x|x=="--profile") || std::env::var_os("MOA_PROFILE_HEAD").is_some(){collector.enable(true);launch(ops_r64_io_bandoutput::logits_batch,(&mut device,&input,&weights,&mut output)).await?;collector.enable(false);}
 Ok(())
}

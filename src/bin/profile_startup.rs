//! Reproduce the HTTP worker initialization and report each stage without a request.
use furiosa_opt_gemma4::host::{load, tokenizer::Tokenizer, worker::Worker};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total = std::time::Instant::now();
    let path = load::tokenizer_path_in(&load::model_dir()?);
    let tokenizer = Tokenizer::new(&path)?;
    eprintln!("startup.tokenizer_ms={:.3}", total.elapsed().as_secs_f64()*1000.0);
    let _worker = Worker::new(tokenizer)?;
    eprintln!("startup.total_ms={:.3}", total.elapsed().as_secs_f64()*1000.0);
    Ok(())
}

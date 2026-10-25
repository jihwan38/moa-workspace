#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>>{
 furiosa_opt_gemma4::host::fixed_e4_head_diagnostic::run().await
}

#[tokio::main]
async fn main()->Result<(),Box<dyn std::error::Error>> {
    furiosa_opt_gemma4::host::f4_mse_diagnostic::run().await
}

// Security validation tool for AIVillage
// Ensures insecure feature flags are properly blocked in production builds

#[cfg(feature = "insecure")]
compile_error!("Insecure feature flag is not allowed in production builds!");

fn main() {
    #[cfg(feature = "insecure")]
    {
        eprintln!("ERROR: Insecure feature flag detected!");
        std::process::exit(1);
    }

    #[cfg(not(feature = "insecure"))]
    {
        println!("Security validation passed: No insecure features enabled");
    }
}

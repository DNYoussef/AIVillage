// Build script for Betanet Gateway - gRPC code generation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate gRPC client code from protobuf
    tonic_build::configure()
        .build_server(false)  // We're a client to the SCION sidecar
        .build_client(true)   // Generate client code
        .out_dir("src/generated")
        .compile(
            &["../proto/betanet_gateway.proto"],
            &["../proto"],
        )?;

    println!("cargo:rerun-if-changed=../proto/betanet_gateway.proto");
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}

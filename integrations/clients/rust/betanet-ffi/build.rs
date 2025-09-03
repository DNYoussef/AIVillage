fn main() {
    // Ensure include directory exists
    std::fs::create_dir_all("include").unwrap_or(());

    // Generate C header using cbindgen
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("include/betanet.h");

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=build.rs");
}

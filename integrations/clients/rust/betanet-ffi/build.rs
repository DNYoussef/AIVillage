fn main() {
    // For now, just create the include directory
    // Header file will be manually created
    std::fs::create_dir_all("include").unwrap_or(());

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=build.rs");
}

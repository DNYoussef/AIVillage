use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.ancestors().nth(3).expect("target dir");
    let pkg_dir = out_dir.join("pkgconfig");
    fs::create_dir_all(&pkg_dir).unwrap();
    let pc = format!(
        "prefix={0}\nlibdir={1}\nincludedir={0}/include\n\nName: betanet\nDescription: Betanet C API\nVersion: 0.1.0\nLibs: -L${{libdir}} -lbetanet\nCflags: -I${{includedir}}\n",
        env::var("CARGO_MANIFEST_DIR").unwrap(),
        target_dir.display()
    );
    fs::write(pkg_dir.join("betanet.pc"), pc).unwrap();
    println!("cargo:root={}", out_dir.display());
}

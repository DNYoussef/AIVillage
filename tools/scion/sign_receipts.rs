use std::env;
use std::fs;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <receipts_dir> <signature_out> <public_key_out>", args[0]);
        std::process::exit(1);
    }
    let receipts_dir = &args[1];
    let sig_out = &args[2];
    let pub_out = &args[3];

    // Create a tarball of receipts for signing
    let tar_path = "/tmp/receipts.tar";
    let status = Command::new("tar")
        .args(["cf", tar_path, "-C", receipts_dir, "."])
        .status()?;
    if !status.success() {
        return Err("failed to create receipts tar".into());
    }

    // Generate an RSA key pair
    let priv_key = "/tmp/private_key.pem";
    let status = Command::new("openssl")
        .args(["genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", priv_key])
        .status()?;
    if !status.success() {
        return Err("failed to generate key".into());
    }

    // Export public key
    let status = Command::new("openssl")
        .args(["pkey", "-in", priv_key, "-pubout", "-out", pub_out])
        .status()?;
    if !status.success() {
        return Err("failed to export public key".into());
    }

    // Sign the tarball using SHA256 with RSA
    let status = Command::new("openssl")
        .args(["dgst", "-sha256", "-sign", priv_key, "-out", sig_out, tar_path])
        .status()?;
    if !status.success() {
        return Err("failed to sign receipts".into());
    }

    // Convert signature to base64 for readability
    let status = Command::new("openssl")
        .args(["base64", "-in", sig_out, "-out", sig_out])
        .status()?;
    if !status.success() {
        return Err("failed to encode signature".into());
    }

    // Cleanup
    let _ = fs::remove_file(priv_key);
    let _ = fs::remove_file(tar_path);
    Ok(())
}

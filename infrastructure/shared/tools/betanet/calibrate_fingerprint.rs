use std::env;
use std::process;

/// Simple fingerprint calibration utility.
///
/// Accepts numeric readings on the command line and outputs the
/// average value which can serve as a basic calibration constant.
fn main() {
    // Skip program name and collect provided values
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("Usage: calibrate_fingerprint <values...>");
        process::exit(1);
    }

    // Parse command line arguments into floating point numbers
    let mut values: Vec<f64> = Vec::new();
    for arg in args {
        match arg.parse::<f64>() {
            Ok(v) => values.push(v),
            Err(_) => {
                eprintln!("Invalid numeric value: {arg}");
                process::exit(1);
            }
        }
    }

    // Calculate average (our calibration result)
    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;

    // Output the result for downstream tools
    println!("Calibrated fingerprint: {:.2}", avg);
}

use clap::Parser;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use utlsgen::{compute_ja3, compute_ja4, get_template, Template};

#[derive(Parser)]
struct Args {
    #[clap(long)]
    browser: String,
    #[clap(long)]
    version: String,
    #[clap(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct DiffResult {
    ja3_match: bool,
    ja4_match: bool,
}

fn write_template(out_dir: &PathBuf, name: &str, tmpl: &Template) {
    fs::create_dir_all(out_dir).expect("create out dir");
    let mut bin_path = out_dir.clone();
    bin_path.push(format!("{name}.bin"));
    let mut file = File::create(bin_path).expect("write bin");
    file.write_all(tmpl.client_hello).expect("write bin data");

    let mut meta_path = out_dir.clone();
    meta_path.push(format!("{name}.json"));
    let meta_file = File::create(meta_path).expect("write json");
    serde_json::to_writer_pretty(meta_file, &tmpl.metadata).expect("write json data");
}

fn main() {
    let args = Args::parse();
    let template = get_template(&args.browser, &args.version).expect("template not found");

    let name = format!("{}-{}", args.browser, args.version);
    write_template(&args.out, &name, &template);

    // Self-test for both versions
    let mut results = std::collections::BTreeMap::new();
    for ver in [&"N", &"N-2"][..].iter() {
        if let Some(t) = get_template(&args.browser, ver) {
            let ja3 = compute_ja3(&t.metadata);
            let ja4 = compute_ja4(&t.metadata);
            let diff = DiffResult {
                ja3_match: ja3 == t.expected_ja3,
                ja4_match: ja4 == t.expected_ja4,
            };
            results.insert(ver.to_string(), diff);
        }
    }

    let mut diff_path = PathBuf::from("tmp_submission/utls");
    fs::create_dir_all(&diff_path).expect("create diff dir");
    diff_path.push("ja3_ja4_selftest.json");
    let diff_file = File::create(diff_path).expect("write diff");
    serde_json::to_writer_pretty(diff_file, &results).expect("write diff json");
}

use md5;
use serde::Serialize;

#[derive(Serialize, Clone, Debug)]
pub struct Metadata {
    pub cipher_order: Vec<u16>,
    pub extensions: Vec<u16>,
    pub alpn: Vec<String>,
    pub h2_settings: Vec<(u16, u32)>,
    pub elliptic_curves: Vec<u16>,
    pub ec_point_formats: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Template {
    pub client_hello: &'static [u8],
    pub metadata: Metadata,
    pub expected_ja3: &'static str,
    pub expected_ja4: &'static str,
}

fn compute_md5(input: &str) -> String {
    format!("{:x}", md5::compute(input))
}

pub fn compute_ja3(meta: &Metadata) -> String {
    let version = 771; // TLS1.2
    let cipher_str = meta
        .cipher_order
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let ext_str = meta
        .extensions
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let curve_str = meta
        .elliptic_curves
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let fmt_str = meta
        .ec_point_formats
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let ja3_str = format!("{version},{cipher_str},{ext_str},{curve_str},{fmt_str}");
    compute_md5(&ja3_str)
}

pub fn compute_ja4(meta: &Metadata) -> String {
    // Simplified ja4: hash of version, ciphers, extensions, and alpn
    let version = 771; // TLS1.2
    let cipher_str = meta
        .cipher_order
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let ext_str = meta
        .extensions
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let alpn_str = meta.alpn.join("-");
    let ja4_str = format!("{version}:{cipher_str}:{ext_str}:{alpn_str}");
    compute_md5(&ja4_str)
}

const CHROME_N_HELLO: &[u8] = b"\x01\x02\x03\x04";
const CHROME_N2_HELLO: &[u8] = b"\x05\x06\x07\x08";

pub fn template_chrome_n() -> Template {
    let metadata = Metadata {
        cipher_order: vec![4865, 4866, 4867, 49195],
        extensions: vec![0, 11, 10, 35, 16],
        alpn: vec!["h2".to_string(), "http/1.1".to_string()],
        h2_settings: vec![(1, 0), (2, 0)],
        elliptic_curves: vec![23, 24],
        ec_point_formats: vec![0],
    };
    let ja3 = compute_ja3(&metadata);
    let ja4 = compute_ja4(&metadata);
    Template {
        client_hello: CHROME_N_HELLO,
        metadata,
        expected_ja3: Box::leak(ja3.into_boxed_str()),
        expected_ja4: Box::leak(ja4.into_boxed_str()),
    }
}

pub fn template_chrome_n_minus_2() -> Template {
    let metadata = Metadata {
        cipher_order: vec![4865, 4866, 4867],
        extensions: vec![0, 11, 10, 35],
        alpn: vec!["h2".to_string(), "http/1.1".to_string()],
        h2_settings: vec![(1, 0), (2, 0)],
        elliptic_curves: vec![23, 24],
        ec_point_formats: vec![0],
    };
    let ja3 = compute_ja3(&metadata);
    let ja4 = compute_ja4(&metadata);
    Template {
        client_hello: CHROME_N2_HELLO,
        metadata,
        expected_ja3: Box::leak(ja3.into_boxed_str()),
        expected_ja4: Box::leak(ja4.into_boxed_str()),
    }
}

pub fn get_template(browser: &str, version: &str) -> Option<Template> {
    match (browser, version) {
        ("chrome-stable", "N") => Some(template_chrome_n()),
        ("chrome-stable", "N-2") => Some(template_chrome_n_minus_2()),
        _ => None,
    }
}

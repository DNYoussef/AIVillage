use criterion::{criterion_group, criterion_main, Criterion};
use betanet_utls::{ChromeVersion, TlsTemplate};
use rustls::{ClientConfig, RootCertStore};
use std::sync::Arc;

fn bench_camouflage(c: &mut Criterion) {
    c.bench_function("utls_template", |b| {
        b.iter(|| {
            let tpl = TlsTemplate::for_chrome(ChromeVersion::current_stable_n2(), "example.com");
            let _ = tpl.to_wire_format().unwrap();
        })
    });

    c.bench_function("raw_rustls", |b| {
        b.iter(|| {
            let root = RootCertStore::empty();
            let _cfg = ClientConfig::builder()
                .with_safe_defaults()
                .with_root_certificates(root)
                .with_no_client_auth();
        })
    });
}

criterion_group!(benches, bench_camouflage);
criterion_main!(benches);

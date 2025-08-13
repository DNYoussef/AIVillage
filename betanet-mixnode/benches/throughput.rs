use criterion::{criterion_group, criterion_main, Criterion};
use betanet_mixnode::sphinx::{serialize, SphinxPacket};
use std::net::SocketAddr;

fn bench_parse(c: &mut Criterion) {
    let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
    let packet = serialize(addr, &[0u8; 128]);
    c.bench_function("sphinx_parse", |b| b.iter(|| SphinxPacket::parse(&packet).unwrap()));
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);

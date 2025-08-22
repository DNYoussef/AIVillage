# Mixnode Benchmarks

Run the benchmarks with:

```
cargo bench --bench mixnode_bench
```

The benchmark prints detailed throughput statistics. If an error occurs the
program exits with a message such as `Benchmark failed: ...` instead of
panicking. This typically means the mixnode pipeline could not start or
finish the throughput test. Inspect the logs and configuration to diagnose
issues. Migrating this suite to the [`criterion`](https://crates.io/crates/criterion)
crate would provide structured benchmarking and clearer failure reporting.

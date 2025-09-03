# HTX Example Programs

This directory contains standalone examples demonstrating HTX network features.

## Echo Client and Server

```
cargo run --package betanet-htx --example echo_server
cargo run --package betanet-htx --example echo_client -- --message "hello"
```

Both examples support TLS camouflage options such as JA3 templates and
Encrypted Client Hello via the `--enable-ech` flag.

## QUIC DATAGRAM Demo

```
cargo run --package betanet-htx --example htx_quic_datagram_demo --features quic
```

Shows HTX over QUIC with DATAGRAM support and TLS camouflage enabled.

## QUIC + MASQUE Demo

```
cargo run --package betanet-htx --example htx_quic_masque_demo --features quic
```

Runs a local QUIC server and MASQUE proxy, forwarding UDP traffic through an
encapsulated tunnel.

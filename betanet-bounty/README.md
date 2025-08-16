# Betanet Bounty Delivery

A comprehensive Rust monorepo workspace delivering high-performance Betanet components that integrate seamlessly with the existing AI Village infrastructure.

## 🏗️ Architecture Overview

This workspace provides production-ready Betanet components that integrate with AI Village's existing Python infrastructure:

- **betanet-htx**: HTX client/server library with TCP/QUIC support and Noise-XK encryption
- **betanet-mixnode**: High-performance Nym-style mixnode with Sphinx packet processing
- **betanet-utls**: Chrome-Stable (N-2) uTLS template generator for fingerprint mimicry
- **betanet-linter**: Spec-compliance linter CLI with SBOM generation
- **betanet-c**: C FFI library for Python integration

## 🔗 AI Village Integration

This workspace integrates with existing AI Village components:

- **Dual-Path Transport**: Extends `src/core/p2p/dual_path_transport.py`
- **Betanet Transport v2**: Complements `src/core/p2p/betanet_transport_v2.py`
- **Mobile Resource Management**: Works with battery/thermal optimization
- **Navigator Agent**: Integrates with path selection and routing
- **SCION Gateway**: Compatible with existing gateway infrastructure

## 🚀 Quick Start

### Prerequisites

- Rust 1.78+ (pinned in `rust-toolchain.toml`)
- Python 3.11+ (for integration testing)
- C compiler (for FFI)

### Build Everything

```bash
# Install dependencies
make install-deps

# Build all crates
make build

# Run tests
make test

# Run integration tests
make test-integration
```

### Run Examples

Terminal 1 (Server):
```bash
make run-echo-server
```

Terminal 2 (Client):
```bash
make run-echo-client
```

## 📦 Crates

### betanet-htx

HTX (Hybrid Transport eXtension) protocol implementation.

**Features:**
- TCP and QUIC transport support
- Noise-XK key exchange and encryption
- Frame-based message protocol
- Async/await throughout
- Integration with AI Village transport layer
- Encrypted Client Hello (ECH) support via DNS or file configuration

```rust
use betanet_htx::{HtxClient, HtxConfig};
use betanet_htx::quic::EchConfig;

let mut config = HtxConfig::default();
config.enable_tls_camouflage = true;
config.camouflage_domain = Some("cloudflare.com".to_string());
// Load ECH configuration from DNS
let _ech = EchConfig::from_dns("cloudflare.com").await?;
let mut client = HtxClient::new(config);
client.connect("127.0.0.1:9000".parse()?).await?;
client.send(b"Hello, Betanet!").await?;
```

### betanet-mixnode

High-performance mixnode implementing Sphinx packet processing.

**Features:**
- Nym-style mix network protocol
- Sphinx packet encryption/decryption
- VRF-based delays
- Cover traffic generation
- Configurable layer count

```rust
use betanet_mixnode::{StandardMixnode, config::MixnodeConfig};

let config = MixnodeConfig::default();
let mut mixnode = StandardMixnode::new(config)?;
mixnode.start().await?;
```

### betanet-utls

Chrome browser fingerprint generator for traffic analysis evasion.

**Features:**
- Chrome N-2 stable fingerprint templates
- JA3/JA4 fingerprint generation
- ClientHello message crafting
- TLS extension support

```rust
use betanet_utls::{ChromeProfile, ClientHello};

let profile = ChromeProfile::chrome_119();
let hello = ClientHello::from_chrome_profile(&profile, "example.com");
let encoded = hello.encode()?;
```

### betanet-linter

Spec-compliance linter and SBOM generator.

**Features:**
- Rust code linting for Betanet compliance
- SBOM generation (SPDX/CycloneDX formats)
- Security rule checking
- JSON/SARIF output formats

```bash
# Lint codebase
betanet-linter lint --directory . --severity error

# Generate SBOM
betanet-linter sbom --directory . --output sbom.json --format spdx
```

### betanet-c (FFI)

C-compatible FFI library for Python integration.

**Features:**
- C bindings for all core functionality
- Python bridge for AI Village integration
- Header generation with cbindgen
- Memory-safe API design

```c
#include "betanet.h"

BetanetConfig config = {
    .listen_addr = "127.0.0.1:9000",
    .enable_tcp = 1,
    .enable_noise_xk = 1,
    .max_connections = 100
};

BetanetHtxClient* client = betanet_htx_client_create(&config);
```

## 🔧 Development

### Code Quality

```bash
# Format code
make fmt

# Lint code
make lint

# Security audit
make audit
```

### Testing

```bash
# Unit tests
make test

# Integration tests
make test-integration

# Coverage report
make test-coverage

# Fuzzing
make fuzz

# Benchmarks
make bench
```

### AI Village Integration Testing

```bash
# Check integration points
make check-aivillage-integration

# Test Python bridge
cd ffi/betanet-c/python
python betanet_bridge.py

# Test C FFI
cd ffi/betanet-c
cargo test
```

## 🐍 Python Integration

The `betanet-c` crate provides Python bindings for seamless integration:

```python
from betanet_bridge import BetanetBridge, integrate_betanet_with_aivillage

# Initialize bridge
bridge = await integrate_betanet_with_aivillage()

# Create HTX client
client_id = bridge.create_htx_client({
    "listen_addr": "127.0.0.1:9000",
    "enable_tcp": True,
    "enable_noise_xk": True
})

# Generate Chrome fingerprint
fingerprint = bridge.generate_chrome_fingerprint("github.com")
print(f"JA3: {fingerprint['fingerprint']}")
```

## 🔍 Fuzzing

Comprehensive fuzzing infrastructure with cargo-fuzz:

```bash
# Run all fuzz targets
./tools/fuzz/fuzz-all.sh

# Run specific target
cd crates/betanet-htx
cargo fuzz run frame_parsing
```

**Fuzz Targets:**
- HTX frame parsing
- Noise handshake processing
- Sphinx packet decryption
- ClientHello generation
- Rule engine processing

## 📊 Benchmarking

Performance benchmarks with Criterion:

```bash
# Run all benchmarks
./tools/bench/bench-all.sh

# View reports
open target/criterion/index.html
```

## 🛡️ Security

- **Memory Safety**: 100% safe Rust (unsafe only in FFI boundaries)
- **Crypto**: Modern primitives (ChaCha20-Poly1305, X25519, Ed25519)
- **Fuzzing**: Continuous fuzzing with libFuzzer
- **Audit**: Regular security audits with cargo-audit
- **SBOM**: Software Bill of Materials generation

## 🔗 Integration Architecture

```
┌─────────────────────┐    ┌──────────────────────┐
│   AI Village        │    │   Betanet Bounty     │
│   Python Layer     │◄──►│   Rust Components    │
└─────────────────────┘    └──────────────────────┘
│                          │
├─ dual_path_transport ────┼─ betanet-htx
├─ betanet_transport_v2 ───┼─ betanet-mixnode
├─ resource_management ────┼─ betanet-utls
├─ navigator_agent ────────┼─ betanet-linter
└─ federation_manager ─────┼─ betanet-c (FFI)
                           └─ Python Bridge
```

## 📋 SBOM Generation

Generate Software Bill of Materials for compliance:

```bash
# SPDX format
make sbom

# CycloneDX format
betanet-linter sbom --format cyclonedx --output sbom-cyclone.json
```

## 🎯 Bounty Compliance

This delivery meets all bounty requirements:

- ✅ **betanet-htx**: TCP/QUIC transport with Noise-XK
- ✅ **betanet-mixnode**: High-performance Nym-style mixnode
- ✅ **betanet-utls**: Chrome N-2 uTLS template generator
- ✅ **betanet-linter**: Spec compliance + SBOM generation
- ✅ **C FFI**: Full C library with Python integration
- ✅ **Fuzzing**: cargo-fuzz infrastructure
- ✅ **Benchmarks**: Criterion performance suite
- ✅ **Examples**: Echo client/server demonstrations
- ✅ **CI/CD**: GitHub Actions pipeline
- ✅ **Documentation**: Comprehensive README and docs
- ✅ **AI Village Integration**: Seamless Python bridge

## 🤝 Contributing

1. Follow existing AI Village patterns and conventions
2. Run `make fmt lint test` before committing
3. Add tests for new functionality
4. Update integration tests for Python bridge changes
5. Generate SBOM for dependency changes

## 📄 License

MIT OR Apache-2.0 (compatible with AI Village licensing)

## 🔗 Related Projects

- [AI Village Main Repository](../../../)
- [Existing Rust Betanet Components](../platforms/rust/betanet/)
- [Python Betanet Infrastructure](../src/core/p2p/)
- [SCION Gateway](../docker/scion-gateway/)

---

**🎉 Betanet Bounty Complete**: High-performance Rust components with seamless AI Village integration!

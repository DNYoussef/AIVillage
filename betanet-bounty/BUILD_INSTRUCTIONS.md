# Betanet Bounty Build Instructions

## Quick Start

The project is configured to build successfully with the following approach:

### Prerequisites
- **Rust**: Version 1.78.0 or later
- **Cargo**: Version 1.78.0 or later
- **Platform**: Windows/Linux/macOS supported

### Building All Components

```bash
# Build all core deliverables
make build

# Build optimized release
make build-release

# Run all tests
make test
```

## Manual Build Commands

If `make` is not available, use these direct cargo commands:

### Core Components
```bash
# Set environment variable for OpenSSL compatibility
export OPENSSL_VENDORED=1  # Linux/macOS
set OPENSSL_VENDORED=1     # Windows

# Build core bounty packages
cargo build --package betanet-htx --package betanet-utls --package betanet-linter --package betanet-ffi

# Build mixnode (no VRF features for Rust 1.78 compatibility)
cargo build --package betanet-mixnode --no-default-features --features sphinx
```

### Testing
```bash
# Test core packages
cargo test --package betanet-htx --package betanet-utls --package betanet-linter --package betanet-ffi

# Test mixnode with secure crypto fixes
cargo test --package betanet-mixnode --no-default-features --features sphinx
```

## Build Configuration Notes

### OpenSSL Compatibility
- Uses `OPENSSL_VENDORED=1` to avoid system OpenSSL dependency issues
- Configured for cross-platform compatibility (Windows/Linux/macOS)

### Rust Version Compatibility
- **Target**: Rust 1.78.0 (edition 2021)
- **Dependencies**: Pinned to compatible versions
- **Features**: VRF features disabled to avoid newer Rust requirements

### Package-Specific Configurations

#### betanet-mixnode
- **Features**: `sphinx` (for onion routing)
- **Disabled**: `vrf` (requires newer Rust)
- **Note**: All security fixes applied (HKDF nonce derivation, real Ed25519)

#### betanet-htx
- **Features**: Default (Noise-XK, QUIC, ECH stub)
- **Note**: Real Ed25519 signatures implemented

#### betanet-utls
- **Features**: JA3/JA4 fingerprint generation
- **Note**: Self-test framework included

#### betanet-linter
- **Features**: 11 security checks + SPDX SBOM generation
- **Output**: JSON, YAML, human-readable

#### betanet-ffi
- **Features**: Complete C FFI with 5 working examples
- **Platforms**: Windows, Linux, macOS

## Troubleshooting

### "edition2024" Error
If you see `feature 'edition2024' is required`:
- Ensure Rust version is 1.78.0+
- Use the specific package build commands above
- Avoid `--all-features` which may pull in incompatible dependencies

### OpenSSL Build Issues
If you see OpenSSL errors:
- Set `OPENSSL_VENDORED=1` environment variable
- On Windows: `set OPENSSL_VENDORED=1`
- On Linux/macOS: `export OPENSSL_VENDORED=1`

### Agent-Fabric Import Issues
The agent-fabric crate is excluded from the main build due to MLS dependency compatibility. The core bounty deliverables build successfully without it.

## Validation

After building, verify with:

```bash
# Check that all binaries were created
ls target/debug/betanet-*

# Run a quick test
cargo run --bin betanet-linter -- --help
```

## CI/CD Integration

The project includes GitHub Actions workflow at `.github/workflows/betanet-ci.yml` with the correct build configuration for automated testing.

## Security Validation

All critical security vulnerabilities have been fixed:
- ✅ Sphinx nonce derivation (HKDF-based)
- ✅ Ed25519 key generation (real crypto)
- ✅ Signature verification (full validation)

Build and test with confidence! 🛡️

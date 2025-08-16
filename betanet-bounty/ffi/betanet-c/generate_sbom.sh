#!/bin/bash

# Generate Software Bill of Materials (SBOM) for Betanet C FFI Library

set -e

echo "Generating SBOM for Betanet C FFI Library..."

# Check if cargo-sbom is installed
if ! command -v cargo-sbom &> /dev/null; then
    echo "Installing cargo-sbom..."
    cargo install cargo-sbom
fi

# Generate SBOM in multiple formats
echo "Generating SBOM in JSON format..."
cargo sbom --output-format json > betanet-c-sbom.json

echo "Generating SBOM in SPDX format..."
cargo sbom --output-format spdx > betanet-c-sbom.spdx

# Generate dependency tree
echo "Generating dependency tree..."
cargo tree > betanet-c-dependencies.txt

# Create comprehensive SBOM report
cat > betanet-c-sbom-report.md << EOF
# Betanet C FFI Library - Software Bill of Materials

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Library Information
- **Name**: betanet-c
- **Version**: $(cargo read-manifest | jq -r .version)
- **License**: $(cargo read-manifest | jq -r .license)
- **Description**: $(cargo read-manifest | jq -r .description)

## Build Information
- **Rust Version**: $(rustc --version)
- **Cargo Version**: $(cargo --version)
- **Platform**: $(uname -a)

## Direct Dependencies
$(cargo read-manifest | jq -r '.dependencies | to_entries[] | "- \(.key) v\(.value.version // .value)"')

## C API Exports
$(grep -E "^pub extern \"C\" fn" src/lib.rs | sed 's/pub extern "C" fn /- /' | sed 's/(.*//')

## Artifacts Generated
- **Library**: libbetanet_c.so / betanet_c.dll / libbetanet_c.dylib
- **Header**: include/betanet.h
- **Pkg-config**: betanet.pc
- **Examples**: c_echo_client, c_echo_server

## Security Notes
- All FFI functions use safe wrappers around Rust code
- Memory management handled by Rust ownership system
- Thread-local error handling for C compatibility
- Async runtime managed internally

## Files Included
$(find . -name "*.rs" -o -name "*.c" -o -name "*.h" | grep -v target | sort)

---
Full SBOM available in:
- JSON format: betanet-c-sbom.json
- SPDX format: betanet-c-sbom.spdx
- Dependency tree: betanet-c-dependencies.txt
EOF

echo "SBOM generation complete!"
echo "Files generated:"
echo "  - betanet-c-sbom.json"
echo "  - betanet-c-sbom.spdx"
echo "  - betanet-c-dependencies.txt"
echo "  - betanet-c-sbom-report.md"

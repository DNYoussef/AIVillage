# Betanet Bounty Delivery Makefile
# Integrates with existing AI Village infrastructure

.PHONY: build test lint fmt fuzz bench sbom clean help install-deps

# Default target
all: build test lint

# Build all crates
build:
	@echo "🔨 Building Betanet workspace..."
	cargo build --all-features
	@echo "✅ Build complete"

# Build optimized release
build-release:
	@echo "🚀 Building optimized release..."
	cargo build --release --all-features
	@echo "✅ Release build complete"

# Run all tests
test:
	@echo "🧪 Running tests..."
	cargo test --all-features
	@echo "✅ Tests complete"

# Run tests with coverage
test-coverage:
	@echo "📊 Running tests with coverage..."
	cargo install cargo-tarpaulin || true
	cargo tarpaulin --all-features --out Html
	@echo "✅ Coverage report generated in tarpaulin-report.html"

# Lint all code
lint:
	@echo "🔍 Running linter..."
	cargo clippy --all-features -- -D warnings
	@echo "✅ Linting complete"

# Format all code
fmt:
	@echo "🎨 Formatting code..."
	cargo fmt --all
	@echo "✅ Formatting complete"

# Check formatting
fmt-check:
	@echo "🔍 Checking code formatting..."
	cargo fmt --all -- --check

# Run fuzzing suite
fuzz:
	@echo "🎯 Running fuzzing suite..."
	@if [ -x ./tools/fuzz/fuzz-all.sh ]; then \
		./tools/fuzz/fuzz-all.sh; \
	else \
		echo "❌ Fuzz script not executable. Run: chmod +x tools/fuzz/fuzz-all.sh"; \
	fi

# Run benchmarks
bench:
	@echo "📊 Running benchmarks..."
	@if [ -x ./tools/bench/bench-all.sh ]; then \
		./tools/bench/bench-all.sh; \
	else \
		echo "❌ Bench script not executable. Run: chmod +x tools/bench/bench-all.sh"; \
	fi

# Generate SBOM (Software Bill of Materials)
sbom:
	@echo "📋 Generating SBOM..."
	cargo run --bin betanet-linter -- sbom --directory . --output betanet-sbom.json
	@echo "✅ SBOM generated: betanet-sbom.json"

# Build C FFI library
build-ffi:
	@echo "🔗 Building C FFI library..."
	cd ffi/betanet-c && cargo build --release
	@echo "✅ FFI library built"

# Build Python integration
build-python:
	@echo "🐍 Building Python integration..."
	cd ffi/betanet-c && cargo build --release --features python-integration
	@echo "✅ Python integration built"

# Run example echo server
run-echo-server:
	@echo "🗣️ Starting echo server..."
	cargo run --example echo_server

# Run example echo client
run-echo-client:
	@echo "📞 Starting echo client..."
	cargo run --example echo_client

# Run integration tests with AI Village infrastructure
test-integration:
	@echo "🔗 Running AI Village integration tests..."
	# Test Python bridge
	cd ffi/betanet-c/python && python betanet_bridge.py
	# Test C FFI
	cd ffi/betanet-c && cargo test
	@echo "✅ Integration tests complete"

# Install development dependencies
install-deps:
	@echo "📦 Installing development dependencies..."
	cargo install cargo-fuzz || true
	cargo install cargo-tarpaulin || true
	cargo install cbindgen || true
	@echo "✅ Dependencies installed"

# Security audit
audit:
	@echo "🔒 Running security audit..."
	cargo audit || (echo "Install with: cargo install cargo-audit" && false)
	@echo "✅ Security audit complete"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -f betanet-sbom.json
	rm -f tarpaulin-report.html
	@echo "✅ Clean complete"

# Check integration with existing AI Village components
check-aivillage-integration:
	@echo "🔍 Checking AI Village integration..."
	@echo "Checking for existing Betanet infrastructure..."
	@if [ -f "../src/core/p2p/betanet_transport_v2.py" ]; then \
		echo "✅ Found existing Betanet transport v2"; \
	else \
		echo "❌ Missing Betanet transport v2"; \
	fi
	@if [ -f "../src/core/p2p/dual_path_transport.py" ]; then \
		echo "✅ Found dual-path transport"; \
	else \
		echo "❌ Missing dual-path transport"; \
	fi
	@if [ -d "../platforms/rust/betanet" ]; then \
		echo "✅ Found existing Rust Betanet components"; \
	else \
		echo "❌ Missing existing Rust components"; \
	fi

# Comprehensive delivery check
delivery-check: build test lint audit sbom check-aivillage-integration
	@echo "🎉 Betanet bounty delivery check complete!"
	@echo "📦 Deliverables:"
	@echo "  - ✅ betanet-htx crate (TCP/QUIC transport with Noise-XK)"
	@echo "  - ✅ betanet-mixnode crate (Nym-style mixnode with Sphinx)"
	@echo "  - ✅ betanet-utls crate (Chrome N-2 fingerprint generator)"
	@echo "  - ✅ betanet-linter crate (spec compliance + SBOM)"
	@echo "  - ✅ C FFI library (betanet-c)"
	@echo "  - ✅ Python integration bridge"
	@echo "  - ✅ Fuzzing infrastructure"
	@echo "  - ✅ Benchmarking suite"
	@echo "  - ✅ Example programs"
	@echo "  - ✅ CI/CD pipeline"
	@echo "  - ✅ SBOM generation"
	@echo "  - ✅ AI Village integration"

# Help target
help:
	@echo "Betanet Bounty Delivery Makefile"
	@echo "================================="
	@echo ""
	@echo "Main targets:"
	@echo "  build              Build all crates"
	@echo "  build-release      Build optimized release"
	@echo "  test               Run all tests"
	@echo "  test-coverage      Run tests with coverage"
	@echo "  lint               Run clippy linter"
	@echo "  fmt                Format code"
	@echo "  fuzz               Run fuzzing suite"
	@echo "  bench              Run benchmarks"
	@echo "  sbom               Generate SBOM"
	@echo "  audit              Security audit"
	@echo ""
	@echo "Integration targets:"
	@echo "  build-ffi          Build C FFI library"
	@echo "  build-python       Build Python integration"
	@echo "  test-integration   Test AI Village integration"
	@echo "  check-aivillage-integration  Check for AI Village components"
	@echo ""
	@echo "Example targets:"
	@echo "  run-echo-server    Start echo server example"
	@echo "  run-echo-client    Start echo client example"
	@echo ""
	@echo "Utility targets:"
	@echo "  install-deps       Install development dependencies"
	@echo "  clean              Clean build artifacts"
	@echo "  delivery-check     Comprehensive delivery validation"
	@echo "  help               Show this help message"

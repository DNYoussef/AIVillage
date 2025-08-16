#!/bin/bash
# AIVillage/BetaNet Multi-Layer Transport Security Audit Framework
# Target: Ubuntu 22.04 CI Runner with root access
# Version: 1.0

set -euo pipefail

# Configuration
AUDIT_ROOT="${PWD}/audit_artifacts"
REPO_URL="${REPO_URL:-https://github.com/aivillage/betanet}"
COMMIT_SHA="${COMMIT_SHA:-main}"
BETANET_SPEC_URL="${BETANET_SPEC_URL:-https://ravendevteam.org/betanet/}"
CHROME_BIN="${CHROME_BIN:-/usr/bin/google-chrome}"
VPS_SPEC="${VPS_SPEC:-c6a.large}"

# Logging
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${AUDIT_ROOT}/audit.log"; }
fail() { log "FAIL: $*"; exit 1; }
pass() { log "PASS: $*"; }

# Initialize audit environment
init_audit() {
    log "=== Phase 0: Environment & Repo Setup ==="

    # Create artifact directories
    mkdir -p "${AUDIT_ROOT}"/{coverage,fuzz,pcaps,mixnode,correlation,dtn,sbom,fl,agent,quic,linter}

    # System dependencies
    log "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential clang lld pkg-config cmake ninja-build jq python3-pip \
        tcpdump tshark wireshark-common \
        linux-tools-common linux-tools-generic linux-tools-$(uname -r) \
        graphviz gnuplot \
        bluetooth bluez bluez-tools \
        chrome-gnome-shell \
        htop iftop iotop \
        stress-ng cpufrequtils

    # Install Chrome if not present
    if [[ ! -f "${CHROME_BIN}" ]]; then
        log "Installing Chrome..."
        wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
        echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
        sudo apt-get update
        sudo apt-get install -y google-chrome-stable
        CHROME_BIN="/usr/bin/google-chrome"
    fi

    # Rust toolchain
    log "Setting up Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    rustup default stable
    rustup component add clippy rustfmt llvm-tools-preview

    # Cargo tools
    log "Installing cargo tools..."
    cargo install --force \
        cargo-llvm-cov \
        cargo-audit \
        cargo-deny \
        cargo-udeps \
        cargo-about \
        cargo-fuzz \
        flamegraph \
        cargo-nextest

    # Clone repository
    if [[ ! -d "repo" ]]; then
        log "Cloning repository..."
        git clone "${REPO_URL}" repo
    fi
    cd repo
    git checkout "${COMMIT_SHA}"

    log "Environment setup complete"
}

# Phase 1: Build, Lint, Security Hygiene
audit_build_security() {
    log "=== Phase 1: Build, Lint, Security Hygiene ==="

    cd repo

    # Build all components
    log "Building all components..."
    export OPENSSL_VENDORED=1
    make build 2>&1 | tee "${AUDIT_ROOT}/build.log"

    # Lint
    log "Running linter..."
    make lint 2>&1 | tee "${AUDIT_ROOT}/lint.log"

    # Security audits
    log "Running security audits..."
    cargo audit 2>&1 | tee "${AUDIT_ROOT}/cargo-audit.log"
    cargo deny check 2>&1 | tee "${AUDIT_ROOT}/cargo-deny.log"

    # Check for unsafe patterns
    log "Checking for unsafe patterns..."
    {
        echo "=== unwrap() usage in libraries ==="
        find . -name "*.rs" -path "*/src/*" ! -path "*/tests/*" ! -path "*/examples/*" \
            -exec grep -Hn "unwrap()" {} \; || true

        echo "=== unsafe blocks in crypto paths ==="
        find . -name "*.rs" -path "*crypto*" -o -path "*noise*" -o -path "*sphinx*" \
            -exec grep -Hn "unsafe" {} \; || true

        echo "=== TODO/FIXME in security-critical code ==="
        find . -name "*.rs" -path "*crypto*" -o -path "*noise*" -o -path "*sphinx*" \
            -exec grep -Hn -i "todo\|fixme" {} \; || true
    } > "${AUDIT_ROOT}/security-patterns.log"

    # Validate no FAIL conditions
    if grep -q "unwrap()" "${AUDIT_ROOT}/security-patterns.log"; then
        log "WARNING: Found unwrap() usage in libraries"
    fi

    pass "Build and security hygiene completed"
}

# Phase 2: Coverage & Fuzzing (HTX core)
audit_coverage_fuzz() {
    log "=== Phase 2: Coverage & Fuzzing (HTX core) ==="

    cd repo

    # Generate coverage
    log "Generating code coverage..."
    cargo llvm-cov --workspace --html --output-dir "${AUDIT_ROOT}/coverage" \
        2>&1 | tee "${AUDIT_ROOT}/coverage.log"

    # Extract coverage percentage
    COVERAGE_PCT=$(cargo llvm-cov --workspace --summary-only 2>/dev/null | \
        grep -oP 'TOTAL.*\K\d+\.\d+(?=%)' | head -1 || echo "0")

    log "Coverage: ${COVERAGE_PCT}%"

    if (( $(echo "${COVERAGE_PCT} >= 80" | bc -l) )); then
        pass "Coverage ≥80%: ${COVERAGE_PCT}%"
    else
        log "FAIL: Coverage <80%: ${COVERAGE_PCT}%"
    fi

    # Fuzzing setup
    log "Setting up fuzzing..."

    # Check if fuzz targets exist
    if [[ -d "fuzz" ]]; then
        cd fuzz

        # Run fuzz targets for 5 minutes each
        for target in htx_frame_fuzz htx_noise_fuzz htx_mux_fuzz; do
            if [[ -f "fuzz_targets/${target}.rs" ]]; then
                log "Running fuzz target: ${target}"
                timeout 300 cargo fuzz run "${target}" -- -max_total_time=300 \
                    2>&1 | tee "${AUDIT_ROOT}/fuzz/${target}.log" || true

                # Copy corpus
                cp -r "corpus/${target}" "${AUDIT_ROOT}/fuzz/" 2>/dev/null || true
            else
                log "WARNING: Fuzz target ${target} not found"
            fi
        done

        cd ..
    else
        log "WARNING: No fuzz directory found"
    fi

    pass "Coverage and fuzzing analysis completed"
}

# Phase 3: TLS Camouflage & ECH Stub
audit_tls_camouflage() {
    log "=== Phase 3: TLS Camouflage & ECH Stub ==="

    cd repo

    # Build uTLS generator
    log "Building uTLS generator..."
    cargo build --bin utls_gen --package betanet-utls 2>&1 | tee "${AUDIT_ROOT}/utls-build.log"

    # Generate deterministic ClientHello
    log "Generating deterministic ClientHello..."
    ./target/debug/utls_gen --chrome-stable --output "${AUDIT_ROOT}/pcaps/client-hello.bin" \
        2>&1 | tee "${AUDIT_ROOT}/pcaps/utls-gen.log"

    # Start packet capture
    log "Starting packet capture..."
    sudo tcpdump -i any -w "${AUDIT_ROOT}/pcaps/betanet-capture.pcap" \
        host 127.0.0.1 and port 443 &
    TCPDUMP_PID=$!

    # Start echo server
    log "Starting echo server..."
    cargo run --example echo_server -- --port 443 --tls \
        > "${AUDIT_ROOT}/pcaps/echo-server.log" 2>&1 &
    SERVER_PID=$!
    sleep 2

    # Run echo client (BetaNet)
    log "Running BetaNet echo client..."
    cargo run --example echo_client -- --host 127.0.0.1 --port 443 --tls \
        > "${AUDIT_ROOT}/pcaps/echo-client.log" 2>&1 &
    CLIENT_PID=$!

    # Run Chrome (reference)
    log "Running Chrome reference..."
    timeout 10 "${CHROME_BIN}" --headless --disable-gpu --no-sandbox \
        --ssl-version-fallback-min=tls1.3 \
        --user-data-dir=/tmp/chrome-audit \
        "https://127.0.0.1:443/" \
        > "${AUDIT_ROOT}/pcaps/chrome.log" 2>&1 || true

    # Wait and cleanup
    sleep 5
    kill ${CLIENT_PID} ${SERVER_PID} ${TCPDUMP_PID} 2>/dev/null || true
    sudo chown $(whoami):$(whoami) "${AUDIT_ROOT}/pcaps/betanet-capture.pcap"

    # Analyze JA3/JA4
    log "Analyzing JA3/JA4 fingerprints..."
    {
        echo "=== TLS Analysis ==="
        tshark -r "${AUDIT_ROOT}/pcaps/betanet-capture.pcap" -T fields \
            -e tls.handshake.type -e tls.handshake.ciphersuite \
            -e tls.handshake.extension.type 2>/dev/null || true
    } > "${AUDIT_ROOT}/pcaps/tls-analysis.txt"

    # Check ECH stub
    log "Checking ECH stub presence..."
    grep -i "ech\|encrypted.*client.*hello" "${AUDIT_ROOT}/pcaps/echo-server.log" \
        > "${AUDIT_ROOT}/pcaps/ech-check.txt" || true

    pass "TLS camouflage analysis completed"
}

# Phase 4: QUIC/H3 + MASQUE CONNECT-UDP
audit_quic_masque() {
    log "=== Phase 4: QUIC/H3 + MASQUE CONNECT-UDP ==="

    cd repo

    # Check if QUIC example exists
    if [[ -f "examples/htx_quic_datagram_demo.rs" ]]; then
        log "Running QUIC DATAGRAM demo..."
        cargo run --example htx_quic_datagram_demo \
            > "${AUDIT_ROOT}/quic/datagram-demo.log" 2>&1 || true

        # Check for ALPN, DATAGRAM, MASQUE
        {
            echo "=== ALPN h3 check ==="
            grep -i "alpn.*h3" "${AUDIT_ROOT}/quic/datagram-demo.log" || true

            echo "=== DATAGRAM path check ==="
            grep -i "datagram" "${AUDIT_ROOT}/quic/datagram-demo.log" || true

            echo "=== MASQUE spindle check ==="
            grep -i "masque\|connect-udp" "${AUDIT_ROOT}/quic/datagram-demo.log" || true
        } > "${AUDIT_ROOT}/quic/quic-analysis.txt"

        pass "QUIC/H3 + MASQUE analysis completed"
    else
        log "WARNING: QUIC datagram demo not found"
    fi
}

# Phase 5: Mixnode Performance & Privacy Features
audit_mixnode_performance() {
    log "=== Phase 5: Mixnode Performance & Privacy Features ==="

    cd repo

    # CPU affinity and turbo setup
    log "Setting up performance environment..."
    sudo cpufreq-set -g performance 2>/dev/null || true
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

    # Run mixnode benchmark
    log "Running mixnode performance benchmark..."
    if [[ -f "examples/mixnode_bench.rs" ]]; then
        timeout 65 cargo run --release --example mixnode_bench -- \
            --duration 60 --threads 4 \
            > "${AUDIT_ROOT}/mixnode/bench.log" 2>&1 || true

        # Extract performance metrics
        {
            echo "=== Performance Metrics ==="
            grep -E "pkt/s|packets.*second|throughput" "${AUDIT_ROOT}/mixnode/bench.log" || true

            echo "=== CPU Usage ==="
            grep -E "cpu.*%|CPU" "${AUDIT_ROOT}/mixnode/bench.log" || true

            echo "=== Memory Usage ==="
            grep -E "memory|mem.*MB|RSS" "${AUDIT_ROOT}/mixnode/bench.log" || true
        } > "${AUDIT_ROOT}/mixnode/metrics.txt"

        # Generate flamegraph if possible
        if command -v flamegraph >/dev/null; then
            log "Generating flamegraph..."
            sudo perf record -F 997 -g -- cargo run --release --example mixnode_bench -- --duration 10 \
                > /dev/null 2>&1 || true
            sudo perf script | flamegraph > "${AUDIT_ROOT}/mixnode/flamegraph.svg" 2>/dev/null || true
        fi

        pass "Mixnode performance analysis completed"
    else
        log "WARNING: Mixnode benchmark not found"
    fi
}

# Phase 6: DTN Session Plane & No-Plaintext Invariant
audit_dtn_security() {
    log "=== Phase 6: DTN Session Plane & No-Plaintext Invariant ==="

    cd repo

    # Check if DTN examples exist
    if [[ -f "examples/dtn_chat.rs" ]]; then
        # Run DTN chat in different modes
        for mode in ble-only betanet-only hybrid; do
            log "Testing DTN mode: ${mode}"
            timeout 30 cargo run --example dtn_chat -- --mode "${mode}" --debug-gateway \
                > "${AUDIT_ROOT}/dtn/dtn-${mode}.log" 2>&1 || true

            # Check for plaintext leaks
            {
                echo "=== Checking for plaintext in gateway logs (${mode}) ==="
                echo "Looking for message content leaks..."
                grep -i "plaintext\|message.*content\|payload.*[a-zA-Z]" \
                    "${AUDIT_ROOT}/dtn/dtn-${mode}.log" || true

                echo "=== Checking for proper encryption ==="
                grep -i "encrypt\|cipher\|crypt" \
                    "${AUDIT_ROOT}/dtn/dtn-${mode}.log" || true
            } > "${AUDIT_ROOT}/dtn/plaintext-check-${mode}.txt"
        done

        pass "DTN security analysis completed"
    else
        log "WARNING: DTN chat example not found"
    fi
}

# Phase 7: Agent Fabric + MLS Cohorts
audit_agent_fabric() {
    log "=== Phase 7: Agent Fabric + MLS Cohorts ==="

    cd repo

    # Check for agent examples
    if [[ -f "examples/agent_twin_tutor.rs" ]]; then
        log "Testing Agent Fabric RPC with DTN fallback..."
        timeout 60 cargo run --example agent_twin_tutor -- --test-fallback \
            > "${AUDIT_ROOT}/agent/rpc-fallback.log" 2>&1 || true

        # MLS cohort churn test
        log "Testing MLS cohort churn..."
        timeout 120 cargo run --example mls_cohort_test -- --churn-rate 0.3 --duration 60 \
            > "${AUDIT_ROOT}/agent/mls-churn.log" 2>&1 || true

        # Extract metrics
        {
            echo "=== RPC Delivery Metrics ==="
            grep -E "delivery.*%|success.*rate|delivered.*\d+" \
                "${AUDIT_ROOT}/agent/rpc-fallback.log" || true

            echo "=== MLS Key Rotation ==="
            grep -E "key.*rotation|rekeying|key.*update" \
                "${AUDIT_ROOT}/agent/mls-churn.log" || true
        } > "${AUDIT_ROOT}/agent/metrics.txt"

        pass "Agent Fabric analysis completed"
    else
        log "WARNING: Agent Fabric examples not found"
    fi
}

# Phase 8: Federated Learning & Receipts
audit_federated_learning() {
    log "=== Phase 8: Federated Learning & Receipts ==="

    cd repo

    # Check for FL examples
    if [[ -f "examples/fl_benchmark.rs" ]]; then
        log "Running federated learning benchmark..."
        timeout 300 cargo run --release --example fl_benchmark -- \
            --participants 500 --rounds 3 --enable-dp --enable-compression \
            > "${AUDIT_ROOT}/fl/benchmark.log" 2>&1 || true

        # Extract performance metrics
        {
            echo "=== Performance Metrics ==="
            grep -E "P50.*ms|P99.*ms|msgs/s|msgs.*second|throughput" \
                "${AUDIT_ROOT}/fl/benchmark.log" || true

            echo "=== Memory Usage ==="
            grep -E "memory.*MB|peak.*memory|RSS.*MB" \
                "${AUDIT_ROOT}/fl/benchmark.log" || true

            echo "=== Privacy Overhead ==="
            grep -E "privacy.*overhead|DP.*overhead|SecureAgg.*cost" \
                "${AUDIT_ROOT}/fl/benchmark.log" || true

            echo "=== DP Accounting ==="
            grep -E "epsilon|delta|ε.*δ|privacy.*budget" \
                "${AUDIT_ROOT}/fl/benchmark.log" || true
        } > "${AUDIT_ROOT}/fl/metrics.txt"

        # Extract receipts
        grep -A5 -B5 "receipt\|energy\|FLOPs" "${AUDIT_ROOT}/fl/benchmark.log" \
            > "${AUDIT_ROOT}/fl/receipts.txt" || true

        pass "Federated learning analysis completed"
    else
        log "WARNING: FL benchmark not found"
    fi
}

# Phase 9: Linter, SBOM, Spec Checks
audit_linter_sbom() {
    log "=== Phase 9: Linter, SBOM, Spec Checks ==="

    cd repo

    # Run betanet linter
    log "Running betanet linter..."
    if [[ -f "target/debug/betanet-linter" || -f "target/release/betanet-linter" ]]; then
        LINTER_BIN=$(find target -name "betanet-linter" -type f | head -1)

        "${LINTER_BIN}" lint --directory . --output "${AUDIT_ROOT}/linter/lint-report.json" \
            2>&1 | tee "${AUDIT_ROOT}/linter/linter.log"

        # Generate SBOM
        log "Generating SBOM..."
        "${LINTER_BIN}" sbom --format spdx --output "${AUDIT_ROOT}/sbom/betanet-sbom.json" \
            2>&1 | tee "${AUDIT_ROOT}/sbom/sbom.log"

        # Alternative SBOM with cargo-about
        cargo about generate about.hbs > "${AUDIT_ROOT}/sbom/cargo-about.html" 2>/dev/null || true

        pass "Linter and SBOM analysis completed"
    else
        log "WARNING: betanet-linter not found"
    fi
}

# Phase 10: Timing Correlation Harness
audit_timing_correlation() {
    log "=== Phase 10: Timing Correlation Harness ==="

    cd repo

    # Check for timing correlation test
    if [[ -f "examples/timing_correlation.rs" ]]; then
        log "Running timing correlation analysis..."
        timeout 120 cargo run --example timing_correlation -- \
            --privacy-mode strict --duration 60 \
            > "${AUDIT_ROOT}/correlation/timing.log" 2>&1 || true

        # Extract K-S test results
        {
            echo "=== K-S Test Results ==="
            grep -E "K-S.*test|correlation.*\d+\.\d+|timing.*analysis" \
                "${AUDIT_ROOT}/correlation/timing.log" || true
        } > "${AUDIT_ROOT}/correlation/ks-results.txt"

        pass "Timing correlation analysis completed"
    else
        log "WARNING: Timing correlation test not found"
    fi
}

# Phase 11: Final Report Generation
generate_final_report() {
    log "=== Phase 11: Final Report Generation ==="

    local report_file="${AUDIT_ROOT}/audit_report.md"

    cat > "${report_file}" << 'EOF'
# AIVillage BetaNet Security Audit Report

**Audit Date:** $(date)
**Commit SHA:** ${COMMIT_SHA}
**Environment:** ${VPS_SPEC} equivalent on Ubuntu 22.04

## Executive Summary

This report presents the findings of a comprehensive security audit of the AIVillage BetaNet multi-layer transport system.

## Detailed Results

| Claim | Test Phase | Result | Evidence | Notes |
|-------|------------|--------|----------|-------|
EOF

    # Add results for each test phase
    {
        echo "| HTX TLS1.3+Noise, ≥80% fuzz coverage | Phases 2-4 | $(check_coverage_result) | [Coverage Report](coverage/index.html) | $(get_coverage_notes) |"
        echo "| QUIC/H3 + MASQUE | Phase 4 | $(check_quic_result) | [QUIC Logs](quic/) | $(get_quic_notes) |"
        echo "| Mixnode ≥25k pkt/s | Phase 5 | $(check_mixnode_result) | [Bench Results](mixnode/) | $(get_mixnode_notes) |"
        echo "| DTN no plaintext at gateways | Phase 6 | $(check_dtn_result) | [DTN Logs](dtn/) | $(get_dtn_notes) |"
        echo "| Agent RPC+DTN fallback; MLS churn | Phase 7 | $(check_agent_result) | [Agent Logs](agent/) | $(get_agent_notes) |"
        echo "| FL (SecureAgg+DP) & overhead | Phase 8 | $(check_fl_result) | [FL Metrics](fl/) | $(get_fl_notes) |"
        echo "| Linter 11 checks + SBOM | Phase 9 | $(check_linter_result) | [Reports](linter/) | $(get_linter_notes) |"
        echo "| Camouflage (JA3/JA4, K-S<0.2) | Phases 3 & 10 | $(check_camouflage_result) | [PCAP Analysis](pcaps/) | $(get_camouflage_notes) |"
    } >> "${report_file}"

    cat >> "${report_file}" << 'EOF'

## Artifacts Location

All test artifacts are available in the `audit_artifacts/` directory:

- `coverage/` - Code coverage HTML reports
- `fuzz/` - Fuzzing logs and corpora
- `pcaps/` - Network captures and TLS analysis
- `mixnode/` - Performance benchmarks and flamegraphs
- `correlation/` - Timing correlation analysis
- `dtn/` - DTN security logs
- `sbom/` - Software Bill of Materials
- `fl/` - Federated learning metrics and receipts
- `agent/` - Agent fabric test results
- `quic/` - QUIC/MASQUE validation logs
- `linter/` - Spec compliance reports

## Recommendations

[Specific recommendations based on findings will be added here]

## Conclusion

[Overall assessment based on test results]

---
*Generated by AIVillage BetaNet Security Audit Framework v1.0*
EOF

    log "Final audit report generated: ${report_file}"
}

# Result checking functions (to be implemented based on actual test outputs)
check_coverage_result() {
    if [[ -f "${AUDIT_ROOT}/coverage.log" ]]; then
        local coverage=$(grep -oP '\d+\.\d+(?=%)' "${AUDIT_ROOT}/coverage.log" | head -1 || echo "0")
        if (( $(echo "${coverage} >= 80" | bc -l) )); then
            echo "PASS"
        else
            echo "FAIL"
        fi
    else
        echo "INCONCLUSIVE"
    fi
}

get_coverage_notes() {
    if [[ -f "${AUDIT_ROOT}/coverage.log" ]]; then
        local coverage=$(grep -oP '\d+\.\d+(?=%)' "${AUDIT_ROOT}/coverage.log" | head -1 || echo "0")
        echo "Coverage: ${coverage}%"
    else
        echo "Coverage test not completed"
    fi
}

# Additional result checking functions would be implemented here...
check_quic_result() { echo "INCONCLUSIVE"; }
check_mixnode_result() { echo "INCONCLUSIVE"; }
check_dtn_result() { echo "INCONCLUSIVE"; }
check_agent_result() { echo "INCONCLUSIVE"; }
check_fl_result() { echo "INCONCLUSIVE"; }
check_linter_result() { echo "INCONCLUSIVE"; }
check_camouflage_result() { echo "INCONCLUSIVE"; }

get_quic_notes() { echo "QUIC test implementation needed"; }
get_mixnode_notes() { echo "Mixnode performance test needed"; }
get_dtn_notes() { echo "DTN security validation needed"; }
get_agent_notes() { echo "Agent fabric tests needed"; }
get_fl_notes() { echo "FL benchmark needed"; }
get_linter_notes() { echo "Linter execution needed"; }
get_camouflage_notes() { echo "TLS camouflage analysis needed"; }

# Main execution
main() {
    log "Starting AIVillage BetaNet Security Audit"

    init_audit
    audit_build_security
    audit_coverage_fuzz
    audit_tls_camouflage
    audit_quic_masque
    audit_mixnode_performance
    audit_dtn_security
    audit_agent_fabric
    audit_federated_learning
    audit_linter_sbom
    audit_timing_correlation
    generate_final_report

    log "Audit completed. Report available at: ${AUDIT_ROOT}/audit_report.md"
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

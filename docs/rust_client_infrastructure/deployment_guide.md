# Rust Client Infrastructure - Deployment Guide

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows (10+)
- **CPU**: x86_64 or ARM64 with 2+ cores
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB available disk space
- **Network**: Internet connectivity for initial setup

#### Development Requirements
- **Rust**: 1.78.0 or later
- **Cargo**: Latest stable version
- **C Compiler**: GCC 7+ or Clang 10+ (for FFI bindings)
- **OpenSSL**: 1.1.1+ or 3.0+ (system installation)

### Platform-Specific Setup

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install OpenSSL via Homebrew (optional)
brew install openssl
export OPENSSL_ROOT_DIR=$(brew --prefix openssl)
```

#### Windows
```powershell
# Install Visual Studio Build Tools or Visual Studio Community
# Download and run: https://visualstudio.microsoft.com/downloads/

# Install Rust via rustup-init.exe
# Download from: https://rustup.rs/

# Verify installation
rustc --version
cargo --version
```

## Installation

### Building from Source

#### Clone Repository
```bash
git clone https://github.com/your-org/aivillage.git
cd aivillage
```

#### Build All Crates
```bash
# Navigate to Rust workspace
cd packages/p2p/betanet-bounty

# Build entire workspace in release mode
cargo build --workspace --release

# Run tests to verify build
cargo test --workspace
```

#### Alternative: Build Specific Components
```bash
# Build only HTX transport
cargo build --package betanet-htx --release

# Build only mixnode
cargo build --package betanet-mixnode --release --no-default-features --features sphinx

# Build with specific features
cargo build --package betanet-htx --features quic,tls-camouflage --release
```

### Installation Targets

#### System Installation
```bash
# Install all binaries to system PATH
cargo install --path crates/betanet-htx --locked
cargo install --path crates/betanet-mixnode --locked
cargo install --path crates/betanet-utls --locked
cargo install --path crates/betanet-linter --locked

# Verify installation
which betanet-htx
which betanet-mixnode
```

#### Development Installation
```bash
# Build for development with debug symbols
cargo build --workspace

# Run directly from workspace
cargo run --package betanet-htx -- --help
cargo run --package betanet-mixnode -- --help
```

## Configuration

### Core Configuration Files

#### HTX Transport Configuration
**File**: `configs/htx.toml`

```toml
[transport]
# Listen address for HTX server
listen_addr = "0.0.0.0:9000"

# Transport protocols to enable
enable_tcp = true
enable_quic = false
enable_websocket = true

# Security settings
enable_noise_xk = true
enable_tickets = true
enable_tls_camouflage = false

# Connection limits
max_connections = 1000
connection_timeout_secs = 30
keepalive_interval_secs = 10

# Frame buffer configuration
frame_buffer_size = 1048576  # 1MB

[noise]
# Static private key (32 bytes hex) - generate with: openssl rand -hex 32
static_private_key = "your-32-byte-hex-key-here"

# Remote static key for client connections (32 bytes hex)
remote_static_key = "remote-32-byte-hex-key-here"

[tls_camouflage]
# Enable TLS fingerprinting evasion
enable = false
camouflage_domain = "example.com"
chrome_version = "119.0.6045.123"

[tickets]
# Access ticket configuration
issuer_keypair_path = "keys/ticket_issuer.key"
ticket_lifetime_secs = 3600
```

#### Mixnode Configuration
**File**: `configs/mixnode.toml`

```toml
[mixnode]
# Node identification
node_id = "mixnode-001"
listen_addr = "0.0.0.0:9001"

# Mixing strategy
mix_strategy = "threshold"  # threshold, timed, binomial
mix_threshold = 10
mix_interval_ms = 1000

# VRF delay parameters
min_delay_ms = 10
max_delay_ms = 1000
vrf_key_path = "keys/vrf_secret.key"

# Cover traffic settings
enable_cover_traffic = true
cover_traffic_rate = 0.1  # packets per second
cover_traffic_size = 1024  # bytes

[crypto]
# Sphinx packet processing
sphinx_private_key_path = "keys/sphinx_private.key"
enable_replay_protection = true
replay_cache_size = 10000

[performance]
# Performance tuning
worker_threads = 4
packet_buffer_size = 1000
processing_timeout_ms = 5000
```

#### DTN Configuration
**File**: `configs/dtn.toml`

```toml
[dtn]
# Node endpoint identifier
node_eid = "dtn://node-001"
storage_path = "/var/lib/aivillage/dtn"

# Bundle lifetime settings
default_lifetime_secs = 86400  # 24 hours
max_lifetime_secs = 604800     # 7 days

# Routing configuration
routing_algorithm = "epidemic"  # epidemic, spray_and_wait, prophet
max_bundle_size = 10485760     # 10MB

[storage]
# Storage backend
backend = "sled"  # sled, sqlite
max_storage_size = 1073741824  # 1GB
cleanup_interval_secs = 3600

[custody]
# Custody transfer settings
enable_custody = true
custody_timeout_secs = 300
max_retransmits = 3
```

### Key Generation

#### Generate Cryptographic Keys
```bash
# Create keys directory
mkdir -p keys

# Generate Ed25519 keys for signatures
openssl genpkey -algorithm Ed25519 -out keys/ed25519_private.pem
openssl pkey -in keys/ed25519_private.pem -pubout -out keys/ed25519_public.pem

# Generate X25519 keys for Noise protocol
openssl genpkey -algorithm X25519 -out keys/x25519_private.pem
openssl pkey -in keys/x25519_private.pem -pubout -out keys/x25519_public.pem

# Generate random 32-byte keys for various uses
openssl rand -hex 32 > keys/static_private.key
openssl rand -hex 32 > keys/vrf_secret.key
```

#### Key Management Script
**File**: `scripts/generate_keys.sh`

```bash
#!/bin/bash
set -euo pipefail

KEYS_DIR="${1:-keys}"
mkdir -p "$KEYS_DIR"

echo "Generating cryptographic keys..."

# Generate Ed25519 signing keys
openssl genpkey -algorithm Ed25519 -out "$KEYS_DIR/signing_private.pem"
openssl pkey -in "$KEYS_DIR/signing_private.pem" -pubout -out "$KEYS_DIR/signing_public.pem"

# Generate X25519 Noise protocol keys  
openssl genpkey -algorithm X25519 -out "$KEYS_DIR/noise_private.pem"
openssl pkey -in "$KEYS_DIR/noise_private.pem" -pubout -out "$KEYS_DIR/noise_public.pem"

# Generate VRF keys
openssl rand -hex 32 > "$KEYS_DIR/vrf_secret.key"

# Generate access ticket issuer keys
openssl genpkey -algorithm Ed25519 -out "$KEYS_DIR/ticket_issuer_private.pem"
openssl pkey -in "$KEYS_DIR/ticket_issuer_private.pem" -pubout -out "$KEYS_DIR/ticket_issuer_public.pem"

# Set appropriate permissions
chmod 600 "$KEYS_DIR"/*.pem "$KEYS_DIR"/*.key

echo "Keys generated successfully in $KEYS_DIR/"
echo "Public key fingerprints:"
openssl pkey -pubin -in "$KEYS_DIR/signing_public.pem" -noout -text | head -2
openssl pkey -pubin -in "$KEYS_DIR/noise_public.pem" -noout -text | head -2
```

## Service Deployment

### Systemd Services (Linux)

#### HTX Transport Service
**File**: `/etc/systemd/system/aivillage-htx.service`

```ini
[Unit]
Description=AIVillage HTX Transport
After=network.target
Wants=network.target

[Service]
Type=simple
User=aivillage
Group=aivillage
WorkingDirectory=/opt/aivillage
ExecStart=/usr/local/bin/betanet-htx server --config /etc/aivillage/htx.toml
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/aivillage

# Resource limits
MemoryMax=1G
TasksMax=1000

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Mixnode Service
**File**: `/etc/systemd/system/aivillage-mixnode.service`

```ini
[Unit]
Description=AIVillage Mixnode
After=network.target
Wants=network.target

[Service]
Type=simple
User=aivillage
Group=aivillage
WorkingDirectory=/opt/aivillage
ExecStart=/usr/local/bin/betanet-mixnode --config /etc/aivillage/mixnode.toml
Restart=always
RestartSec=5

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/aivillage

# Resource limits
MemoryMax=2G
TasksMax=2000

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Services
```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable aivillage-htx
sudo systemctl enable aivillage-mixnode

# Start services
sudo systemctl start aivillage-htx
sudo systemctl start aivillage-mixnode

# Check status
sudo systemctl status aivillage-htx
sudo systemctl status aivillage-mixnode

# View logs
sudo journalctl -u aivillage-htx -f
sudo journalctl -u aivillage-mixnode -f
```

### Docker Deployment

#### Dockerfile
**File**: `docker/Dockerfile`

```dockerfile
FROM rust:1.78-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Build release binaries
RUN cargo build --workspace --release

# Runtime image
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aivillage && useradd -r -g aivillage aivillage

# Copy binaries from builder
COPY --from=builder /app/target/release/betanet-htx /usr/local/bin/
COPY --from=builder /app/target/release/betanet-mixnode /usr/local/bin/
COPY --from=builder /app/target/release/betanet-utls /usr/local/bin/
COPY --from=builder /app/target/release/betanet-linter /usr/local/bin/

# Create directories
RUN mkdir -p /etc/aivillage /var/lib/aivillage/keys /var/lib/aivillage/data
RUN chown -R aivillage:aivillage /var/lib/aivillage

# Switch to non-root user
USER aivillage

# Expose ports
EXPOSE 9000 9001

# Default command
CMD ["betanet-htx", "server", "--config", "/etc/aivillage/htx.toml"]
```

#### Docker Compose
**File**: `docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  htx-transport:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "9000:9000"
    volumes:
      - ./configs:/etc/aivillage:ro
      - htx-data:/var/lib/aivillage
    environment:
      - RUST_LOG=info
    command: ["betanet-htx", "server", "--config", "/etc/aivillage/htx.toml"]
    restart: unless-stopped

  mixnode:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "9001:9001"
    volumes:
      - ./configs:/etc/aivillage:ro
      - mixnode-data:/var/lib/aivillage
    environment:
      - RUST_LOG=info
    command: ["betanet-mixnode", "--config", "/etc/aivillage/mixnode.toml"]
    restart: unless-stopped
    depends_on:
      - htx-transport

  dtn-node:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "9002:9002"
    volumes:
      - ./configs:/etc/aivillage:ro
      - dtn-data:/var/lib/aivillage
    environment:
      - RUST_LOG=info
    command: ["betanet-dtn", "--config", "/etc/aivillage/dtn.toml"]
    restart: unless-stopped

volumes:
  htx-data:
  mixnode-data:
  dtn-data:

networks:
  default:
    name: aivillage-net
```

#### Deploy with Docker Compose
```bash
# Build and start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Scale mixnodes
docker-compose -f docker/docker-compose.yml up -d --scale mixnode=3

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Kubernetes Deployment

#### ConfigMap
**File**: `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aivillage-config
  namespace: aivillage
data:
  htx.toml: |
    [transport]
    listen_addr = "0.0.0.0:9000"
    enable_tcp = true
    enable_quic = false
    enable_noise_xk = true
    max_connections = 1000
    
    [noise]
    static_private_key = "${NOISE_PRIVATE_KEY}"
    
  mixnode.toml: |
    [mixnode]
    node_id = "${POD_NAME}"
    listen_addr = "0.0.0.0:9001"
    mix_strategy = "threshold"
    mix_threshold = 10
    
    [crypto]
    sphinx_private_key_path = "/etc/keys/sphinx_private.key"
```

#### Deployment
**File**: `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-htx
  namespace: aivillage
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aivillage-htx
  template:
    metadata:
      labels:
        app: aivillage-htx
    spec:
      containers:
      - name: htx-transport
        image: aivillage/rust-client:latest
        ports:
        - containerPort: 9000
        env:
        - name: RUST_LOG
          value: "info"
        - name: NOISE_PRIVATE_KEY
          valueFrom:
            secretKeyRef:
              name: aivillage-keys
              key: noise-private-key
        volumeMounts:
        - name: config
          mountPath: /etc/aivillage
          readOnly: true
        - name: keys
          mountPath: /etc/keys
          readOnly: true
        command: ["betanet-htx", "server", "--config", "/etc/aivillage/htx.toml"]
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          tcpSocket:
            port: 9000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          tcpSocket:
            port: 9000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: aivillage-config
      - name: keys
        secret:
          secretName: aivillage-keys
```

#### Service
**File**: `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: aivillage-htx-service
  namespace: aivillage
spec:
  selector:
    app: aivillage-htx
  ports:
  - name: htx
    port: 9000
    targetPort: 9000
    protocol: TCP
  type: LoadBalancer
```

## Monitoring and Observability

### Metrics Configuration

#### Prometheus Integration
```rust
// Add to Cargo.toml
prometheus = "0.13"
tokio-metrics = "0.1"

// In main application
use prometheus::{Counter, Histogram, Gauge, Registry};

pub struct Metrics {
    pub packets_processed: Counter,
    pub processing_time: Histogram,
    pub active_connections: Gauge,
}

impl Metrics {
    pub fn new(registry: &Registry) -> Self {
        let packets_processed = Counter::new(
            "aivillage_packets_processed_total",
            "Total number of packets processed"
        ).unwrap();
        
        let processing_time = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "aivillage_processing_duration_seconds",
                "Packet processing time in seconds"
            )
        ).unwrap();
        
        let active_connections = Gauge::new(
            "aivillage_active_connections",
            "Number of active connections"
        ).unwrap();
        
        registry.register(Box::new(packets_processed.clone())).unwrap();
        registry.register(Box::new(processing_time.clone())).unwrap();
        registry.register(Box::new(active_connections.clone())).unwrap();
        
        Self {
            packets_processed,
            processing_time,
            active_connections,
        }
    }
}
```

#### Health Check Endpoints
```rust
use warp::Filter;

pub fn health_routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    health_check()
        .or(metrics())
        .or(ready_check())
}

fn health_check() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path("health")
        .and(warp::get())
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "status": "healthy",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "version": env!("CARGO_PKG_VERSION")
            }))
        })
}

fn metrics() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path("metrics")
        .and(warp::get())
        .map(|| {
            let encoder = prometheus::TextEncoder::new();
            let metric_families = prometheus::gather();
            let output = encoder.encode_to_string(&metric_families).unwrap();
            warp::reply::with_header(output, "content-type", "text/plain; charset=utf-8")
        })
}
```

### Logging Configuration

#### Structured Logging
```bash
# Environment variables for logging
export RUST_LOG=info,betanet_htx=debug,betanet_mixnode=debug
export RUST_LOG_STYLE=json

# Or in configuration file
echo 'RUST_LOG=info' >> /etc/aivillage/environment
echo 'RUST_LOG_STYLE=json' >> /etc/aivillage/environment
```

```rust
use tracing::{info, warn, error, debug};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .init();
}

// Usage in application
#[tracing::instrument(fields(packet_id = %packet.id))]
pub async fn process_packet(&self, packet: SphinxPacket) -> Result<()> {
    debug!("Processing packet");
    
    let start = std::time::Instant::now();
    let result = self.inner_process_packet(packet).await;
    let duration = start.elapsed();
    
    match result {
        Ok(_) => {
            info!(
                processing_time_ms = duration.as_millis(),
                "Packet processed successfully"
            );
        }
        Err(ref e) => {
            error!(
                error = %e,
                processing_time_ms = duration.as_millis(),
                "Packet processing failed"
            );
        }
    }
    
    result
}
```

## Security Hardening

### File Permissions
```bash
# Create aivillage user and group
sudo groupadd -r aivillage
sudo useradd -r -g aivillage -s /bin/false -M aivillage

# Set directory permissions
sudo mkdir -p /etc/aivillage /var/lib/aivillage
sudo chown root:aivillage /etc/aivillage
sudo chmod 750 /etc/aivillage
sudo chown aivillage:aivillage /var/lib/aivillage
sudo chmod 750 /var/lib/aivillage

# Set key permissions
sudo chmod 600 /etc/aivillage/keys/*
sudo chown aivillage:aivillage /etc/aivillage/keys/*
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 9000/tcp comment 'AIVillage HTX'
sudo ufw allow 9001/tcp comment 'AIVillage Mixnode'
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 9000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9001 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### SELinux/AppArmor Policies
```bash
# SELinux context (RHEL/CentOS)
sudo setsebool -P nis_enabled 1
sudo semanage port -a -t http_port_t -p tcp 9000
sudo semanage port -a -t http_port_t -p tcp 9001

# AppArmor profile (Ubuntu)
cat > /etc/apparmor.d/usr.local.bin.betanet-htx << 'EOF'
#include <tunables/global>

/usr/local/bin/betanet-htx {
  #include <abstractions/base>
  #include <abstractions/nameservice>
  
  capability net_bind_service,
  capability setuid,
  capability setgid,
  
  /usr/local/bin/betanet-htx mr,
  /etc/aivillage/** r,
  /var/lib/aivillage/** rw,
  
  network inet stream,
  network inet dgram,
  
  deny @{HOME}/** rw,
  deny /tmp/** rw,
}
EOF

sudo apparmor_parser -r /etc/apparmor.d/usr.local.bin.betanet-htx
```

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# OpenSSL not found
export OPENSSL_DIR=/usr/local/ssl
export OPENSSL_LIB_DIR=/usr/local/ssl/lib
export OPENSSL_INCLUDE_DIR=/usr/local/ssl/include

# Link errors on Windows
set OPENSSL_VENDORED=1
cargo build --release

# Missing system libraries
sudo apt install build-essential pkg-config libssl-dev
```

#### Runtime Errors
```bash
# Permission denied on key files
sudo chown aivillage:aivillage /etc/aivillage/keys/*
sudo chmod 600 /etc/aivillage/keys/*

# Port already in use
sudo lsof -i :9000
sudo netstat -tulpn | grep 9000

# Memory/resource limits
ulimit -n 65536  # Increase file descriptor limit
echo 'aivillage soft nofile 65536' >> /etc/security/limits.conf
```

### Debug Mode
```bash
# Enable debug logging
RUST_LOG=debug cargo run --package betanet-htx

# Enable backtraces
RUST_BACKTRACE=1 cargo run --package betanet-htx

# Profile performance
cargo install flamegraph
cargo flamegraph --package betanet-htx
```

### Validation Commands
```bash
# Test connectivity
cargo run --package betanet-htx -- client --target 127.0.0.1:9000

# Validate configuration
cargo run --package betanet-linter -- lint --config /etc/aivillage/htx.toml

# Generate SBOM
cargo run --package betanet-linter -- sbom --format spdx --output aivillage-sbom.json

# Performance benchmark
cargo bench --package betanet-htx
```

This deployment guide provides comprehensive instructions for deploying the AIVillage Rust Client Infrastructure across various environments with proper security hardening and monitoring capabilities.
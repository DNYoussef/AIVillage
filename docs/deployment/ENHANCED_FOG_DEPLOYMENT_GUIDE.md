# Enhanced Fog Computing Platform - Deployment Guide

## Overview

This comprehensive deployment guide covers the complete setup, configuration, and deployment of AIVillage's Enhanced Fog Computing Platform. The platform provides enterprise-grade privacy-first fog cloud capabilities with 8 advanced security components.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Installation Methods](#installation-methods)
4. [Component Configuration](#component-configuration)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Scaling and High Availability](#scaling-and-high-availability)

## System Requirements

### Minimum Requirements (Development)
- **OS**: Linux (Ubuntu 20.04+), macOS (12.0+), Windows 10/11
- **CPU**: 4 cores (x86_64)
- **Memory**: 8GB RAM
- **Storage**: 50GB available space
- **Network**: Broadband internet connection
- **Python**: 3.9+ (3.11 recommended)
- **Docker**: 20.10+ (optional but recommended)

### Recommended Requirements (Production)
- **OS**: Linux (Ubuntu 22.04 LTS, RHEL 8+, CentOS Stream 8+)
- **CPU**: 16+ cores (x86_64 with TEE support)
- **Memory**: 64GB+ RAM
- **Storage**: 1TB+ SSD storage
- **Network**: Gigabit ethernet, low-latency connection
- **Python**: 3.11
- **Docker**: 24.0+
- **Kubernetes**: 1.28+ (for cluster deployment)

### Hardware TEE Support (Optional but Recommended)
- **AMD**: EPYC 7003/7004 series with SEV-SNP
- **Intel**: Xeon Scalable (3rd/4th Gen) with TDX
- **ARM**: Cortex-A78+ with TrustZone
- **Alternative**: Software isolation with gVisor/Firecracker

### Network Requirements
- **Ports**: 8000 (API), 8001 (WebSocket), 443 (HTTPS)
- **Firewall**: Allow inbound connections on required ports
- **DNS**: Proper DNS resolution for external APIs
- **Bandwidth**: 100Mbps+ for production workloads

## Pre-deployment Checklist

### 1. Environment Preparation
```bash
# Check system compatibility
python --version  # Should be 3.9+
docker --version  # Should be 20.10+
git --version    # For source code management

# Check available resources
free -h          # Memory availability
df -h           # Disk space
nproc           # CPU cores
```

### 2. Dependency Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-pip python3.11-venv
sudo apt install -y docker.io docker-compose-plugin
sudo apt install -y build-essential libffi-dev libssl-dev

# RHEL/CentOS/Fedora
sudo dnf update
sudo dnf install -y python3.11 python3.11-pip
sudo dnf install -y docker docker-compose
sudo dnf install -y gcc openssl-devel libffi-devel

# macOS (with Homebrew)
brew install python@3.11 docker docker-compose
brew install openssl libffi
```

### 3. TEE Hardware Detection (Optional)
```bash
# Check for AMD SEV-SNP
dmesg | grep -i sev
cat /sys/module/kvm_amd/parameters/sev
cat /sys/module/kvm_amd/parameters/sev_es

# Check for Intel TDX
dmesg | grep -i tdx
cat /proc/cpuinfo | grep -i tdx

# Alternative: Use detection script
python scripts/detect_tee_hardware.py
```

### 4. Security Configuration
```bash
# Generate JWT secret key
openssl rand -hex 32 > .jwt_secret

# Create SSL certificates (for production)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Set up firewall rules (example for ufw)
sudo ufw allow 8000/tcp
sudo ufw allow 8001/tcp
sudo ufw allow 443/tcp
```

## Installation Methods

### Method 1: Automated Deployment (Recommended)

#### Quick Start Deployment
```bash
# Clone repository
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage

# Run automated deployment script
python scripts/deploy_enhanced_fog_system.py

# The script will:
# - Check prerequisites
# - Install dependencies
# - Configure components
# - Start services
# - Validate deployment
# - Provide access URLs
```

#### Production Deployment
```bash
# Production deployment with custom configuration
python scripts/deploy_enhanced_fog_system.py \
    --environment production \
    --config config/production.yaml \
    --enable-tee \
    --enable-clustering \
    --ssl-cert /path/to/cert.pem \
    --ssl-key /path/to/key.pem
```

### Method 2: Manual Deployment

#### Step 1: Environment Setup
```bash
# Create project directory
mkdir -p /opt/aivillage
cd /opt/aivillage

# Clone repository
git clone https://github.com/DNYoussef/AIVillage.git .

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-production.txt
```

#### Step 2: Configuration
```bash
# Copy configuration templates
cp config/production.yaml.template config/production.yaml
cp config/fog_config.yaml.template config/fog_config.yaml

# Set environment variables
export PYTHONPATH=/opt/aivillage
export FOG_CONFIG_PATH=/opt/aivillage/config/fog_config.yaml
export JWT_SECRET_KEY=$(cat .jwt_secret)
export AIVILLAGE_ENV=production
```

#### Step 3: Database Setup (Optional)
```bash
# Initialize database for persistent storage
python scripts/init_database.py --config config/production.yaml

# Run database migrations
python scripts/migrate_database.py --upgrade
```

#### Step 4: Component Initialization
```bash
# Initialize fog components
python -m infrastructure.fog.integration.fog_system_manager --init

# Validate component configuration
python scripts/validate_fog_config.py --config config/fog_config.yaml

# Start components individually (for debugging)
python -m infrastructure.fog.tee.tee_runtime_manager --start
python -m infrastructure.fog.proofs.proof_generator --start
python -m infrastructure.fog.zk.zk_predicates --start
# ... (repeat for all 8 components)
```

#### Step 5: API Gateway Startup
```bash
# Start enhanced API gateway
python infrastructure/gateway/enhanced_unified_api_gateway.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --config config/production.yaml
```

### Method 3: Docker Deployment

#### Docker Compose (Recommended for Development)
```yaml
# docker-compose.yml
version: '3.8'

services:
  fog-platform:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - AIVILLAGE_ENV=production
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - FOG_CONFIG_PATH=/app/config/fog_config.yaml
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f fog-platform

# Scale services
docker-compose up -d --scale fog-platform=3
```

#### Kubernetes Deployment
```yaml
# k8s/fog-platform-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fog-platform
  namespace: aivillage
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fog-platform
  template:
    metadata:
      labels:
        app: fog-platform
    spec:
      containers:
      - name: fog-platform
        image: aivillage/fog-platform:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        env:
        - name: AIVILLAGE_ENV
          value: "production"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: fog-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: fog-platform-service
  namespace: aivillage
spec:
  selector:
    app: fog-platform
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: websocket
    port: 8001
    targetPort: 8001
  type: LoadBalancer
```

```bash
# Deploy to Kubernetes
kubectl create namespace aivillage
kubectl apply -f k8s/
kubectl get pods -n aivillage -w
```

## Component Configuration

### 1. TEE Runtime Configuration
```yaml
# config/tee_config.yaml
tee_runtime:
  preferred_tee_types:
    - "amd_sev_snp"
    - "intel_tdx" 
    - "software_isolation"
  
  enclave_defaults:
    memory_mb: 2048
    cpu_cores: 2
    measurement_policy: "strict"
    network_isolation: true
    attestation_timeout_seconds: 300
  
  security_policies:
    allow_debug_enclaves: false
    require_signed_code: true
    enforce_memory_encryption: true
  
  performance:
    max_concurrent_enclaves: 16
    attestation_cache_ttl_seconds: 3600
    metrics_collection_interval: 60
```

### 2. Cryptographic Proof System
```yaml
# config/proof_config.yaml
proof_system:
  supported_proof_types:
    - "proof_of_execution"
    - "proof_of_audit"
    - "proof_of_sla"
  
  blockchain_anchoring:
    enabled: true
    blockchain: "ethereum"
    contract_address: "0x1234567890abcdef..."
    gas_price_gwei: 20
    confirmation_blocks: 6
  
  merkle_tree:
    hash_algorithm: "sha256"
    tree_depth: 20
    batch_size: 100
  
  performance:
    proof_generation_timeout_seconds: 300
    verification_timeout_seconds: 60
    batch_processing_enabled: true
```

### 3. Zero-Knowledge Predicates
```yaml
# config/zk_config.yaml
zk_predicates:
  available_predicates:
    network_policy_compliance:
      proof_size_kb: 2
      verification_time_ms: 800
      circuit_file: "circuits/network_policy.r1cs"
    
    content_classification:
      proof_size_kb: 1.5
      verification_time_ms: 600
      circuit_file: "circuits/content_class.r1cs"
    
    model_integrity:
      proof_size_kb: 3
      verification_time_ms: 1200
      circuit_file: "circuits/model_integrity.r1cs"
  
  circuit_compiler:
    backend: "bellman"
    curve: "bn256"
    trusted_setup: "circuits/trusted_setup.ptau"
  
  performance:
    max_concurrent_verifications: 10
    proof_cache_size: 1000
    verification_timeout_seconds: 30
```

### 4. Market-Based Pricing
```yaml
# config/market_config.yaml
market_pricing:
  auction_settings:
    default_duration_minutes: 15
    min_bid_increment: 0.01
    max_bid_amount: 1000.0
    anti_griefing_deposit_percent: 10
  
  price_bands:
    cpu_core_hour:
      min: 0.50
      max: 5.00
      default: 1.25
    memory_gb_hour:
      min: 0.10
      max: 1.00
      default: 0.25
    storage_gb_hour:
      min: 0.01
      max: 0.10
      default: 0.05
  
  sla_multipliers:
    bronze: 1.0
    silver: 2.5
    gold: 5.0
  
  privacy_multipliers:
    public: 1.0
    private: 1.5
    confidential: 2.0
    secret: 3.0
```

### 5. Job Scheduler (NSGA-II)
```yaml
# config/scheduler_config.yaml
job_scheduler:
  nsga_ii_settings:
    population_size: 100
    max_generations: 50
    crossover_rate: 0.9
    mutation_rate: 0.1
  
  optimization_objectives:
    cost_minimization:
      weight: 0.3
      enabled: true
    latency_minimization:
      weight: 0.4
      enabled: true
    resource_efficiency:
      weight: 0.3
      enabled: true
  
  resource_limits:
    max_cpu_cores_per_job: 64
    max_memory_gb_per_job: 512
    max_storage_gb_per_job: 10240
    max_job_duration_hours: 168
  
  queue_management:
    max_queued_jobs: 1000
    priority_levels: 5
    scheduling_interval_seconds: 30
```

### 6. Heterogeneous Quorum Manager
```yaml
# config/quorum_config.yaml
quorum_manager:
  sla_tiers:
    bronze:
      min_nodes: 1
      uptime_guarantee: 97.0
      max_latency_ms: 2500
      
    silver:
      min_nodes: 2
      uptime_guarantee: 99.0
      max_latency_ms: 1200
      diversity_requirements:
        asn_diversity: 2
        
    gold:
      min_nodes: 3
      uptime_guarantee: 99.9
      max_latency_ms: 400
      diversity_requirements:
        asn_diversity: 3
        tee_vendor_diversity: 2
        geographic_diversity: 2
        power_region_diversity: 2
  
  consensus:
    protocol: "raft"
    election_timeout_ms: 5000
    heartbeat_interval_ms: 1000
    log_compaction_threshold: 10000
  
  health_monitoring:
    node_health_check_interval_seconds: 30
    consensus_latency_threshold_ms: 1000
    partition_detection_enabled: true
```

### 7. Onion Routing Integration
```yaml
# config/onion_config.yaml
onion_routing:
  privacy_levels:
    public:
      hops: 0
      latency_overhead_ms: 0
      
    private:
      hops: 3
      latency_overhead_ms: 300
      
    confidential:
      hops: 5
      latency_overhead_ms: 850
      mixnet_enabled: true
      
    secret:
      hops: 7
      latency_overhead_ms: 1200
      cover_traffic_enabled: true
      cover_traffic_rate_kbps: 10
  
  network_settings:
    directory_servers: 8
    relay_selection_algorithm: "bandwidth_weighted"
    circuit_build_timeout_seconds: 30
    circuit_lifetime_minutes: 60
  
  performance:
    max_concurrent_circuits: 100
    bandwidth_limit_mbps: 1000
    connection_pool_size: 50
```

### 8. Bayesian Reputation System
```yaml
# config/reputation_config.yaml
reputation_system:
  beta_distribution:
    initial_alpha: 1
    initial_beta: 1
    confidence_threshold: 0.8
  
  reputation_tiers:
    diamond:
      min_score: 0.95
      min_interactions: 1000
      benefits:
        - "priority_scheduling"
        - "premium_pricing"
        - "reduced_deposits"
    
    platinum:
      min_score: 0.85
      min_interactions: 500
      benefits:
        - "fast_track_approval"
        - "reduced_deposits"
    
    gold:
      min_score: 0.75
      min_interactions: 200
      benefits:
        - "standard_access"
        - "normal_rates"
    
    silver:
      min_score: 0.60
      min_interactions: 50
      benefits:
        - "limited_access"
        - "higher_deposits"
    
    bronze:
      min_score: 0.0
      min_interactions: 0
      benefits:
        - "basic_access"
        - "maximum_deposits"
  
  time_decay:
    enabled: true
    decay_rate: 0.95
    decay_interval_days: 30
  
  analytics:
    trend_analysis_window_days: 7
    volatility_threshold: 0.1
    risk_assessment_enabled: true
```

### 9. VRF Neighbor Selection
```yaml
# config/vrf_config.yaml
vrf_neighbor_selection:
  network_topology:
    target_degree: 8
    max_degree: 16
    min_degree: 4
    reselection_interval_minutes: 60
  
  vrf_settings:
    key_size_bits: 256
    proof_verification_timeout_ms: 100
    entropy_requirement: 4.0
  
  security:
    eclipse_resistance_threshold: 0.8
    sybil_resistance_enabled: true
    partition_healing_enabled: true
  
  selection_criteria:
    geographic_diversity_weight: 0.3
    reputation_weight: 0.4
    latency_weight: 0.2
    bandwidth_weight: 0.1
  
  topology_analysis:
    expander_graph_validation: true
    spectral_gap_threshold: 0.3
    clustering_coefficient_target: 0.35
```

## Production Deployment

### Load Balancer Configuration (Nginx)
```nginx
# /etc/nginx/sites-available/fog-platform
upstream fog_backend {
    least_conn;
    server 127.0.0.1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 weight=1 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name fog.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://fog_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API endpoints
    location / {
        proxy_pass http://fog_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Request buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # Static files
    location /static/ {
        alias /opt/aivillage/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
}
```

### Systemd Service Configuration
```ini
# /etc/systemd/system/fog-platform.service
[Unit]
Description=Enhanced Fog Computing Platform
After=network.target
Requires=network.target

[Service]
Type=simple
User=aivillage
Group=aivillage
WorkingDirectory=/opt/aivillage
Environment=PYTHONPATH=/opt/aivillage
Environment=AIVILLAGE_ENV=production
Environment=JWT_SECRET_KEY_FILE=/opt/aivillage/.jwt_secret
ExecStart=/opt/aivillage/venv/bin/python infrastructure/gateway/enhanced_unified_api_gateway.py --config config/production.yaml
ExecReload=/bin/kill -USR1 $MAINPID
Restart=always
RestartSec=5
StartLimitInterval=0

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/aivillage/data /opt/aivillage/logs

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable fog-platform
sudo systemctl start fog-platform
sudo systemctl status fog-platform
```

### Database Configuration (PostgreSQL)
```sql
-- Create database and user
CREATE DATABASE fog_platform;
CREATE USER fog_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE fog_platform TO fog_user;

-- Configure connection pooling (config/database.yaml)
database:
  url: "postgresql://fog_user:secure_password@localhost:5432/fog_platform"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  echo: false
```

### Redis Configuration (Caching and Sessions)
```yaml
# config/redis.yaml
redis:
  url: "redis://localhost:6379/0"
  connection_pool:
    max_connections: 50
    retry_on_timeout: true
  
  caching:
    default_ttl_seconds: 3600
    key_prefix: "fog:"
  
  sessions:
    ttl_seconds: 86400
    key_prefix: "session:"
```

### Monitoring Configuration (Prometheus)
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fog-platform'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

## Monitoring and Maintenance

### Health Checks
```bash
# Basic health check
curl -f http://localhost:8000/health

# Component-specific health checks
curl -f http://localhost:8000/v1/fog/system/status
curl -f http://localhost:8000/v1/fog/tee/status
curl -f http://localhost:8000/v1/fog/proofs/status

# Performance metrics
curl -s http://localhost:8000/metrics | grep fog_
```

### Log Management
```bash
# View application logs
tail -f /opt/aivillage/logs/fog-platform.log

# View component logs
tail -f /opt/aivillage/logs/tee-runtime.log
tail -f /opt/aivillage/logs/proof-system.log
tail -f /opt/aivillage/logs/zk-predicates.log

# Log rotation (logrotate configuration)
# /etc/logrotate.d/fog-platform
/opt/aivillage/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
```

### Performance Monitoring
```bash
# System resource monitoring
htop
iotop
netstat -tulpn

# Application-specific monitoring
python scripts/monitor_fog_performance.py --duration 3600

# Database performance
psql -U fog_user -d fog_platform -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC LIMIT 10;"
```

### Backup and Recovery
```bash
# Database backup
pg_dump -U fog_user fog_platform > backup_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz config/ .env

# Full system backup
tar -czf full_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    /opt/aivillage/

# Automated backup script (cron)
# 0 2 * * * /opt/aivillage/scripts/backup.sh
```

### Security Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python dependencies
pip install --upgrade -r requirements.txt

# Security audit
pip audit
bandit -r infrastructure/ -f json -o security_report.json

# Check for vulnerabilities
safety check --json
```

### Performance Tuning
```bash
# Python performance tuning
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Memory optimization
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Network optimization
echo 'net.core.somaxconn=65536' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog=65536' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Service Fails to Start
**Symptoms**: Service exits immediately or fails to bind to port
**Diagnostic Commands**:
```bash
# Check service status
sudo systemctl status fog-platform

# View detailed logs
journalctl -u fog-platform -f

# Check port availability
sudo netstat -tulpn | grep :8000

# Validate configuration
python scripts/validate_config.py --config config/production.yaml
```

**Common Solutions**:
- Verify port 8000 is not in use by another service
- Check configuration file syntax and paths
- Ensure all dependencies are installed
- Verify environment variables are set correctly

#### 2. TEE Components Fail
**Symptoms**: TEE status shows hardware unavailable or enclaves fail to start
**Diagnostic Commands**:
```bash
# Check TEE hardware support
python scripts/detect_tee_hardware.py

# Verify kernel modules
lsmod | grep kvm
lsmod | grep sev

# Check SEV/TDX status
dmesg | grep -i sev
dmesg | grep -i tdx
```

**Common Solutions**:
- Ensure BIOS settings enable SEV-SNP/TDX
- Load required kernel modules
- Fall back to software isolation if hardware unavailable
- Update firmware and kernel drivers

#### 3. High Memory Usage
**Symptoms**: System becomes unresponsive, OOM killer activated
**Diagnostic Commands**:
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head -10

# Check application memory usage
python scripts/memory_profiler.py --pid $(pgrep -f fog-platform)

# Analyze memory leaks
valgrind --tool=memcheck --leak-check=full python infrastructure/gateway/enhanced_unified_api_gateway.py
```

**Common Solutions**:
- Reduce enclave memory limits in configuration
- Implement memory limits in systemd service
- Tune garbage collection settings
- Add more RAM or enable swap

#### 4. Database Connection Issues
**Symptoms**: API requests fail with database connection errors
**Diagnostic Commands**:
```bash
# Test database connection
psql -U fog_user -d fog_platform -c "SELECT version();"

# Check connection pool status
python scripts/check_db_pool.py

# Monitor database performance
sudo -u postgres psql -c "
  SELECT pid, query, state, query_start 
  FROM pg_stat_activity 
  WHERE datname='fog_platform';"
```

**Common Solutions**:
- Increase database connection pool size
- Optimize slow queries
- Add database indexes
- Restart PostgreSQL service

#### 5. WebSocket Connection Failures
**Symptoms**: Dashboard shows stale data, real-time updates not working
**Diagnostic Commands**:
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/fog

# Check nginx configuration (if using proxy)
nginx -t

# Monitor WebSocket connections
netstat -an | grep :8001
```

**Common Solutions**:
- Configure proxy_pass settings for WebSockets
- Check firewall rules for WebSocket ports
- Verify WebSocket upgrade headers
- Restart nginx/proxy service

### Performance Optimization

#### 1. Database Optimization
```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_proofs_timestamp ON proofs(timestamp);
CREATE INDEX idx_reputation_entity_id ON reputation(entity_id);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM jobs WHERE status = 'running';

-- Update table statistics
ANALYZE;
```

#### 2. Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    'tee_status': {'ttl': 60, 'key_pattern': 'tee:status:{node_id}'},
    'market_prices': {'ttl': 30, 'key_pattern': 'market:prices:{resource_type}'},
    'reputation_scores': {'ttl': 300, 'key_pattern': 'rep:score:{entity_id}'},
}
```

#### 3. Resource Limits
```yaml
# systemd service limits
[Service]
# Memory limit: 8GB
MemoryLimit=8G
# CPU quota: 4 cores
CPUQuota=400%
# File descriptor limit
LimitNOFILE=65536
# Process limit
LimitNPROC=4096
```

### Disaster Recovery

#### 1. Backup Strategy
```bash
# Automated backup script
#!/bin/bash
# /opt/aivillage/scripts/backup.sh

BACKUP_DIR="/backups/aivillage"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Database backup
pg_dump -U fog_user fog_platform > $BACKUP_DIR/$DATE/database.sql

# Configuration backup
tar -czf $BACKUP_DIR/$DATE/config.tar.gz config/ .env

# Data directory backup
tar -czf $BACKUP_DIR/$DATE/data.tar.gz data/

# Logs backup
tar -czf $BACKUP_DIR/$DATE/logs.tar.gz logs/

# Remove backups older than 30 days
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;
```

#### 2. Recovery Procedures
```bash
# Database recovery
sudo systemctl stop fog-platform
psql -U fog_user fog_platform < /backups/aivillage/20250828_120000/database.sql

# Configuration recovery
cd /opt/aivillage
tar -xzf /backups/aivillage/20250828_120000/config.tar.gz

# Data recovery
tar -xzf /backups/aivillage/20250828_120000/data.tar.gz

# Restart services
sudo systemctl start fog-platform
```

## Scaling and High Availability

### Horizontal Scaling

#### Load Balancer Setup (HAProxy)
```
# /etc/haproxy/haproxy.cfg
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend fog_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/fog-platform.pem
    redirect scheme https if !{ ssl_fc }
    default_backend fog_backend

backend fog_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server fog1 10.0.1.10:8000 check inter 30s
    server fog2 10.0.1.11:8000 check inter 30s
    server fog3 10.0.1.12:8000 check inter 30s

backend fog_websocket
    balance source
    option httpchk GET /ws/health
    
    server fog1_ws 10.0.1.10:8001 check inter 30s
    server fog2_ws 10.0.1.11:8001 check inter 30s
    server fog3_ws 10.0.1.12:8001 check inter 30s
```

#### Auto-scaling Configuration
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fog-platform-hpa
  namespace: aivillage
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fog-platform
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

### Database Clustering

#### PostgreSQL Streaming Replication
```bash
# Primary server configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
synchronous_commit = on
synchronous_standby_names = 'standby1'

# Standby server setup
pg_basebackup -h primary_server -D /var/lib/postgresql/13/main -U replicator -P -v -R -W -C -S standby1
```

#### Connection Pooling (PgBouncer)
```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
fog_platform = host=localhost port=5432 dbname=fog_platform

[pgbouncer]
pool_mode = session
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
max_client_conn = 200
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 5
server_lifetime = 3600
server_idle_timeout = 600
```

### Redis Clustering
```bash
# Redis Cluster setup
redis-cli --cluster create \
  127.0.0.1:7000 \
  127.0.0.1:7001 \
  127.0.0.1:7002 \
  127.0.0.1:7003 \
  127.0.0.1:7004 \
  127.0.0.1:7005 \
  --cluster-replicas 1
```

### Multi-Region Deployment

#### DNS Configuration (Route 53)
```yaml
# Route 53 health checks and failover
Type: A
Name: fog.yourdomain.com
Routing Policy: Failover
Primary:
  Value: 1.2.3.4 (us-east-1)
  Health Check: fog-us-east-health-check
Secondary:
  Value: 5.6.7.8 (us-west-2)
  Health Check: fog-us-west-health-check
```

#### Data Synchronization
```python
# Cross-region data sync
import asyncio
import aioredis
from typing import List

class CrossRegionSync:
    def __init__(self, redis_clusters: List[str]):
        self.clusters = redis_clusters
        
    async def sync_reputation_data(self):
        """Sync reputation scores across regions"""
        for cluster in self.clusters:
            redis = await aioredis.create_redis_pool(cluster)
            # Implement CRDT-based sync logic
            
    async def sync_market_data(self):
        """Sync market pricing data"""
        # Implement eventual consistency sync
        pass
```

---

This comprehensive deployment guide provides all necessary information for deploying and maintaining the Enhanced Fog Computing Platform in both development and production environments. For additional support, consult the API documentation and system logs for detailed troubleshooting information.
# Enhanced Fog Computing Dashboard User Guide

## Overview

The Enhanced Fog Computing Dashboard provides comprehensive monitoring and management capabilities for AIVillage's privacy-first fog cloud platform. This guide covers all features of the integrated admin interface with detailed instructions for using the fog computing components.

## Accessing the Dashboard

### Prerequisites
- Enhanced Fog Computing Platform running (port 8000)
- Valid authentication credentials
- Modern web browser with JavaScript enabled

### Access Methods
1. **Direct URL**: `http://localhost:8000/admin_interface.html`
2. **Production URL**: `https://your-domain.com/admin_interface.html`
3. **Via API Root**: Navigate to system root and click "Admin Interface" link

## Dashboard Overview

### Main Interface Layout
```
┌─────────────────────────────────────────────────────────────────┐
│                  Enhanced Fog Computing Platform                │
├─────────────────────────────────────────────────────────────────┤
│  [System Monitor] [Agent Forge] [P2P Network] [Fog Computing]   │
├─────────────────────────────────────────────────────────────────┤
│                        Main Content Area                        │
│                     (Tab-specific content)                      │
│                                                                 │
│                        Status Indicators                        │
│                        Control Panels                          │
│                        Real-time Metrics                       │
└─────────────────────────────────────────────────────────────────┘
```

### Tab Navigation
- **System Monitor**: Overall system health and performance metrics
- **Agent Forge**: AI model training and development pipeline
- **P2P Network**: Peer-to-peer networking and BitChat status
- **Fog Computing**: Complete 8-component fog platform management ⭐

## System Monitor Tab

### System Health Overview
**Location**: System Monitor → Health Status Panel

**Features**:
- **Overall System Health**: Green/Yellow/Red status indicator
- **Uptime Tracking**: System uptime with automatic refresh
- **Component Status Grid**: 9x3 grid showing all fog components
- **Performance Metrics**: CPU, memory, network utilization
- **Error Rate Monitoring**: Real-time error rate tracking

**Key Indicators**:
```
System Health: ● HEALTHY
Uptime: 2d 14h 32m
Components: 9/9 Operational
CPU Usage: 45.2% | Memory: 67.5% | Network: 12.3 MB/s
Error Rate: 0.02% (target: <0.1%)
```

### Performance Dashboard
**Location**: System Monitor → Performance Metrics

**Real-time Charts**:
- **Request Rate**: API requests per second over time
- **Response Times**: P95 latency trends
- **Resource Utilization**: CPU, memory, disk usage
- **Network Throughput**: Inbound/outbound traffic

**Interactive Features**:
- Hover over charts for detailed metrics
- Click time periods for different granularities (1h, 6h, 24h, 7d)
- Export performance data as CSV/JSON

## Agent Forge Tab

### 7-Phase AI Pipeline
**Location**: Agent Forge → Pipeline Management

**Phase Overview**:
1. **Data Collection**: Dataset gathering and preparation
2. **Preprocessing**: Data cleaning and augmentation
3. **Model Architecture**: Neural network design
4. **Training**: Model training with progress tracking
5. **Evaluation**: Performance assessment
6. **Optimization**: Model compression and optimization
7. **Deployment**: Production deployment

**Training Controls**:
- **Start Training**: Begin new training session
- **Pause/Resume**: Control training execution
- **Stop Training**: Terminate current session
- **View Logs**: Real-time training logs
- **Export Model**: Download trained models

### Model Management
**Location**: Agent Forge → Model Repository

**Features**:
- **Model Library**: Browse trained models
- **Version Control**: Track model versions
- **Performance Metrics**: Accuracy, loss, training time
- **Model Comparison**: Side-by-side comparisons
- **Deployment Status**: Production deployment tracking

## P2P Network Tab

### Network Topology
**Location**: P2P Network → Topology Viewer

**Features**:
- **Interactive Network Graph**: Visual representation of peer connections
- **Node Information**: Click nodes for detailed information
- **Connection Quality**: Color-coded connection strength
- **Geographic Distribution**: World map showing peer locations
- **Network Statistics**: Connection count, latency, bandwidth

### BitChat Integration
**Location**: P2P Network → Mobile Bridge

**Mobile Features**:
- **Connected Devices**: List of mobile devices
- **Battery Optimization**: Power management status
- **Data Synchronization**: Sync status and progress
- **Offline Capabilities**: Local storage and queue management
- **Security Status**: Encryption and authentication status

## Fog Computing Tab ⭐

The **Fog Computing Tab** is the centerpiece of the enhanced dashboard, providing comprehensive management of all 8 fog computing components.

### Component Status Matrix
**Location**: Fog Computing → Component Overview

**9x3 Status Grid**:
```
┌─────────────────┬─────────┬─────────┬──────────────┐
│   Component     │ Status  │  Load   │   Health     │
├─────────────────┼─────────┼─────────┼──────────────┤
│ TEE Runtime     │ ● READY │  45%    │ ● EXCELLENT  │
│ Crypto Proofs   │ ● READY │  23%    │ ● EXCELLENT  │
│ ZK Predicates   │ ● READY │  67%    │ ● GOOD       │
│ Market Engine   │ ● READY │  34%    │ ● EXCELLENT  │
│ Job Scheduler   │ ● READY │  78%    │ ● GOOD       │
│ Quorum Manager  │ ● READY │  12%    │ ● EXCELLENT  │
│ Onion Router    │ ● READY │  56%    │ ● GOOD       │
│ Reputation Sys  │ ● READY │  23%    │ ● EXCELLENT  │
│ VRF Topology    │ ● READY │  34%    │ ● EXCELLENT  │
└─────────────────┴─────────┴─────────┴──────────────┘
```

**Status Indicators**:
- ● **GREEN (READY)**: Component operational and healthy
- ◐ **YELLOW (DEGRADED)**: Component operational but with issues
- ● **RED (FAILED)**: Component non-operational
- ○ **GRAY (UNKNOWN)**: Component status unknown

### 1. TEE Runtime Management
**Location**: Fog Computing → TEE Runtime

**Capabilities Overview**:
- **Hardware Detection**: Automatic TEE hardware discovery
- **Enclave Management**: Create, monitor, and destroy secure enclaves
- **Attestation Services**: Generate and verify cryptographic attestations
- **Performance Monitoring**: Real-time enclave performance metrics

**Interactive Controls**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                         TEE Runtime Control                     │
├─────────────────────────────────────────────────────────────────┤
│  Available TEE Hardware:                                        │
│    ☑ AMD SEV-SNP (16GB, 16 enclaves max)                      │
│    ☐ Intel TDX (Not detected)                                  │
│    ☑ Software Isolation (Unlimited)                            │
│                                                                 │
│  Active Enclaves: 3/16                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ enclave_ml_training_1  │ RUNNING │ 2GB │ 85% CPU        │   │
│  │ enclave_data_proc_2    │ RUNNING │ 1GB │ 45% CPU        │   │
│  │ enclave_crypto_svc_3   │ PAUSED  │ 512MB │ 0% CPU       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Create Enclave] [Generate Attestation] [View Metrics]        │
└─────────────────────────────────────────────────────────────────┘
```

**Create Enclave Dialog**:
- **Name**: Human-readable enclave name
- **Memory**: Memory allocation (64MB - 16GB)
- **CPU Cores**: Number of CPU cores (1-32)
- **Security Policy**: Strict/Permissive measurement policy
- **Network Access**: Allow/deny network connections
- **Code Hash**: SHA-256 hash of code to load

**Attestation Management**:
- **Generate Report**: Create new attestation report
- **Verify Report**: Validate existing attestation
- **Export Certificate**: Download certificate chain
- **View Measurements**: Display PCR/MRENCLAVE values

### 2. Cryptographic Proof System
**Location**: Fog Computing → Crypto Proofs

**Proof Types Available**:
- **Proof-of-Execution (PoE)**: Task completion verification
- **Proof-of-Audit (PoA)**: AI auditor consensus validation
- **Proof-of-SLA (PoSLA)**: Performance compliance verification

**Interactive Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                    Cryptographic Proof System                  │
├─────────────────────────────────────────────────────────────────┤
│  Active Proofs: 24 │ Verified: 23 │ Pending: 1 │ Failed: 0    │
│                                                                 │
│  Recent Proofs:                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ prf_ml_training_001   │ PoE │ VERIFIED │ 10:30:25        │   │
│  │ prf_data_audit_002    │ PoA │ VERIFIED │ 10:29:45        │   │
│  │ prf_sla_check_003     │PoSLA│ PENDING  │ 10:31:10        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Blockchain Anchoring: Ethereum │ Last Block: 18,234,567       │
│  Anchored Proofs: 156 │ Pending Confirmations: 3               │
│                                                                 │
│  [Generate Proof] [Verify Proof] [Anchor to Blockchain]        │
└─────────────────────────────────────────────────────────────────┘
```

**Generate Proof Workflow**:
1. Select proof type (PoE/PoA/PoSLA)
2. Choose enclave/task to prove
3. Configure proof parameters
4. Generate proof with witness data
5. Optional blockchain anchoring

**Proof Verification**:
- **Upload Proof**: Drag-and-drop proof files
- **Verify Signature**: Cryptographic signature validation
- **Check Merkle Path**: Merkle tree verification
- **Blockchain Confirmation**: On-chain verification status

### 3. Zero-Knowledge Predicates
**Location**: Fog Computing → ZK Predicates

**Available Predicates**:
- **Network Policy Compliance**: Verify access without revealing topology
- **Content Classification**: Validate content without accessing data
- **Model Integrity**: Verify AI models without exposing weights
- **Regulatory Compliance**: Privacy-preserving audit compliance

**ZK Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                   Zero-Knowledge Predicates                     │
├─────────────────────────────────────────────────────────────────┤
│  Predicate Library:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Network Policy      │ 2KB proof │ ~800ms verify │ READY │   │
│  │ Content Class.      │1.5KB proof│ ~600ms verify │ READY │   │
│  │ Model Integrity     │ 3KB proof │~1200ms verify │ READY │   │
│  │ GDPR Compliance     │2.5KB proof│ ~900ms verify │ READY │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Recent Verifications: 45 │ Success Rate: 98.9%                │
│  Average Verification Time: 750ms                               │
│                                                                 │
│  [Verify Predicate] [Create Custom] [View Proofs]              │
└─────────────────────────────────────────────────────────────────┘
```

**Custom Predicate Builder**:
- **Predicate Logic**: Visual circuit builder
- **Public Inputs**: Define public parameters
- **Private Inputs**: Specify private witness data
- **Constraints**: Add verification constraints
- **Testing**: Test predicate with sample data

### 4. Market-Based Pricing
**Location**: Fog Computing → Market Engine

**Market Overview Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                     Market-Based Pricing Engine                 │
├─────────────────────────────────────────────────────────────────┤
│  Market Conditions: MODERATE │ Avg Price: $1.25/core-hour      │
│  Provider Availability: 234 │ Available Capacity: 1,280 cores  │
│                                                                 │
│  Active Auctions: 8                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ auc_ml_train_001  │ 8 cores │ 4h │ $7.50 │ 5 bidders    │   │
│  │ auc_data_proc_002 │16 cores │ 2h │ $15.20│ 3 bidders    │   │
│  │ auc_inference_003 │ 4 cores │ 1h │ $3.80 │ 7 bidders    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Price Trends (24h): CPU +2.3% │ Memory -1.1% │ Storage +5.7%  │
│  Privacy Premium: 1.8x │ Gold SLA Premium: 2.5x                │
│                                                                 │
│  [Request Quote] [Create Auction] [View Market Data]           │
└─────────────────────────────────────────────────────────────────┘
```

**Quote Request Form**:
- **Resource Specification**: CPU, memory, storage requirements
- **Duration**: Time period for resource allocation
- **Privacy Level**: Public/Private/Confidential/Secret
- **SLA Tier**: Bronze/Silver/Gold service level
- **Budget Limit**: Maximum price per hour
- **Preferred Regions**: Geographic preferences

**Auction Management**:
- **Create Reverse Auction**: Providers bid lower prices
- **Monitor Bidding**: Real-time bid updates
- **Anti-Griefing Deposits**: Automatic deposit management
- **Winner Selection**: Second-price auction mechanics

### 5. Job Scheduler (NSGA-II)
**Location**: Fog Computing → Job Scheduler

**Scheduler Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                      NSGA-II Job Scheduler                      │
├─────────────────────────────────────────────────────────────────┤
│  Queue Status: 24 Total │ 8 Queued │ 12 Running │ 4 Complete   │
│  Average Wait Time: 15 minutes │ Utilization: 72.3%            │
│                                                                 │
│  Job Queue:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ job_ml_train_001 │ RUNNING │ 35% │ 2h 15m rem │ Gold SLA │   │
│  │ job_data_proc_002│ QUEUED  │  0% │ Pos: #3    │Silver SLA│   │
│  │ job_inference_003│ RUNNING │ 90% │ 5m rem     │Bronze SLA│   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Multi-Objective Optimization:                                  │
│  ● Cost Minimization: Weight 30% │ Current Avg: $2.45/h       │
│  ● Latency Minimization: Weight 40% │ Current P95: 1.2s       │
│  ● Resource Efficiency: Weight 30% │ Current Util: 72%        │
│                                                                 │
│  [Submit Job] [View Queue] [Optimization Settings]             │
└─────────────────────────────────────────────────────────────────┘
```

**Job Submission Form**:
- **Job Specification**: Docker image, resource requirements
- **Environment Variables**: Configuration parameters
- **Input/Output**: Data sources and destinations
- **Privacy Requirements**: TEE, onion routing preferences
- **SLA Requirements**: Performance and availability targets
- **Optimization Preferences**: Cost vs performance trade-offs

**Real-time Monitoring**:
- **Job Progress**: Visual progress bars with ETA
- **Resource Usage**: CPU, memory, GPU utilization
- **Cost Tracking**: Real-time cost accumulation
- **Log Streaming**: Live job execution logs
- **Performance Metrics**: Throughput, latency, error rates

### 6. Heterogeneous Quorum Manager
**Location**: Fog Computing → Quorum Manager

**Diversity Compliance Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                   Heterogeneous Quorum Manager                  │
├─────────────────────────────────────────────────────────────────┤
│  Gold SLA Compliance: ✅ COMPLIANT                              │
│                                                                 │
│  Infrastructure Diversity:                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ASN Diversity      │ Req: 3 │ Current: 5 │ ✅ PASS      │   │
│  │ TEE Vendor Div.    │ Req: 2 │ Current: 3 │ ✅ PASS      │   │
│  │ Geographic Div.    │ Req: 2 │ Current: 4 │ ✅ PASS      │   │
│  │ Power Region Div.  │ Req: 2 │ Current: 3 │ ✅ PASS      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Active Quorum: 12 nodes │ Consensus Latency: 250ms            │
│  Fault Tolerance: 4 node failures │ Byzantine Tolerance: 3      │
│                                                                 │
│  Regional Distribution:                                          │
│  US-East: 3 nodes │ US-West: 3 nodes │ EU-Central: 3 nodes     │
│  AP-Southeast: 3 nodes                                          │
│                                                                 │
│  [Validate Diversity] [View Topology] [SLA Calculator]         │
└─────────────────────────────────────────────────────────────────┘
```

**SLA Tier Validation**:
- **Bronze Tier**: Single instance, basic availability
- **Silver Tier**: Primary + canary, enhanced monitoring
- **Gold Tier**: 3+ instances with full diversity requirements
- **Custom Tiers**: Define specific diversity requirements

**Topology Visualization**:
- **Network Map**: Geographic distribution of nodes
- **Diversity Matrix**: Visual representation of diversity compliance
- **Failure Scenarios**: Simulate various failure modes
- **Recommendation Engine**: Optimal node placement suggestions

### 7. Onion Routing Integration
**Location**: Fog Computing → Onion Router

**Privacy Circuit Management**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                    Onion Routing Integration                    │
├─────────────────────────────────────────────────────────────────┤
│  Privacy Levels Available:                                      │
│  ● PUBLIC: Direct routing (no overhead)                         │
│  ● PRIVATE: 3-hop onion routing (~300ms latency)               │
│  ● CONFIDENTIAL: 5-hop + mixnet (~850ms latency)               │
│  ● SECRET: Full anonymity + cover traffic (~1200ms latency)     │
│                                                                 │
│  Active Circuits: 7                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ circ_ml_001 │CONFIDENTIAL│ 5 hops│ 125MB transferred      │   │
│  │ circ_data_002│ PRIVATE   │ 3 hops│ 45MB transferred       │   │
│  │ circ_comm_003│ SECRET    │ 7 hops│ 8MB + cover traffic    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Network Stats: 1,247 relay nodes │ 89 exit nodes              │
│  Directory Services: 8 │ Hidden Services: 12 active            │
│                                                                 │
│  [Create Circuit] [Monitor Traffic] [Hidden Services]          │
└─────────────────────────────────────────────────────────────────┘
```

**Circuit Creation Wizard**:
- **Privacy Level**: Select from 4 privacy tiers
- **Bandwidth Requirements**: Low/Medium/High bandwidth
- **Geographic Constraints**: Avoid/prefer specific regions
- **Circuit Lifetime**: Duration before automatic renewal
- **Cover Traffic**: Enable additional anonymity protection

**Traffic Analysis Dashboard**:
- **Circuit Performance**: Latency, throughput, reliability
- **Data Flow Visualization**: Real-time traffic flow
- **Anonymity Metrics**: Entropy and unlinkability scores
- **Security Alerts**: Potential traffic analysis attacks

### 8. Bayesian Reputation System
**Location**: Fog Computing → Reputation System

**Trust Management Dashboard**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                   Bayesian Reputation System                    │
├─────────────────────────────────────────────────────────────────┤
│  Entity Statistics:                                             │
│  Total Entities: 602 │ Tracked Interactions: 15,678            │
│                                                                 │
│  Reputation Tiers:                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Diamond (≥0.95) │  23 entities │ Priority scheduling     │   │
│  │ Platinum(≥0.85) │  89 entities │ Reduced deposits        │   │
│  │ Gold    (≥0.75) │ 245 entities │ Standard access         │   │
│  │ Silver  (≥0.60) │ 178 entities │ Higher deposits         │   │
│  │ Bronze  (≥0.00) │  67 entities │ Maximum deposits        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Trust Analytics (24h):                                         │
│  Average Reputation: 0.78 │ Median: 0.82 │ Volatility: 0.05   │
│  Improving: 145 │ Declining: 67 │ Stable: 390                  │
│                                                                 │
│  [Entity Search] [Trust Analytics] [Reputation History]        │
└─────────────────────────────────────────────────────────────────┘
```

**Entity Reputation Viewer**:
- **Search Interface**: Find specific providers/users
- **Reputation Score**: Current score with confidence intervals
- **Interaction History**: Detailed transaction history
- **Trust Trend**: Reputation changes over time
- **Beta Parameters**: Statistical model parameters (α, β)

**Trust Analytics**:
- **Reputation Distribution**: Histogram of reputation scores
- **Temporal Trends**: Reputation changes over time periods
- **Risk Assessment**: High/medium/low risk entity identification
- **Correlation Analysis**: Factors affecting reputation scores

### 9. VRF Neighbor Selection
**Location**: Fog Computing → VRF Topology

**Secure Topology Management**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                     VRF Neighbor Selection                      │
├─────────────────────────────────────────────────────────────────┤
│  Network Topology Health: ✅ EXCELLENT                         │
│  Eclipse Resistance: 89% │ Sybil Resistance: 92%              │
│                                                                 │
│  Network Statistics:                                            │
│  Total Nodes: 1,247 │ Avg Degree: 8.5 │ Diameter: 6          │
│  Clustering Coeff: 0.35 │ Connectivity: 95%                   │
│                                                                 │
│  Topology Properties:                                           │
│  ☑ Expander Graph │ Expansion Ratio: 0.76                     │
│  ☑ Spectral Gap: 0.42 │ Random Walk Mixing: Fast              │
│                                                                 │
│  Recent Selections: 234 │ Verification Success: 100%          │
│  Last Reselection: 15 minutes ago │ Next: 45 minutes          │
│                                                                 │
│  Node View: [Your Node: node_abc123] │ Neighbors: 8/8          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ node_def456 │ Dist: 0.23 │ Quality: 95% │ Rep: 0.87      │   │
│  │ node_ghi789 │ Dist: 0.31 │ Quality: 89% │ Rep: 0.91      │   │
│  │ node_jkl012 │ Dist: 0.28 │ Quality: 93% │ Rep: 0.85      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Trigger Reselection] [View Topology] [Security Analysis]     │
└─────────────────────────────────────────────────────────────────┘
```

**Topology Visualization**:
- **Interactive Network Graph**: Zoom and pan network visualization
- **Node Details**: Click nodes to view detailed information
- **Connection Analysis**: Edge weights and quality indicators
- **Geographic Overlay**: Show nodes on world map
- **Selection History**: Track neighbor selection changes

**Security Analysis Tools**:
- **Eclipse Attack Simulation**: Test resistance to eclipse attacks
- **Sybil Detection**: Identify potential sybil nodes
- **Partition Resistance**: Analyze network partition tolerance
- **Entropy Calculation**: Measure randomness of selections

## Real-Time Features

### WebSocket Integration
All dashboard components feature real-time updates via WebSocket connections:

**Auto-Refresh Capabilities**:
- **System Status**: Updates every 5 seconds
- **Job Progress**: Updates every 2 seconds
- **Market Prices**: Updates every 10 seconds
- **Network Topology**: Updates every 30 seconds
- **Performance Metrics**: Updates every 15 seconds

**Event Notifications**:
- **System Alerts**: Critical system events
- **Job Completion**: Training and processing completion
- **Auction Updates**: Bid changes and auction results
- **Security Events**: Attestation failures, circuit drops
- **Network Changes**: Topology changes, node failures

### Interactive Testing Interface

Each fog component includes an interactive testing panel for API endpoint testing:

**Test Interface Features**:
```html
┌─────────────────────────────────────────────────────────────────┐
│                    Interactive API Testing                      │
├─────────────────────────────────────────────────────────────────┤
│  Select Endpoint: [TEE Create Enclave ▼]                       │
│                                                                 │
│  Parameters:                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Name: [test_enclave_001                             ]     │   │
│  │ Memory (MB): [2048                                  ]     │   │
│  │ CPU Cores: [4                                       ]     │   │
│  │ Security Policy: [Strict ▼]                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Send Request] [Clear] [Load Example]                         │
│                                                                 │
│  Response:                                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ {                                                         │   │
│  │   "success": true,                                        │   │
│  │   "data": {                                               │   │
│  │     "enclave_id": "enclave_test_001_xyz",                │   │
│  │     "state": "created",                                   │   │
│  │     "tee_type": "amd_sev_snp"                            │   │
│  │   }                                                       │   │
│  │ }                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Testing Features**:
- **Endpoint Selection**: Dropdown with all available endpoints
- **Parameter Forms**: Dynamic forms based on API specification
- **Example Data**: Load example requests for quick testing
- **Response Formatting**: Syntax-highlighted JSON responses
- **Request History**: Previous test requests and responses
- **Performance Timing**: Response time measurement

## Monitoring and Alerts

### Alert System
**Location**: All tabs → Alert Panel (top-right)

**Alert Types**:
- 🔴 **Critical**: System failures, security breaches
- 🟡 **Warning**: Performance degradation, resource limits
- 🔵 **Info**: Routine operations, status changes
- 🟢 **Success**: Successful operations, achievements

**Alert Management**:
- **Real-time Notifications**: Browser notifications for critical alerts
- **Alert History**: View past 24 hours of alerts
- **Alert Filtering**: Filter by severity, component, time range
- **Acknowledgment**: Mark alerts as acknowledged
- **Auto-Resolution**: Alerts automatically resolve when conditions improve

### Performance Monitoring
**Location**: System Monitor → Advanced Metrics

**Comprehensive Metrics**:
- **System Performance**: CPU, memory, disk, network utilization
- **Application Metrics**: Request rates, response times, error rates
- **Fog Component Metrics**: Component-specific performance indicators
- **Business Metrics**: Jobs completed, revenue generated, SLA compliance
- **Security Metrics**: Authentication failures, suspicious activities

**Visualization Options**:
- **Time Series Charts**: Line charts with multiple time granularities
- **Heat Maps**: Performance heat maps by component and time
- **Scatter Plots**: Correlation analysis between metrics
- **Distribution Charts**: Histograms and percentile distributions
- **Geographic Maps**: Performance by geographic region

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Dashboard Not Loading
**Symptoms**: White screen or loading indefinitely
**Solutions**:
- Check if fog system is running on port 8000
- Verify JavaScript is enabled in browser
- Clear browser cache and cookies
- Check browser console for JavaScript errors

#### 2. Component Status Shows FAILED
**Symptoms**: Red status indicators in component matrix
**Solutions**:
- Check component logs via API endpoints
- Restart specific components via system controls
- Verify resource availability (memory, CPU)
- Check network connectivity

#### 3. Real-time Updates Not Working
**Symptoms**: Static data, no auto-refresh
**Solutions**:
- Check WebSocket connection status
- Verify firewall allows WebSocket connections
- Refresh browser page to reestablish connection
- Check network connectivity and proxies

#### 4. API Tests Failing
**Symptoms**: Error responses in testing interface
**Solutions**:
- Verify authentication credentials
- Check API endpoint availability
- Validate request parameters
- Review API rate limiting

#### 5. Performance Charts Not Updating
**Symptoms**: Stale or missing chart data
**Solutions**:
- Verify metrics collection is active
- Check time range settings
- Refresh dashboard components
- Verify system resources for metrics storage

### Logging and Diagnostics

#### Browser Console
Access browser developer tools (F12) for detailed logging:
- **Console Tab**: JavaScript errors and debug messages
- **Network Tab**: API request/response analysis
- **Application Tab**: WebSocket connection status
- **Performance Tab**: Dashboard performance analysis

#### System Logs
Access system logs via API endpoints:
- **System Logs**: `/v1/fog/system/logs`
- **Component Logs**: `/v1/fog/{component}/logs`
- **Audit Logs**: `/v1/fog/system/audit`
- **Security Logs**: `/v1/fog/system/security`

## Best Practices

### Dashboard Usage Recommendations

#### 1. Monitoring Workflow
- **Start with System Monitor**: Get overall health overview
- **Check Component Status**: Verify all components operational
- **Review Performance Metrics**: Identify potential bottlenecks
- **Monitor Active Jobs**: Track running workloads
- **Set Up Alerts**: Configure notifications for critical events

#### 2. Security Best Practices
- **Regular Attestation**: Generate and verify TEE attestations
- **Monitor Reputation**: Track entity reputation scores
- **Review Security Logs**: Check for suspicious activities
- **Update Credentials**: Rotate authentication tokens regularly
- **Audit Access**: Review user access and permissions

#### 3. Performance Optimization
- **Monitor Resource Usage**: Track CPU, memory, storage utilization
- **Optimize Job Scheduling**: Balance cost vs performance
- **Review Market Conditions**: Adjust resource allocation based on pricing
- **Analyze Network Performance**: Monitor latency and throughput
- **Update Components**: Keep fog components up to date

#### 4. Cost Management
- **Track Spending**: Monitor resource costs and budgets
- **Optimize Auctions**: Use reverse auctions for better pricing
- **Review SLA Tiers**: Balance performance requirements vs costs
- **Monitor Market Trends**: Adjust resource timing based on price trends
- **Implement Budgets**: Set spending limits and alerts

### Advanced Usage Tips

#### 1. Custom Monitoring
- **Create Custom Dashboards**: Use API data for specific views
- **Set Up External Monitoring**: Integrate with Prometheus/Grafana
- **Export Metrics**: Regular data exports for analysis
- **Automated Reports**: Generate performance and cost reports

#### 2. Integration Development
- **API Integration**: Use REST APIs for custom applications
- **WebSocket Integration**: Real-time updates in custom apps
- **Webhook Configuration**: Event-driven integrations
- **SDK Usage**: Leverage provided SDKs for development

#### 3. Scaling Operations
- **Multi-Region Deployment**: Distribute components geographically
- **Load Balancing**: Implement load balancers for high availability
- **Auto-Scaling**: Configure automatic resource scaling
- **Disaster Recovery**: Implement backup and recovery procedures

---

This comprehensive guide covers all aspects of the Enhanced Fog Computing Dashboard. For additional support, consult the API documentation at `/docs` or the system logs for detailed troubleshooting information.
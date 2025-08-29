# Digital Twin Architecture - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the complete Digital Twin Architecture, including on-device Digital Twin Concierge, fog network meta-agent sharding, and distributed RAG coordination.

## 📋 Prerequisites

### System Requirements

**Local Device Requirements:**
- **Memory**: Minimum 4GB RAM (2GB available for AIVillage)
- **Storage**: 500MB for digital twin system + 1GB for data storage
- **CPU**: Dual-core processor or better
- **OS**: Python 3.10+ compatible (Windows, macOS, Linux)

**Mobile Device Requirements:**
- **Android**: API level 26+ (Android 8.0+), 2GB RAM minimum
- **iOS**: iOS 15.0+, 3GB RAM minimum
- **Permissions**: Location, Contacts, App Usage, Storage access

**Fog Network Requirements:**
- **Minimum Nodes**: 3 devices for reliable sharding
- **Network**: P2P mesh capability (BitChat/BetaNet)
- **Total Resources**: 8GB RAM, 10GB storage across fog network

### Dependencies

**Python Dependencies:**
```bash
pip install -r requirements.txt

# Core dependencies (automatically installed)
numpy>=1.21.0
sqlite3  # Built-in to Python
asyncio  # Built-in to Python
pydantic>=1.8.0
cryptography>=3.4.8
psutil>=5.8.0
```

**Mobile Dependencies:**
```bash
# Android (in android/app/build.gradle)
implementation 'androidx.core:core:1.7.0'
implementation 'androidx.lifecycle:lifecycle-extensions:2.2.0'
implementation 'org.json:json:20210307'

# iOS (in Podfile)
pod 'CoreData'
pod 'CoreLocation'
pod 'UserNotifications'
```

## 🚀 Core Deployment

### Step 1: Environment Setup

**1.1 Clone and Setup Repository:**
```bash
cd /path/to/AIVillage
python -m venv digital_twin_env
source digital_twin_env/bin/activate  # On Windows: digital_twin_env\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

**1.2 Configure Environment Variables:**
```bash
# Create .env file
cat > .env << 'EOF'
# Digital Twin Configuration
DIGITAL_TWIN_DATA_DIR=/path/to/twin_data
DIGITAL_TWIN_PRIVACY_MODE=balanced
DIGITAL_TWIN_RETENTION_HOURS=24

# P2P Network Configuration
P2P_BITCHAT_ENABLED=true
P2P_BETANET_ENABLED=true
P2P_NODE_ID=auto

# Fog Compute Configuration
FOG_COORDINATOR_ENABLED=true
FOG_MIN_BATTERY_LEVEL=20
FOG_THERMAL_THRESHOLD=50

# Security Configuration
ENCRYPTION_ENABLED=true
BIOMETRIC_AUTH_REQUIRED=false
AUTO_DELETE_SENSITIVE=true
EOF
```

**1.3 Verify System Compatibility:**
```bash
# Run compatibility check
python scripts/verify_digital_twin_compatibility.py

# Expected output:
# ✅ Python version: 3.11.x (compatible)
# ✅ Required packages: All installed
# ✅ SQLite support: Available
# ✅ Cryptography: Available
# ✅ Storage space: X.XGB available
# ✅ System ready for Digital Twin deployment
```

### Step 2: Core System Deployment

**2.1 Initialize Digital Twin Concierge:**
```python
# deploy_digital_twin.py
import asyncio
from pathlib import Path
from packages.edge.mobile.digital_twin_concierge import (
    DigitalTwinConcierge, 
    UserPreferences, 
    DataSource
)

async def deploy_digital_twin():
    # Configure user preferences
    preferences = UserPreferences(
        enabled_sources={
            DataSource.CONVERSATIONS,
            DataSource.LOCATION, 
            DataSource.APP_USAGE,
            DataSource.PURCHASES
        },
        max_data_retention_hours=24,
        privacy_mode="balanced",
        learning_enabled=True,
        surprise_threshold=0.4
    )
    
    # Initialize digital twin
    data_dir = Path("./digital_twin_data")
    concierge = DigitalTwinConcierge(data_dir, preferences)
    
    print("✅ Digital Twin Concierge deployed successfully")
    return concierge

# Run deployment
if __name__ == "__main__":
    asyncio.run(deploy_digital_twin())
```

**2.2 Deploy Meta-Agent Sharding Coordinator:**
```python
# deploy_meta_agents.py
import asyncio
from packages.agents.distributed.meta_agent_sharding_coordinator import (
    MetaAgentShardingCoordinator
)
from packages.edge.fog_compute.fog_coordinator import FogCoordinator
from packages.p2p.core.transport_manager import UnifiedTransportManager

async def deploy_meta_agents():
    # Initialize components
    fog_coordinator = FogCoordinator()
    transport_manager = UnifiedTransportManager()
    sharding_engine = ModelShardingEngine()  # From existing infrastructure
    
    # Get digital twin from previous step
    digital_twin = await get_digital_twin_instance()
    
    # Create coordinator
    coordinator = MetaAgentShardingCoordinator(
        fog_coordinator=fog_coordinator,
        transport_manager=transport_manager, 
        sharding_engine=sharding_engine,
        digital_twin=digital_twin
    )
    
    # Create deployment plan
    target_agents = [
        "digital_twin_concierge",
        "king_agent",
        "magi_agent", 
        "oracle_agent",
        "sage_agent"
    ]
    
    plan = await coordinator.create_deployment_plan(target_agents)
    print(f"📋 Deployment plan: {len(plan.local_agents)} local, {len(plan.fog_agents)} fog")
    
    # Execute deployment
    results = await coordinator.deploy_agents(plan)
    
    success_count = sum(1 for success in results.values() if success)
    print(f"🚀 Deployed {success_count}/{len(results)} agents successfully")
    
    return coordinator

if __name__ == "__main__":
    asyncio.run(deploy_meta_agents())
```

**2.3 Deploy Distributed RAG Coordinator:**
```python
# deploy_distributed_rag.py
import asyncio
from packages.rag.distributed.distributed_rag_coordinator import (
    DistributedRAGCoordinator
)
from packages.rag.core.hyper_rag import HyperRAG

async def deploy_distributed_rag():
    # Initialize HyperRAG (existing system)
    hyper_rag = HyperRAG()
    
    # Get coordinators from previous steps
    fog_coordinator = await get_fog_coordinator()
    transport_manager = await get_transport_manager()
    
    # Create distributed RAG
    distributed_rag = DistributedRAGCoordinator(
        hyper_rag=hyper_rag,
        fog_coordinator=fog_coordinator,
        transport_manager=transport_manager
    )
    
    # Initialize distributed system
    success = await distributed_rag.initialize_distributed_system()
    
    if success:
        print("✅ Distributed RAG system initialized successfully")
        print(f"📚 {len(distributed_rag.knowledge_shards)} knowledge shards created")
    else:
        print("❌ Failed to initialize distributed RAG system")
        
    return distributed_rag

if __name__ == "__main__":
    asyncio.run(deploy_distributed_rag())
```

### Step 3: System Integration

**3.1 Deploy Complete Integrated System:**
```python
# deploy_complete_system.py
import asyncio
from pathlib import Path

async def deploy_complete_digital_twin_system():
    """Deploy complete integrated Digital Twin Architecture"""
    
    print("🚀 Starting Digital Twin Architecture deployment...")
    
    # 1. Deploy Digital Twin Concierge
    print("\n📱 Step 1: Deploying Digital Twin Concierge...")
    concierge = await deploy_digital_twin()
    
    # 2. Deploy Meta-Agent Coordinator  
    print("\n🧠 Step 2: Deploying Meta-Agent Sharding...")
    coordinator = await deploy_meta_agents()
    
    # 3. Deploy Distributed RAG
    print("\n📚 Step 3: Deploying Distributed RAG...")
    distributed_rag = await deploy_distributed_rag()
    
    # 4. Register Mini-RAG with distributed system
    print("\n🔗 Step 4: Integrating systems...")
    device_id = "local_device_001"
    await distributed_rag.register_mini_rag(device_id, concierge.mini_rag)
    
    # 5. Verify deployment
    print("\n✅ Step 5: Verifying deployment...")
    status = await coordinator.get_deployment_status()
    privacy_report = coordinator.get_privacy_report()
    
    print(f"   Local agents: {status['local_agents']}")
    print(f"   Fog agents: {status['fog_agents']}")  
    print(f"   Total shards: {status['total_shards']}")
    print(f"   Privacy status: {privacy_report['digital_twin_privacy']['status']}")
    
    print("\n🎉 Digital Twin Architecture deployed successfully!")
    
    return {
        "concierge": concierge,
        "coordinator": coordinator, 
        "distributed_rag": distributed_rag
    }

if __name__ == "__main__":
    system = asyncio.run(deploy_complete_digital_twin_system())
```

## 📱 Mobile Client Deployment

### Android Deployment

**4.1 Android Studio Setup:**
```bash
# 1. Open Android project
cd clients/mobile/android
# Open in Android Studio or use command line

# 2. Configure build.gradle (app level)
android {
    compileSdk 33
    
    defaultConfig {
        applicationId "com.aivillage.digitaltwin"
        minSdk 26
        targetSdk 33
        versionCode 1
        versionName "1.0"
    }
    
    buildFeatures {
        dataBinding true
    }
}

dependencies {
    implementation 'androidx.core:core:1.9.0'
    implementation 'androidx.lifecycle:lifecycle-extensions:2.2.0'
    implementation 'androidx.work:work-runtime:2.8.1'
    implementation 'org.json:json:20220924'
}
```

**4.2 Configure Android Permissions:**
```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Required permissions -->
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
    <uses-permission android:name="android.permission.READ_CALL_LOG" />
    <uses-permission android:name="android.permission.READ_SMS" />
    <uses-permission android:name="android.permission.PACKAGE_USAGE_STATS" />
    <uses-permission android:name="android.permission.BLUETOOTH" />
    <uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
    
    <!-- Optional permissions -->
    <uses-permission android:name="android.permission.READ_CONTACTS" />
    <uses-permission android:name="android.permission.READ_CALENDAR" />
    
    <application
        android:name=".DigitalTwinApplication"
        android:label="AIVillage Digital Twin"
        android:theme="@style/AppTheme">
        
        <!-- Digital Twin Service -->
        <service android:name=".DigitalTwinService"
                 android:exported="false" />
        
        <!-- Main Activity -->
        <activity android:name=".MainActivity"
                  android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

**4.3 Build and Deploy Android App:**
```bash
# Build debug version
./gradlew assembleDebug

# Install on device
adb install app/build/outputs/apk/debug/app-debug.apk

# Or build release version
./gradlew assembleRelease

# Verify installation
adb shell am start -n com.aivillage.digitaltwin/.MainActivity
```

### iOS Deployment

**4.4 Xcode Project Setup:**
```bash
# 1. Open iOS project
cd clients/mobile/ios
open DigitalTwin.xcodeproj

# 2. Configure Info.plist permissions
```

```xml
<!-- ios/DigitalTwin/Info.plist -->
<dict>
    <!-- Privacy permissions -->
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>Digital Twin uses location to learn your movement patterns locally on your device</string>
    
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>Digital Twin uses location to learn your movement patterns locally on your device</string>
    
    <key>NSContactsUsageDescription</key>
    <string>Digital Twin analyzes contact patterns locally to improve predictions</string>
    
    <key>NSCalendarsUsageDescription</key>
    <string>Digital Twin learns from your schedule to provide better assistance</string>
    
    <key>NSLocalNetworkUsageDescription</key>
    <string>Digital Twin uses local network for P2P mesh communication</string>
    
    <!-- App capabilities -->
    <key>UIBackgroundModes</key>
    <array>
        <string>background-processing</string>
        <string>location</string>
    </array>
</dict>
```

**4.5 Build and Deploy iOS App:**
```bash
# Build for simulator
xcodebuild -project DigitalTwin.xcodeproj \
           -scheme DigitalTwin \
           -destination 'platform=iOS Simulator,name=iPhone 14' \
           build

# Build for device (requires developer certificate)
xcodebuild -project DigitalTwin.xcodeproj \
           -scheme DigitalTwin \
           -destination 'platform=iOS,id=YOUR_DEVICE_ID' \
           build

# Or use Xcode IDE for easier deployment
```

## 🌐 Fog Network Deployment

### Step 5: Multi-Device Fog Network Setup

**5.1 Fog Node Configuration:**
```python
# configure_fog_node.py
import asyncio
from packages.edge.fog_compute.fog_coordinator import FogCoordinator, ComputeCapacity

async def configure_fog_node(node_id: str, node_type: str = "standard"):
    """Configure individual fog node for meta-agent hosting"""
    
    # Assess local device capabilities  
    local_capacity = ComputeCapacity(
        cpu_cores=4,  # Detected automatically in production
        cpu_utilization=0.2,
        memory_mb=8000,
        memory_used_mb=2000,
        gpu_available=False,  # Detected automatically
        gpu_memory_mb=0,
        battery_powered=True,
        battery_percent=75,
        is_charging=False,
        thermal_state="normal"
    )
    
    # Initialize fog coordinator
    fog_coordinator = FogCoordinator()
    success = await fog_coordinator.register_fog_node(
        node_id=node_id,
        capacity=local_capacity,
        node_type=node_type
    )
    
    if success:
        print(f"✅ Fog node {node_id} registered successfully")
        print(f"   Capacity: {local_capacity.available_memory_mb}MB RAM, {local_capacity.cpu_cores} cores")
    else:
        print(f"❌ Failed to register fog node {node_id}")
    
    return fog_coordinator

# Deploy fog node
if __name__ == "__main__":
    import socket
    node_id = f"fog_node_{socket.gethostname()}"
    asyncio.run(configure_fog_node(node_id))
```

**5.2 Multi-Node Deployment Script:**
```bash
#!/bin/bash
# deploy_fog_network.sh

echo "🌐 Deploying AIVillage Fog Network..."

# Configuration
FOG_NODES=("fog-node-1" "fog-node-2" "fog-node-3")
PYTHON_ENV="digital_twin_env"

# Deploy to each fog node
for node in "${FOG_NODES[@]}"; do
    echo "📡 Deploying to $node..."
    
    # SSH to node and run deployment
    ssh $node << 'EOF'
        cd /path/to/AIVillage
        source digital_twin_env/bin/activate
        
        # Configure this node
        python configure_fog_node.py
        
        # Start fog services
        python -m packages.edge.fog_compute.fog_service --node-id=$(hostname)
        
        echo "✅ Fog node $(hostname) deployed"
EOF
done

echo "🎉 Fog network deployment complete!"
```

## 🔧 Configuration

### Step 6: System Configuration

**6.1 Privacy Configuration:**
```python
# configure_privacy.py
from packages.edge.mobile.digital_twin_concierge import UserPreferences, DataSource

def configure_privacy_settings():
    """Configure privacy settings for different user profiles"""
    
    # Minimal privacy (least data collection)
    minimal_privacy = UserPreferences(
        enabled_sources={DataSource.APP_USAGE},  # Only app usage
        max_data_retention_hours=4,  # 4 hours retention
        privacy_mode="minimal",
        learning_enabled=True,
        surprise_threshold=0.2,  # Only learn from very surprising events
        require_biometric=True,
        auto_delete_sensitive=True,
        encrypt_all_data=True
    )
    
    # Balanced privacy (moderate data collection)
    balanced_privacy = UserPreferences(
        enabled_sources={
            DataSource.CONVERSATIONS,
            DataSource.LOCATION, 
            DataSource.APP_USAGE
        },
        max_data_retention_hours=24,  # 24 hours retention
        privacy_mode="balanced",
        learning_enabled=True,
        surprise_threshold=0.4,
        require_biometric=False,
        auto_delete_sensitive=True,
        encrypt_all_data=True
    )
    
    # Comprehensive (full data collection, still privacy-preserving)
    comprehensive_privacy = UserPreferences(
        enabled_sources={
            DataSource.CONVERSATIONS,
            DataSource.LOCATION,
            DataSource.APP_USAGE, 
            DataSource.PURCHASES,
            DataSource.CALENDAR,
            DataSource.VOICE
        },
        max_data_retention_hours=48,  # 48 hours retention
        privacy_mode="comprehensive",
        learning_enabled=True,
        surprise_threshold=0.6,  # Learn from more events
        require_biometric=False,
        auto_delete_sensitive=True,
        encrypt_all_data=True
    )
    
    return {
        "minimal": minimal_privacy,
        "balanced": balanced_privacy,
        "comprehensive": comprehensive_privacy
    }
```

**6.2 Performance Configuration:**
```python
# configure_performance.py
def configure_performance_settings():
    """Configure performance settings for different device types"""
    
    # High-end device configuration
    high_end_config = {
        "learning_cycle_frequency": 300,  # 5 minutes
        "data_collection_interval": 60,   # 1 minute
        "max_concurrent_agents": 8,
        "enable_gpu_acceleration": True,
        "thermal_throttling_threshold": 65,
        "battery_optimization": False
    }
    
    # Standard device configuration  
    standard_config = {
        "learning_cycle_frequency": 900,  # 15 minutes
        "data_collection_interval": 300,  # 5 minutes
        "max_concurrent_agents": 4,
        "enable_gpu_acceleration": False,
        "thermal_throttling_threshold": 55,
        "battery_optimization": True
    }
    
    # Low-end device configuration
    low_end_config = {
        "learning_cycle_frequency": 1800,  # 30 minutes
        "data_collection_interval": 600,   # 10 minutes  
        "max_concurrent_agents": 2,
        "enable_gpu_acceleration": False,
        "thermal_throttling_threshold": 45,
        "battery_optimization": True
    }
    
    return {
        "high_end": high_end_config,
        "standard": standard_config,
        "low_end": low_end_config
    }
```

## ✅ Verification and Testing

### Step 7: Deployment Verification

**7.1 System Health Check:**
```python
# verify_deployment.py
import asyncio

async def verify_digital_twin_deployment():
    """Comprehensive deployment verification"""
    
    print("🔍 Verifying Digital Twin Architecture deployment...")
    
    checks = []
    
    # 1. Digital Twin Concierge
    try:
        from packages.edge.mobile.digital_twin_concierge import DigitalTwinConcierge
        concierge = await get_digital_twin_instance()
        
        # Test basic functionality
        test_context = {"conversation": True, "time_of_day": 14}
        prediction = await concierge.predict_user_response(test_context)
        
        if prediction and "confidence" in prediction:
            checks.append("✅ Digital Twin Concierge: WORKING")
        else:
            checks.append("❌ Digital Twin Concierge: FAILED")
            
    except Exception as e:
        checks.append(f"❌ Digital Twin Concierge: ERROR - {e}")
    
    # 2. Meta-Agent Coordinator
    try:
        coordinator = await get_meta_agent_coordinator()
        status = await coordinator.get_deployment_status()
        
        if status["local_agents"] > 0 or status["fog_agents"] > 0:
            checks.append("✅ Meta-Agent Coordinator: WORKING")
        else:
            checks.append("❌ Meta-Agent Coordinator: NO AGENTS DEPLOYED")
            
    except Exception as e:
        checks.append(f"❌ Meta-Agent Coordinator: ERROR - {e}")
    
    # 3. Distributed RAG
    try:
        distributed_rag = await get_distributed_rag()
        
        if len(distributed_rag.knowledge_shards) > 0:
            checks.append("✅ Distributed RAG: WORKING")
        else:
            checks.append("❌ Distributed RAG: NO SHARDS FOUND")
            
    except Exception as e:
        checks.append(f"❌ Distributed RAG: ERROR - {e}")
    
    # 4. P2P Network
    try:
        transport_manager = await get_transport_manager()
        network_status = await transport_manager.get_network_status()
        
        if network_status["connected_peers"] > 0:
            checks.append("✅ P2P Network: CONNECTED")
        else:
            checks.append("⚠️ P2P Network: NO PEERS (expected for single device)")
            
    except Exception as e:
        checks.append(f"❌ P2P Network: ERROR - {e}")
    
    # 5. Privacy Compliance
    try:
        privacy_report = coordinator.get_privacy_report()
        
        if privacy_report["digital_twin_privacy"]:
            checks.append("✅ Privacy Compliance: VERIFIED")
        else:
            checks.append("❌ Privacy Compliance: FAILED")
            
    except Exception as e:
        checks.append(f"❌ Privacy Compliance: ERROR - {e}")
    
    # Print results
    print("\n📊 Deployment Verification Results:")
    for check in checks:
        print(f"   {check}")
    
    # Overall status
    success_count = sum(1 for check in checks if check.startswith("✅"))
    total_checks = len(checks)
    
    if success_count == total_checks:
        print(f"\n🎉 All {total_checks} checks passed! Deployment successful!")
        return True
    elif success_count >= total_checks * 0.8:
        print(f"\n⚠️ {success_count}/{total_checks} checks passed. Deployment mostly successful.")
        return True
    else:
        print(f"\n❌ Only {success_count}/{total_checks} checks passed. Deployment needs attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_digital_twin_deployment())
    exit(0 if success else 1)
```

**7.2 End-to-End Test:**
```python
# test_end_to_end.py
import asyncio
from datetime import datetime

async def test_complete_digital_twin_workflow():
    """Test complete end-to-end Digital Twin workflow"""
    
    print("🧪 Running end-to-end Digital Twin test...")
    
    # 1. Initialize system
    concierge = await get_digital_twin_instance()
    coordinator = await get_meta_agent_coordinator()
    distributed_rag = await get_distributed_rag()
    
    # 2. Test learning cycle
    print("\n📚 Testing learning cycle...")
    from packages.edge.mobile.resource_manager import MobileDeviceProfile
    
    device_profile = MobileDeviceProfile(
        timestamp=datetime.now().timestamp(),
        device_id="test_device",
        battery_percent=75,
        battery_charging=False,
        cpu_temp_celsius=35.0,
        cpu_percent=25.0,
        ram_used_mb=2000,
        ram_available_mb=2000,
        ram_total_mb=4000
    )
    
    cycle = await concierge.run_learning_cycle(device_profile)
    print(f"   Learning cycle completed: {cycle.data_points_count} data points")
    print(f"   Average surprise: {cycle.average_surprise:.3f}")
    print(f"   Prediction accuracy: {cycle.improvement_score:.3f}")
    
    # 3. Test personal knowledge query
    print("\n🔍 Testing personal knowledge query...")
    results = await concierge.query_personal_knowledge("user habits and patterns")
    print(f"   Retrieved {len(results)} relevant knowledge pieces")
    
    # 4. Test meta-agent communication
    print("\n🧠 Testing meta-agent coordination...")
    status = await coordinator.get_deployment_status()
    print(f"   Active agents: {len(status['agents'])}")
    
    # 5. Test privacy compliance
    print("\n🔒 Testing privacy compliance...")
    privacy_report = concierge.get_privacy_report()
    print(f"   Data location: {privacy_report['data_location']}")
    print(f"   Encryption enabled: {privacy_report['encryption_enabled']}")
    print(f"   Auto deletion: {privacy_report['auto_deletion']}")
    
    print("\n✅ End-to-end test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_complete_digital_twin_workflow())
```

## 🔧 Troubleshooting

### Common Deployment Issues

**Issue 1: Digital Twin Concierge fails to start**
```bash
# Check Python environment
python --version  # Should be 3.10+
pip list | grep numpy  # Should show numpy

# Check data directory permissions
ls -la digital_twin_data/
chmod 755 digital_twin_data/

# Check SQLite availability
python -c "import sqlite3; print('SQLite available')"
```

**Issue 2: Mobile permissions not granted**
```bash
# Android - Check permissions via ADB
adb shell dumpsys package com.aivillage.digitaltwin | grep permission

# Grant permissions manually
adb shell pm grant com.aivillage.digitaltwin android.permission.ACCESS_FINE_LOCATION
adb shell pm grant com.aivillage.digitaltwin android.permission.READ_CALL_LOG
```

**Issue 3: Fog network connection issues**
```bash
# Check P2P network connectivity
python -c "
from packages.p2p.core.transport_manager import UnifiedTransportManager
tm = UnifiedTransportManager()
print('P2P network status:', tm.get_network_status())
"

# Check fog coordinator status
python -c "
from packages.edge.fog_compute.fog_coordinator import FogCoordinator
fc = FogCoordinator()
print('Fog network status:', fc.get_system_status())
"
```

**Issue 4: Meta-agent deployment fails**
```bash
# Check available resources
python -c "
import psutil
print('Available RAM:', psutil.virtual_memory().available / 1024**3, 'GB')
print('CPU count:', psutil.cpu_count())
print('Battery:', psutil.sensors_battery())
"

# Check model sharding engine
python -c "
from src.production.distributed_inference.model_sharding_engine import ModelShardingEngine
mse = ModelShardingEngine()
print('Sharding engine initialized successfully')
"
```

## 📊 Performance Monitoring

### Step 8: Monitoring Setup

**8.1 Performance Monitoring:**
```python
# monitor_performance.py
import asyncio
import time
import psutil
from datetime import datetime

async def monitor_digital_twin_performance():
    """Monitor Digital Twin system performance"""
    
    print("📊 Starting Digital Twin performance monitoring...")
    
    while True:
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            battery = psutil.sensors_battery()
            
            # Digital Twin metrics
            concierge = await get_digital_twin_instance()
            coordinator = await get_meta_agent_coordinator()
            
            # Log metrics
            timestamp = datetime.now().isoformat()
            print(f"\n[{timestamp}] System Performance:")
            print(f"  CPU: {cpu_percent:.1f}%")
            print(f"  Memory: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
            
            if battery:
                print(f"  Battery: {battery.percent:.1f}% ({'charging' if battery.power_plugged else 'discharging'})")
            
            # Digital Twin specific metrics
            if hasattr(coordinator, 'metrics'):
                metrics = coordinator.metrics
                print(f"  Local agents: {metrics['local_agents_count']}")
                print(f"  Fog agents: {metrics['fog_agents_count']}")
                print(f"  Total shards: {metrics['total_shards']}")
                print(f"  Avg latency: {metrics.get('average_latency_ms', 0):.1f}ms")
            
            # Privacy status
            privacy_report = concierge.get_privacy_report()
            print(f"  Privacy: {privacy_report['data_location']} (encryption: {privacy_report['encryption_enabled']})")
            
            # Wait 30 seconds before next check
            await asyncio.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(monitor_digital_twin_performance())
```

**8.2 Health Check Script:**
```bash
#!/bin/bash
# health_check.sh

echo "🏥 Digital Twin Health Check"
echo "============================"

# Check Python processes
echo "📊 System Processes:"
ps aux | grep -E "(digital_twin|meta_agent|fog_coordinator)" | grep -v grep

# Check resource usage
echo -e "\n💾 Resource Usage:"
df -h | grep -E "(/$|digital_twin)"
free -h

# Check network connectivity
echo -e "\n🌐 Network Status:"
python -c "
from packages.p2p.core.transport_manager import UnifiedTransportManager
tm = UnifiedTransportManager()
try:
    status = tm.get_network_status()
    print(f'P2P Status: {status}')
except Exception as e:
    print(f'P2P Error: {e}')
"

# Check database integrity
echo -e "\n🗄️ Database Status:"
python -c "
import sqlite3
from pathlib import Path

db_path = Path('digital_twin_data/twin_data.db')
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.execute('SELECT COUNT(*) FROM data_points')
    count = cursor.fetchone()[0]
    print(f'Data points in database: {count}')
    conn.close()
else:
    print('Database not found')
"

echo -e "\n✅ Health check complete"
```

---

This deployment guide provides comprehensive instructions for deploying the complete Digital Twin Architecture. Follow the steps in order, verify each stage, and use the troubleshooting section for any issues that arise.

The system is designed to be privacy-first and resource-aware, automatically adapting to device capabilities while maintaining strong privacy guarantees.
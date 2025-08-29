# Digital Twin Concierge Deployment Guide

## Overview

The Digital Twin Concierge is a privacy-preserving on-device AI assistant that learns from user behavior to provide personalized assistance. This system processes data locally, never transmits personal information to external servers, and automatically deletes training data after each cycle.

## System Architecture

```
┌─────────────────────────────────────────┐
│           Digital Twin Concierge        │
├─────────────────────────────────────────┤
│  Privacy-Preserving Data Collection     │
│  ├── Conversation Analysis              │
│  ├── Location Pattern Recognition       │
│  ├── App Usage Monitoring              │
│  └── Purchase Behavior Analysis        │
├─────────────────────────────────────────┤
│       Surprise-Based Learning           │
│  ├── Prediction Engine                 │
│  ├── Surprise Score Calculation        │
│  └── Pattern Recognition               │
├─────────────────────────────────────────┤
│        Mini-RAG System                  │
│  ├── Personal Knowledge Base           │
│  ├── Context-Aware Retrieval           │
│  └── Global Knowledge Elevation        │
├─────────────────────────────────────────┤
│      Mobile Resource Management         │
│  ├── Battery-Aware Processing          │
│  ├── Thermal Throttling                │
│  └── Memory Optimization               │
└─────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements
- **iOS**: iPhone 12 or newer (A14 Bionic minimum)
- **Android**: Device with 4GB+ RAM, ARM64 processor, Android 10+
- **Storage**: 2GB free space for model and data
- **Battery**: Charging capability for training cycles

### Software Dependencies
```bash
# Python Environment
Python >= 3.11
numpy >= 1.24.0
sqlite3 (built-in)
asyncio (built-in)

# iOS Development
Xcode >= 14.0
iOS SDK >= 16.0
Swift >= 5.7

# Android Development
Android Studio >= 2022.2.1
Android SDK >= 33
Kotlin >= 1.8.0
```

### System Permissions Required
```json
{
  "ios": {
    "NSLocationWhenInUseUsageDescription": "Learn location patterns for suggestions",
    "NSContactsUsageDescription": "Analyze communication patterns",
    "NSCalendarsUsageDescription": "Learn scheduling preferences",
    "NSAppleEventsUsageDescription": "Monitor app usage patterns",
    "NSBiometricUsageDescription": "Secure personal AI data access"
  },
  "android": {
    "android.permission.ACCESS_FINE_LOCATION": "Learn location patterns",
    "android.permission.READ_SMS": "Analyze messaging patterns",
    "android.permission.PACKAGE_USAGE_STATS": "Monitor app usage",
    "android.permission.USE_BIOMETRIC": "Secure personal data access",
    "android.permission.WRITE_EXTERNAL_STORAGE": "Store encrypted training data"
  }
}
```

## Deployment Steps

### Step 1: Environment Setup

```bash
# Create deployment directory
mkdir digital_twin_deployment
cd digital_twin_deployment

# Clone the AIVillage repository
git clone https://github.com/your-org/AIVillage.git
cd AIVillage

# Set up Python environment
python -m venv digital_twin_env
source digital_twin_env/bin/activate  # Linux/Mac
# or
digital_twin_env\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configure Privacy Settings

Create `digital_twin_config.json`:
```json
{
  "privacy_settings": {
    "data_retention_hours": 24,
    "encryption_enabled": true,
    "biometric_required": false,
    "auto_delete_sensitive": true,
    "privacy_mode": "balanced"
  },
  "data_sources": {
    "conversations": true,
    "location": true,
    "app_usage": true,
    "purchases": false,
    "calendar": true,
    "voice": false
  },
  "learning_parameters": {
    "surprise_threshold": 0.3,
    "learning_enabled": true,
    "suggestion_frequency": "moderate",
    "min_battery_for_training": 20
  }
}
```

### Step 3: Initialize Digital Twin System

```python
# Example initialization script
import asyncio
from pathlib import Path
from ui.mobile.shared.digital_twin_concierge import (
    DigitalTwinConcierge,
    UserPreferences,
    DataSource
)

async def initialize_digital_twin():
    # Configure user preferences
    preferences = UserPreferences(
        enabled_sources={
            DataSource.CONVERSATIONS,
            DataSource.LOCATION,
            DataSource.APP_USAGE
        },
        max_data_retention_hours=24,
        privacy_mode="balanced",
        learning_enabled=True,
        surprise_threshold=0.3,
        require_biometric=True,
        encrypt_all_data=True
    )

    # Initialize data directory
    data_dir = Path("./user_digital_twin_data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create Digital Twin Concierge
    concierge = DigitalTwinConcierge(data_dir, preferences)

    print("✅ Digital Twin Concierge initialized successfully")
    return concierge

# Run initialization
if __name__ == "__main__":
    asyncio.run(initialize_digital_twin())
```

### Step 4: Configure Mobile Resource Manager

```python
# Mobile optimization configuration
from infrastructure.fog.edge.mobile.resource_manager import (
    MobileResourceManager,
    ResourcePolicy,
    MobileDeviceProfile
)

def setup_mobile_optimization():
    # Configure resource policy for mobile devices
    policy = ResourcePolicy(
        battery_critical=10,      # Critical battery threshold
        battery_low=20,           # Low battery threshold
        thermal_critical=65.0,    # Thermal throttling temperature
        memory_low_gb=2.0,        # Low memory threshold
        data_cost_low=100         # Daily data usage threshold (MB)
    )

    # Initialize resource manager with fog computing support
    resource_manager = MobileResourceManager(
        policy=policy,
        harvest_enabled=True,     # Enable idle resource harvesting
        token_rewards_enabled=True # Enable token rewards for contributions
    )

    return resource_manager

# Example device profile for testing
def create_test_profile():
    return MobileDeviceProfile(
        timestamp=time.time(),
        device_id="deployment_test_device",
        battery_percent=75,
        battery_charging=False,
        cpu_temp_celsius=35.0,
        cpu_percent=25.0,
        ram_used_mb=2000,
        ram_available_mb=2000,
        ram_total_mb=4000,
        network_type="wifi",
        is_foreground=True
    )
```

### Step 5: iOS Deployment

#### Xcode Project Setup

```swift
// DigitalTwinBridge.swift
import Foundation
import UserNotifications
import CoreLocation
import Contacts

class DigitalTwinBridge: NSObject, ObservableObject {
    private var pythonProcess: Process?
    private let dataDirectory: URL

    override init() {
        // Create secure data directory
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory,
                                                 in: .userDomainMask).first!
        self.dataDirectory = appSupport.appendingPathComponent("DigitalTwin",
                                                               isDirectory: true)
        super.init()
        setupDataDirectory()
    }

    private func setupDataDirectory() {
        do {
            try FileManager.default.createDirectory(at: dataDirectory,
                                                   withIntermediateDirectories: true)

            // Set file protection to complete unless open
            try FileManager.default.setAttributes([
                .protectionKey: FileProtectionType.completeUnlessOpen
            ], ofItemAtPath: dataDirectory.path)
        } catch {
            print("Failed to create data directory: \\(error)")
        }
    }

    func startDigitalTwin() async -> Bool {
        guard await requestPermissions() else {
            print("Required permissions not granted")
            return false
        }

        return await startPythonEngine()
    }

    private func requestPermissions() async -> Bool {
        let locationManager = CLLocationManager()
        let contactStore = CNContactStore()

        // Request location permission
        locationManager.requestWhenInUseAuthorization()

        // Request contacts permission
        let contactsStatus = await contactStore.requestAccess(for: .contacts)

        // Request notification permission
        let notificationCenter = UNUserNotificationCenter.current()
        let notificationStatus = await notificationCenter.requestAuthorization(
            options: [.alert, .badge, .sound]
        )

        return contactsStatus && notificationStatus
    }

    private func startPythonEngine() async -> Bool {
        // This would integrate with Python runtime
        // Using frameworks like Python.framework or embedded Python
        return true
    }
}
```

#### iOS Integration Configuration

```json
{
  "ios_config": {
    "bundle_identifier": "com.aivillage.digitaltwin",
    "minimum_os_version": "16.0",
    "required_device_capabilities": ["arm64"],
    "background_modes": ["background-processing"],
    "data_protection": "NSFileProtectionComplete",
    "app_transport_security": {
      "NSAllowsArbitraryLoads": false,
      "NSAllowsLocalNetworking": true
    }
  }
}
```

### Step 6: Android Deployment

#### Android Manifest Configuration

```xml
<!-- AndroidManifest.xml -->
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
    <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
    <uses-permission android:name="android.permission.READ_SMS" />
    <uses-permission android:name="android.permission.PACKAGE_USAGE_STATS" />
    <uses-permission android:name="android.permission.USE_BIOMETRIC" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />

    <application
        android:name=".DigitalTwinApplication"
        android:allowBackup="false"
        android:dataExtractionRules="@xml/data_extraction_rules"
        android:fullBackupContent="false">

        <service android:name=".DigitalTwinService"
                android:enabled="true"
                android:exported="false"
                android:foregroundServiceType="dataSync" />

        <receiver android:name=".DigitalTwinReceiver"
                 android:enabled="true"
                 android:exported="false" />
    </application>
</manifest>
```

#### Android Service Implementation

```kotlin
// DigitalTwinService.kt
class DigitalTwinService : Service() {
    private lateinit var pythonInstance: Python
    private var digitalTwinModule: PyObject? = null

    override fun onCreate() {
        super.onCreate()
        initializePython()
        startForegroundNotification()
    }

    private fun initializePython() {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        pythonInstance = Python.getInstance()

        // Load Digital Twin module
        try {
            digitalTwinModule = pythonInstance.getModule("digital_twin_concierge")
        } catch (e: PyException) {
            Log.e("DigitalTwin", "Failed to load Python module: ${e.message}")
        }
    }

    private fun startForegroundNotification() {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Digital Twin Active")
            .setContentText("Learning your preferences securely")
            .setSmallIcon(R.drawable.ic_brain)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val CHANNEL_ID = "digital_twin_channel"
        private const val NOTIFICATION_ID = 1001
    }
}
```

### Step 7: Configure Surprise-Based Learning

```python
# Create learning configuration
learning_config = {
    "surprise_thresholds": {
        "very_low": 0.1,    # Excellent prediction
        "low": 0.3,         # Good prediction
        "medium": 0.7,      # Poor prediction
        "high": 1.0         # Very poor prediction
    },
    "retraining_triggers": {
        "average_surprise_threshold": 0.5,
        "prediction_accuracy_threshold": 0.6,
        "min_data_points": 10
    },
    "learning_schedule": {
        "active_hours": [8, 22],    # Learn between 8 AM and 10 PM
        "min_battery_percent": 20,   # Don't learn when battery < 20%
        "max_thermal_temp": 45.0,    # Don't learn when device hot
        "wifi_only": True            # Only learn on Wi-Fi
    }
}
```

### Step 8: Deploy Mini-RAG Integration

```python
# Configure Mini-RAG for personal knowledge
from ui.mobile.shared.mini_rag_system import MiniRAGSystem

async def setup_mini_rag():
    # Initialize Mini-RAG with local storage
    mini_rag = MiniRAGSystem(
        storage_path="./personal_knowledge",
        instance_id=f"user_{user_id}",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Lightweight
        max_knowledge_items=10000,    # Reasonable limit for mobile
        cleanup_interval_hours=168,   # Weekly cleanup
        enable_global_elevation=True  # Share anonymized insights
    )

    # Configure knowledge elevation criteria
    elevation_config = {
        "min_confidence_score": 0.8,
        "min_usage_frequency": 5,
        "exclude_personal_data": True,
        "anonymization_level": "high"
    }

    await mini_rag.configure_global_elevation(elevation_config)
    return mini_rag
```

## Configuration Examples

### Production Configuration

```yaml
# digital_twin_production.yaml
digital_twin:
  privacy:
    encryption_algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2"
    biometric_authentication: true
    secure_enclave: true  # iOS only

  performance:
    max_memory_usage_mb: 500
    max_cpu_usage_percent: 30
    battery_optimization: true
    thermal_throttling: true

  data_collection:
    sampling_rate: 0.1  # Collect 10% of interactions
    batch_size: 32
    max_collection_interval: 3600  # 1 hour

  learning:
    training_frequency: "daily"
    max_training_time_minutes: 5
    surprise_sensitivity: 0.3
    pattern_retention_days: 30

  fog_computing:
    enable_harvesting: true
    min_contribution_hours: 2
    token_rewards: true
    p2p_mesh: true
```

### Development Configuration

```yaml
# digital_twin_development.yaml
digital_twin:
  privacy:
    encryption_algorithm: "AES-128-GCM"  # Lighter for development
    biometric_authentication: false

  performance:
    max_memory_usage_mb: 1000  # More generous for debugging
    logging_level: "DEBUG"

  data_collection:
    sampling_rate: 1.0  # Collect all interactions for testing
    synthetic_data: true  # Generate test data

  learning:
    training_frequency: "immediate"  # Train immediately for testing
    max_training_time_minutes: 1

  fog_computing:
    enable_harvesting: false  # Disable for development
    mock_p2p: true
```

## Performance Optimization

### Battery Optimization

```python
# Battery-aware processing configuration
battery_config = {
    "critical_battery_threshold": 10,  # Stop all processing
    "low_battery_threshold": 20,       # Reduce processing
    "charging_boost": 1.5,             # Increase processing when charging
    "background_processing_limit": 0.1, # 10% CPU max in background
}

# Thermal throttling configuration
thermal_config = {
    "temperature_monitoring": True,
    "throttle_at_celsius": 45.0,
    "shutdown_at_celsius": 65.0,
    "cooling_wait_seconds": 60,
}
```

### Memory Management

```python
# Memory optimization settings
memory_config = {
    "max_sqlite_cache_mb": 50,
    "embedding_cache_size": 1000,
    "knowledge_chunk_size": 512,
    "garbage_collection_frequency": 300,  # 5 minutes
    "low_memory_cleanup_threshold": 100,  # MB
}
```

## Security Implementation

### Data Encryption

```python
# Encryption configuration for sensitive data
encryption_config = {
    "algorithm": "AES-256-GCM",
    "key_derivation": {
        "algorithm": "PBKDF2",
        "iterations": 100000,
        "salt_length": 32
    },
    "secure_key_storage": {
        "ios": "Keychain",
        "android": "EncryptedSharedPreferences"
    }
}
```

### Privacy Controls

```python
# Privacy protection mechanisms
privacy_controls = {
    "data_minimization": True,
    "purpose_limitation": True,
    "storage_limitation": 24,  # hours
    "accuracy_maintenance": True,
    "integrity_verification": True,
    "confidentiality_protection": True,
    "availability_assurance": True
}
```

## Monitoring and Metrics

### Health Metrics

```python
# Digital Twin health monitoring
health_metrics = {
    "prediction_accuracy": 0.75,      # Target: >70%
    "average_surprise_score": 0.25,   # Target: <30%
    "processing_latency_ms": 150,     # Target: <200ms
    "memory_usage_mb": 200,           # Target: <300MB
    "battery_impact_percent": 2.5,    # Target: <5%
    "data_retention_compliance": True,
    "encryption_status": "enabled",
    "last_successful_training": "2024-01-15T10:30:00Z"
}
```

### Performance Dashboard

```python
# Dashboard configuration for monitoring
dashboard_config = {
    "metrics_collection_interval": 300,  # 5 minutes
    "alert_thresholds": {
        "prediction_accuracy": 0.6,      # Alert if below 60%
        "memory_usage_mb": 400,           # Alert if above 400MB
        "processing_errors": 5,           # Alert if >5 errors/hour
        "privacy_violations": 0           # Alert on any violation
    },
    "retention_days": 7,
    "export_format": "json"
}
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied Errors
```bash
# Check permissions
adb shell pm list permissions | grep aivillage
# Reset permissions
adb shell pm reset-permissions com.aivillage.digitaltwin
```

#### 2. High Battery Usage
```python
# Check resource usage
resource_manager = MobileResourceManager()
status = resource_manager.get_status()
print(f"Current optimization: {status['current_optimization']}")
print(f"Battery policies: {status['active_policies']}")
```

#### 3. Storage Issues
```bash
# Check storage usage
du -sh ./user_digital_twin_data/
# Clean up old data
python -c "
from ui.mobile.shared.digital_twin_concierge import OnDeviceDataCollector
collector = OnDeviceDataCollector('./data', preferences)
collector.cleanup_old_data()
"
```

#### 4. Model Not Learning
```python
# Check learning cycle status
async def debug_learning():
    concierge = DigitalTwinConcierge(data_dir, preferences)
    cycle = await concierge.run_learning_cycle(device_profile)
    print(f"Data points: {cycle.data_points_count}")
    print(f"Average surprise: {cycle.average_surprise}")
    print(f"Improvement: {cycle.improvement_score}")
```

### Log Analysis

```bash
# View Digital Twin logs
tail -f ./logs/digital_twin.log

# Filter for errors
grep -i error ./logs/digital_twin.log

# Check privacy compliance
grep -i privacy ./logs/digital_twin.log
```

## Production Deployment Checklist

### Pre-Deployment
- [ ] Privacy policy updated and approved
- [ ] Security audit completed
- [ ] Performance benchmarks met
- [ ] Battery impact < 5%
- [ ] Memory usage < 300MB
- [ ] Data encryption verified
- [ ] Biometric authentication tested
- [ ] Auto-deletion mechanisms verified

### Deployment
- [ ] App store compliance verified
- [ ] Distribution certificates valid
- [ ] Crash reporting configured
- [ ] Analytics implementation (privacy-compliant)
- [ ] Remote configuration system ready
- [ ] Rollback procedures tested

### Post-Deployment
- [ ] Monitor prediction accuracy metrics
- [ ] Track battery usage reports
- [ ] Review privacy compliance logs
- [ ] Monitor user feedback
- [ ] Analyze performance telemetry
- [ ] Verify data deletion schedules

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review prediction accuracy metrics
2. **Monthly**: Update surprise-based learning thresholds
3. **Quarterly**: Security audit and penetration testing
4. **Annually**: Full privacy impact assessment

### Performance Monitoring

```python
# Automated monitoring script
import asyncio
import json
from datetime import datetime

async def monitor_digital_twin():
    metrics = await collect_health_metrics()

    if metrics['prediction_accuracy'] < 0.6:
        send_alert("Low prediction accuracy", metrics)

    if metrics['memory_usage_mb'] > 400:
        send_alert("High memory usage", metrics)

    # Log metrics for analysis
    with open('digital_twin_metrics.log', 'a') as f:
        f.write(f"{datetime.now()}: {json.dumps(metrics)}\\n")

# Run monitoring every 5 minutes
if __name__ == "__main__":
    while True:
        asyncio.run(monitor_digital_twin())
        await asyncio.sleep(300)
```

## Conclusion

The Digital Twin Concierge provides a privacy-preserving AI assistant that learns user patterns without compromising personal data. The system's surprise-based learning approach ensures continuous improvement while maintaining strict privacy controls.

Key deployment success factors:
- Proper permission management
- Battery and thermal optimization
- Secure data encryption
- Regular performance monitoring
- Privacy compliance verification

For additional support, refer to the troubleshooting section or contact the development team through the established support channels.

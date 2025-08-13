# BitChat iOS Performance Measurement Guide

## Overview

This guide provides manual testing procedures and XCTest automation hooks for measuring BitChat iOS performance. Due to iOS platform restrictions, some measurements require manual coordination between devices.

## Automated XCTest Integration

### Setup XCTest Performance Suite

Add to your `BitChatUITests.swift`:

```swift
// MARK: - Performance Measurement Tests

func testMeasureDiscoveryPerformance() throws {
    let expectation = XCTestExpectation(description: "Discovery performance measurement")
    let startTime = CFAbsoluteTimeGetCurrent()

    var discoveryMetrics: [String: Any] = [:]

    // Start both managers
    manager1.startMeshNetworking()
    manager2.startMeshNetworking()

    // Monitor discovery time
    let cancellable = manager1.$connectedPeers.sink { peers in
        if !peers.isEmpty {
            let discoveryTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000 // ms
            discoveryMetrics["discovery_time_ms"] = discoveryTime
            discoveryMetrics["discovered_peers"] = peers.count
            discoveryMetrics["timestamp"] = Date().timeIntervalSince1970

            // Save metrics to file for analysis
            self.saveMetrics("discovery", metrics: discoveryMetrics)
            expectation.fulfill()
        }
    }

    wait(for: [expectation], timeout: 30.0)
    cancellable.cancel()
}

func testMeasureHopLatency() throws {
    let messageCount = 50
    var latencyResults: [Double] = []

    // Establish connection first
    let connectionExpectation = XCTestExpectation(description: "Connection")

    manager1.startMeshNetworking()
    manager2.startMeshNetworking()

    manager2.$connectedPeers.sink { peers in
        if !peers.isEmpty { connectionExpectation.fulfill() }
    }.store(in: &cancellables)

    wait(for: [connectionExpectation], timeout: 20.0)

    // Measure message latency
    for i in 0..<messageCount {
        let sendTime = CFAbsoluteTimeGetCurrent()
        let testMessage = "latency_test_\(i)_\(Int(sendTime))".data(using: .utf8)!

        let messageExpectation = XCTestExpectation(description: "Message \(i)")

        let cancellable = manager2.$receivedMessages.sink { messages in
            if let latestMessage = messages.last,
               String(data: latestMessage.content, encoding: .utf8)?.contains("latency_test_\(i)") == true {
                let receiveTime = CFAbsoluteTimeGetCurrent()
                let latency = (receiveTime - sendTime) * 1000 // ms
                latencyResults.append(latency)
                messageExpectation.fulfill()
            }
        }

        manager1.sendMessage(testMessage)
        wait(for: [messageExpectation], timeout: 5.0)
        cancellable.cancel()
    }

    // Calculate statistics
    let medianLatency = latencyResults.sorted()[latencyResults.count / 2]
    let avgLatency = latencyResults.reduce(0, +) / Double(latencyResults.count)
    let deliveryRatio = Double(latencyResults.count) / Double(messageCount)

    let latencyMetrics: [String: Any] = [
        "hop_count": 1,
        "median_latency_ms": medianLatency,
        "average_latency_ms": avgLatency,
        "delivery_ratio": deliveryRatio,
        "successful_messages": latencyResults.count,
        "total_messages": messageCount,
        "timestamp": Date().timeIntervalSince1970
    ]

    saveMetrics("hop_latency", metrics: latencyMetrics)

    // Validate against targets
    XCTAssertLessThan(medianLatency, 1000, "1-hop latency should be <1000ms")
    XCTAssertGreaterThan(deliveryRatio, 0.95, "Delivery ratio should be >95%")
}

func testMeasureBatteryImpact() throws {
    // Battery measurement requires longer test duration
    // This test provides framework for measurement

    let testDuration: TimeInterval = 300 // 5 minutes for CI/CD
    let measurementInterval: TimeInterval = 60 // 1 minute

    var batteryReadings: [Float] = []
    let startTime = Date()

    // Enable battery monitoring
    UIDevice.current.isBatteryMonitoringEnabled = true

    manager1.startMeshNetworking()

    // Take periodic battery readings
    let timer = Timer.scheduledTimer(withTimeInterval: measurementInterval, repeats: true) { _ in
        let batteryLevel = UIDevice.current.batteryLevel
        batteryReadings.append(batteryLevel)

        print("Battery level: \(batteryLevel * 100)%")
    }

    // Wait for test duration
    let expectation = XCTestExpectation(description: "Battery test duration")
    DispatchQueue.main.asyncAfter(deadline: .now() + testDuration) {
        expectation.fulfill()
    }

    wait(for: [expectation], timeout: testDuration + 10)
    timer.invalidate()

    // Calculate battery impact
    let initialBattery = batteryReadings.first ?? 0
    let finalBattery = batteryReadings.last ?? 0
    let batteryDrain = initialBattery - finalBattery
    let testHours = testDuration / 3600
    let drainRatePerHour = Double(batteryDrain) / testHours * 100 // percent per hour

    let batteryMetrics: [String: Any] = [
        "test_duration_hours": testHours,
        "initial_battery_percent": Double(initialBattery * 100),
        "final_battery_percent": Double(finalBattery * 100),
        "drain_rate_percent_per_hour": drainRatePerHour,
        "target_drain_rate": 3.0,
        "meets_target": drainRatePerHour < 3.0,
        "readings_count": batteryReadings.count,
        "timestamp": Date().timeIntervalSince1970
    ]

    saveMetrics("battery_consumption", metrics: batteryMetrics)

    UIDevice.current.isBatteryMonitoringEnabled = false

    print("Battery drain rate: \(drainRatePerHour)%/hour (target: <3%/hour)")
}

// Helper function to save metrics
private func saveMetrics(_ testType: String, metrics: [String: Any]) {
    let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let metricsDir = documentsPath.appendingPathComponent("bitchat_metrics")

    try? FileManager.default.createDirectory(at: metricsDir, withIntermediateDirectories: true)

    let timestamp = Int(Date().timeIntervalSince1970)
    let filename = "\(testType)_metrics_\(timestamp).json"
    let fileURL = metricsDir.appendingPathComponent(filename)

    do {
        let jsonData = try JSONSerialization.data(withJSONObject: metrics, options: .prettyPrinted)
        try jsonData.write(to: fileURL)
        print("Saved \(testType) metrics to: \(fileURL.path)")
    } catch {
        print("Failed to save \(testType) metrics: \(error)")
    }
}
```

## Manual Testing Procedures

### 1. Multi-Device Discovery Test

**Required**: 2-5 iOS devices with BitChat installed

**Procedure**:
1. Install BitChat on all test devices
2. Clear app data and restart devices
3. Open BitChat on all devices simultaneously
4. Record time to discover all peers
5. Note transport types used (WiFi, Bluetooth)
6. Measure connection establishment time

**Expected Results**:
- Discovery time: <30 seconds for all peers
- Connection establishment: <10 seconds per peer
- Transport preference: WiFi > Bluetooth > BLE

**Metrics to Capture**:
```json
{
  "device_count": 3,
  "discovery_time_seconds": 12.5,
  "connections_established": 2,
  "transport_breakdown": {
    "wifi_direct": 1,
    "bluetooth": 1,
    "ble": 0
  }
}
```

### 2. 7-Hop Message Relay Test

**Required**: 7+ iOS devices arranged in linear chain

**Setup**:
```
Device A ↔ Device B ↔ Device C ↔ Device D ↔ Device E ↔ Device F ↔ Device G
```

**Procedure**:
1. Arrange devices in line with limited range (force multi-hop)
2. Start BitChat on all devices
3. Wait for mesh formation
4. Send test message from Device A
5. Verify message reaches Device G
6. Measure total delivery time
7. Repeat with different message sizes

**Expected Results**:
- 7-hop delivery: <7 seconds
- Success rate: >70% for 7-hop chain
- TTL protection: Messages expire after 7 hops

### 3. Background Reconnection Test

**Required**: 2+ iOS devices

**Procedure**:
1. Establish BitChat connection between devices
2. Background the app on one device (home button)
3. Wait 2-5 minutes
4. Return app to foreground
5. Measure reconnection time
6. Test message delivery after reconnection

**Expected Results**:
- Reconnection time: <60 seconds
- Message delivery: Functional after reconnection
- Queue processing: Offline messages delivered

### 4. Chunked Message Delivery Test

**Required**: 2+ iOS devices with good connectivity

**Procedure**:
1. Establish BitChat connection
2. Send messages of increasing size:
   - 1 KB (small message)
   - 100 KB (medium message)
   - 500 KB (large message requiring chunking)
   - 1 MB (multiple chunks)
3. Measure delivery time for each size
4. Verify message integrity

**Expected Results**:
- Small messages: <1 second delivery
- Large messages: <10 seconds delivery
- Chunking threshold: 256 KB
- Message integrity: 100% accuracy

### 5. Battery Impact Measurement

**Required**: 1 iOS device, power measurement tools

**Procedure**:
1. Fully charge device to 100%
2. Baseline measurement: Record battery with BitChat disabled (1 hour)
3. Enable BitChat beacon mode only
4. Record battery drain over 4-6 hours
5. Calculate additional drain from BitChat

**Expected Results**:
- Additional battery drain: <3% per hour
- Idle beacon mode: Minimal impact
- Active mesh relay: <5% per hour additional

**Monitoring Script**:
```bash
#!/bin/bash
# iOS Battery Monitoring (requires device connection)

DEVICE_UDID="your_device_udid"
LOG_FILE="ios_battery_log.txt"
DURATION_HOURS=4

echo "Starting iOS battery monitoring for $DURATION_HOURS hours..."

for i in $(seq 1 $((DURATION_HOURS * 12))); do  # Every 5 minutes
    BATTERY_LEVEL=$(ideviceinfo -u $DEVICE_UDID -k BatteryCurrentCapacity)
    TIMESTAMP=$(date)
    echo "$TIMESTAMP: Battery Level: $BATTERY_LEVEL%" >> $LOG_FILE
    sleep 300  # 5 minutes
done

echo "Battery monitoring complete. Results in $LOG_FILE"
```

## Performance Data Collection

### XCTest Results Export

Add to your test scheme's post-action script:

```bash
#!/bin/bash

# Export BitChat performance metrics
DERIVED_DATA_PATH="$BUILD_DIR"
METRICS_SOURCE="$HOME/Library/Developer/CoreSimulator/Devices/*/data/Containers/Data/Application/*/Documents/bitchat_metrics"
RESULTS_DIR="$PROJECT_DIR/tmp_bitchat/ios/measurements"

mkdir -p "$RESULTS_DIR"

# Copy metrics files from simulator
find $METRICS_SOURCE -name "*.json" -exec cp {} "$RESULTS_DIR/" \; 2>/dev/null || true

# Generate summary report
cat > "$RESULTS_DIR/ios_test_summary.md" << EOF
# BitChat iOS Test Results

**Generated**: $(date)
**Simulator**: iOS $(xcrun simctl list runtimes | grep iOS | tail -1 | awk '{print $4}')

## Performance Metrics

$(ls -la "$RESULTS_DIR"/*.json 2>/dev/null | while read line; do
    FILE=$(echo $line | awk '{print $9}')
    echo "### $(basename $FILE .json)"
    echo "\`\`\`json"
    cat "$FILE"
    echo "\`\`\`"
    echo ""
done)

EOF

echo "📊 iOS test results exported to: $RESULTS_DIR"
```

### Real Device Testing Commands

```bash
# Run performance tests on connected iOS device
xcodebuild test \
  -scheme Bitchat \
  -destination 'platform=iOS,name=Your Device Name' \
  -only-testing:BitchatUITests/testMeasureDiscoveryPerformance \
  -only-testing:BitchatUITests/testMeasureHopLatency \
  -only-testing:BitchatUITests/testMeasureBatteryImpact

# Export test results
xcrun xccov view --report --json DerivedData/Build/Logs/Test/*.xcresult > ios_test_results.json
```

## Key Performance Indicators (KPIs)

### Discovery & Connection
- **Peer discovery time**: <30 seconds
- **Connection establishment**: <10 seconds
- **Multi-device success rate**: >90%

### Message Delivery
- **1-hop latency**: <1 second
- **3-hop latency**: <3 seconds
- **7-hop success rate**: >70%
- **Message integrity**: 100%

### Battery & Resources
- **Idle battery drain**: <3%/hour
- **Active relay battery**: <5%/hour additional
- **Memory usage**: <50 MB
- **Background reconnection**: <60 seconds

### Platform-Specific Metrics
- **Background suspension**: Expected behavior
- **Foreground recovery**: <60 seconds
- **MultipeerConnectivity efficiency**: >85%
- **Chunked delivery**: >95% success for large messages

## Integration with CI/CD

### GitHub Actions iOS Testing

```yaml
name: BitChat iOS Performance Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  ios-performance:
    runs-on: macos-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Xcode
      uses: maxim-lobanov/setup-xcode@v1
      with:
        xcode-version: latest-stable

    - name: Run BitChat Performance Tests
      run: |
        cd ios/Bitchat
        xcodebuild test \
          -scheme Bitchat \
          -destination 'platform=iOS Simulator,name=iPhone 15' \
          -only-testing:BitchatUITests/testMeasureDiscoveryPerformance \
          -only-testing:BitchatUITests/testMeasureHopLatency

    - name: Upload Performance Results
      uses: actions/upload-artifact@v3
      with:
        name: ios-performance-results
        path: tmp_bitchat/ios/measurements/
```

## Troubleshooting

### Common Issues

1. **MultipeerConnectivity not discovering peers**
   - Check WiFi and Bluetooth permissions
   - Verify devices are on same network segment
   - Restart networking services

2. **Background reconnection failing**
   - Expected iOS behavior - apps suspend in background
   - Test foreground restoration instead
   - Use app lifecycle notifications

3. **Battery measurement inconsistencies**
   - Use physical devices, not simulator
   - Allow sufficient test duration (4+ hours)
   - Account for baseline device battery usage

### Performance Optimization Tips

1. **Discovery Optimization**
   - Reduce discovery info payload size
   - Implement peer caching
   - Use efficient advertising intervals

2. **Message Delivery Optimization**
   - Implement message priority queues
   - Optimize chunk sizes for device capabilities
   - Add delivery confirmations

3. **Battery Optimization**
   - Reduce beacon frequency in background
   - Implement connection pooling
   - Use efficient serialization formats

---

*BitChat iOS measurement procedures - optimized for iOS platform constraints and MultipeerConnectivity framework*

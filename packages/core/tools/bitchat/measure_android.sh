#!/bin/bash

# BitChat Android Performance Measurement Script
#
# Measures key performance indicators for BitChat mesh networking:
# - Median hop latency by radio type (BLE, Wi-Fi, Bluetooth)
# - Delivery ratio by hop count (1-7 hops)
# - Idle beacon battery consumption
# - Network formation and discovery times

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/tmp_bitchat/android/measurements"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/android_measurement_$TIMESTAMP.log"

# Configuration
ADB_DEVICE=""
PACKAGE_NAME="com.aivillage.bitchat"
TEST_DURATION_SECONDS=300  # 5 minutes per test
BEACON_TEST_DURATION=3600  # 1 hour for battery test
PEER_COUNT=3
HOP_COUNT_MAX=7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Setup functions
setup_measurement_environment() {
    log_info "Setting up BitChat Android measurement environment..."

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Check ADB connection
    if ! command -v adb &> /dev/null; then
        log_error "ADB not found. Please install Android SDK tools."
        exit 1
    fi

    # Check for connected devices
    DEVICES=$(adb devices | grep -v "List of devices" | grep "device" | wc -l)
    if [ "$DEVICES" -eq 0 ]; then
        log_error "No Android devices connected via ADB."
        exit 1
    elif [ "$DEVICES" -eq 1 ]; then
        log_warning "Only 1 device connected. Multi-device tests will be limited."
        PEER_COUNT=1
    else
        log_success "Found $DEVICES connected Android devices."
        PEER_COUNT=$(($DEVICES < 5 ? $DEVICES : 5))  # Cap at 5 devices
    fi

    # Check if BitChat app is installed
    if ! adb shell pm list packages | grep -q "$PACKAGE_NAME"; then
        log_error "BitChat app not installed. Please install the app first."
        log_info "Install with: ./gradlew :app:installDebug"
        exit 1
    fi

    log_success "Measurement environment setup complete"
}

# Performance measurement functions
measure_discovery_performance() {
    log_info "Measuring peer discovery performance..."

    local discovery_results="$RESULTS_DIR/discovery_results_$TIMESTAMP.json"

    # Start BitChat service on all devices
    adb shell am start-foreground-service -a START_MESH "$PACKAGE_NAME/.BitChatService"

    # Monitor discovery times
    local start_time=$(date +%s%3N)
    local discovered_peers=0
    local max_wait_time=30000  # 30 seconds in milliseconds

    while [ $discovered_peers -lt $((PEER_COUNT - 1)) ] && [ $(($(date +%s%3N) - start_time)) -lt $max_wait_time ]; do
        # Query connected peer count via ADB logcat
        discovered_peers=$(adb shell "logcat -d -s BitChatService | grep 'Connected to peer' | wc -l")
        sleep 1
    done

    local discovery_time=$(($(date +%s%3N) - start_time))

    # Extract radio type statistics from logs
    local wifi_discoveries=$(adb shell "logcat -d -s BitChatService | grep -c 'NEARBY_WIFI' || echo 0")
    local bluetooth_discoveries=$(adb shell "logcat -d -s BitChatService | grep -c 'NEARBY_BLUETOOTH' || echo 0")
    local ble_discoveries=$(adb shell "logcat -d -s BitChatService | grep -c 'BLE_BEACON' || echo 0")

    # Create results JSON
    cat > "$discovery_results" << EOF
{
    "timestamp": "$TIMESTAMP",
    "peer_count": $PEER_COUNT,
    "discovery_time_ms": $discovery_time,
    "discovered_peers": $discovered_peers,
    "radio_stats": {
        "wifi_discoveries": $wifi_discoveries,
        "bluetooth_discoveries": $bluetooth_discoveries,
        "ble_discoveries": $ble_discoveries
    },
    "success_rate": $(echo "scale=2; $discovered_peers / ($PEER_COUNT - 1)" | bc)
}
EOF

    log_success "Discovery measurement completed: $discovered_peers/$((PEER_COUNT - 1)) peers in ${discovery_time}ms"

    # Clear logcat for next test
    adb shell logcat -c
}

measure_hop_latency() {
    log_info "Measuring hop latency by radio type..."

    local latency_results="$RESULTS_DIR/latency_results_$TIMESTAMP.json"
    local message_count=50
    local hop_results=()

    # Test different hop counts (1-7)
    for hop_count in $(seq 1 $HOP_COUNT_MAX); do
        if [ $hop_count -gt $PEER_COUNT ]; then
            log_warning "Skipping $hop_count hops (not enough devices: $PEER_COUNT)"
            continue
        fi

        log_info "Testing $hop_count hop latency..."

        # Send test messages and measure latency
        local total_latency=0
        local successful_messages=0

        for msg_num in $(seq 1 $message_count); do
            local send_time=$(date +%s%3N)

            # Send test message via ADB intent
            adb shell am start-foreground-service \
                -a SEND_MESSAGE \
                --es message "hop_test_${hop_count}_${msg_num}_${send_time}" \
                "$PACKAGE_NAME/.BitChatService"

            # Wait for message delivery confirmation (polling logcat)
            local delivered=false
            local timeout=$((send_time + 5000))  # 5 second timeout

            while [ $(date +%s%3N) -lt $timeout ]; do
                if adb shell "logcat -d -s BitChatService | grep -q 'hop_test_${hop_count}_${msg_num}_${send_time}'"; then
                    local receive_time=$(date +%s%3N)
                    local message_latency=$((receive_time - send_time))
                    total_latency=$((total_latency + message_latency))
                    successful_messages=$((successful_messages + 1))
                    delivered=true
                    break
                fi
                sleep 0.1
            done

            if [ "$delivered" = false ]; then
                log_warning "Message $msg_num at $hop_count hops timed out"
            fi
        done

        local median_latency=0
        if [ $successful_messages -gt 0 ]; then
            median_latency=$((total_latency / successful_messages))
        fi

        local delivery_ratio=$(echo "scale=3; $successful_messages / $message_count" | bc)

        hop_results+=("{\"hop_count\":$hop_count,\"median_latency_ms\":$median_latency,\"delivery_ratio\":$delivery_ratio,\"successful_messages\":$successful_messages}")

        log_success "$hop_count hops: ${median_latency}ms median latency, ${delivery_ratio} delivery ratio"

        # Brief pause between hop tests
        sleep 2
    done

    # Create comprehensive latency results
    printf '%s\n' "${hop_results[@]}" | jq -s . > "$latency_results.tmp"
    cat > "$latency_results" << EOF
{
    "timestamp": "$TIMESTAMP",
    "test_duration_seconds": $TEST_DURATION_SECONDS,
    "message_count_per_hop": $message_count,
    "hop_results": $(cat "$latency_results.tmp")
}
EOF

    rm "$latency_results.tmp"

    log_success "Hop latency measurement completed"
}

measure_battery_consumption() {
    log_info "Measuring idle beacon battery consumption..."

    local battery_results="$RESULTS_DIR/battery_results_$TIMESTAMP.json"

    # Get initial battery level
    local initial_battery=$(adb shell dumpsys battery | grep level | cut -d: -f2 | tr -d ' ')
    local initial_time=$(date +%s)

    log_info "Initial battery level: ${initial_battery}%"
    log_info "Starting $((BEACON_TEST_DURATION / 60)) minute battery test..."

    # Start BitChat in beacon-only mode
    adb shell am start-foreground-service -a START_MESH "$PACKAGE_NAME/.BitChatService"

    # Monitor battery level periodically
    local battery_samples=()
    local measurement_interval=300  # 5 minutes
    local elapsed_time=0

    while [ $elapsed_time -lt $BEACON_TEST_DURATION ]; do
        sleep $measurement_interval
        elapsed_time=$((elapsed_time + measurement_interval))

        local current_battery=$(adb shell dumpsys battery | grep level | cut -d: -f2 | tr -d ' ')
        local current_time=$(date +%s)
        local elapsed_minutes=$(((current_time - initial_time) / 60))

        battery_samples+=("{\"elapsed_minutes\":$elapsed_minutes,\"battery_level\":$current_battery}")

        log_info "Battery level at ${elapsed_minutes}min: ${current_battery}%"

        # Early exit if battery drops too much (device protection)
        if [ $current_battery -lt 20 ]; then
            log_warning "Battery level critical ($current_battery%), ending test early"
            break
        fi
    done

    # Calculate battery drain rate
    local final_battery=$(adb shell dumpsys battery | grep level | cut -d: -f2 | tr -d ' ')
    local final_time=$(date +%s)
    local total_elapsed_hours=$(echo "scale=3; ($final_time - $initial_time) / 3600" | bc)
    local battery_drain=$((initial_battery - final_battery))
    local drain_rate_per_hour=$(echo "scale=3; $battery_drain / $total_elapsed_hours" | bc)

    # Create battery results
    printf '%s\n' "${battery_samples[@]}" | jq -s . > "$battery_results.tmp"
    cat > "$battery_results" << EOF
{
    "timestamp": "$TIMESTAMP",
    "test_duration_hours": $total_elapsed_hours,
    "initial_battery_percent": $initial_battery,
    "final_battery_percent": $final_battery,
    "total_battery_drain_percent": $battery_drain,
    "drain_rate_percent_per_hour": $drain_rate_per_hour,
    "battery_samples": $(cat "$battery_results.tmp"),
    "target_drain_rate": 3.0,
    "meets_target": $(echo "$drain_rate_per_hour < 3.0" | bc)
}
EOF

    rm "$battery_results.tmp"

    log_success "Battery consumption measurement completed"
    log_info "Battery drain rate: ${drain_rate_per_hour}%/hour (target: <3.0%/hour)"

    # Stop BitChat service
    adb shell am stopservice "$PACKAGE_NAME/.BitChatService"
}

measure_transport_performance() {
    log_info "Measuring transport-specific performance..."

    local transport_results="$RESULTS_DIR/transport_results_$TIMESTAMP.json"

    # Test different transport types if available
    declare -A transport_tests=(
        ["BLE"]="ble_beacon_test"
        ["WIFI"]="nearby_wifi_test"
        ["BLUETOOTH"]="nearby_bluetooth_test"
    )

    local transport_results_array=()

    for transport in "${!transport_tests[@]}"; do
        log_info "Testing $transport transport performance..."

        # Configure BitChat to prefer specific transport (simulation)
        adb shell am start-foreground-service \
            -a START_MESH \
            --es transport_preference "$transport" \
            "$PACKAGE_NAME/.BitChatService"

        # Wait for stabilization
        sleep 10

        # Send test messages
        local test_message_count=20
        local transport_latency_total=0
        local successful_transport_messages=0

        for i in $(seq 1 $test_message_count); do
            local send_time=$(date +%s%3N)

            adb shell am start-foreground-service \
                -a SEND_MESSAGE \
                --es message "transport_test_${transport}_${i}_${send_time}" \
                "$PACKAGE_NAME/.BitChatService"

            # Monitor for delivery confirmation
            local timeout=$((send_time + 10000))  # 10 second timeout
            while [ $(date +%s%3N) -lt $timeout ]; do
                if adb shell "logcat -d -s BitChatService | grep -q 'transport_test_${transport}_${i}_${send_time}'"; then
                    local receive_time=$(date +%s%3N)
                    local latency=$((receive_time - send_time))
                    transport_latency_total=$((transport_latency_total + latency))
                    successful_transport_messages=$((successful_transport_messages + 1))
                    break
                fi
                sleep 0.1
            done
        done

        local avg_latency=0
        if [ $successful_transport_messages -gt 0 ]; then
            avg_latency=$((transport_latency_total / successful_transport_messages))
        fi

        local success_rate=$(echo "scale=3; $successful_transport_messages / $test_message_count" | bc)

        transport_results_array+=("{\"transport\":\"$transport\",\"avg_latency_ms\":$avg_latency,\"success_rate\":$success_rate,\"test_messages\":$test_message_count}")

        log_success "$transport: ${avg_latency}ms avg latency, ${success_rate} success rate"

        # Clear logs for next transport
        adb shell logcat -c
        sleep 5
    done

    # Create transport performance results
    printf '%s\n' "${transport_results_array[@]}" | jq -s . > "$transport_results.tmp"
    cat > "$transport_results" << EOF
{
    "timestamp": "$TIMESTAMP",
    "transport_tests": $(cat "$transport_results.tmp")
}
EOF

    rm "$transport_results.tmp"

    log_success "Transport performance measurement completed"
}

generate_measurement_report() {
    log_info "Generating comprehensive measurement report..."

    local report_file="$RESULTS_DIR/bitchat_android_report_$TIMESTAMP.md"

    cat > "$report_file" << EOF
# BitChat Android Performance Measurement Report

**Generated**: $(date)
**Test Duration**: $(($TEST_DURATION_SECONDS / 60)) minutes per test
**Devices Used**: $PEER_COUNT Android devices
**Package**: $PACKAGE_NAME

## Executive Summary

This report contains performance measurements for BitChat Android mesh networking implementation, focusing on key performance indicators (KPIs) for local device-to-device communication.

## Key Performance Indicators

### Discovery Performance
$(if [ -f "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" ]; then
    echo "- **Peer Discovery Time**: $(cat "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" | jq -r '.discovery_time_ms')ms"
    echo "- **Discovery Success Rate**: $(cat "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" | jq -r '.success_rate')"
    echo "- **Wi-Fi Discoveries**: $(cat "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" | jq -r '.radio_stats.wifi_discoveries')"
    echo "- **Bluetooth Discoveries**: $(cat "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" | jq -r '.radio_stats.bluetooth_discoveries')"
    echo "- **BLE Discoveries**: $(cat "$RESULTS_DIR/discovery_results_$TIMESTAMP.json" | jq -r '.radio_stats.ble_discoveries')"
else
    echo "- Discovery test data not available"
fi)

### Hop Latency Performance
$(if [ -f "$RESULTS_DIR/latency_results_$TIMESTAMP.json" ]; then
    echo "$(cat "$RESULTS_DIR/latency_results_$TIMESTAMP.json" | jq -r '.hop_results[] | "- **\(.hop_count) hops**: \(.median_latency_ms)ms median latency, \(.delivery_ratio) delivery ratio"')"
else
    echo "- Latency test data not available"
fi)

### Battery Consumption
$(if [ -f "$RESULTS_DIR/battery_results_$TIMESTAMP.json" ]; then
    echo "- **Idle Beacon Drain Rate**: $(cat "$RESULTS_DIR/battery_results_$TIMESTAMP.json" | jq -r '.drain_rate_percent_per_hour')%/hour"
    echo "- **Target Compliance**: $(cat "$RESULTS_DIR/battery_results_$TIMESTAMP.json" | jq -r 'if .meets_target == 1 then "✅ PASS" else "❌ FAIL" end') (target: <3.0%/hour)"
    echo "- **Test Duration**: $(cat "$RESULTS_DIR/battery_results_$TIMESTAMP.json" | jq -r '.test_duration_hours') hours"
else
    echo "- Battery test data not available"
fi)

## Transport Performance Analysis

$(if [ -f "$RESULTS_DIR/transport_results_$TIMESTAMP.json" ]; then
    cat "$RESULTS_DIR/transport_results_$TIMESTAMP.json" | jq -r '.transport_tests[] | "### \(.transport) Transport\n- **Average Latency**: \(.avg_latency_ms)ms\n- **Success Rate**: \(.success_rate)\n- **Test Messages**: \(.test_messages)\n"'
else
    echo "Transport test data not available"
fi)

## Test Configuration

- **Test Script**: $0
- **Results Directory**: $RESULTS_DIR
- **Log File**: $LOG_FILE
- **ADB Devices**: $PEER_COUNT
- **Maximum Hop Count**: $HOP_COUNT_MAX

## Raw Data Files

$(ls -la "$RESULTS_DIR"/*_$TIMESTAMP.* 2>/dev/null | awk '{print "- " $9}' || echo "No raw data files found")

## Recommendations

### Performance Optimization
- Monitor hop latency to ensure <500ms target for 3-hop delivery
- Validate delivery ratio maintains >90% success rate
- Optimize battery consumption to stay under 3%/hour target

### Next Steps
1. Run multi-device mesh formation tests
2. Validate performance under network stress
3. Test background operation limitations
4. Measure performance across different Android versions

---

*Report generated by BitChat Android measurement tools*
EOF

    log_success "Measurement report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting BitChat Android performance measurement suite..."
    log_info "Results will be saved to: $RESULTS_DIR"

    setup_measurement_environment

    # Run measurement tests
    measure_discovery_performance
    measure_hop_latency
    measure_transport_performance

    # Battery test is optional due to long duration
    if [ "${1:-}" = "--include-battery" ]; then
        measure_battery_consumption
    else
        log_info "Skipping battery test (use --include-battery to run)"
        log_info "Battery test takes $((BEACON_TEST_DURATION / 60)) minutes"
    fi

    generate_measurement_report

    log_success "BitChat Android measurement suite completed!"
    log_info "View results in: $RESULTS_DIR"
}

# Error handling
trap 'log_error "Measurement script interrupted"; exit 1' INT TERM

# Run main function
main "$@"

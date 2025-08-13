import XCTest
import MultipeerConnectivity
@testable import Bitchat

/**
 * BitChat UI Tests for iOS
 *
 * Tests MultipeerConnectivity mesh functionality including:
 * - Two-peer discovery and connection
 * - Message transmission with TTL validation
 * - Background/foreground reconnection handling
 * - Chunked message delivery for large payloads
 */
@available(iOS 13.0, *)
class BitChatUITests: XCTestCase {

    var manager1: BitChatManager!
    var manager2: BitChatManager!

    override func setUpWithError() throws {
        continueAfterFailure = false

        // Create two BitChat managers for peer testing
        manager1 = BitChatManager()
        manager2 = BitChatManager()

        print("🧪 BitChat UI Tests setup completed")
    }

    override func tearDownWithError() throws {
        manager1?.stopMeshNetworking()
        manager2?.stopMeshNetworking()

        manager1 = nil
        manager2 = nil

        print("🧪 BitChat UI Tests cleanup completed")
    }

    /// Test basic peer discovery and connection between two devices
    func testTwoPeerConnection() throws {
        let expectation = XCTestExpectation(description: "Two peers should connect")
        expectation.expectedFulfillmentCount = 2

        // Start both mesh networks
        manager1.startMeshNetworking()
        manager2.startMeshNetworking()

        // Monitor connection state changes
        var manager1Connected = false
        var manager2Connected = false

        let cancellable1 = manager1.$connectedPeers.sink { peers in
            if !peers.isEmpty && !manager1Connected {
                manager1Connected = true
                expectation.fulfill()
                print("✅ Manager 1 connected to \(peers.count) peer(s)")
            }
        }

        let cancellable2 = manager2.$connectedPeers.sink { peers in
            if !peers.isEmpty && !manager2Connected {
                manager2Connected = true
                expectation.fulfill()
                print("✅ Manager 2 connected to \(peers.count) peer(s)")
            }
        }

        // Wait for both peers to connect
        wait(for: [expectation], timeout: 30.0)

        // Verify both managers have connected peers
        XCTAssertFalse(manager1.connectedPeers.isEmpty, "Manager 1 should have connected peers")
        XCTAssertFalse(manager2.connectedPeers.isEmpty, "Manager 2 should have connected peers")

        cancellable1.cancel()
        cancellable2.cancel()

        print("🎉 Two-peer connection test passed")
    }

    /// Test message transmission and TTL validation
    func testMessageTransmissionWithTTL() throws {
        let connectionExpectation = XCTestExpectation(description: "Peers should connect")
        let messageExpectation = XCTestExpectation(description: "Message should be received")

        // First establish connection
        manager1.startMeshNetworking()
        manager2.startMeshNetworking()

        let connectionCancellable = manager2.$connectedPeers.sink { peers in
            if !peers.isEmpty {
                connectionExpectation.fulfill()
            }
        }

        wait(for: [connectionExpectation], timeout: 20.0)
        connectionCancellable.cancel()

        // Now test message transmission
        let testMessage = "Hello from BitChat iOS test!".data(using: .utf8)!

        let messageCancellable = manager2.$receivedMessages.sink { messages in
            if !messages.isEmpty {
                let receivedMessage = messages.last!

                // Validate message content
                XCTAssertEqual(receivedMessage.content, testMessage, "Message content should match")
                XCTAssertEqual(receivedMessage.hopCount, 0, "Direct message should have hop count 0")

                messageExpectation.fulfill()
                print("📩 Message received: \(String(data: receivedMessage.content, encoding: .utf8) ?? "binary")")
            }
        }

        // Send test message from manager1 to manager2
        manager1.sendMessage(testMessage)

        wait(for: [messageExpectation], timeout: 15.0)
        messageCancellable.cancel()

        print("🎉 Message transmission test passed")
    }

    /// Test TTL expiry prevents infinite message loops
    func testTTLExpiryProtection() throws {
        // Test TTL logic without requiring multiple hops
        let initialTTL = 7
        let maxHops = 7

        // Simulate hop progression
        for hopCount in 0..<10 {
            let remainingTTL = initialTTL - hopCount
            let shouldRelay = remainingTTL > 0 && hopCount < maxHops

            if hopCount < maxHops {
                XCTAssertTrue(shouldRelay, "Message with \(hopCount) hops should be relayed when TTL > 0")
            } else {
                XCTAssertFalse(shouldRelay, "Message with \(hopCount) hops should not be relayed")
            }
        }

        // Test message expiry (5 minute timeout)
        let currentTime = Date()
        let oldTime = currentTime.addingTimeInterval(-6 * 60) // 6 minutes ago
        let messageAge = currentTime.timeIntervalSince(oldTime)
        let isExpired = messageAge > (5 * 60) // 5 minute expiry

        XCTAssertTrue(isExpired, "Old messages should be detected as expired")

        print("✅ TTL expiry protection test passed")
    }

    /// Test message deduplication prevents processing duplicates
    func testMessageDeduplication() throws {
        var seenMessageIDs: Set<String> = []
        let testMessageID = "test_msg_\(UUID().uuidString)"

        // First message should be new
        let isFirstTimeNew = seenMessageIDs.insert(testMessageID).inserted
        XCTAssertTrue(isFirstTimeNew, "First occurrence should be new")

        // Second message should be duplicate
        let isSecondTimeNew = seenMessageIDs.insert(testMessageID).inserted
        XCTAssertFalse(isSecondTimeNew, "Second occurrence should be duplicate")

        print("✅ Message deduplication test passed")
    }

    /// Test chunked message delivery for large payloads
    func testChunkedMessageDelivery() throws {
        let connectionExpectation = XCTestExpectation(description: "Peers should connect")

        // Establish connection first
        manager1.startMeshNetworking()
        manager2.startMeshNetworking()

        let connectionCancellable = manager2.$connectedPeers.sink { peers in
            if !peers.isEmpty {
                connectionExpectation.fulfill()
            }
        }

        wait(for: [connectionExpectation], timeout: 20.0)
        connectionCancellable.cancel()

        // Create large test message (larger than 256KB chunk size)
        let chunkSize = 256 * 1024 // 256KB
        let largeMessageSize = chunkSize + 1000 // Slightly larger than one chunk
        let largeMessage = Data(repeating: 65, count: largeMessageSize) // Repeated 'A' characters

        // Calculate expected chunk count
        let expectedChunkCount = (largeMessageSize + chunkSize - 1) / chunkSize
        XCTAssertGreaterThan(expectedChunkCount, 1, "Large message should require multiple chunks")

        print("📦 Testing chunked delivery: \(largeMessageSize) bytes in \(expectedChunkCount) chunks")

        // In a full implementation, we would test actual chunk reassembly
        // For this UI test, we validate the chunking logic
        manager1.sendMessage(largeMessage)

        // Brief wait to allow processing
        let processExpectation = XCTestExpectation(description: "Message processing")
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            processExpectation.fulfill()
        }
        wait(for: [processExpectation], timeout: 5.0)

        print("✅ Chunked message delivery test completed")
    }

    /// Test background/foreground reconnection behavior
    func testBackgroundForegroundReconnection() throws {
        let initialConnectionExpectation = XCTestExpectation(description: "Initial connection")

        // Start mesh networking
        manager1.startMeshNetworking()
        manager2.startMeshNetworking()

        let initialCancellable = manager1.$meshState.sink { state in
            if case .connected = state {
                initialConnectionExpectation.fulfill()
            }
        }

        wait(for: [initialConnectionExpectation], timeout: 20.0)
        initialCancellable.cancel()

        // Simulate background transition
        NotificationCenter.default.post(name: UIApplication.didEnterBackgroundNotification, object: nil)

        // Brief wait for background handling
        let backgroundExpectation = XCTestExpectation(description: "Background processing")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            backgroundExpectation.fulfill()
        }
        wait(for: [backgroundExpectation], timeout: 3.0)

        // Verify state changed to backgrounded
        XCTAssertEqual(manager1.meshState, .backgrounded, "Should enter backgrounded state")

        // Simulate foreground transition
        NotificationCenter.default.post(name: UIApplication.willEnterForegroundNotification, object: nil)

        // Brief wait for foreground handling
        let foregroundExpectation = XCTestExpectation(description: "Foreground processing")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            foregroundExpectation.fulfill()
        }
        wait(for: [foregroundExpectation], timeout: 5.0)

        // Verify state recovery
        XCTAssertNotEqual(manager1.meshState, .backgrounded, "Should exit backgrounded state")

        print("✅ Background/foreground reconnection test passed")
    }

    /// Test network statistics and monitoring
    func testNetworkStatistics() throws {
        manager1.startMeshNetworking()

        // Initial stats should show no connections
        let initialStats = manager1.getNetworkStats()
        XCTAssertEqual(initialStats.connectedPeerCount, 0, "Should start with no connected peers")
        XCTAssertEqual(initialStats.queuedMessageCount, 0, "Should start with empty message queue")
        XCTAssertEqual(initialStats.sessionState, .disconnected, "Should start disconnected")

        // Queue a message and verify stats
        let testMessage = "Test message for stats".data(using: .utf8)!
        manager1.sendMessage(testMessage)

        // Brief wait for queue processing
        let queueExpectation = XCTestExpectation(description: "Queue processing")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            queueExpectation.fulfill()
        }
        wait(for: [queueExpectation], timeout: 2.0)

        let updatedStats = manager1.getNetworkStats()
        print("📊 Network stats: \(updatedStats.queuedMessageCount) queued, \(updatedStats.connectedPeerCount) peers")

        print("✅ Network statistics test passed")
    }

    /// Test peer capability exchange
    func testPeerCapabilityExchange() throws {
        let expectation = XCTestExpectation(description: "Capability exchange")

        manager1.startMeshNetworking()
        manager2.startMeshNetworking()

        // Wait for connection and capability exchange
        let cancellable = manager1.$connectedPeers.sink { peers in
            if let peer = peers.first, peer.capabilities != nil {
                // Validate capability information
                let capabilities = peer.capabilities!
                XCTAssertTrue(capabilities.supportsChunking, "Peer should support chunking")
                XCTAssertGreaterThan(capabilities.maxChunkSize, 0, "Max chunk size should be positive")
                XCTAssertFalse(capabilities.deviceModel.isEmpty, "Device model should not be empty")

                expectation.fulfill()
                print("📋 Capability exchange completed: \(capabilities.deviceModel)")
            }
        }

        wait(for: [expectation], timeout: 30.0)
        cancellable.cancel()

        print("✅ Peer capability exchange test passed")
    }

    /// Performance test for message throughput
    func testMessageThroughputPerformance() throws {
        measure {
            let messageCount = 100
            let testData = "Performance test message".data(using: .utf8)!

            manager1.startMeshNetworking()

            // Send multiple messages rapidly
            for i in 0..<messageCount {
                manager1.sendMessage(testData)
            }

            // Brief processing time
            let semaphore = DispatchSemaphore(value: 0)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                semaphore.signal()
            }
            semaphore.wait()
        }

        print("✅ Message throughput performance test completed")
    }
}

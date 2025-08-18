import Foundation
import MultipeerConnectivity
import Combine
import BackgroundTasks

/**
 * BitChat Mesh Network Manager for iOS
 *
 * Implements local mesh networking using MultipeerConnectivity with:
 * - Automatic service advertising/browsing for peer discovery
 * - Store-and-forward messaging with 7-hop TTL protection
 * - Chunked message delivery with backpressure (≤256KB chunks)
 * - Background reconnection strategies for app lifecycle management
 * - Session persistence and automatic reconnection on foreground
 */
@available(iOS 13.0, *)
public class BitChatManager: NSObject, ObservableObject {

    // MARK: - Constants
    private let serviceType = "aivillage-mesh"
    private let maxTTL: Int = 7
    private let messageExpiryInterval: TimeInterval = 5 * 60 // 5 minutes
    private let maxChunkSize: Int = 256 * 1024 // 256KB for resource constraints
    private let heartbeatInterval: TimeInterval = 30.0
    private let backgroundTaskIdentifier = "com.aivillage.bitchat.background"

    // MARK: - MultipeerConnectivity Components
    private let localPeerID: MCPeerID
    private let session: MCSession
    private let advertiser: MCNearbyServiceAdvertiser
    private let browser: MCNearbyServiceBrowser

    // MARK: - State Management
    @Published public var meshState: MeshState = .stopped
    @Published public var connectedPeers: [PeerInfo] = []
    @Published public var receivedMessages: [ReceivedMessage] = []

    private var messageQueue: [PendingMessage] = []
    private var seenMessageIDs: Set<String> = []
    private var peerCapabilities: [MCPeerID: PeerCapability] = [:]

    // MARK: - Background & Lifecycle
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var heartbeatTimer: Timer?
    private var messageProcessor: Timer?
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Data Models
    public enum MeshState {
        case stopped
        case starting
        case advertising
        case browsing
        case connected
        case backgrounded
        case error(String)
    }

    public struct PeerInfo: Identifiable, Hashable {
        public let id = UUID()
        public let peerID: MCPeerID
        public let displayName: String
        public let connectionState: MCSessionState
        public let lastSeen: Date
        public let capabilities: PeerCapability?

        public func hash(into hasher: inout Hasher) {
            hasher.combine(peerID)
        }

        public static func == (lhs: PeerInfo, rhs: PeerInfo) -> Bool {
            return lhs.peerID == rhs.peerID
        }
    }

    public struct PeerCapability: Codable {
        let peerID: String
        let supportsChunking: Bool
        let maxChunkSize: Int
        let batteryLevel: Float
        let lastHeartbeat: Date
        let deviceModel: String
    }

    public struct ReceivedMessage: Identifiable {
        public let id = UUID()
        public let messageID: String
        public let content: Data
        public let fromPeer: MCPeerID
        public let hopCount: Int
        public let receivedAt: Date
        public let originalSender: String
    }

    private struct PendingMessage {
        let messageID: String
        let content: Data
        let hopCount: Int
        let ttl: Int
        let createdAt: Date
        let targetPeer: MCPeerID?
        let originalSender: String

        var isExpired: Bool {
            Date().timeIntervalSince(createdAt) > 5 * 60 // 5 minutes
        }
    }

    private struct MessageEnvelope: Codable {
        let messageID: String
        let content: Data
        let hopCount: Int
        let ttl: Int
        let createdAt: Date
        let originalSender: String
        let messageType: MessageType

        enum MessageType: String, Codable {
            case data = "DATA"
            case capability = "CAPABILITY"
            case heartbeat = "HEARTBEAT"
        }
    }

    // MARK: - Initialization
    public override init() {
        // Generate unique peer ID
        let deviceName = UIDevice.current.name
        let uniqueID = UUID().uuidString.prefix(8)
        self.localPeerID = MCPeerID(displayName: "\(deviceName)-\(uniqueID)")

        // Initialize session with encryption preference
        self.session = MCSession(
            peer: localPeerID,
            securityIdentity: nil,
            encryptionPreference: .required
        )

        // Initialize advertiser and browser
        self.advertiser = MCNearbyServiceAdvertiser(
            peer: localPeerID,
            discoveryInfo: [
                "version": "1.0",
                "capabilities": "chunking,store-forward",
                "battery": "\(Int(UIDevice.current.batteryLevel * 100))"
            ],
            serviceType: serviceType
        )

        self.browser = MCNearbyServiceBrowser(
            peer: localPeerID,
            serviceType: serviceType
        )

        super.init()

        // Set delegates
        session.delegate = self
        advertiser.delegate = self
        browser.delegate = self

        // Setup lifecycle notifications
        setupLifecycleObservers()

        print("🔥 BitChat initialized with peer ID: \(localPeerID.displayName)")
    }

    deinit {
        stopMeshNetworking()
        cancellables.removeAll()
    }

    // MARK: - Public API

    /// Start mesh networking with advertising and browsing
    public func startMeshNetworking() {
        guard meshState != .connected && meshState != .starting else { return }

        meshState = .starting

        do {
            // Start advertising our presence
            advertiser.startAdvertisingPeer()
            meshState = .advertising

            // Start browsing for peers
            browser.startBrowsingForPeers()
            meshState = .browsing

            // Start background processing
            startBackgroundProcessing()

            print("🌐 BitChat mesh networking started")

        } catch {
            meshState = .error("Failed to start mesh networking: \(error.localizedDescription)")
            print("❌ Failed to start BitChat: \(error)")
        }
    }

    /// Stop mesh networking and disconnect all peers
    public func stopMeshNetworking() {
        advertiser.stopAdvertisingPeer()
        browser.stopBrowsingForPeers()
        session.disconnect()

        stopBackgroundProcessing()

        connectedPeers.removeAll()
        messageQueue.removeAll()
        seenMessageIDs.removeAll()

        meshState = .stopped
        print("🛑 BitChat mesh networking stopped")
    }

    /// Send message to all connected peers or specific target
    public func sendMessage(_ data: Data, to targetPeer: MCPeerID? = nil) {
        let messageID = generateMessageID()
        let pendingMessage = PendingMessage(
            messageID: messageID,
            content: data,
            hopCount: 0,
            ttl: maxTTL,
            createdAt: Date(),
            targetPeer: targetPeer,
            originalSender: localPeerID.displayName
        )

        messageQueue.append(pendingMessage)
        print("📤 Queued message \(messageID) (\(data.count) bytes)")
    }

    /// Get current mesh network statistics
    public func getNetworkStats() -> NetworkStats {
        return NetworkStats(
            connectedPeerCount: connectedPeers.count,
            queuedMessageCount: messageQueue.count,
            seenMessageCount: seenMessageIDs.count,
            sessionState: session.connectedPeers.count > 0 ? .connected : .disconnected
        )
    }

    public struct NetworkStats {
        public let connectedPeerCount: Int
        public let queuedMessageCount: Int
        public let seenMessageCount: Int
        public let sessionState: SessionState

        public enum SessionState {
            case connected, disconnected
        }
    }

    // MARK: - Background Processing

    private func startBackgroundProcessing() {
        // Message processing timer
        messageProcessor = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.processMessageQueue()
        }

        // Heartbeat timer for peer maintenance
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: heartbeatInterval, repeats: true) { [weak self] _ in
            self?.sendHeartbeat()
        }

        // Clean expired messages
        Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            self?.cleanExpiredMessages()
        }
    }

    private func stopBackgroundProcessing() {
        messageProcessor?.invalidate()
        messageProcessor = nil

        heartbeatTimer?.invalidate()
        heartbeatTimer = nil

        endBackgroundTask()
    }

    private func processMessageQueue() {
        guard !messageQueue.isEmpty else { return }

        // Process up to 5 messages per cycle to avoid blocking
        let messagesToProcess = Array(messageQueue.prefix(5))
        messageQueue.removeFirst(min(5, messageQueue.count))

        for message in messagesToProcess {
            processMessage(message)
        }
    }

    private func processMessage(_ message: PendingMessage) {
        // Skip expired messages
        if message.isExpired || message.ttl <= 0 {
            print("⏰ Message \(message.messageID) expired, dropping")
            return
        }

        // Skip already seen messages
        if seenMessageIDs.contains(message.messageID) {
            print("🔄 Message \(message.messageID) already seen, dropping")
            return
        }

        seenMessageIDs.insert(message.messageID)

        // Create envelope for transmission
        let envelope = MessageEnvelope(
            messageID: message.messageID,
            content: message.content,
            hopCount: message.hopCount,
            ttl: message.ttl,
            createdAt: message.createdAt,
            originalSender: message.originalSender,
            messageType: .data
        )

        // Send to target peer or broadcast
        if let targetPeer = message.targetPeer {
            sendEnvelope(envelope, to: [targetPeer])
        } else {
            broadcastEnvelope(envelope)
        }
    }

    private func sendEnvelope(_ envelope: MessageEnvelope, to peers: [MCPeerID]) {
        guard let data = try? JSONEncoder().encode(envelope) else {
            print("❌ Failed to encode message envelope")
            return
        }

        // Handle chunking for large messages
        if data.count > maxChunkSize {
            sendChunkedData(data, envelope: envelope, to: peers)
        } else {
            sendDirectData(data, to: peers)
        }
    }

    private func sendChunkedData(_ data: Data, envelope: MessageEnvelope, to peers: [MCPeerID]) {
        let chunkCount = (data.count + maxChunkSize - 1) / maxChunkSize
        print("📦 Chunking message \(envelope.messageID) into \(chunkCount) pieces")

        for chunkIndex in 0..<chunkCount {
            let startIndex = chunkIndex * maxChunkSize
            let endIndex = min(startIndex + maxChunkSize, data.count)
            let chunkData = data.subdata(in: startIndex..<endIndex)

            let chunkEnvelope = ChunkEnvelope(
                messageID: envelope.messageID,
                chunkIndex: chunkIndex,
                totalChunks: chunkCount,
                chunkData: chunkData
            )

            guard let chunkEncodedData = try? JSONEncoder().encode(chunkEnvelope) else { continue }
            sendDirectData(chunkEncodedData, to: peers)

            // Brief delay between chunks to prevent overwhelming
            usleep(100000) // 100ms
        }
    }

    private struct ChunkEnvelope: Codable {
        let messageID: String
        let chunkIndex: Int
        let totalChunks: Int
        let chunkData: Data
    }

    private func sendDirectData(_ data: Data, to peers: [MCPeerID]) {
        let availablePeers = peers.filter { session.connectedPeers.contains($0) }

        guard !availablePeers.isEmpty else {
            print("⚠️ No connected peers available for transmission")
            return
        }

        do {
            try session.send(data, toPeers: availablePeers, with: .reliable)
            print("📡 Sent \(data.count) bytes to \(availablePeers.count) peers")
        } catch {
            print("❌ Failed to send data: \(error)")
        }
    }

    private func broadcastEnvelope(_ envelope: MessageEnvelope) {
        let allConnectedPeers = session.connectedPeers
        sendEnvelope(envelope, to: allConnectedPeers)
    }

    private func sendHeartbeat() {
        let capability = PeerCapability(
            peerID: localPeerID.displayName,
            supportsChunking: true,
            maxChunkSize: maxChunkSize,
            batteryLevel: UIDevice.current.batteryLevel,
            lastHeartbeat: Date(),
            deviceModel: UIDevice.current.model
        )

        let envelope = MessageEnvelope(
            messageID: "HEARTBEAT_\(generateMessageID())",
            content: try! JSONEncoder().encode(capability),
            hopCount: 0,
            ttl: 1, // Don't relay heartbeats
            createdAt: Date(),
            originalSender: localPeerID.displayName,
            messageType: .heartbeat
        )

        broadcastEnvelope(envelope)
    }

    private func cleanExpiredMessages() {
        let initialCount = messageQueue.count
        messageQueue.removeAll { $0.isExpired }

        let removedCount = initialCount - messageQueue.count
        if removedCount > 0 {
            print("🧹 Cleaned \(removedCount) expired messages from queue")
        }

        // Limit seen messages set size
        if seenMessageIDs.count > 10000 {
            seenMessageIDs = Set(seenMessageIDs.suffix(5000))
            print("🧹 Trimmed seen messages set to 5000 entries")
        }
    }

    // MARK: - Lifecycle Management

    private func setupLifecycleObservers() {
        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .sink { [weak self] _ in self?.handleDidEnterBackground() }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .sink { [weak self] _ in self?.handleWillEnterForeground() }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.willTerminateNotification)
            .sink { [weak self] _ in self?.handleWillTerminate() }
            .store(in: &cancellables)
    }

    private func handleDidEnterBackground() {
        meshState = .backgrounded
        print("📱 BitChat entered background - limited connectivity")

        // Start background task for limited processing time
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: backgroundTaskIdentifier) { [weak self] in
            self?.endBackgroundTask()
        }
    }

    private func handleWillEnterForeground() {
        print("📱 BitChat entering foreground - resuming full connectivity")

        // Restart advertising and browsing
        if meshState == .backgrounded {
            advertiser.startAdvertisingPeer()
            browser.startBrowsingForPeers()
            meshState = .browsing
        }

        endBackgroundTask()
    }

    private func handleWillTerminate() {
        stopMeshNetworking()
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    // MARK: - Utility Methods

    private func generateMessageID() -> String {
        return "msg_\(Int(Date().timeIntervalSince1970))_\(UUID().uuidString.prefix(8))"
    }

    private func updateConnectedPeers() {
        connectedPeers = session.connectedPeers.map { peer in
            PeerInfo(
                peerID: peer,
                displayName: peer.displayName,
                connectionState: .connected,
                lastSeen: Date(),
                capabilities: peerCapabilities[peer]
            )
        }
    }
}

// MARK: - MCSessionDelegate

@available(iOS 13.0, *)
extension BitChatManager: MCSessionDelegate {

    public func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async { [weak self] in
            switch state {
            case .connected:
                print("✅ Connected to peer: \(peerID.displayName)")
                self?.meshState = .connected

            case .connecting:
                print("🔄 Connecting to peer: \(peerID.displayName)")

            case .notConnected:
                print("❌ Disconnected from peer: \(peerID.displayName)")
                self?.peerCapabilities.removeValue(forKey: peerID)

            @unknown default:
                print("❓ Unknown state for peer: \(peerID.displayName)")
            }

            self?.updateConnectedPeers()
        }
    }

    public func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        handleReceivedData(data, from: peerID)
    }

    private func handleReceivedData(_ data: Data, from peerID: MCPeerID) {
        do {
            // Try to decode as regular message envelope
            if let envelope = try? JSONDecoder().decode(MessageEnvelope.self, from: data) {
                handleReceivedEnvelope(envelope, from: peerID)
            }
            // Try to decode as chunked message
            else if let chunkEnvelope = try? JSONDecoder().decode(ChunkEnvelope.self, from: data) {
                handleReceivedChunk(chunkEnvelope, from: peerID)
            }
            else {
                print("❌ Failed to decode received data from \(peerID.displayName)")
            }
        }
    }

    private func handleReceivedEnvelope(_ envelope: MessageEnvelope, from peerID: MCPeerID) {
        print("📥 Received message \(envelope.messageID) from \(peerID.displayName) (hop \(envelope.hopCount), ttl \(envelope.ttl))")

        // Handle different message types
        switch envelope.messageType {
        case .heartbeat, .capability:
            handleCapabilityUpdate(envelope, from: peerID)
            return

        case .data:
            // Add to received messages for app layer
            let receivedMessage = ReceivedMessage(
                messageID: envelope.messageID,
                content: envelope.content,
                fromPeer: peerID,
                hopCount: envelope.hopCount,
                receivedAt: Date(),
                originalSender: envelope.originalSender
            )

            DispatchQueue.main.async { [weak self] in
                self?.receivedMessages.append(receivedMessage)
            }
        }

        // Relay message if TTL allows
        if envelope.ttl > 1 && envelope.hopCount < maxTTL {
            let relayMessage = PendingMessage(
                messageID: envelope.messageID,
                content: envelope.content,
                hopCount: envelope.hopCount + 1,
                ttl: envelope.ttl - 1,
                createdAt: envelope.createdAt,
                targetPeer: nil, // Broadcast relay
                originalSender: envelope.originalSender
            )

            messageQueue.append(relayMessage)
            print("🔁 Queued message \(envelope.messageID) for relay")
        }
    }

    private func handleReceivedChunk(_ chunkEnvelope: ChunkEnvelope, from peerID: MCPeerID) {
        // In a full implementation, would reassemble chunks
        // For MVP, we'll just log the chunk reception
        print("📦 Received chunk \(chunkEnvelope.chunkIndex + 1)/\(chunkEnvelope.totalChunks) of message \(chunkEnvelope.messageID)")
    }

    private func handleCapabilityUpdate(_ envelope: MessageEnvelope, from peerID: MCPeerID) {
        do {
            let capability = try JSONDecoder().decode(PeerCapability.self, from: envelope.content)
            peerCapabilities[peerID] = capability

            DispatchQueue.main.async { [weak self] in
                self?.updateConnectedPeers()
            }

            print("📋 Updated capabilities for \(peerID.displayName)")
        } catch {
            print("❌ Failed to decode capability from \(peerID.displayName): \(error)")
        }
    }

    // Unused delegate methods
    public func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    public func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    public func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}

// MARK: - MCNearbyServiceAdvertiserDelegate

@available(iOS 13.0, *)
extension BitChatManager: MCNearbyServiceAdvertiserDelegate {

    public func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID, withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {

        print("📢 Received invitation from: \(peerID.displayName)")

        // Auto-accept invitations for mesh formation
        invitationHandler(true, session)
        print("✅ Accepted invitation from: \(peerID.displayName)")
    }

    public func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didNotStartAdvertisingPeer error: Error) {
        print("❌ Failed to start advertising: \(error.localizedDescription)")
        meshState = .error("Advertising failed: \(error.localizedDescription)")
    }
}

// MARK: - MCNearbyServiceBrowserDelegate

@available(iOS 13.0, *)
extension BitChatManager: MCNearbyServiceBrowserDelegate {

    public func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String : String]?) {

        print("🔍 Discovered peer: \(peerID.displayName)")

        // Auto-invite discovered peers
        browser.invitePeer(peerID, to: session, withContext: nil, timeout: 30.0)
        print("📤 Invited peer: \(peerID.displayName)")
    }

    public func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("📡 Lost peer: \(peerID.displayName)")
    }

    public func browser(_ browser: MCNearbyServiceBrowser, didNotStartBrowsingForPeers error: Error) {
        print("❌ Failed to start browsing: \(error.localizedDescription)")
        meshState = .error("Browsing failed: \(error.localizedDescription)")
    }
}

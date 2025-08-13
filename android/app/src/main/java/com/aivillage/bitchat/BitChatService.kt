package com.aivillage.bitchat

import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.bluetooth.le.*
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.os.ParcelUuid
import android.util.Log
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.connection.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.nio.ByteBuffer
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.LinkedBlockingQueue

/**
 * BitChat Mesh Network Service
 *
 * Implements local mesh networking with:
 * - Nearby Connections for discovery + Wi-Fi/BT upgrades (P2P_CLUSTER strategy)
 * - BLE GATT for low-bandwidth beacons/heartbeats
 * - Store-and-forward queue with TTL and deduplication
 * - 7-hop message relay with TTL expiry protection
 */
class BitChatService : Service() {

    companion object {
        private const val TAG = "BitChatService"
        private const val SERVICE_ID = "aivillage_bitchat_v1"
        private const val BLE_SERVICE_UUID = "12345678-1234-5678-9012-123456789ABC"
        private const val MAX_TTL = 7
        private const val MESSAGE_EXPIRY_MS = 5 * 60 * 1000L // 5 minutes
        private const val BEACON_INTERVAL_MS = 30_000L // 30 seconds
        private const val MAX_CHUNK_SIZE = 512 // Bytes for BLE compatibility
    }

    // Service components
    private lateinit var nearbyConnectionsClient: ConnectionsClient
    private lateinit var bluetoothAdapter: BluetoothAdapter
    private lateinit var bleAdvertiser: BluetoothLeAdvertiser
    private lateinit var bleScanner: BluetoothLeScanner

    // Coroutine management
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val _meshState = MutableStateFlow(MeshState.STOPPED)
    val meshState: StateFlow<MeshState> = _meshState

    // Message management
    private val messageQueue = LinkedBlockingQueue<PendingMessage>()
    private val seenMessages = Collections.synchronizedSet(mutableSetOf<String>())
    private val connectedPeers = ConcurrentHashMap<String, PeerConnection>()
    private val peerCapabilities = ConcurrentHashMap<String, PeerCapability>()

    // Local peer info
    private val localPeerId = generatePeerId()
    private var isAdvertising = false
    private var isDiscovering = false

    enum class MeshState {
        STOPPED, STARTING, RUNNING, ERROR
    }

    data class PendingMessage(
        val messageId: String,
        val payload: ByteArray,
        val hopCount: Int,
        val ttl: Int,
        val createdAt: Long,
        val targetPeer: String? = null
    )

    data class PeerConnection(
        val endpointId: String,
        val connectionType: ConnectionType,
        val lastSeen: Long,
        var isHealthy: Boolean = true
    )

    enum class ConnectionType {
        BLE_BEACON, NEARBY_WIFI, NEARBY_BLUETOOTH
    }

    data class PeerCapability(
        val peerId: String,
        val supportsNearby: Boolean,
        val supportsBle: Boolean,
        val batteryLevel: Int,
        val lastHeartbeat: Long
    )

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "BitChat service created with peer ID: $localPeerId")
        initializeComponents()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "START_MESH" -> startMeshNetworking()
            "STOP_MESH" -> stopMeshNetworking()
            "SEND_MESSAGE" -> {
                val message = intent.getStringExtra("message")
                val target = intent.getStringExtra("target")
                if (message != null) {
                    sendMessage(message.toByteArray(), target)
                }
            }
        }
        return START_STICKY
    }

    private fun initializeComponents() {
        try {
            // Initialize Nearby Connections
            nearbyConnectionsClient = Nearby.getConnectionsClient(this)

            // Initialize Bluetooth components
            val bluetoothManager = getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
            bluetoothAdapter = bluetoothManager.adapter
            bleAdvertiser = bluetoothAdapter.bluetoothLeAdvertiser
            bleScanner = bluetoothAdapter.bluetoothLeScanner

            Log.i(TAG, "BitChat components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize BitChat components", e)
            _meshState.value = MeshState.ERROR
        }
    }

    private fun startMeshNetworking() {
        if (_meshState.value == MeshState.RUNNING) return

        serviceScope.launch {
            try {
                _meshState.value = MeshState.STARTING

                // Start discovery and advertising in parallel
                launch { startNearbyDiscovery() }
                launch { startNearbyAdvertising() }
                launch { startBleBeacons() }
                launch { processMessageQueue() }

                _meshState.value = MeshState.RUNNING
                Log.i(TAG, "BitChat mesh networking started")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to start mesh networking", e)
                _meshState.value = MeshState.ERROR
            }
        }
    }

    private fun stopMeshNetworking() {
        serviceScope.launch {
            try {
                stopNearbyOperations()
                stopBleOperations()
                connectedPeers.clear()
                seenMessages.clear()

                _meshState.value = MeshState.STOPPED
                Log.i(TAG, "BitChat mesh networking stopped")

            } catch (e: Exception) {
                Log.e(TAG, "Error stopping mesh networking", e)
            }
        }
    }

    // Nearby Connections Implementation
    private suspend fun startNearbyDiscovery() {
        val discoveryOptions = DiscoveryOptions.Builder()
            .setStrategy(Strategy.P2P_CLUSTER)
            .build()

        nearbyConnectionsClient.startDiscovery(
            SERVICE_ID,
            endpointDiscoveryCallback,
            discoveryOptions
        ).addOnSuccessListener {
            isDiscovering = true
            Log.i(TAG, "Nearby discovery started")
        }.addOnFailureListener { e ->
            Log.e(TAG, "Failed to start nearby discovery", e)
        }
    }

    private suspend fun startNearbyAdvertising() {
        val advertisingOptions = AdvertisingOptions.Builder()
            .setStrategy(Strategy.P2P_CLUSTER)
            .build()

        nearbyConnectionsClient.startAdvertising(
            localPeerId,
            SERVICE_ID,
            connectionLifecycleCallback,
            advertisingOptions
        ).addOnSuccessListener {
            isAdvertising = true
            Log.i(TAG, "Nearby advertising started")
        }.addOnFailureListener { e ->
            Log.e(TAG, "Failed to start nearby advertising", e)
        }
    }

    private val endpointDiscoveryCallback = object : EndpointDiscoveryCallback() {
        override fun onEndpointFound(endpointId: String, info: DiscoveredEndpointInfo) {
            Log.i(TAG, "Discovered peer: $endpointId (${info.endpointName})")

            // Request connection for mesh formation
            nearbyConnectionsClient.requestConnection(
                localPeerId,
                endpointId,
                connectionLifecycleCallback
            ).addOnSuccessListener {
                Log.i(TAG, "Connection requested to $endpointId")
            }.addOnFailureListener { e ->
                Log.w(TAG, "Failed to request connection to $endpointId", e)
            }
        }

        override fun onEndpointLost(endpointId: String) {
            Log.i(TAG, "Lost peer: $endpointId")
            connectedPeers.remove(endpointId)
        }
    }

    private val connectionLifecycleCallback = object : ConnectionLifecycleCallback() {
        override fun onConnectionInitiated(endpointId: String, connectionInfo: ConnectionInfo) {
            Log.i(TAG, "Connection initiated with $endpointId")

            // Auto-accept connections for mesh formation
            nearbyConnectionsClient.acceptConnection(endpointId, payloadCallback)
        }

        override fun onConnectionResult(endpointId: String, result: ConnectionResolution) {
            when (result.status.statusCode) {
                ConnectionsStatusCodes.STATUS_OK -> {
                    Log.i(TAG, "Connected to peer: $endpointId")

                    val peerConnection = PeerConnection(
                        endpointId = endpointId,
                        connectionType = ConnectionType.NEARBY_WIFI, // Will be determined by strategy
                        lastSeen = System.currentTimeMillis()
                    )
                    connectedPeers[endpointId] = peerConnection

                    // Send capability exchange
                    sendCapabilityInfo(endpointId)
                }
                else -> {
                    Log.w(TAG, "Connection failed to $endpointId: ${result.status}")
                }
            }
        }

        override fun onDisconnected(endpointId: String) {
            Log.i(TAG, "Disconnected from peer: $endpointId")
            connectedPeers.remove(endpointId)
        }
    }

    private val payloadCallback = object : PayloadCallback() {
        override fun onPayloadReceived(endpointId: String, payload: Payload) {
            when (payload.type) {
                Payload.Type.BYTES -> {
                    payload.asBytes()?.let { data ->
                        handleReceivedMessage(endpointId, data)
                    }
                }
                else -> {
                    Log.w(TAG, "Received unsupported payload type from $endpointId")
                }
            }
        }

        override fun onPayloadTransferUpdate(endpointId: String, update: PayloadTransferUpdate) {
            if (update.status == PayloadTransferUpdate.Status.SUCCESS) {
                Log.d(TAG, "Payload transfer completed to $endpointId")
            } else if (update.status == PayloadTransferUpdate.Status.FAILURE) {
                Log.w(TAG, "Payload transfer failed to $endpointId")
            }
        }
    }

    // BLE Beacon Implementation
    private suspend fun startBleBeacons() {
        if (!bluetoothAdapter.isEnabled) {
            Log.w(TAG, "Bluetooth not enabled, skipping BLE beacons")
            return
        }

        // Start BLE advertising for discovery
        val settings = AdvertiseSettings.Builder()
            .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_BALANCED)
            .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM)
            .setConnectable(false)
            .build()

        val data = AdvertiseData.Builder()
            .setIncludeDeviceName(false)
            .addServiceUuid(ParcelUuid.fromString(BLE_SERVICE_UUID))
            .addServiceData(
                ParcelUuid.fromString(BLE_SERVICE_UUID),
                localPeerId.toByteArray()
            )
            .build()

        bleAdvertiser.startAdvertising(settings, data, bleAdvertiseCallback)

        // Start BLE scanning
        val scanSettings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_BALANCED)
            .build()

        val filters = listOf(
            ScanFilter.Builder()
                .setServiceUuid(ParcelUuid.fromString(BLE_SERVICE_UUID))
                .build()
        )

        bleScanner.startScan(filters, scanSettings, bleScanCallback)
    }

    private val bleAdvertiseCallback = object : AdvertiseCallback() {
        override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
            Log.i(TAG, "BLE advertising started successfully")
        }

        override fun onStartFailure(errorCode: Int) {
            Log.e(TAG, "BLE advertising failed with error: $errorCode")
        }
    }

    private val bleScanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val serviceData = result.scanRecord?.getServiceData(
                ParcelUuid.fromString(BLE_SERVICE_UUID)
            )

            serviceData?.let { data ->
                val peerId = String(data)
                if (peerId != localPeerId && !connectedPeers.containsKey(peerId)) {
                    Log.i(TAG, "Discovered BLE peer: $peerId")

                    // For BLE, we add as beacon-only connection
                    val peerConnection = PeerConnection(
                        endpointId = peerId,
                        connectionType = ConnectionType.BLE_BEACON,
                        lastSeen = System.currentTimeMillis()
                    )
                    connectedPeers[peerId] = peerConnection
                }
            }
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "BLE scan failed with error: $errorCode")
        }
    }

    // Message Processing
    private suspend fun processMessageQueue() {
        while (_meshState.value == MeshState.RUNNING) {
            try {
                val message = withTimeoutOrNull(1000) {
                    messageQueue.take()
                } ?: continue

                // Check TTL expiry
                if (message.ttl <= 0 ||
                    System.currentTimeMillis() - message.createdAt > MESSAGE_EXPIRY_MS) {
                    Log.d(TAG, "Message ${message.messageId} expired, dropping")
                    continue
                }

                // Check if already seen (deduplication)
                if (seenMessages.contains(message.messageId)) {
                    Log.d(TAG, "Message ${message.messageId} already seen, dropping")
                    continue
                }

                seenMessages.add(message.messageId)

                // Relay to appropriate peers
                relayMessage(message)

            } catch (e: Exception) {
                Log.e(TAG, "Error processing message queue", e)
                delay(1000) // Brief pause on error
            }
        }
    }

    private suspend fun relayMessage(message: PendingMessage) {
        val envelope = createMessageEnvelope(
            messageId = message.messageId,
            payload = message.payload,
            hopCount = message.hopCount + 1,
            ttl = message.ttl - 1
        )

        val targetPeers = if (message.targetPeer != null) {
            listOfNotNull(connectedPeers[message.targetPeer])
        } else {
            // Broadcast to all connected peers
            connectedPeers.values.filter { it.isHealthy }
        }

        for (peer in targetPeers) {
            try {
                when (peer.connectionType) {
                    ConnectionType.NEARBY_WIFI, ConnectionType.NEARBY_BLUETOOTH -> {
                        val payload = Payload.fromBytes(envelope)
                        nearbyConnectionsClient.sendPayload(peer.endpointId, payload)
                    }
                    ConnectionType.BLE_BEACON -> {
                        // For BLE, we would need GATT connection for larger messages
                        // For now, just log - BLE is primarily for discovery
                        Log.d(TAG, "BLE peer ${peer.endpointId} discovered but no data path")
                    }
                }

                Log.d(TAG, "Relayed message ${message.messageId} to ${peer.endpointId}")

            } catch (e: Exception) {
                Log.w(TAG, "Failed to relay message to ${peer.endpointId}", e)
                peer.isHealthy = false
            }
        }
    }

    private fun handleReceivedMessage(fromPeer: String, data: ByteArray) {
        try {
            val envelope = parseMessageEnvelope(data)

            Log.i(TAG, "Received message ${envelope.messageId} from $fromPeer " +
                      "(hop ${envelope.hopCount}/${MAX_TTL}, ttl ${envelope.ttl})")

            // Update peer last seen
            connectedPeers[fromPeer]?.let { peer ->
                connectedPeers[fromPeer] = peer.copy(lastSeen = System.currentTimeMillis())
            }

            // Check if this is a capability exchange message
            if (envelope.messageId.startsWith("CAP_")) {
                handleCapabilityExchange(fromPeer, envelope.payload)
                return
            }

            // Add to message queue for processing/relay
            val pendingMessage = PendingMessage(
                messageId = envelope.messageId,
                payload = envelope.payload,
                hopCount = envelope.hopCount,
                ttl = envelope.ttl,
                createdAt = envelope.createdAt
            )

            messageQueue.offer(pendingMessage)

            // Notify application layer of received message
            broadcastMessageReceived(envelope)

        } catch (e: Exception) {
            Log.e(TAG, "Error handling received message from $fromPeer", e)
        }
    }

    // Public API Methods
    fun sendMessage(payload: ByteArray, targetPeer: String? = null) {
        val messageId = generateMessageId()
        val message = PendingMessage(
            messageId = messageId,
            payload = payload,
            hopCount = 0,
            ttl = MAX_TTL,
            createdAt = System.currentTimeMillis(),
            targetPeer = targetPeer
        )

        messageQueue.offer(message)
        Log.i(TAG, "Queued message $messageId for transmission")
    }

    fun getConnectedPeers(): List<PeerConnection> = connectedPeers.values.toList()

    fun getPeerCapabilities(): List<PeerCapability> = peerCapabilities.values.toList()

    // Helper Methods
    private fun sendCapabilityInfo(endpointId: String) {
        val capability = PeerCapability(
            peerId = localPeerId,
            supportsNearby = true,
            supportsBle = bluetoothAdapter.isEnabled,
            batteryLevel = getBatteryLevel(),
            lastHeartbeat = System.currentTimeMillis()
        )

        val capabilityData = serializeCapability(capability)
        val capabilityMessage = PendingMessage(
            messageId = "CAP_${generateMessageId()}",
            payload = capabilityData,
            hopCount = 0,
            ttl = 1, // Don't relay capability messages
            createdAt = System.currentTimeMillis(),
            targetPeer = endpointId
        )

        messageQueue.offer(capabilityMessage)
    }

    private fun handleCapabilityExchange(fromPeer: String, data: ByteArray) {
        try {
            val capability = deserializeCapability(data)
            peerCapabilities[fromPeer] = capability
            Log.i(TAG, "Updated capabilities for peer $fromPeer")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse capability data from $fromPeer", e)
        }
    }

    private fun stopNearbyOperations() {
        if (isDiscovering) {
            nearbyConnectionsClient.stopDiscovery()
            isDiscovering = false
        }
        if (isAdvertising) {
            nearbyConnectionsClient.stopAdvertising()
            isAdvertising = false
        }
        nearbyConnectionsClient.stopAllEndpoints()
    }

    private fun stopBleOperations() {
        if (::bleAdvertiser.isInitialized) {
            bleAdvertiser.stopAdvertising(bleAdvertiseCallback)
        }
        if (::bleScanner.isInitialized) {
            bleScanner.stopScan(bleScanCallback)
        }
    }

    private fun broadcastMessageReceived(envelope: MessageEnvelope) {
        val intent = Intent("com.aivillage.bitchat.MESSAGE_RECEIVED").apply {
            putExtra("messageId", envelope.messageId)
            putExtra("payload", envelope.payload)
            putExtra("hopCount", envelope.hopCount)
            putExtra("ttl", envelope.ttl)
        }
        sendBroadcast(intent)
    }

    // Utility functions
    private fun generatePeerId(): String = "peer_${UUID.randomUUID().toString().take(8)}"
    private fun generateMessageId(): String = "msg_${System.currentTimeMillis()}_${UUID.randomUUID().toString().take(8)}"

    private fun getBatteryLevel(): Int {
        // Simplified battery level - in real implementation would use BatteryManager
        return 80
    }

    // Protocol message structures and serialization
    data class MessageEnvelope(
        val messageId: String,
        val createdAt: Long,
        val hopCount: Int,
        val ttl: Int,
        val payload: ByteArray
    )

    private fun createMessageEnvelope(
        messageId: String,
        payload: ByteArray,
        hopCount: Int,
        ttl: Int
    ): ByteArray {
        val createdAt = System.currentTimeMillis()

        // Simple binary format: [msgId_len][msgId][createdAt][hopCount][ttl][payload_len][payload]
        val msgIdBytes = messageId.toByteArray()
        val buffer = ByteBuffer.allocate(4 + msgIdBytes.size + 8 + 4 + 4 + 4 + payload.size)

        buffer.putInt(msgIdBytes.size)
        buffer.put(msgIdBytes)
        buffer.putLong(createdAt)
        buffer.putInt(hopCount)
        buffer.putInt(ttl)
        buffer.putInt(payload.size)
        buffer.put(payload)

        return buffer.array()
    }

    private fun parseMessageEnvelope(data: ByteArray): MessageEnvelope {
        val buffer = ByteBuffer.wrap(data)

        val msgIdLen = buffer.int
        val msgIdBytes = ByteArray(msgIdLen)
        buffer.get(msgIdBytes)
        val messageId = String(msgIdBytes)

        val createdAt = buffer.long
        val hopCount = buffer.int
        val ttl = buffer.int

        val payloadLen = buffer.int
        val payload = ByteArray(payloadLen)
        buffer.get(payload)

        return MessageEnvelope(messageId, createdAt, hopCount, ttl, payload)
    }

    private fun serializeCapability(capability: PeerCapability): ByteArray {
        // Simple JSON serialization for capabilities
        val json = """
            {
                "peerId": "${capability.peerId}",
                "supportsNearby": ${capability.supportsNearby},
                "supportsBle": ${capability.supportsBle},
                "batteryLevel": ${capability.batteryLevel},
                "lastHeartbeat": ${capability.lastHeartbeat}
            }
        """.trimIndent()
        return json.toByteArray()
    }

    private fun deserializeCapability(data: ByteArray): PeerCapability {
        // Simple JSON parsing - in production would use proper JSON library
        val json = String(data)
        val peerId = json.substringAfter("\"peerId\": \"").substringBefore("\"")
        val supportsNearby = json.contains("\"supportsNearby\": true")
        val supportsBle = json.contains("\"supportsBle\": true")
        val batteryLevel = json.substringAfter("\"batteryLevel\": ").substringBefore(",").toInt()
        val lastHeartbeat = json.substringAfter("\"lastHeartbeat\": ").substringBefore("}").toLong()

        return PeerCapability(peerId, supportsNearby, supportsBle, batteryLevel, lastHeartbeat)
    }

    override fun onDestroy() {
        serviceScope.cancel()
        stopMeshNetworking()
        super.onDestroy()
        Log.i(TAG, "BitChat service destroyed")
    }
}

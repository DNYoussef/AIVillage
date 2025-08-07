package ai.atlantis.aivillage.mesh

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.UUID
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import android.util.Log

/**
 * LibP2P Mesh Network Service for Android
 *
 * Replaces the broken Bluetooth-based MeshNetwork.kt with a LibP2P implementation
 * that communicates with the Python LibP2P bridge via HTTP REST API and WebSocket.
 *
 * Features:
 * - Reliable P2P communication via LibP2P
 * - Peer discovery through mDNS
 * - DHT-based routing
 * - Transport agnostic (TCP, WebSocket, with Bluetooth/WiFi Direct fallback)
 * - Support for all message types: DATA_MESSAGE, AGENT_TASK, PARAMETER_UPDATE, GRADIENT_SHARING
 * - Real-time messaging via WebSocket
 * - Automatic peer discovery and connection management
 */

/**
 * Mesh network states
 */
sealed class MeshState {
    object IDLE : MeshState()
    object INITIALIZING : MeshState()
    object STARTING : MeshState()
    object ACTIVE : MeshState()
    object DISCOVERING : MeshState()
    object CONNECTED : MeshState()
    data class ERROR(val message: String) : MeshState()
}

/**
 * Message types supported by the mesh network
 */
enum class MessageType(val value: String) {
    DATA_MESSAGE("DATA_MESSAGE"),
    AGENT_TASK("AGENT_TASK"),
    PARAMETER_UPDATE("PARAMETER_UPDATE"),
    GRADIENT_SHARING("GRADIENT_SHARING"),
    PEER_DISCOVERY("PEER_DISCOVERY"),
    HEARTBEAT("HEARTBEAT")
}

/**
 * Represents a node in the LibP2P mesh network
 */
@Serializable
data class MeshNode(
    val nodeId: String,
    val addresses: List<String>,
    val port: Int,
    val capabilities: Map<String, String> = emptyMap(),
    val lastSeen: Long = System.currentTimeMillis(),
    val latencyMs: Float = 0.0f,
    val trustScore: Float = 0.5f
) {
    /**
     * Check if node supports evolution tasks
     */
    fun isEvolutionCapable(): Boolean {
        return capabilities["can_evolve"]?.toBoolean() ?: true &&
               capabilities["available_for_evolution"]?.toBoolean() ?: true
    }

    /**
     * Get resource capacity
     */
    fun getResourceCapacity(): Float {
        return capabilities["evolution_capacity"]?.toFloatOrNull() ?: 1.0f
    }
}

/**
 * Mesh network message
 */
@Serializable
data class MeshMessage(
    val id: String = UUID.randomUUID().toString(),
    val type: String,
    val sender: String,
    val recipient: String? = null, // null for broadcast
    val payload: String,
    val ttl: Int = 5,
    val timestamp: Long = System.currentTimeMillis(),
    val hopCount: Int = 0
)

/**
 * Bridge configuration
 */
data class BridgeConfiguration(
    val bridgePort: Int = 8080,
    val nodeId: String? = null,
    val listenPort: Int = 4001,
    val maxPeers: Int = 50,
    val transports: List<String> = listOf("tcp", "ws"),
    val fallbackTransports: List<String> = listOf("bluetooth", "wifi_direct"),
    val enableMDNS: Boolean = true,
    val enableDHT: Boolean = true
)

/**
 * LibP2P Mesh Network Service
 */
class LibP2PMeshService(
    private val context: Context,
    private val config: BridgeConfiguration = BridgeConfiguration()
) {
    private val TAG = "LibP2PMeshService"

    // State management
    private val _meshState = MutableStateFlow(MeshState.IDLE)
    val meshState: StateFlow<MeshState> = _meshState.asStateFlow()

    private val _connectedNodes = MutableStateFlow<Set<MeshNode>>(emptySet())
    val connectedNodes: StateFlow<Set<MeshNode>> = _connectedNodes.asStateFlow()

    private val _discoveredPeers = MutableStateFlow<List<MeshNode>>(emptyList())
    val discoveredPeers: StateFlow<List<MeshNode>> = _discoveredPeers.asStateFlow()

    // HTTP client for REST API communication
    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // WebSocket for real-time messaging
    private var webSocket: WebSocket? = null
    private val messageHandlers = mutableMapOf<MessageType, suspend (MeshMessage) -> Unit>()

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val coroutineScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // Bridge connection info
    private val bridgeBaseUrl = "http://localhost:${config.bridgePort}"
    private val bridgeWsUrl = "ws://localhost:${config.bridgePort}/ws"

    // Node information
    private var nodeId: String = config.nodeId ?: UUID.randomUUID().toString()
    private var bridgeStarted = false

    /**
     * Initialize the mesh network
     */
    fun initialize() {
        Log.i(TAG, "Initializing LibP2P Mesh Service")
        _meshState.value = MeshState.INITIALIZING

        // Start the Python bridge if not already running
        coroutineScope.launch {
            initializeBridge()
        }
    }

    /**
     * Start mesh discovery and networking
     */
    suspend fun startMesh(): Result<Unit> {
        return try {
            Log.i(TAG, "Starting LibP2P mesh network")
            _meshState.value = MeshState.STARTING

            // Ensure bridge is running
            if (!bridgeStarted) {
                initializeBridge()
                delay(2000) // Wait for bridge to start
            }

            // Start mesh network via bridge
            val startResult = startMeshNetwork()
            if (startResult.isFailure) {
                _meshState.value = MeshState.ERROR(startResult.exceptionOrNull()?.message ?: "Unknown error")
                return startResult
            }

            // Connect WebSocket for real-time messaging
            connectWebSocket()

            // Start background tasks
            startPeerDiscovery()
            startStatusUpdates()

            _meshState.value = MeshState.ACTIVE
            Log.i(TAG, "LibP2P mesh network started successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start mesh network", e)
            _meshState.value = MeshState.ERROR(e.message ?: "Unknown error")
            Result.failure(e)
        }
    }

    /**
     * Stop mesh networking
     */
    suspend fun stopMesh(): Result<Unit> {
        return try {
            Log.i(TAG, "Stopping LibP2P mesh network")

            // Disconnect WebSocket
            webSocket?.close(1000, "Service stopping")
            webSocket = null

            // Stop mesh network via bridge
            val request = Request.Builder()
                .url("$bridgeBaseUrl/mesh/stop")
                .post("{}".toRequestBody("application/json".toMediaType()))
                .build()

            val response = httpClient.newCall(request).execute()
            if (!response.isSuccessful) {
                throw IOException("Failed to stop mesh: ${response.code}")
            }

            _meshState.value = MeshState.IDLE
            _connectedNodes.value = emptySet()
            Log.i(TAG, "LibP2P mesh network stopped")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop mesh network", e)
            _meshState.value = MeshState.ERROR(e.message ?: "Unknown error")
            Result.failure(e)
        }
    }

    /**
     * Send message through mesh network
     */
    suspend fun sendMessage(message: MeshMessage): Result<String> {
        return try {
            if (_meshState.value !is MeshState.ACTIVE && _meshState.value !is MeshState.CONNECTED) {
                throw IllegalStateException("Mesh network not active")
            }

            val requestBody = json.encodeToString(message)

            val request = Request.Builder()
                .url("$bridgeBaseUrl/mesh/send")
                .post(requestBody.toRequestBody("application/json".toMediaType()))
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw IOException("Failed to send message: ${response.code} - $responseBody")
            }

            val result = json.decodeFromString<Map<String, Any>>(responseBody)
            val success = result["success"] as? Boolean ?: false

            if (success) {
                val messageId = result["message_id"] as? String ?: message.id
                Log.d(TAG, "Message sent successfully: $messageId")
                Result.success(messageId)
            } else {
                throw IOException("Message sending failed")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to send message", e)
            Result.failure(e)
        }
    }

    /**
     * Send message with specific type and payload
     */
    suspend fun sendMessage(
        type: MessageType,
        recipient: String? = null,
        payload: String,
        sender: String = nodeId
    ): Result<String> {
        val message = MeshMessage(
            type = type.value,
            sender = sender,
            recipient = recipient,
            payload = payload
        )
        return sendMessage(message)
    }

    /**
     * Broadcast message to all peers
     */
    suspend fun broadcastMessage(type: MessageType, payload: String): Result<String> {
        return sendMessage(type, null, payload)
    }

    /**
     * Register message handler for specific type
     */
    fun registerMessageHandler(type: MessageType, handler: suspend (MeshMessage) -> Unit) {
        messageHandlers[type] = handler
        Log.d(TAG, "Registered handler for message type: ${type.value}")
    }

    /**
     * Connect to specific peer
     */
    suspend fun connectToPeer(peerAddress: String): Result<Unit> {
        return try {
            val requestBody = json.encodeToString(mapOf("address" to peerAddress))

            val request = Request.Builder()
                .url("$bridgeBaseUrl/mesh/connect")
                .post(requestBody.toRequestBody("application/json".toMediaType()))
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw IOException("Failed to connect peer: ${response.code} - $responseBody")
            }

            val result = json.decodeFromString<Map<String, Any>>(responseBody)
            val success = result["success"] as? Boolean ?: false

            if (success) {
                Log.i(TAG, "Successfully connected to peer: $peerAddress")
                Result.success(Unit)
            } else {
                throw IOException("Peer connection failed")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to peer $peerAddress", e)
            Result.failure(e)
        }
    }

    /**
     * Get mesh network status
     */
    suspend fun getMeshStatus(): Result<Map<String, Any>> {
        return try {
            val request = Request.Builder()
                .url("$bridgeBaseUrl/mesh/status")
                .get()
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw IOException("Failed to get status: ${response.code}")
            }

            val status = json.decodeFromString<Map<String, Any>>(responseBody)
            Log.d(TAG, "Mesh status: $status")
            Result.success(status)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get mesh status", e)
            Result.failure(e)
        }
    }

    /**
     * Store value in DHT
     */
    suspend fun dhtStore(key: String, value: String): Result<Unit> {
        return try {
            val requestBody = json.encodeToString(mapOf("key" to key, "value" to value))

            val request = Request.Builder()
                .url("$bridgeBaseUrl/dht/store")
                .post(requestBody.toRequestBody("application/json".toMediaType()))
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw IOException("DHT store failed: ${response.code} - $responseBody")
            }

            val result = json.decodeFromString<Map<String, Any>>(responseBody)
            val success = result["success"] as? Boolean ?: false

            if (success) {
                Log.d(TAG, "DHT store successful: $key")
                Result.success(Unit)
            } else {
                throw IOException("DHT store operation failed")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to store in DHT", e)
            Result.failure(e)
        }
    }

    /**
     * Get value from DHT
     */
    suspend fun dhtGet(key: String): Result<String?> {
        return try {
            val request = Request.Builder()
                .url("$bridgeBaseUrl/dht/get/$key")
                .get()
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (response.code == 404) {
                // Key not found
                Result.success(null)
            } else if (!response.isSuccessful) {
                throw IOException("DHT get failed: ${response.code} - $responseBody")
            } else {
                val result = json.decodeFromString<Map<String, String>>(responseBody)
                val value = result["value"]
                Log.d(TAG, "DHT get successful: $key -> $value")
                Result.success(value)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get from DHT", e)
            Result.failure(e)
        }
    }

    // Private implementation methods

    private suspend fun initializeBridge() {
        // In a real implementation, this would start the Python bridge process
        // For now, assume the bridge is running externally
        // You could use ProcessBuilder to start the Python script
        bridgeStarted = true
        Log.i(TAG, "Bridge initialized (assuming external process)")
    }

    private suspend fun startMeshNetwork(): Result<Unit> {
        return try {
            val meshConfig = mapOf(
                "node_id" to nodeId,
                "listen_port" to config.listenPort,
                "max_peers" to config.maxPeers,
                "transports" to config.transports
            )

            val requestBody = json.encodeToString(meshConfig)

            val request = Request.Builder()
                .url("$bridgeBaseUrl/mesh/start")
                .post(requestBody.toRequestBody("application/json".toMediaType()))
                .build()

            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw IOException("Failed to start mesh: ${response.code} - $responseBody")
            }

            val result = json.decodeFromString<Map<String, Any>>(responseBody)
            val status = result["status"] as? String

            if (status == "started" || status == "already_running") {
                nodeId = result["node_id"] as? String ?: nodeId
                Log.i(TAG, "Mesh network started with node ID: $nodeId")
                Result.success(Unit)
            } else {
                throw IOException("Unexpected start response: $status")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start mesh network", e)
            Result.failure(e)
        }
    }

    private fun connectWebSocket() {
        val request = Request.Builder()
            .url(bridgeWsUrl)
            .build()

        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i(TAG, "WebSocket connected")

                // Send initial subscription
                val subscribeMessage = json.encodeToString(mapOf(
                    "type" to "subscribe",
                    "message_types" to MessageType.values().map { it.value }
                ))
                webSocket.send(subscribeMessage)
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                coroutineScope.launch {
                    handleWebSocketMessage(text)
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i(TAG, "WebSocket closed: $code - $reason")
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket error", t)

                // Attempt to reconnect after delay
                coroutineScope.launch {
                    delay(5000)
                    if (_meshState.value is MeshState.ACTIVE) {
                        connectWebSocket()
                    }
                }
            }
        })
    }

    private suspend fun handleWebSocketMessage(text: String) {
        try {
            val messageData = json.decodeFromString<Map<String, Any>>(text)
            val type = messageData["type"] as? String

            when (type) {
                "mesh_message" -> {
                    val meshMessage = MeshMessage(
                        id = messageData["message_id"] as? String ?: UUID.randomUUID().toString(),
                        type = messageData["message_type"] as? String ?: "DATA_MESSAGE",
                        sender = messageData["sender"] as? String ?: "",
                        recipient = messageData["recipient"] as? String,
                        payload = messageData["payload"] as? String ?: "",
                        timestamp = (messageData["timestamp"] as? Double)?.toLong() ?: System.currentTimeMillis(),
                        hopCount = (messageData["hop_count"] as? Double)?.toInt() ?: 0
                    )

                    // Route to appropriate handler
                    val messageType = MessageType.values().find { it.value == meshMessage.type }
                    if (messageType != null && messageHandlers.containsKey(messageType)) {
                        messageHandlers[messageType]?.invoke(meshMessage)
                    }

                    Log.d(TAG, "Received mesh message: ${meshMessage.type} from ${meshMessage.sender}")
                }

                "pong" -> {
                    Log.d(TAG, "Received pong from bridge")
                }

                "error" -> {
                    val error = messageData["message"] as? String ?: "Unknown error"
                    Log.e(TAG, "Bridge error: $error")
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to handle WebSocket message", e)
        }
    }

    private fun startPeerDiscovery() {
        coroutineScope.launch {
            while (_meshState.value is MeshState.ACTIVE) {
                try {
                    // Get current peers
                    val request = Request.Builder()
                        .url("$bridgeBaseUrl/mesh/peers")
                        .get()
                        .build()

                    val response = httpClient.newCall(request).execute()
                    val responseBody = response.body?.string() ?: ""

                    if (response.isSuccessful) {
                        val peersData = json.decodeFromString<Map<String, Any>>(responseBody)
                        val peersList = peersData["peers"] as? List<Map<String, Any>> ?: emptyList()

                        val nodes = peersList.map { peerData ->
                            val capabilities = peerData["capabilities"] as? Map<String, Any> ?: emptyMap()
                            MeshNode(
                                nodeId = peerData["peer_id"] as? String ?: "",
                                addresses = listOf("unknown"), // Would be populated from peer data
                                port = capabilities["listen_port"] as? Int ?: 4001,
                                capabilities = capabilities.mapValues { it.value.toString() },
                                lastSeen = (capabilities["last_seen"] as? Double)?.toLong() ?: System.currentTimeMillis()
                            )
                        }.toSet()

                        _connectedNodes.value = nodes

                        if (nodes.isNotEmpty() && _meshState.value is MeshState.ACTIVE) {
                            _meshState.value = MeshState.CONNECTED
                        }
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Peer discovery error", e)
                }

                delay(30000) // Update every 30 seconds
            }
        }
    }

    private fun startStatusUpdates() {
        coroutineScope.launch {
            while (_meshState.value is MeshState.ACTIVE || _meshState.value is MeshState.CONNECTED) {
                try {
                    // Send periodic ping to keep connection alive
                    webSocket?.send(json.encodeToString(mapOf("type" to "ping")))

                } catch (e: Exception) {
                    Log.e(TAG, "Status update error", e)
                }

                delay(30000) // Ping every 30 seconds
            }
        }
    }

    /**
     * Get current node ID
     */
    fun getNodeId(): String = nodeId

    /**
     * Get current peer count
     */
    fun getPeerCount(): Int = _connectedNodes.value.size

    /**
     * Check if mesh is active
     */
    fun isActive(): Boolean = _meshState.value is MeshState.ACTIVE || _meshState.value is MeshState.CONNECTED

    /**
     * Get evolution-capable peers
     */
    fun getEvolutionCapablePeers(): List<MeshNode> {
        return _connectedNodes.value.filter { it.isEvolutionCapable() }
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        Log.i(TAG, "Cleaning up LibP2P Mesh Service")

        coroutineScope.cancel()
        webSocket?.close(1000, "Service cleanup")
        httpClient.dispatcher.executorService.shutdown()

        _meshState.value = MeshState.IDLE
        _connectedNodes.value = emptySet()
    }
}

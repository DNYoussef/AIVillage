package ai.atlantis.aivillage.core

/**
 * SDK Configuration
 */
data class Configuration(
    val deviceId: String,
    val meshConfig: MeshConfiguration = MeshConfiguration(),
    val agentConfig: AgentConfiguration = AgentConfiguration(),
    val flConfig: FLConfiguration = FLConfiguration(),
    val autoStart: Boolean = true,
    val enableLogging: Boolean = true,
    val dataDirectory: String = "aivillage",
    val maxMemoryMB: Int = 512,
    val maxCpuPercent: Int = 50
) {

    class Builder {
        private var deviceId: String = ""
        private var meshConfig = MeshConfiguration()
        private var agentConfig = AgentConfiguration()
        private var flConfig = FLConfiguration()
        private var autoStart = true
        private var enableLogging = true
        private var dataDirectory = "aivillage"
        private var maxMemoryMB = 512
        private var maxCpuPercent = 50

        fun deviceId(id: String) = apply { deviceId = id }
        fun meshConfig(config: MeshConfiguration) = apply { meshConfig = config }
        fun agentConfig(config: AgentConfiguration) = apply { agentConfig = config }
        fun flConfig(config: FLConfiguration) = apply { flConfig = config }
        fun autoStart(enabled: Boolean) = apply { autoStart = enabled }
        fun enableLogging(enabled: Boolean) = apply { enableLogging = enabled }
        fun dataDirectory(dir: String) = apply { dataDirectory = dir }
        fun maxMemoryMB(mb: Int) = apply { maxMemoryMB = mb }
        fun maxCpuPercent(percent: Int) = apply { maxCpuPercent = percent }

        fun build(): Configuration {
            require(deviceId.isNotEmpty()) { "Device ID must be set" }
            return Configuration(
                deviceId, meshConfig, agentConfig, flConfig,
                autoStart, enableLogging, dataDirectory, maxMemoryMB, maxCpuPercent
            )
        }
    }
}

/**
 * Mesh network configuration
 */
data class MeshConfiguration(
    val nodeId: String = java.util.UUID.randomUUID().toString(),
    val maxConnections: Int = 8,
    val discoveryTimeout: Long = 30000,
    val messageTimeout: Long = 5000,
    val enableEncryption: Boolean = true
)

/**
 * Agent configuration
 */
data class AgentConfiguration(
    val enabledAgents: Set<String> = setOf("translator", "classifier"),
    val maxConcurrentTasks: Int = 5,
    val taskTimeout: Long = 30000,
    val modelUpdateInterval: Long = 86400000 // 24 hours
)

/**
 * Federated Learning configuration
 */
data class FLConfiguration(
    val clientId: String = java.util.UUID.randomUUID().toString(),
    val serverUrl: String = "https://fl.aivillage.ai",
    val participationRate: Float = 0.3f,
    val minBatteryLevel: Float = 0.2f,
    val allowMobileData: Boolean = false,
    val localEpochs: Int = 5,
    val batchSize: Int = 32,
    val learningRate: Float = 0.01f,
    val roundInterval: Long = 3600000, // 1 hour
    val retryDelay: Long = 60000 // 1 minute
)

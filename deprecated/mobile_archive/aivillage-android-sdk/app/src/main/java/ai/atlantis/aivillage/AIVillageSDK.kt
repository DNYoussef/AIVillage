package ai.atlantis.aivillage

import android.content.Context
import ai.atlantis.aivillage.core.Configuration
import ai.atlantis.aivillage.mesh.MeshNetwork
import ai.atlantis.aivillage.agents.AgentManager
import ai.atlantis.aivillage.fl.FederatedLearningClient

/**
 * Main entry point for AIVillage SDK
 */
class AIVillageSDK private constructor(
    private val context: Context,
    private val configuration: Configuration
) {
    
    private val meshNetwork: MeshNetwork by lazy {
        MeshNetwork(context, configuration.meshConfig)
    }
    
    private val agentManager: AgentManager by lazy {
        AgentManager(context, configuration.agentConfig)
    }
    
    private val flClient: FederatedLearningClient by lazy {
        FederatedLearningClient(context, configuration.flConfig)
    }
    
    /**
     * Initialize the SDK
     */
    fun initialize() {
        // Initialize components
        meshNetwork.initialize()
        agentManager.initialize()
        flClient.initialize()
        
        // Start background services
        if (configuration.autoStart) {
            start()
        }
    }
    
    /**
     * Start SDK services
     */
    fun start() {
        meshNetwork.startDiscovery()
        agentManager.startAgents()
        flClient.startParticipation()
    }
    
    /**
     * Stop SDK services
     */
    fun stop() {
        meshNetwork.stopDiscovery()
        agentManager.stopAgents()
        flClient.stopParticipation()
    }
    
    /**
     * Get mesh network interface
     */
    fun getMeshNetwork(): MeshNetwork = meshNetwork
    
    /**
     * Get agent manager interface
     */
    fun getAgentManager(): AgentManager = agentManager
    
    /**
     * Get federated learning client
     */
    fun getFLClient(): FederatedLearningClient = flClient
    
    companion object {
        @Volatile
        private var INSTANCE: AIVillageSDK? = null
        
        /**
         * Initialize the SDK with configuration
         */
        fun initialize(context: Context, configuration: Configuration): AIVillageSDK {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: AIVillageSDK(context.applicationContext, configuration).also {
                    INSTANCE = it
                    it.initialize()
                }
            }
        }
        
        /**
         * Get SDK instance
         */
        fun getInstance(): AIVillageSDK {
            return INSTANCE ?: throw IllegalStateException(
                "AIVillageSDK not initialized. Call initialize() first."
            )
        }
    }
}

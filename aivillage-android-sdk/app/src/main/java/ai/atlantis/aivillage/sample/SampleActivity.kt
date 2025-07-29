package ai.atlantis.aivillage.sample

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import ai.atlantis.aivillage.AIVillageSDK
import ai.atlantis.aivillage.core.Configuration
import ai.atlantis.aivillage.core.MeshConfiguration
import ai.atlantis.aivillage.core.AgentConfiguration
import ai.atlantis.aivillage.agents.TranslationTask
import ai.atlantis.aivillage.agents.AgentResult
import ai.atlantis.aivillage.agents.TranslationResult
import ai.atlantis.aivillage.core.FLConfiguration
import kotlinx.coroutines.*

class SampleActivity : AppCompatActivity() {
    
    private lateinit var aiVillageSDK: AIVillageSDK
    private val activityScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize SDK
        initializeAIVillage()
        
        // Example: Translate text
        translateText("Hello World", "en", "es")
        
        // Example: Monitor mesh network
        monitorMeshNetwork()
        
        // Example: Monitor federated learning
        monitorFederatedLearning()
    }
    
    private fun initializeAIVillage() {
        val config = Configuration.Builder()
            .deviceId(getDeviceId())
            .meshConfig(
                MeshConfiguration(
                    maxConnections = 5,
                    enableEncryption = true
                )
            )
            .agentConfig(
                AgentConfiguration(
                    enabledAgents = setOf("translator", "classifier")
                )
            )
            .flConfig(
                FLConfiguration(
                    participationRate = 0.5f,
                    minBatteryLevel = 0.3f
                )
            )
            .maxMemoryMB(256)
            .build()
        
        aiVillageSDK = AIVillageSDK.initialize(this, config)
    }
    
    private fun translateText(text: String, sourceLang: String, targetLang: String) {
        activityScope.launch {
            val agentManager = aiVillageSDK.getAgentManager()
            val task = TranslationTask(text, sourceLang, targetLang)
            
            when (val result = agentManager.processTask(task)) {
                is AgentResult.Success -> {
                    val translation = result.data as TranslationResult
                    println("Translated: ${translation.translatedText}")
                }
                is AgentResult.Error -> {
                    println("Translation error: ${result.message}")
                }
            }
        }
    }
    
    private fun monitorMeshNetwork() {
        activityScope.launch {
            aiVillageSDK.getMeshNetwork().connectedNodes.collect { nodes ->
                println("Connected nodes: ${nodes.size}")
                nodes.forEach { node ->
                    println("  Node ${node.nodeId}: RSSI=${node.rssi}")
                }
            }
        }
    }
    
    private fun monitorFederatedLearning() {
        activityScope.launch {
            aiVillageSDK.getFLClient().flState.collect { state ->
                println("FL State: $state")
            }
        }
    }
    
    private fun getDeviceId(): String {
        // Generate or retrieve unique device ID
        return "device_${System.currentTimeMillis()}"
    }
    
    override fun onDestroy() {
        super.onDestroy()
        activityScope.cancel()
        aiVillageSDK.stop()
    }
}

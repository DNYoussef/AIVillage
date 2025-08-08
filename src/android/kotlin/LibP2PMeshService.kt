package ai.atlantis.aivillage.mesh

import android.content.Context
import androidx.room.*
import java.util.Base64
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

/**
 * Configuration for the LibP2P mesh service
 */
data class MeshConfig(
    val nodeId: String,
    val listenPort: Int = 4001,
    val enableMDNS: Boolean = true,
    val enableDHT: Boolean = true
)

/**
 * Status returned after attempting message delivery
 */
enum class DeliveryStatus {
    DELIVERED,
    QUEUED,
    FAILED
}

/**
 * Mesh message container
 */
data class MeshMessage(
    val id: String = UUID.randomUUID().toString(),
    val sender: String,
    val recipient: String?,
    val payload: ByteArray,
    val timestamp: Long = System.currentTimeMillis(),
    val ttl: Int = 5
)

/**
 * Room entity for persisting queued messages while offline
 */
@Entity(tableName = "queued_messages")
data class QueuedMessage(
    @PrimaryKey val id: String,
    val recipient: String?,
    val payload: ByteArray,
    val timestamp: Long
)

@Dao
interface MessageDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insert(msg: QueuedMessage)

    @Query("SELECT * FROM queued_messages ORDER BY timestamp ASC")
    fun getAll(): List<QueuedMessage>

    @Query("DELETE FROM queued_messages WHERE id = :id")
    fun delete(id: String)
}

@Database(entities = [QueuedMessage::class], version = 1)
abstract class MessageQueueDb : RoomDatabase() {
    abstract fun dao(): MessageDao
}

/**
 * Simple persistent queue backed by Room
 */
class PersistentMessageQueue(context: Context) {
    private val db = Room.databaseBuilder(
        context.applicationContext,
        MessageQueueDb::class.java,
        "mesh_queue.db"
    ).build()

    private val executor = Executors.newSingleThreadExecutor()

    fun enqueue(message: MeshMessage) {
        executor.execute {
            db.dao().insert(
                QueuedMessage(
                    id = message.id,
                    recipient = message.recipient,
                    payload = message.payload,
                    timestamp = message.timestamp
                )
            )
        }
    }

    fun dequeueAll(): List<MeshMessage> {
        val msgs = db.dao().getAll()
        db.runInTransaction {
            msgs.forEach { db.dao().delete(it.id) }
        }
        return msgs.map {
            MeshMessage(
                id = it.id,
                sender = "", // sender is not persisted
                recipient = it.recipient,
                payload = it.payload,
                timestamp = it.timestamp
            )
        }
    }
}

/**
 * Basic peer manager with no hard peer limit (supports 50+ peers)
 */
class PeerManager {
    private val peers = ConcurrentHashMap.newKeySet<String>()

    fun addPeer(id: String) = peers.add(id)
    fun removePeer(id: String) = peers.remove(id)
    fun allPeers(): Set<String> = peers
}

/**
 * Bridge to Python LibP2P implementation via JNI
 */
class LibP2PMeshService(private val context: Context) {
    private val bridge = LibP2PJNIBridge()
    private val peerManager = PeerManager()
    private val messageQueue = PersistentMessageQueue(context)
    private val connected = AtomicBoolean(false)
    private val executor = Executors.newSingleThreadScheduledExecutor()

    /**
     * Initialize LibP2P and supporting services.
     * Includes retry logic and triggers discovery mechanisms.
     */
    fun initializeP2P(config: MeshConfig): Boolean {
        var attempts = 0
        while (attempts < 3) {
            if (bridge.initialize(config.toJson())) {
                connected.set(true)
                flushQueue()
                return true
            }
            attempts++
            Thread.sleep(1000L * attempts)
        }
        return false
    }

    /**
     * Send message with retry and offline queueing
     */
    fun sendMessage(message: MeshMessage): CompletableFuture<DeliveryStatus> {
        val result = CompletableFuture<DeliveryStatus>()

        executor.execute {
            if (!connected.get()) {
                messageQueue.enqueue(message)
                result.complete(DeliveryStatus.QUEUED)
                return@execute
            }

            var backoff = 500L
            repeat(5) { _ ->
                if (bridge.sendMessage(message.toJson())) {
                    result.complete(DeliveryStatus.DELIVERED)
                    return@execute
                }
                Thread.sleep(backoff)
                backoff = min(backoff * 2, TimeUnit.SECONDS.toMillis(30))
            }

            // If we failed all retries, queue for later
            messageQueue.enqueue(message)
            result.complete(DeliveryStatus.QUEUED)
        }

        return result
    }

    private fun flushQueue() {
        executor.execute {
            val queued = messageQueue.dequeueAll()
            queued.forEach { sendMessage(it) }
        }
    }
}

// Helper JSON serializers used by JNI layer
fun MeshConfig.toJson(): String {
    return "{" +
        "\"nodeId\":\"$nodeId\"," +
        "\"listenPort\":$listenPort," +
        "\"enableMDNS\":$enableMDNS," +
        "\"enableDHT\":$enableDHT" +
        "}"
}

fun MeshMessage.toJson(): String {
    val payloadStr = Base64.getEncoder().encodeToString(payload)
    val recipientJson = recipient?.let { "\"$it\"" } ?: "null"
    return "{" +
        "\"id\":\"$id\"," +
        "\"sender\":\"$sender\"," +
        "\"recipient\":$recipientJson," +
        "\"payload\":\"$payloadStr\"," +
        "\"timestamp\":$timestamp," +
        "\"ttl\":$ttl" +
        "}"
}


package ai.atlantis.aivillage.mesh;

/**
 * JNI interface to the Python LibP2P bridge.
 * The native implementation lives in {@code libp2p_bridge.cpp} and wraps
 * calls into the Python runtime. This class exposes a minimal API used by
 * {@link LibP2PMeshService} for initialization and message passing.
 */
public class LibP2PJNIBridge {

    static {
        try {
            System.loadLibrary("libp2p_bridge");
        } catch (UnsatisfiedLinkError e) {
            // In test environments the native library may be absent.
        }
    }

    /** Initialise the underlying libp2p mesh with the given configuration. */
    public native boolean initialize(String configJson);

    /** Send a mesh message encoded as JSON to the Python layer. */
    public native boolean sendMessage(String messageJson);

    /** Register a handler for incoming messages from native/Python layer. */
    public void registerHandler(MessageHandler handler) {
        this.handler = handler;
        registerNativeHandler();
    }

    private MessageHandler handler;

    private native void registerNativeHandler();

    /** Called from native code when a message is received. */
    private void handleIncoming(String messageJson) {
        if (handler != null) {
            handler.onMessage(messageJson);
        }
    }

    /** Simple functional interface for callbacks. */
    public interface MessageHandler {
        void onMessage(String messageJson);
    }
}

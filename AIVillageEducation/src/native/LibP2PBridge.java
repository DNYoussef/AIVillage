package com.aivillageeducation;

import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableMapKeySetIterator;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * React Native bridge exposing a very small subset of the LibP2P mesh
 * functionality used by the education app. The implementation here is a
 * lightweight stub that mimics the behaviour of a LibP2P node so that the
 * JavaScript layer can be exercised in tests without requiring the full native
 * stack to be present.
 */
public class LibP2PBridge extends ReactContextBaseJavaModule {
    private final ReactApplicationContext reactContext;
    private Callback peerFoundCallback;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public LibP2PBridge(ReactApplicationContext context) {
        super(context);
        this.reactContext = context;
    }

    @NonNull
    @Override
    public String getName() {
        return "LibP2PBridge";
    }

    /**
     * Initialise the (mock) LibP2P node.  In the real application this would
     * spin up the networking stack; here we simply schedule a synthetic peer
     * discovery event so the JavaScript side can observe behaviour.
     */
    @ReactMethod
    public void initialize(WritableMap config, Promise promise) {
        try {
            final int interval = config != null && config.hasKey("discoveryInterval")
                    ? config.getInt("discoveryInterval")
                    : 5000;

            executor.execute(() -> {
                try {
                    Thread.sleep(interval);
                    String peerId = "peer-" + UUID.randomUUID();
                    emitPeerFound(peerId);
                } catch (InterruptedException ignored) {
                }
            });

            promise.resolve(null);
        } catch (Exception e) {
            promise.reject("INIT_ERROR", e);
        }
    }

    /**
     * Register a callback that will be triggered whenever a peer discovery
     * event occurs.
     */
    @ReactMethod
    public void onPeerFound(Callback callback) {
        this.peerFoundCallback = callback;
    }

    /**
     * Serialize and "send" a message to the given peer.  The message is
     * converted to JSON and logged – in a production build this JSON payload
     * would be handed off to the LibP2P layer for transmission across the mesh.
     */
    @ReactMethod
    public void sendMessage(String peerId, WritableMap message, Promise promise) {
        try {
            JSONObject json = new JSONObject();
            json.put("peerId", peerId);
            json.put("message", readableMapToJson(message));

            Log.d("LibP2PBridge", "sendMessage: " + json.toString());
            promise.resolve(null);
        } catch (JSONException e) {
            promise.reject("SEND_ERROR", e);
        }
    }

    /** Emit a peer discovery event to both the registered callback and the
     * React Native event emitter so JavaScript can subscribe using either
     * approach. */
    private void emitPeerFound(String peerId) {
        if (peerFoundCallback != null) {
            peerFoundCallback.invoke(peerId);
        }
        WritableMap map = Arguments.createMap();
        map.putString("id", peerId);
        reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
                .emit("peerFound", map);
    }

    private JSONObject readableMapToJson(ReadableMap map) throws JSONException {
        JSONObject json = new JSONObject();
        ReadableMapKeySetIterator iterator = map.keySetIterator();
        while (iterator.hasNextKey()) {
            String key = iterator.nextKey();
            ReadableType type = map.getType(key);
            switch (type) {
                case Null:
                    json.put(key, JSONObject.NULL);
                    break;
                case Boolean:
                    json.put(key, map.getBoolean(key));
                    break;
                case Number:
                    json.put(key, map.getDouble(key));
                    break;
                case String:
                    json.put(key, map.getString(key));
                    break;
                case Map:
                    json.put(key, readableMapToJson(map.getMap(key)));
                    break;
                case Array:
                    json.put(key, readableArrayToJson(map.getArray(key)));
                    break;
            }
        }
        return json;
    }

    private JSONArray readableArrayToJson(ReadableArray array) throws JSONException {
        JSONArray json = new JSONArray();
        for (int i = 0; i < array.size(); i++) {
            ReadableType type = array.getType(i);
            switch (type) {
                case Null:
                    json.put(JSONObject.NULL);
                    break;
                case Boolean:
                    json.put(array.getBoolean(i));
                    break;
                case Number:
                    json.put(array.getDouble(i));
                    break;
                case String:
                    json.put(array.getString(i));
                    break;
                case Map:
                    json.put(readableMapToJson(array.getMap(i)));
                    break;
                case Array:
                    json.put(readableArrayToJson(array.getArray(i)));
                    break;
            }
        }
        return json;
    }
}


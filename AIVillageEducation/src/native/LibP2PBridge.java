package com.aivillageeducation;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import org.json.JSONException;
import org.json.JSONObject;

import ai.atlantis.aivillage.mesh.LibP2PJNIBridge;

public class LibP2PBridge extends ReactContextBaseJavaModule {
    private final ReactApplicationContext reactContext;
    private final LibP2PJNIBridge nativeBridge = new LibP2PJNIBridge();
    private Callback peerFoundCallback;

    public LibP2PBridge(ReactApplicationContext context) {
        super(context);
        this.reactContext = context;
    }

    @NonNull
    @Override
    public String getName() {
        return "LibP2PBridge";
    }

    @ReactMethod
    public void initialize(WritableMap config, Promise promise) {
        try {
            String configJson = mapToJson(config);
            boolean ok = nativeBridge.initialize(configJson);
            if (ok) {
                nativeBridge.registerHandler(this::handleNativeMessage);
                promise.resolve(null);
            } else {
                promise.reject("INIT_FAILED", "Failed to initialize LibP2P node");
            }
        } catch (Exception e) {
            promise.reject("INIT_ERROR", e);
        }
    }

    @ReactMethod
    public void onPeerFound(Callback callback) {
        this.peerFoundCallback = callback;
    }

    private void handleNativeMessage(String messageJson) {
        try {
            JSONObject obj = new JSONObject(messageJson);
            if ("peerFound".equals(obj.optString("type"))) {
                WritableMap peer = Arguments.createMap();
                peer.putString("id", obj.getString("peerId"));
                if (peerFoundCallback != null) {
                    peerFoundCallback.invoke(peer);
                }
                sendEvent("peerFound", peer);
            }
        } catch (Exception ignored) {
        }
    }

    private void sendEvent(String eventName, WritableMap params) {
        reactContext
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
            .emit(eventName, params);
    }

    @ReactMethod
    public void sendMessage(String peerId, WritableMap message, Promise promise) {
        try {
            JSONObject obj = new JSONObject();
            obj.put("peerId", peerId);
            obj.put("message", new JSONObject(message.toHashMap()));
            boolean ok = nativeBridge.sendMessage(obj.toString());
            if (ok) {
                promise.resolve(null);
            } else {
                promise.reject("SEND_FAILED", "Failed to send message");
            }
        } catch (JSONException e) {
            promise.reject("SEND_ERROR", e);
        }
    }

    private String mapToJson(WritableMap map) throws JSONException {
        return new JSONObject(map.toHashMap()).toString();
    }
}

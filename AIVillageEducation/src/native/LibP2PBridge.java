package com.aivillageeducation;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableMap;

public class LibP2PBridge extends ReactContextBaseJavaModule {
    public LibP2PBridge(ReactApplicationContext context) {
        super(context);
    }

    @NonNull
    @Override
    public String getName() {
        return "LibP2PBridge";
    }

    @ReactMethod
    public void initialize(WritableMap config, Promise promise) {
        // TODO: integrate LibP2P
        promise.resolve(null);
    }

    @ReactMethod
    public void onPeerFound(String peerId) {
        // Placeholder callback registration
    }

    @ReactMethod
    public void sendMessage(String peerId, WritableMap message, Promise promise) {
        // TODO: send message via mesh
        promise.resolve(null);
    }
}

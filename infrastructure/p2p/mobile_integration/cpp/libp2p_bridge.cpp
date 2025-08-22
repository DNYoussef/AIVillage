#include <jni.h>
#include <Python.h>

/**
 * Minimal native bridge that forwards calls from the Android runtime to the
 * Python implementation located in {@code src/android/jni/libp2p_mesh_bridge.py}.
 * The implementation intentionally avoids complex state and relies on the
 * Python module to manage libp2p resources and peer connections.
 */

namespace {
bool ensure_python() {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
    return Py_IsInitialized();
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_atlantis_aivillage_mesh_LibP2PJNIBridge_initialize(
    JNIEnv* env, jobject /*thiz*/, jstring config_json) {
    if (!ensure_python()) return JNI_FALSE;

    const char* config = env->GetStringUTFChars(config_json, nullptr);
    PyObject* module = PyImport_ImportModule("src.android.jni.libp2p_mesh_bridge");
    if (!module) {
        env->ReleaseStringUTFChars(config_json, config);
        return JNI_FALSE;
    }
    PyObject* func = PyObject_GetAttrString(module, "initialize_bridge");
    PyObject* args = Py_BuildValue("(s)", config);
    PyObject* result = func ? PyObject_CallObject(func, args) : nullptr;
    bool success = result && PyObject_IsTrue(PyObject_GetAttrString(result, "success"));
    Py_XDECREF(result);
    Py_XDECREF(func);
    Py_DECREF(args);
    Py_DECREF(module);
    env->ReleaseStringUTFChars(config_json, config);
    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_atlantis_aivillage_mesh_LibP2PJNIBridge_sendMessage(
    JNIEnv* env, jobject /*thiz*/, jstring message_json) {
    if (!ensure_python()) return JNI_FALSE;

    const char* msg = env->GetStringUTFChars(message_json, nullptr);
    PyObject* module = PyImport_ImportModule("src.android.jni.libp2p_mesh_bridge");
    if (!module) {
        env->ReleaseStringUTFChars(message_json, msg);
        return JNI_FALSE;
    }
    PyObject* func = PyObject_GetAttrString(module, "send_message_via_bridge");
    PyObject* args = Py_BuildValue("(s)", msg);
    PyObject* result = func ? PyObject_CallObject(func, args) : nullptr;
    bool success = result && PyObject_IsTrue(result);
    Py_XDECREF(result);
    Py_XDECREF(func);
    Py_DECREF(args);
    Py_DECREF(module);
    env->ReleaseStringUTFChars(message_json, msg);
    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_atlantis_aivillage_mesh_LibP2PJNIBridge_registerNativeHandler(
    JNIEnv* /*env*/, jobject /*thiz*/) {
    // In this minimal implementation, messages are delivered via the existing
    // WebSocket bridge. A direct JNI callback is left as future work.
}

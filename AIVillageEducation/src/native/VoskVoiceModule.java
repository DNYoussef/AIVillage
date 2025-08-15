package com.aivillageeducation;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.speech.tts.TextToSpeech;
import android.speech.tts.Voice;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;

import org.json.JSONException;
import org.json.JSONObject;
import org.vosk.Model;
import org.vosk.Recognizer;

import java.io.File;
import java.io.IOException;
import java.util.Locale;

public class VoskVoiceModule extends ReactContextBaseJavaModule {
    private Model model;
    private TextToSpeech tts;
    private boolean isListening = false;

    public VoskVoiceModule(ReactApplicationContext context) {
        super(context);
    }

    @NonNull
    @Override
    public String getName() {
        return "VoskVoiceModule";
    }

    @ReactMethod
    public void loadModel(String name, Promise promise) {
        try {
            ReactApplicationContext ctx = getReactApplicationContext();
            File modelDir = new File(ctx.getFilesDir(), name);
            model = new Model(modelDir.getAbsolutePath());
            promise.resolve(null);
        } catch (IOException e) {
            promise.reject("MODEL_LOAD_ERROR", e);
        }
    }

    @ReactMethod
    public void createTTS(ReadableMap config, Promise promise) {
        ReactApplicationContext ctx = getReactApplicationContext();
        tts = new TextToSpeech(ctx, status -> {
            if (status != TextToSpeech.SUCCESS) {
                promise.reject("TTS_INIT_ERROR", "Initialization failed");
                return;
            }
            if (config != null) {
                if (config.hasKey("language")) {
                    Locale locale = Locale.forLanguageTag(config.getString("language"));
                    if (locale != null) {
                        tts.setLanguage(locale);
                    }
                }
                if (config.hasKey("rate")) {
                    tts.setSpeechRate((float) config.getDouble("rate"));
                }
                if (config.hasKey("voice") && Build.VERSION.SDK_INT >= 21) {
                    String voiceName = config.getString("voice");
                    for (Voice voice : tts.getVoices()) {
                        if (voice.getName().contains(voiceName)) {
                            tts.setVoice(voice);
                            break;
                        }
                    }
                }
            }
            promise.resolve("tts");
        });
    }

    @ReactMethod
    public void startListening(Callback callback) {
        if (model == null || isListening) {
            return;
        }
        isListening = true;
        new Thread(() -> {
            int sampleRate = 16000;
            Recognizer recognizer;
            try {
                recognizer = new Recognizer(model, sampleRate);
            } catch (IOException e) {
                callback.invoke("" );
                isListening = false;
                return;
            }

            int bufferSize = AudioRecord.getMinBufferSize(sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            AudioRecord recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize);
            recorder.startRecording();
            byte[] buffer = new byte[bufferSize];
            while (isListening) {
                int nread = recorder.read(buffer, 0, buffer.length);
                if (nread < 0) {
                    break;
                }
                if (recognizer.acceptWaveForm(buffer, nread)) {
                    String result = recognizer.getResult();
                    try {
                        String text = new JSONObject(result).optString("text");
                        if (!text.isEmpty()) {
                            callback.invoke(text);
                        }
                    } catch (JSONException e) {
                        callback.invoke(result);
                    }
                } else {
                    String partial = recognizer.getPartialResult();
                    try {
                        String text = new JSONObject(partial).optString("partial");
                        if (!text.isEmpty()) {
                            callback.invoke(text);
                        }
                    } catch (JSONException e) {
                        callback.invoke(partial);
                    }
                }
            }
            recorder.stop();
            recorder.release();
            recognizer.close();
            isListening = false;
        }).start();
    }

    @ReactMethod
    public void speak(String ttsId, String text, Promise promise) {
        if (tts == null) {
            promise.reject("TTS_NOT_INITIALIZED", "TTS not initialized");
            return;
        }
        int result = tts.speak(text, TextToSpeech.QUEUE_ADD, null, ttsId);
        if (result == TextToSpeech.ERROR) {
            promise.reject("TTS_SPEAK_ERROR", "Error speaking");
        } else {
            promise.resolve(null);
        }
    }
}

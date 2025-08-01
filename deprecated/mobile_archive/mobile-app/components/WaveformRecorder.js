import React, { useState, useRef } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Animated, Easing } from 'react-native';
import { Audio } from 'expo-av';

export default function WaveformRecorder({ onStop, placeholder }) {
  const [recording, setRecording] = useState(null);
  const waveAnim = useRef(new Animated.Value(0)).current;

  const startWave = () => {
    waveAnim.setValue(0);
    Animated.loop(
      Animated.timing(waveAnim, {
        toValue: 1,
        duration: 800,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    ).start();
  };

  const stopWave = () => {
    waveAnim.stopAnimation();
    waveAnim.setValue(0);
  };

  const startRecording = async () => {
    const perm = await Audio.requestPermissionsAsync();
    if (!perm.granted) return;
    await Audio.setAudioModeAsync({ allowsRecordingIOS: true, playsInSilentModeIOS: true });
    const rec = new Audio.Recording();
    await rec.prepareToRecordAsync(Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY);
    await rec.startAsync();
    setRecording(rec);
    startWave();
  };

  const stopRecording = async () => {
    await recording.stopAndUnloadAsync();
    const uri = recording.getURI();
    setRecording(null);
    stopWave();
    onStop(uri);
  };

  const waveScale = waveAnim.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: [1, 1.5, 1],
  });

  return (
    <View style={styles.container}>
      {recording && (
        <Animated.View style={[styles.waveform, { transform: [{ scaleY: waveScale }] }]} />
      )}
      <TouchableOpacity
        onPress={recording ? stopRecording : startRecording}
        style={[styles.button, recording ? styles.buttonStop : styles.buttonStart]}
      >
        <Text style={styles.buttonText}>
          {recording ? '🔴 Stop' : placeholder || '🎙️ Speak'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    margin: 12,
  },
  waveform: {
    width: '80%',
    height: 50,
    backgroundColor: '#4F46E5',
    marginBottom: 12,
    borderRadius: 4,
    opacity: 0.6,
  },
  button: {
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  buttonStart: {
    backgroundColor: '#4F46E5',
  },
  buttonStop: {
    backgroundColor: '#DC2626',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});

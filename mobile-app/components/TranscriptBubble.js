import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { Audio } from 'expo-av';

export default function TranscriptBubble({ speaker, text, audioUri }) {
  const isTwin = speaker === 'Twin';
  return (
    <View style={{
      alignSelf: isTwin ? 'flex-start' : 'flex-end',
      backgroundColor: isTwin ? '#E0E7FF' : '#DCFCE7',
      borderRadius: 8, padding: 8, marginVertical: 4, maxWidth: '80%'
    }}>
      <Text style={{ fontWeight: '600', marginBottom: 4 }}>{speaker}</Text>
      <Text>{text}</Text>
      {audioUri && (
        <TouchableOpacity onPress={async () => {
          const { sound } = await Audio.Sound.createAsync({ uri: audioUri });
          await sound.playAsync();
        }} style={{ marginTop: 6 }}>
          <Text style={{ color: '#4F46E5' }}>🔊 Play</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

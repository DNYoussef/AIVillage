import React from 'react';
import { View, StyleSheet } from 'react-native';

export default function ProgressBar({ progress }) {
  return (
    <View style={styles.container}>
      <View style={[styles.fill, { flex: progress }]} />
      <View style={[styles.empty, { flex: 1 - progress }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', height: 8, width: '100%', backgroundColor: '#EEE', borderRadius: 4, overflow: 'hidden' },
  fill: { backgroundColor: '#4F46E5' },
  empty: { backgroundColor: '#DDD' }
});

import React from 'react';
import { View, Text } from 'react-native';

export default function EvidencePanel({ tier, text }) {
  const colors = {
    low: '#e5e7eb',
    medium: '#fef08a',
    high: '#bbf7d0'
  };
  return (
    <View style={{ padding: 8, backgroundColor: colors[tier] || colors.low }}>
      <Text>{text}</Text>
    </View>
  );
}

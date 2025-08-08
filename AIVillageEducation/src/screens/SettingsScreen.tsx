import React from 'react';
import { View, Text } from 'react-native';
import ResourceAwareManager from '../services/ResourceAwareManager';

export default function SettingsScreen() {
  const manager = new ResourceAwareManager();
  manager.optimizeForDevice();
  return (
    <View>
      <Text>Settings</Text>
    </View>
  );
}

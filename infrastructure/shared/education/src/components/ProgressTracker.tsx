import React from 'react';
import { View, Text } from 'react-native';

interface Props {
  completed: number;
  total: number;
}

export default function ProgressTracker({ completed, total }: Props) {
  const percent = total === 0 ? 0 : Math.round((completed / total) * 100);
  return (
    <View>
      <Text>Progress: {percent}% ({completed}/{total})</Text>
    </View>
  );
}

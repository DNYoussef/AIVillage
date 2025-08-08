import React from 'react';
import { View, Text, Button } from 'react-native';
import VoiceTutor from '../components/VoiceTutor';

export default function HomeScreen({ navigation }: any) {
  return (
    <View>
      <Text>Welcome to AI Village Education</Text>
      <Button title="Go to Lessons" onPress={() => navigation.navigate('Lesson')} />
      <VoiceTutor onNavigate={(target) => navigation.navigate(target)} />
    </View>
  );
}

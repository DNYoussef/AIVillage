import React from 'react';
import { View, Text, Image, ScrollView } from 'react-native';
import OfflineIndicator from './OfflineIndicator';

interface Lesson {
  id: string;
  title: string;
  content: string;
  images?: string[];
}

interface Props {
  lesson: Lesson;
  offline?: boolean;
}

export default function LessonDisplay({ lesson, offline }: Props) {
  return (
    <ScrollView>
      {offline && <OfflineIndicator />}
      <Text>{lesson.title}</Text>
      <Text>{lesson.content}</Text>
      {lesson.images?.map((img, idx) => (
        <Image key={idx} source={{ uri: img }} style={{ width: '100%', height: 200 }} />
      ))}
    </ScrollView>
  );
}

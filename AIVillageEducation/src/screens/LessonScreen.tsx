import React, { useEffect, useState } from 'react';
import { View } from 'react-native';
import LessonDisplay from '../components/LessonDisplay';
import ProgressTracker from '../components/ProgressTracker';
import OfflineManager from '../services/OfflineManager';

const lessonStub = {
  id: '1',
  title: 'Sample Lesson',
  content: 'This is a sample lesson.',
  gradeLevel: 1,
  subject: 'general'
};

export default function LessonScreen() {
  const [offline, setOffline] = useState(false);
  const offlineManager = new OfflineManager();

  useEffect(() => {
    offlineManager.initialize();
    offlineManager.cacheLesson(lessonStub);
  }, []);

  return (
    <View>
      <LessonDisplay lesson={lessonStub} offline={offline} />
      <ProgressTracker completed={1} total={10} />
    </View>
  );
}

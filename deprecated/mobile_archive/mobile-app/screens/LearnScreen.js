import React, { useState } from 'react';
import { View, Text, Button } from 'react-native';
import WaveformRecorder from '../components/WaveformRecorder';
import Flashcard from '../components/Flashcard';
import ProgressBar from '../components/ProgressBar';
import twinApi from '../services/twinApi';
import { useTranslation } from 'react-i18next';

export default function LearnScreen() {
  const { t } = useTranslation();
  const [card, setCard] = useState(null);
  const [feedback, setFb] = useState(null);
  const [progress, setProgress] = useState(0);

  const loadCard = async () => {
    const next = await twinApi.fetchNextFlashcard();
    setFb(null);
    setCard(next);
    setProgress(next.progress || 0);
  };

  const grade = async uri => {
    const result = await twinApi.gradeAnswer(card.id, uri);
    setFb(result);
  };

  return (
    <View style={{ flex:1, padding:16, justifyContent:'space-around' }}>
      <ProgressBar progress={progress} />
      {!card ? (
        <Button title={t('learn.start')} onPress={loadCard} />
      ) : feedback ? (
        <View style={{ alignItems:'center' }}>
          <Text style={{ fontSize:24, marginBottom:8 }}>
            { feedback.correct ? t('learn.correct') : t('learn.wrong') }
          </Text>
          <Text style={{ textAlign:'center', marginBottom:16 }}>
            {feedback.explanation}
          </Text>
          <Button title={t('learn.next')} onPress={loadCard} />
        </View>
      ) : (
        <>
          <Flashcard front={card.prompt} back={card.answer} />
          <WaveformRecorder onStop={grade} />
        </>
      )}
    </View>
  );
}

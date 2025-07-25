import React, { useState } from 'react';
import { View, ScrollView, ActivityIndicator } from 'react-native';
import WaveformRecorder from '../components/WaveformRecorder';
import TranscriptBubble from '../components/TranscriptBubble';
import EvidencePanel from '../components/EvidencePanel';
import twinApi from '../services/twinApi';
import { useTranslation } from 'react-i18next';

export default function TwinScreen() {
  const { t } = useTranslation();
  const [msgs, setMsgs] = useState([]);
  const [evidence, setEvidence] = useState([]);
  const [loading, setLoad] = useState(false);

  const onStop = async uri => {
    setLoad(true);
    try {
      const { transcript, response, audioUri } = await twinApi.queryTwin(uri);
      const evid = await twinApi.fetchExplanation('latest');
      setMsgs(arr => [
        ...arr,
        { from: 'You', text: transcript },
        { from: 'Twin', text: response, audioUri }
      ]);
      setEvidence(evid);
    } catch(e) {
      // handle error...
    }
    setLoad(false);
  };

  return (
    <View style={{ flex:1 }}>
      <ScrollView style={{ flex:1, padding:12 }}>
        {msgs.map((m,i) => (
          <TranscriptBubble key={i} speaker={m.from} text={m.text} audioUri={m.audioUri}/>
        ))}
        {evidence.map((ev,i) => (
          <EvidencePanel key={i} tier={ev.confidence_tier} text={ev.chunks[0].text} />
        ))}
      </ScrollView>
      {loading && (
        <ActivityIndicator size="large" color="#4F46E5" style={{ margin:12 }} />
      )}
      <WaveformRecorder onStop={onStop} placeholder={t('twin.hint')} />
    </View>
  );
}

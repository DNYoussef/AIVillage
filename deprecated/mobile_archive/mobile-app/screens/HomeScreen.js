import React from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';
import logoBase64 from '../assets/logoBase64';
import { useTranslation } from 'react-i18next';

export default function HomeScreen({ navigation }) {
  const { t } = useTranslation();
  return (
    <View style={{
      flex:1, justifyContent:'center', alignItems:'center', padding:24
    }}>
      <Image
        source={{ uri: logoBase64 }}
        style={{ width:120, height:120, marginBottom:20 }}
      />
      <Text style={{ fontSize:28, fontWeight:'bold', marginBottom:10 }}>
        {t('appName')}
      </Text>
      <Text style={{
        fontSize:16, textAlign:'center', marginBottom:30, color:'#555'
      }}>
        {t('home.title')}
      </Text>

      <TouchableOpacity
        onPress={() => navigation.navigate('Twin')}
        style={styles.buttonPrimary}
      >
        <Text style={styles.buttonText}>{t('home.talkBtn')}</Text>
      </TouchableOpacity>

      <TouchableOpacity
        onPress={() => navigation.navigate('Learn')}
        style={styles.buttonSecondary}
      >
        <Text style={styles.buttonText}>{t('home.learnBtn')}</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = {
  buttonPrimary: {
    backgroundColor: '#4F46E5',
    paddingVertical:14, paddingHorizontal:32,
    borderRadius:8, marginBottom:16
  },
  buttonSecondary: {
    backgroundColor: '#10B981',
    paddingVertical:14, paddingHorizontal:32,
    borderRadius:8
  },
  buttonText: {
    color:'#fff', fontSize:18, fontWeight:'600'
  }
};

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import MainTabs from './navigation/MainTabs';
import { I18nextProvider } from 'react-i18next';
import i18n from './i18n/i18n';

export default function App() {
  return (
    <I18nextProvider i18n={i18n}>
      <NavigationContainer>
        <MainTabs />
      </NavigationContainer>
    </I18nextProvider>
  );
}

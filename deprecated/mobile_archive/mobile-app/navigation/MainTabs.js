import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from '../screens/HomeScreen';
import TwinScreen from '../screens/TwinScreen';
import LearnScreen from '../screens/LearnScreen';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';

const Tab = createBottomTabNavigator();

export default function MainTabs() {
  const { t } = useTranslation();
  return (
    <Tab.Navigator
      initialRouteName="Home"
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ color, size }) => {
          let icon = 'ellipse';
          if (route.name === 'Home') icon = 'home';
          if (route.name === 'Twin') icon = 'mic';
          if (route.name === 'Learn') icon = 'book';
          return <Ionicons name={icon} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} options={{ title: t('nav.home') }} />
      <Tab.Screen name="Twin" component={TwinScreen} options={{ title: t('nav.talk') }} />
      <Tab.Screen name="Learn" component={LearnScreen} options={{ title: t('nav.learn') }} />
    </Tab.Navigator>
  );
}

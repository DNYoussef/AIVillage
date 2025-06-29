import React, { useRef, useState } from 'react';
import { Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';

export default function Flashcard({ front, back }) {
  const anim = useRef(new Animated.Value(0)).current;
  const [flipped, setFlipped] = useState(false);

  const flip = () => {
    Animated.timing(anim, {
      toValue: flipped ? 0 : 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
    setFlipped(!flipped);
  };

  const frontStyle = {
    transform: [
      { rotateY: anim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '180deg'] }) }
    ]
  };
  const backStyle = {
    transform: [
      { rotateY: anim.interpolate({ inputRange: [0, 1], outputRange: ['180deg', '360deg'] }) }
    ],
    position: 'absolute',
    top: 0
  };

  return (
    <TouchableOpacity onPress={flip}>
      <Animated.View style={[styles.card, frontStyle]}>
        <Text style={styles.text}>{front}</Text>
      </Animated.View>
      <Animated.View style={[styles.card, backStyle]}>
        <Text style={styles.text}>{back || 'Tap to reveal answer'}</Text>
      </Animated.View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    width: '100%', minHeight: 120,
    backgroundColor: '#fff', borderRadius: 8,
    justifyContent: 'center', alignItems: 'center',
    backfaceVisibility: 'hidden', padding: 16
  },
  text: { fontSize: 18, textAlign: 'center' }
});

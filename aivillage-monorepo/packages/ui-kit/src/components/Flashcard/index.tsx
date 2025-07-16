import React, { useRef, useState } from 'react'
import { Text, TouchableOpacity, StyleSheet, Animated } from 'react-native'
import theme from '../../theme'

export interface FlashcardProps {
  front: string
  back: string
  onFlip?: (isFlipped: boolean) => void
  style?: any
  frontStyle?: any
  backStyle?: any
}

export default function Flashcard({
  front,
  back,
  onFlip,
  style,
  frontStyle,
  backStyle
}: FlashcardProps) {
  const anim = useRef(new Animated.Value(0)).current
  const [flipped, setFlipped] = useState(false)

  const flip = () => {
    Animated.timing(anim, {
      toValue: flipped ? 0 : 1,
      duration: 500,
      useNativeDriver: true,
    }).start()

    const newFlipped = !flipped
    setFlipped(newFlipped)
    onFlip?.(newFlipped)
  }

  const frontAnimationStyle = {
    transform: [
      {
        rotateY: anim.interpolate({
          inputRange: [0, 1],
          outputRange: ['0deg', '180deg']
        })
      }
    ],
    opacity: anim.interpolate({
      inputRange: [0, 0.5, 1],
      outputRange: [1, 0, 0]
    })
  }

  const backAnimationStyle = {
    transform: [
      {
        rotateY: anim.interpolate({
          inputRange: [0, 1],
          outputRange: ['180deg', '360deg']
        })
      }
    ],
    opacity: anim.interpolate({
      inputRange: [0, 0.5, 1],
      outputRange: [0, 0, 1]
    })
  }

  return (
    <TouchableOpacity
      onPress={flip}
      style={[styles.container, style]}
      activeOpacity={0.8}
    >
      <Animated.View style={[
        styles.card,
        styles.front,
        frontAnimationStyle,
        frontStyle
      ]}>
        <Text style={[styles.text, styles.frontText]}>
          {front}
        </Text>
      </Animated.View>

      <Animated.View style={[
        styles.card,
        styles.back,
        backAnimationStyle,
        backStyle
      ]}>
        <Text style={[styles.text, styles.backText]}>
          {back || 'Tap to reveal answer'}
        </Text>
      </Animated.View>
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  container: {
    width: '100%',
    aspectRatio: 1.6,
    position: 'relative',
  },
  card: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.md,
    backfaceVisibility: 'hidden',
    ...theme.shadows.md,
    borderWidth: 1,
    borderColor: theme.colors.gray[200],
  },
  front: {
    backgroundColor: theme.colors.primary,
  },
  back: {
    backgroundColor: theme.colors.secondary,
  },
  text: {
    fontSize: theme.typography.fontSizes.lg,
    textAlign: 'center',
    lineHeight: theme.typography.lineHeights.normal,
    fontWeight: theme.typography.fontWeights.medium,
  },
  frontText: {
    color: theme.colors.background.primary,
  },
  backText: {
    color: theme.colors.background.primary,
  }
})

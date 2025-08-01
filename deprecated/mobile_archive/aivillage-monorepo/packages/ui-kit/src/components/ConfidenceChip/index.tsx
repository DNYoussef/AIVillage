import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import theme from '../../theme'

export interface ConfidenceChipProps {
  tier: 'low' | 'medium' | 'high'
  size?: 'sm' | 'md' | 'lg'
  style?: any
}

export default function ConfidenceChip({
  tier,
  size = 'md',
  style
}: ConfidenceChipProps) {
  const confidence = theme.colors.confidence[tier]

  const sizeStyles = {
    sm: { paddingHorizontal: 6, paddingVertical: 2, fontSize: 11 },
    md: { paddingHorizontal: 8, paddingVertical: 4, fontSize: 12 },
    lg: { paddingHorizontal: 12, paddingVertical: 6, fontSize: 14 }
  }

  return (
    <View style={[
      styles.container,
      {
        backgroundColor: confidence.bg,
        borderColor: confidence.border,
        paddingHorizontal: sizeStyles[size].paddingHorizontal,
        paddingVertical: sizeStyles[size].paddingVertical,
      },
      style
    ]}>
      <Text style={[
        styles.text,
        {
          color: confidence.text,
          fontSize: sizeStyles[size].fontSize,
        }
      ]}>
        {tier}
      </Text>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    borderRadius: theme.borderRadius.full,
    borderWidth: 1,
    alignSelf: 'flex-start'
  },
  text: {
    fontWeight: theme.typography.fontWeights.semibold,
    textTransform: 'capitalize'
  }
})

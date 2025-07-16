import React from 'react'
import { TouchableOpacity, Text, StyleSheet, ActivityIndicator } from 'react-native'
import theme from '../../theme'

export interface ButtonProps {
  title: string
  onPress: () => void
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  style?: any
  textStyle?: any
  icon?: React.ReactNode
}

export default function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  style,
  textStyle,
  icon
}: ButtonProps) {
  const variantStyles = {
    primary: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,
      borderWidth: 1,
    },
    secondary: {
      backgroundColor: theme.colors.secondary,
      borderColor: theme.colors.secondary,
      borderWidth: 1,
    },
    outline: {
      backgroundColor: 'transparent',
      borderColor: theme.colors.primary,
      borderWidth: 1,
    },
    ghost: {
      backgroundColor: 'transparent',
      borderColor: 'transparent',
      borderWidth: 0,
    }
  }

  const textVariantStyles = {
    primary: { color: theme.colors.background.primary },
    secondary: { color: theme.colors.background.primary },
    outline: { color: theme.colors.primary },
    ghost: { color: theme.colors.primary }
  }

  const sizeStyles = {
    sm: {
      paddingHorizontal: 12,
      paddingVertical: 6,
      minHeight: 32
    },
    md: {
      paddingHorizontal: 16,
      paddingVertical: 12,
      minHeight: 44
    },
    lg: {
      paddingHorizontal: 24,
      paddingVertical: 16,
      minHeight: 52
    }
  }

  const textSizeStyles = {
    sm: { fontSize: theme.typography.fontSizes.sm },
    md: { fontSize: theme.typography.fontSizes.md },
    lg: { fontSize: theme.typography.fontSizes.lg }
  }

  const isDisabled = disabled || loading

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isDisabled}
      style={[
        styles.button,
        variantStyles[variant],
        sizeStyles[size],
        isDisabled && styles.disabled,
        style
      ]}
      activeOpacity={0.8}
    >
      {loading ? (
        <ActivityIndicator
          size="small"
          color={textVariantStyles[variant].color}
        />
      ) : (
        <>
          {icon}
          <Text style={[
            styles.text,
            textVariantStyles[variant],
            textSizeStyles[size],
            isDisabled && styles.disabledText,
            textStyle
          ]}>
            {title}
          </Text>
        </>
      )}
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: theme.borderRadius.md,
    gap: theme.spacing.sm,
  },
  text: {
    fontWeight: theme.typography.fontWeights.semibold,
    textAlign: 'center',
  },
  disabled: {
    opacity: 0.5,
  },
  disabledText: {
    opacity: 0.7,
  }
})

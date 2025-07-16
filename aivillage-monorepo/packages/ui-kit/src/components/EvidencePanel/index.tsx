import React from 'react'
import { View, Text, StyleSheet, ScrollView } from 'react-native'
import ConfidenceChip from '../ConfidenceChip'
import theme from '../../theme'

export interface Evidence {
  id: string
  confidence_tier: 'low' | 'medium' | 'high'
  text: string
  source?: string
  chunks?: Array<{
    text: string
    score?: number
    source_uri?: string
  }>
}

export interface EvidencePanelProps {
  evidences: Evidence[]
  showSources?: boolean
  maxHeight?: number
  style?: any
}

export default function EvidencePanel({
  evidences = [],
  showSources = false,
  maxHeight = 300,
  style
}: EvidencePanelProps) {
  if (!evidences.length) {
    return null
  }

  return (
    <ScrollView
      style={[styles.container, { maxHeight }, style]}
      showsVerticalScrollIndicator={false}
    >
      <View style={styles.content}>
        {evidences.map((evidence, index) => (
          <View key={evidence.id || index} style={styles.evidenceItem}>
            <View style={styles.header}>
              <ConfidenceChip
                tier={evidence.confidence_tier}
                size="sm"
              />
              {showSources && evidence.source && (
                <Text style={styles.source}>
                  {evidence.source}
                </Text>
              )}
            </View>

            <Text style={styles.text}>
              {evidence.text || evidence.chunks?.[0]?.text}
            </Text>

            {evidence.chunks && evidence.chunks.length > 1 && (
              <View style={styles.additionalChunks}>
                {evidence.chunks.slice(1).map((chunk, chunkIndex) => (
                  <Text
                    key={chunkIndex}
                    style={[styles.text, styles.additionalChunk]}
                  >
                    {chunk.text}
                  </Text>
                ))}
              </View>
            )}
          </View>
        ))}
      </View>
    </ScrollView>
  )
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.colors.background.primary,
    borderRadius: theme.borderRadius.md,
    borderWidth: 1,
    borderColor: theme.colors.gray[200],
  },
  content: {
    padding: theme.spacing.md,
  },
  evidenceItem: {
    marginBottom: theme.spacing.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
    gap: theme.spacing.sm,
  },
  source: {
    fontSize: theme.typography.fontSizes.xs,
    color: theme.colors.gray[500],
    fontStyle: 'italic',
  },
  text: {
    fontSize: theme.typography.fontSizes.sm,
    lineHeight: theme.typography.lineHeights.relaxed,
    color: theme.colors.gray[700],
  },
  additionalChunks: {
    marginTop: theme.spacing.sm,
    paddingTop: theme.spacing.sm,
    borderTopWidth: 1,
    borderTopColor: theme.colors.gray[100],
  },
  additionalChunk: {
    marginTop: theme.spacing.xs,
    color: theme.colors.gray[600],
  }
})

/**
 * Phase 3 Quiet-STaR Integration Tests
 * Phase 4 - Testing BitNet preservation of Quiet-STaR reasoning capabilities
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { QuietSTaRProcessor } from '../../../src/phase3_quietstar/QuietSTaRProcessor';
import { ReasoningPreserver } from '../../../src/phase4_bitnet/ReasoningPreserver';
import { TokenEmbedding } from '../../../src/phase3_quietstar/TokenEmbedding';

describe('Phase 3 Quiet-STaR Integration', () => {
  let quietSTaRProcessor: QuietSTaRProcessor;
  let reasoningPreserver: ReasoningPreserver;
  let tokenEmbedding: TokenEmbedding;

  beforeEach(() => {
    quietSTaRProcessor = new QuietSTaRProcessor({
      maxReasoningLength: 256,
      thinkingTokens: ['<think>', '</think>'],
      enableParallelReasoning: true,
      preserveReasoningPatterns: true
    });

    reasoningPreserver = new ReasoningPreserver({
      quantizationBits: 1,
      preserveLogicalStructure: true,
      maintainReasoningFlow: true
    });

    tokenEmbedding = new TokenEmbedding({
      vocabSize: 50000,
      embeddingDim: 512,
      enableQuietSTaR: true
    });
  });

  describe('Reasoning Pattern Preservation', () => {
    it('should preserve quiet reasoning patterns during quantization', async () => {
      const reasoningSequence = [
        'The problem requires <think> analyzing the data structure </think> careful consideration',
        'We need to <think> implement a sorting algorithm that handles edge cases </think> proceed step by step',
        'The solution involves <think> dynamic programming with memoization </think> optimization'
      ];

      // Process with original Quiet-STaR
      const originalResults = await Promise.all(
        reasoningSequence.map(seq => quietSTaRProcessor.processReasoning(seq))
      );

      // Convert to BitNet and process
      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);
      const bitNetResults = await Promise.all(
        reasoningSequence.map(seq => bitNetProcessor.processReasoning(seq))
      );

      // Verify reasoning preservation
      for (let i = 0; i < originalResults.length; i++) {
        const original = originalResults[i];
        const converted = bitNetResults[i];

        // Check reasoning structure preservation
        expect(converted.reasoningTokens).toHaveLength(original.reasoningTokens.length);
        expect(converted.logicalFlow.steps).toHaveLength(original.logicalFlow.steps);

        // Verify reasoning quality metrics
        const reasoningCorrelation = calculateReasoningCorrelation(
          original.reasoningEmbeddings,
          converted.reasoningEmbeddings
        );
        expect(reasoningCorrelation).toBeGreaterThan(0.85);

        // Check logical consistency
        expect(converted.logicalConsistency).toBeGreaterThan(0.9);
      }
    });

    it('should maintain thinking token recognition after quantization', async () => {
      const testTexts = [
        'Simple reasoning <think> this requires analysis </think> continues here',
        'Nested <think> outer thought <think> inner reflection </think> back to outer </think> complete',
        'Multiple <think> first thought </think> and <think> second thought </think> reasoning blocks'
      ];

      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);

      for (const text of testTexts) {
        const result = await bitNetProcessor.processReasoning(text);

        // Verify all thinking tokens are detected
        const thinkingBlocks = text.match(/<think>.*?<\/think>/g) || [];
        expect(result.reasoningTokens).toHaveLength(thinkingBlocks.length);

        // Verify token positions are accurate
        result.reasoningTokens.forEach(token => {
          expect(token.startPosition).toBeGreaterThanOrEqual(0);
          expect(token.endPosition).toBeGreaterThan(token.startPosition);
          expect(text.substring(token.startPosition, token.endPosition + 1))
            .toContain('<think>');
        });
      }
    });
  });

  describe('Token Embedding Integration', () => {
    it('should preserve token embeddings during BitNet conversion', async () => {
      const testTokens = ['reasoning', 'analysis', 'thinking', 'solution', 'problem'];

      // Generate original embeddings
      const originalEmbeddings = await Promise.all(
        testTokens.map(token => tokenEmbedding.getEmbedding(token))
      );

      // Convert token embedding to BitNet
      const bitNetEmbedding = await reasoningPreserver.convertEmbeddingToBitNet(tokenEmbedding);

      // Generate quantized embeddings
      const quantizedEmbeddings = await Promise.all(
        testTokens.map(token => bitNetEmbedding.getEmbedding(token))
      );

      // Verify embedding preservation
      for (let i = 0; i < testTokens.length; i++) {
        const original = originalEmbeddings[i];
        const quantized = quantizedEmbeddings[i];

        expect(quantized).toHaveLength(original.length);

        // Calculate cosine similarity
        const similarity = calculateCosineSimilarity(original, quantized);
        expect(similarity).toBeGreaterThan(0.8);

        // Verify embedding magnitude is reasonable
        const magnitudeRatio = calculateMagnitudeRatio(original, quantized);
        expect(magnitudeRatio).toBeGreaterThan(0.7);
        expect(magnitudeRatio).toBeLessThan(1.3);
      }
    });

    it('should handle special reasoning tokens correctly', async () => {
      const specialTokens = ['<think>', '</think>', '<reason>', '</reason>', '<analyze>', '</analyze>'];

      const bitNetEmbedding = await reasoningPreserver.convertEmbeddingToBitNet(tokenEmbedding);

      for (const token of specialTokens) {
        const embedding = await bitNetEmbedding.getEmbedding(token);

        expect(embedding).toBeDefined();
        expect(embedding.length).toBe(512); // Expected embedding dimension

        // Special tokens should have unique representations
        const uniqueness = await bitNetEmbedding.calculateTokenUniqueness(token);
        expect(uniqueness).toBeGreaterThan(0.7);
      }
    });
  });

  describe('Reasoning Flow Preservation', () => {
    it('should maintain logical reasoning flow in complex scenarios', async () => {
      const complexReasoningText = `
        To solve this problem, we need to <think>
        First, let's identify the key components:
        1. Input validation
        2. Data transformation
        3. Algorithm selection

        The input appears to be a graph structure, so we should consider:
        - Graph traversal algorithms (DFS, BFS)
        - Shortest path algorithms if distance matters
        - Cycle detection if loops are a concern

        Given the constraints, I think BFS would be most appropriate because:
        - It explores nodes level by level
        - It finds shortest paths in unweighted graphs
        - It has predictable memory usage
        </think>
        Based on this analysis, we'll implement a BFS solution.
      `;

      // Process with original system
      const originalResult = await quietSTaRProcessor.processReasoning(complexReasoningText);

      // Process with BitNet system
      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);
      const bitNetResult = await bitNetProcessor.processReasoning(complexReasoningText);

      // Verify reasoning structure preservation
      expect(bitNetResult.logicalFlow.hierarchicalStructure)
        .toEqual(originalResult.logicalFlow.hierarchicalStructure);

      expect(bitNetResult.logicalFlow.steps).toHaveLength(originalResult.logicalFlow.steps);

      // Check reasoning depth and complexity
      expect(bitNetResult.reasoningDepth).toBeGreaterThanOrEqual(
        originalResult.reasoningDepth * 0.9
      );

      expect(bitNetResult.conceptualLinks).toHaveLength(
        originalResult.conceptualLinks.length
      );
    });

    it('should preserve causal reasoning chains', async () => {
      const causalReasoningText = `
        <think>
        If we increase the temperature, then the reaction rate will increase.
        This is because higher temperature means more kinetic energy.
        More kinetic energy leads to more frequent molecular collisions.
        More collisions result in higher probability of successful reactions.
        Therefore, the overall reaction rate increases.
        </think>
        The temperature increase will accelerate the chemical reaction.
      `;

      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);
      const result = await bitNetProcessor.processReasoning(causalReasoningText);

      // Verify causal chain preservation
      expect(result.causalChain).toBeDefined();
      expect(result.causalChain.links).toHaveLength(4); // Four causal connections

      // Check logical consistency of causal relationships
      result.causalChain.links.forEach(link => {
        expect(link.confidence).toBeGreaterThan(0.8);
        expect(link.causality_strength).toBeGreaterThan(0.7);
      });

      // Verify conclusion follows from reasoning
      expect(result.conclusionAlignment).toBeGreaterThan(0.9);
    });
  });

  describe('Performance with Reasoning Tasks', () => {
    it('should maintain reasoning performance after BitNet conversion', async () => {
      const reasoningTasks = generateReasoningTasks(50); // 50 diverse reasoning tasks

      // Benchmark original system
      const originalStart = performance.now();
      const originalResults = await Promise.all(
        reasoningTasks.map(task => quietSTaRProcessor.processReasoning(task))
      );
      const originalTime = performance.now() - originalStart;

      // Benchmark BitNet system
      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);
      const bitNetStart = performance.now();
      const bitNetResults = await Promise.all(
        reasoningTasks.map(task => bitNetProcessor.processReasoning(task))
      );
      const bitNetTime = performance.now() - bitNetStart;

      // Performance should be comparable or better
      expect(bitNetTime).toBeLessThan(originalTime * 1.2); // At most 20% slower

      // Quality should be preserved
      const qualityCorrelation = calculateBatchReasoningCorrelation(
        originalResults,
        bitNetResults
      );
      expect(qualityCorrelation).toBeGreaterThan(0.85);

      console.log(`Performance ratio (BitNet/Original): ${(bitNetTime / originalTime).toFixed(3)}`);
      console.log(`Quality correlation: ${qualityCorrelation.toFixed(3)}`);
    });

    it('should handle concurrent reasoning efficiently', async () => {
      const concurrentTasks = Array.from({ length: 20 }, (_, i) =>
        `Task ${i}: <think> This requires analysis of problem ${i} with complexity level ${i % 5} </think> solution`
      );

      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);

      // Process tasks concurrently
      const startTime = performance.now();
      const results = await Promise.allSettled(
        concurrentTasks.map(task => bitNetProcessor.processReasoning(task))
      );
      const endTime = performance.now();

      // All tasks should complete successfully
      const successfulResults = results.filter(result => result.status === 'fulfilled');
      expect(successfulResults).toHaveLength(concurrentTasks.length);

      // Should complete efficiently
      const avgTimePerTask = (endTime - startTime) / concurrentTasks.length;
      expect(avgTimePerTask).toBeLessThan(50); // Less than 50ms per task on average
    });
  });

  describe('Memory Efficiency with Reasoning', () => {
    it('should reduce memory usage while preserving reasoning quality', async () => {
      const memoryProfiler = new MemoryProfiler();

      // Measure original system memory usage
      const originalMemoryBefore = memoryProfiler.getCurrentMemoryUsage();
      const originalProcessor = quietSTaRProcessor;
      const originalMemoryAfter = memoryProfiler.getCurrentMemoryUsage();
      const originalMemoryUsage = originalMemoryAfter.heapUsed - originalMemoryBefore.heapUsed;

      // Measure BitNet system memory usage
      const bitNetMemoryBefore = memoryProfiler.getCurrentMemoryUsage();
      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);
      const bitNetMemoryAfter = memoryProfiler.getCurrentMemoryUsage();
      const bitNetMemoryUsage = bitNetMemoryAfter.heapUsed - bitNetMemoryBefore.heapUsed;

      // BitNet should use significantly less memory
      const memoryReduction = originalMemoryUsage / bitNetMemoryUsage;
      expect(memoryReduction).toBeGreaterThan(4); // At least 4x reduction

      // Test reasoning quality is preserved
      const testReasoning = '<think> Complex reasoning requires multiple steps of analysis </think> conclusion';

      const originalResult = await originalProcessor.processReasoning(testReasoning);
      const bitNetResult = await bitNetProcessor.processReasoning(testReasoning);

      const qualityCorrelation = calculateReasoningCorrelation(
        originalResult.reasoningEmbeddings,
        bitNetResult.reasoningEmbeddings
      );

      expect(qualityCorrelation).toBeGreaterThan(0.8);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle malformed reasoning tokens gracefully', async () => {
      const malformedInputs = [
        'Unclosed thinking <think> this has no end',
        'Nested without proper closure <think> outer <think> inner </think> missing close',
        'Empty reasoning blocks <think></think> should work',
        'Multiple consecutive <think></think><think></think> blocks'
      ];

      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);

      for (const input of malformedInputs) {
        const result = await bitNetProcessor.processReasoning(input);

        expect(result).toBeDefined();
        expect(result.errorHandling.gracefulDegradation).toBe(true);
        expect(result.processingSuccess).toBe(true);
      }
    });

    it('should validate reasoning quality and detect degradation', async () => {
      const qualityTestCases = [
        {
          text: '<think> This is high-quality reasoning with clear logical steps </think>',
          expectedQuality: 0.9
        },
        {
          text: '<think> low quality unclear reasoning </think>',
          expectedQuality: 0.4
        },
        {
          text: '<think> Detailed analysis: 1) Identify problem 2) Analyze constraints 3) Develop solution 4) Validate approach </think>',
          expectedQuality: 0.95
        }
      ];

      const bitNetProcessor = await reasoningPreserver.convertToBitNet(quietSTaRProcessor);

      for (const testCase of qualityTestCases) {
        const result = await bitNetProcessor.processReasoning(testCase.text);

        expect(result.qualityMetrics.overallQuality).toBeCloseTo(
          testCase.expectedQuality,
          1 // Within 0.1 tolerance
        );

        expect(result.qualityMetrics.logicalCoherence).toBeGreaterThan(0.7);
        expect(result.qualityMetrics.reasoningDepth).toBeGreaterThan(0.5);
      }
    });
  });
});

// Helper functions
function calculateReasoningCorrelation(embedding1: Float32Array, embedding2: Float32Array): number {
  if (embedding1.length !== embedding2.length) {
    return 0;
  }

  return calculateCosineSimilarity(Array.from(embedding1), Array.from(embedding2));
}

function calculateCosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  return magnitudeA === 0 || magnitudeB === 0 ? 0 : dotProduct / (magnitudeA * magnitudeB);
}

function calculateMagnitudeRatio(a: Float32Array | number[], b: Float32Array | number[]): number {
  const arrayA = Array.isArray(a) ? a : Array.from(a);
  const arrayB = Array.isArray(b) ? b : Array.from(b);

  const magnitudeA = Math.sqrt(arrayA.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(arrayB.reduce((sum, val) => sum + val * val, 0));

  return magnitudeA === 0 ? (magnitudeB === 0 ? 1 : 0) : magnitudeB / magnitudeA;
}

function generateReasoningTasks(count: number): string[] {
  const templates = [
    'Analyze <think> breaking down the problem into components: {analysis} </think> the solution',
    'Consider <think> evaluating options: {options} leading to {conclusion} </think> the best approach',
    'Examine <think> step-by-step reasoning: {steps} </think> the methodology',
    'Investigate <think> causal relationships: {causes} result in {effects} </think> the implications'
  ];

  return Array.from({ length: count }, (_, i) => {
    const template = templates[i % templates.length];
    return template
      .replace('{analysis}', `analysis_${i}`)
      .replace('{options}', `option_A_${i}, option_B_${i}`)
      .replace('{conclusion}', `conclusion_${i}`)
      .replace('{steps}', `step1_${i}, step2_${i}, step3_${i}`)
      .replace('{causes}', `cause_${i}`)
      .replace('{effects}', `effect_${i}`);
  });
}

function calculateBatchReasoningCorrelation(original: any[], converted: any[]): number {
  if (original.length !== converted.length) {
    return 0;
  }

  const correlations = original.map((orig, i) => {
    const conv = converted[i];
    return calculateReasoningCorrelation(orig.reasoningEmbeddings, conv.reasoningEmbeddings);
  });

  return correlations.reduce((sum, corr) => sum + corr, 0) / correlations.length;
}

class MemoryProfiler {
  getCurrentMemoryUsage() {
    return process.memoryUsage();
  }
}
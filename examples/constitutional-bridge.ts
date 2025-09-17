/**
 * AIVillage Bridge System Usage Examples
 * Demonstrates comprehensive integration and monitoring capabilities
 */

import BridgeOrchestrator, { BridgeConfig } from '../src/bridge/BridgeOrchestrator';
import AIVillageInterface from '../src/bridge/interfaces/AIVillageInterface';
import BetaNetInterface from '../src/bridge/interfaces/BetaNetInterface';
import ConstitutionalInterface from '../src/bridge/interfaces/ConstitutionalInterface';

/**
 * Example 1: Basic Bridge Setup and Initialization
 */
async function exampleBasicSetup() {
  console.log('=== Example 1: Basic Bridge Setup ===');

  const config: BridgeConfig = {
    healthMonitoring: {
      enabled: true,
      checkInterval: 30000,
      thresholds: {
        responseTimeWarning: 1000,
        responseTimeCritical: 5000,
        errorRateWarning: 0.05,
        errorRateCritical: 0.1,
        consecutiveFailures: 3
      }
    },
    metricsCollection: {
      enabled: true,
      collectionInterval: 10000,
      retentionPeriod: 7 * 24 * 60 * 60 * 1000 // 7 days
    },
    integrations: {
      aivillageCore: {
        enabled: true,
        endpoint: 'http://localhost:3000',
        timeout: 30000,
        retryPolicy: {
          maxAttempts: 3,
          initialDelay: 1000,
          maxDelay: 5000,
          backoffMultiplier: 2
        },
        authentication: {
          type: 'bearer',
          credentials: { token: 'your-aivillage-token' }
        }
      },
      betanetTransport: {
        enabled: true,
        endpoint: 'ws://localhost:8545',
        timeout: 15000,
        retryPolicy: {
          maxAttempts: 5,
          initialDelay: 2000,
          maxDelay: 10000,
          backoffMultiplier: 1.5
        }
      },
      constitutionalAI: {
        enabled: true,
        endpoint: 'http://localhost:4000',
        timeout: 20000,
        retryPolicy: {
          maxAttempts: 3,
          initialDelay: 1000,
          maxDelay: 8000,
          backoffMultiplier: 2
        },
        authentication: {
          type: 'api_key',
          credentials: { apiKey: 'your-constitutional-api-key' }
        }
      },
      externalMonitoring: {
        enabled: true,
        endpoint: 'http://localhost:9090',
        timeout: 10000,
        retryPolicy: {
          maxAttempts: 2,
          initialDelay: 500,
          maxDelay: 2000,
          backoffMultiplier: 2
        }
      }
    },
    recovery: {
      autoRecoveryEnabled: true,
      maxRecoveryAttempts: 3,
      backoffStrategy: 'exponential'
    }
  };

  const orchestrator = new BridgeOrchestrator(config);

  try {
    // Start the orchestrator
    await orchestrator.start();

    // Get initial system status
    const status = await orchestrator.getSystemStatus();
    console.log('System Status:', JSON.stringify(status, null, 2));

    // Export diagnostics
    const diagnostics = await orchestrator.exportDiagnostics();
    console.log('Diagnostics exported with', Object.keys(diagnostics).length, 'sections');

  } catch (error) {
    console.error('Setup failed:', error);
  } finally {
    await orchestrator.stop();
  }
}

/**
 * Example 2: AIVillage Integration Usage
 */
async function exampleAIVillageIntegration() {
  console.log('=== Example 2: AIVillage Integration ===');

  const aivillage = new AIVillageInterface(
    'http://localhost:3000',
    'your-aivillage-token'
  );

  try {
    // Health check
    const health = await aivillage.healthCheck();
    console.log('AIVillage Health:', health);

    // Get available agents
    const agents = await aivillage.getAgents();
    console.log(`Found ${agents.length} agents:`, agents.map(a => a.name));

    // Create a conversation
    const conversation = await aivillage.createConversation(
      'System Integration Test',
      [agents[0]?.id || 'test-agent']
    );
    console.log('Created conversation:', conversation.id);

    // Send a message
    if (agents.length > 0) {
      const response = await aivillage.sendMessage(
        conversation.id,
        agents[0].id,
        'Hello from the bridge system!'
      );
      console.log('Agent response:', response.content);

      // Get conversation analytics
      const analytics = await aivillage.getConversationAnalytics(conversation.id);
      console.log('Conversation analytics:', analytics);
    }

    // Get system metrics
    const metrics = await aivillage.getMetrics();
    console.log('AIVillage metrics:', metrics);

    // Subscribe to real-time events
    const unsubscribe = aivillage.subscribeToEvents(
      ['agent_response', 'conversation_update'],
      (event) => {
        console.log('AIVillage event:', event.type, event.data);
      }
    );

    // Cleanup after 10 seconds
    setTimeout(() => {
      unsubscribe();
      console.log('Unsubscribed from AIVillage events');
    }, 10000);

  } catch (error) {
    console.error('AIVillage integration error:', error);
  }
}

/**
 * Example 3: BetaNet Network Integration
 */
async function exampleBetaNetIntegration() {
  console.log('=== Example 3: BetaNet Integration ===');

  const betanet = new BetaNetInterface({
    nodeId: 'bridge-node-001',
    networkId: 'aivillage-network',
    bootstrapNodes: ['node1.betanet.local:8545', 'node2.betanet.local:8545'],
    maxConnections: 50,
    syncTimeout: 60000,
    heartbeatInterval: 30000
  });

  try {
    // Connect to network
    await betanet.connect();
    console.log('Connected to BetaNet');

    // Get network state
    const networkState = await betanet.getNetworkState();
    console.log('Network state:', networkState);

    // Get connected nodes
    const nodes = await betanet.getConnectedNodes();
    console.log(`Connected to ${nodes.length} nodes`);

    // Subscribe to network events
    betanet.on('nodeJoined', (data) => {
      console.log('Node joined network:', data.nodeId);
    });

    betanet.on('transactionUpdate', (data) => {
      console.log('Transaction update:', data.transactionId, data.status);
    });

    // Send a message to specific nodes
    if (nodes.length > 0) {
      const messageId = await betanet.sendMessage({
        sourceNodeId: 'bridge-node-001',
        destinationNodeIds: [nodes[0].id],
        messageType: 'data',
        payload: { greeting: 'Hello from bridge', timestamp: new Date() },
        priority: 'normal',
        encryption: false,
        ttl: 300
      });
      console.log('Message sent:', messageId);
    }

    // Broadcast a message
    const broadcastId = await betanet.broadcastMessage(
      'control',
      { type: 'system_status', status: 'operational' },
      'normal'
    );
    console.log('Broadcast message:', broadcastId);

    // Initiate a transaction
    if (nodes.length >= 2) {
      const transaction = await betanet.initiateTransaction(
        [nodes[0].id, nodes[1].id],
        { operation: 'data_sync', data: 'test-data' }
      );
      console.log('Transaction initiated:', transaction.id);

      // Confirm transaction
      await betanet.confirmTransaction(transaction.id);
      console.log('Transaction confirmed');
    }

    // Get metrics
    const metrics = await betanet.getMetrics();
    console.log('BetaNet metrics:', metrics);

  } catch (error) {
    console.error('BetaNet integration error:', error);
  } finally {
    await betanet.disconnect();
  }
}

/**
 * Example 4: Constitutional AI Validation
 */
async function exampleConstitutionalValidation() {
  console.log('=== Example 4: Constitutional AI Validation ===');

  const constitutional = new ConstitutionalInterface(
    'http://localhost:4000',
    'your-constitutional-api-key'
  );

  try {
    // Health check
    const health = await constitutional.healthCheck();
    console.log('Constitutional AI health:', health);

    // Get available rules
    const rules = await constitutional.getRules();
    console.log(`Found ${rules.length} constitutional rules`);

    // Validate a message
    const messageValidation = await constitutional.validateMessage(
      'Hello, can you help me with this task?',
      {
        userId: 'user-123',
        sessionId: 'session-456',
        agentId: 'agent-789',
        timestamp: new Date()
      }
    );
    console.log('Message validation result:', messageValidation.overallResult);
    if (messageValidation.violations.length > 0) {
      console.log('Violations found:', messageValidation.violations);
    }

    // Validate an action
    const actionValidation = await constitutional.validateAction(
      {
        type: 'file_access',
        target: '/home/user/documents/report.pdf',
        operation: 'read'
      },
      {
        userId: 'user-123',
        agentId: 'agent-789',
        timestamp: new Date()
      }
    );
    console.log('Action validation result:', actionValidation.overallResult);

    // Get compliance score
    const complianceScore = await constitutional.getComplianceScore();
    console.log('Compliance score:', complianceScore);

    // Create a new rule
    const newRule = await constitutional.createRule({
      name: 'Bridge System Safety Rule',
      description: 'Prevents dangerous bridge operations',
      category: 'safety',
      priority: 'high',
      conditions: [
        {
          field: 'operation',
          operator: 'equals',
          value: 'system_shutdown'
        }
      ],
      actions: [
        {
          type: 'block',
          severity: 'high',
          message: 'System shutdown operations require manual approval'
        }
      ],
      enabled: true
    });
    console.log('Created new rule:', newRule.id);

    // Test the rule
    const testResult = await constitutional.testRule(
      newRule.id,
      { operation: 'system_shutdown', user: 'automated' }
    );
    console.log('Rule test result:', testResult.matched);

    // Get violation statistics
    const violationStats = await constitutional.getViolationStats();
    console.log('Violation statistics:', violationStats);

    // Bulk validate multiple requests
    const bulkResults = await constitutional.bulkValidate([
      {
        content: { text: 'This is a safe message' },
        contentType: 'message',
        context: { timestamp: new Date() },
        requestedValidations: ['safety', 'content'],
        priority: 'normal'
      },
      {
        content: { action: 'read_file', file: 'config.json' },
        contentType: 'action',
        context: { timestamp: new Date() },
        requestedValidations: ['safety', 'compliance'],
        priority: 'high'
      }
    ]);
    console.log('Bulk validation results:', bulkResults.map(r => r.overallResult));

  } catch (error) {
    console.error('Constitutional AI validation error:', error);
  }
}

/**
 * Example 5: Comprehensive Error Recovery Scenarios
 */
async function exampleErrorRecovery() {
  console.log('=== Example 5: Error Recovery Scenarios ===');

  const config: BridgeConfig = {
    healthMonitoring: {
      enabled: true,
      checkInterval: 5000, // Faster for demo
      thresholds: {
        responseTimeWarning: 500,
        responseTimeCritical: 2000,
        errorRateWarning: 0.02,
        errorRateCritical: 0.05,
        consecutiveFailures: 2
      }
    },
    metricsCollection: {
      enabled: true,
      collectionInterval: 5000,
      retentionPeriod: 60000 // 1 minute for demo
    },
    integrations: {
      aivillageCore: {
        enabled: true,
        endpoint: 'http://localhost:3000',
        timeout: 5000,
        retryPolicy: {
          maxAttempts: 2,
          initialDelay: 500,
          maxDelay: 2000,
          backoffMultiplier: 2
        }
      },
      betanetTransport: {
        enabled: false, // Disabled for demo
        endpoint: 'ws://localhost:8545',
        timeout: 5000,
        retryPolicy: { maxAttempts: 1, initialDelay: 1000, maxDelay: 1000, backoffMultiplier: 1 }
      },
      constitutionalAI: {
        enabled: true,
        endpoint: 'http://localhost:4000',
        timeout: 5000,
        retryPolicy: {
          maxAttempts: 2,
          initialDelay: 500,
          maxDelay: 2000,
          backoffMultiplier: 2
        }
      },
      externalMonitoring: {
        enabled: false, // Disabled for demo
        endpoint: 'http://localhost:9090',
        timeout: 5000,
        retryPolicy: { maxAttempts: 1, initialDelay: 500, maxDelay: 500, backoffMultiplier: 1 }
      }
    },
    recovery: {
      autoRecoveryEnabled: true,
      maxRecoveryAttempts: 2,
      backoffStrategy: 'exponential'
    }
  };

  const orchestrator = new BridgeOrchestrator(config);

  try {
    await orchestrator.start();
    console.log('Bridge orchestrator started');

    // Monitor system status for 30 seconds
    const statusInterval = setInterval(async () => {
      try {
        const status = await orchestrator.getSystemStatus();
        console.log(`System Status: ${status.overall} (${JSON.stringify(status.components)})`);

        // Check for degraded or unhealthy components
        const unhealthyComponents = Object.entries(status.components)
          .filter(([_, status]) => status === 'error' || status === 'disconnected')
          .map(([name, _]) => name);

        if (unhealthyComponents.length > 0) {
          console.log('Unhealthy components detected:', unhealthyComponents);

          // Trigger manual recovery for the first unhealthy component
          const component = unhealthyComponents[0];
          console.log(`Triggering recovery for ${component}...`);
          const recoveryId = await orchestrator.triggerRecovery(component, 'restart');
          console.log(`Recovery action started: ${recoveryId}`);
        }

        // Show capacity predictions if available
        if (status.capacity.predictions.length > 0) {
          console.log('Capacity predictions:', status.capacity.predictions.map(p =>
            `${p.metric}: ${Math.round(p.timeToCapacity / (24 * 60 * 60 * 1000))} days to capacity`
          ));
        }

        // Show recommendations
        if (status.capacity.recommendations.length > 0) {
          console.log('Recommendations:', status.capacity.recommendations.slice(0, 2));
        }

      } catch (error) {
        console.error('Status check error:', error);
      }
    }, 10000);

    // Stop monitoring after 30 seconds
    setTimeout(() => {
      clearInterval(statusInterval);
    }, 30000);

    // Wait for monitoring period
    await new Promise(resolve => setTimeout(resolve, 35000));

    // Get recovery history
    const recoveryHistory = orchestrator.getRecoveryHistory(10);
    if (recoveryHistory.length > 0) {
      console.log('Recovery actions performed:');
      recoveryHistory.forEach(action => {
        console.log(`- ${action.id}: ${action.component} ${action.action} (${action.status})`);
      });
    }

  } catch (error) {
    console.error('Error recovery example failed:', error);
  } finally {
    await orchestrator.stop();
  }
}

/**
 * Example 6: Performance Monitoring and Metrics Analysis
 */
async function examplePerformanceMonitoring() {
  console.log('=== Example 6: Performance Monitoring ===');

  const config: BridgeConfig = {
    healthMonitoring: {
      enabled: true,
      checkInterval: 2000,
      thresholds: {
        responseTimeWarning: 1000,
        responseTimeCritical: 3000,
        errorRateWarning: 0.1,
        errorRateCritical: 0.2,
        consecutiveFailures: 5
      }
    },
    metricsCollection: {
      enabled: true,
      collectionInterval: 1000, // Collect every second
      retentionPeriod: 300000 // 5 minutes
    },
    integrations: {
      aivillageCore: {
        enabled: true,
        endpoint: 'http://localhost:3000',
        timeout: 10000,
        retryPolicy: { maxAttempts: 3, initialDelay: 1000, maxDelay: 5000, backoffMultiplier: 2 }
      },
      betanetTransport: {
        enabled: false,
        endpoint: 'ws://localhost:8545',
        timeout: 10000,
        retryPolicy: { maxAttempts: 1, initialDelay: 1000, maxDelay: 1000, backoffMultiplier: 1 }
      },
      constitutionalAI: {
        enabled: true,
        endpoint: 'http://localhost:4000',
        timeout: 10000,
        retryPolicy: { maxAttempts: 3, initialDelay: 1000, maxDelay: 5000, backoffMultiplier: 2 }
      },
      externalMonitoring: {
        enabled: false,
        endpoint: 'http://localhost:9090',
        timeout: 5000,
        retryPolicy: { maxAttempts: 1, initialDelay: 500, maxDelay: 500, backoffMultiplier: 1 }
      }
    },
    recovery: {
      autoRecoveryEnabled: false, // Disabled for monitoring focus
      maxRecoveryAttempts: 1,
      backoffStrategy: 'linear'
    }
  };

  const orchestrator = new BridgeOrchestrator(config);

  try {
    await orchestrator.start();
    console.log('Performance monitoring started');

    // Run for 20 seconds collecting metrics
    await new Promise(resolve => setTimeout(resolve, 20000));

    // Get comprehensive diagnostics
    const diagnostics = await orchestrator.exportDiagnostics();

    console.log('\n=== Performance Analysis ===');
    console.log('System uptime:', Math.round(diagnostics.systemStatus.metrics.uptime / 1000), 'seconds');
    console.log('Total requests:', diagnostics.systemStatus.metrics.totalRequests);
    console.log('Error rate:', (diagnostics.systemStatus.metrics.errorRate * 100).toFixed(2), '%');
    console.log('Avg response time:', diagnostics.systemStatus.metrics.avgResponseTime, 'ms');

    console.log('\n=== Resource Usage ===');
    const metrics = diagnostics.metricsSnapshot;
    console.log('CPU usage:', metrics.systemMetrics.cpu.usage.toFixed(2), '%');
    console.log('Memory usage:', (metrics.systemMetrics.memory.heapUsed / metrics.systemMetrics.memory.heapTotal * 100).toFixed(2), '%');
    console.log('Load average:', metrics.systemMetrics.cpu.loadAverage.map(l => l.toFixed(2)).join(', '));

    console.log('\n=== Component Health ===');
    diagnostics.healthDetails.components.forEach(component => {
      console.log(`${component.component}: ${component.status} (${component.responseTime}ms, ${component.errorCount} errors)`);
    });

    console.log('\n=== Capacity Predictions ===');
    if (diagnostics.systemStatus.capacity.predictions.length > 0) {
      diagnostics.systemStatus.capacity.predictions.forEach(prediction => {
        const daysToCapacity = Math.round(prediction.timeToCapacity / (24 * 60 * 60 * 1000));
        console.log(`${prediction.metric}: ${daysToCapacity} days to capacity (${(prediction.confidence * 100).toFixed(1)}% confidence)`);
        prediction.recommendations.forEach(rec => console.log(`  - ${rec}`));
      });
    } else {
      console.log('No capacity concerns detected');
    }

    console.log('\n=== System Recommendations ===');
    diagnostics.systemStatus.capacity.recommendations.forEach(rec => {
      console.log(`- ${rec}`);
    });

  } catch (error) {
    console.error('Performance monitoring error:', error);
  } finally {
    await orchestrator.stop();
  }
}

/**
 * Run all examples
 */
async function runAllExamples() {
  console.log('AIVillage Bridge System - Comprehensive Examples\n');

  const examples = [
    exampleBasicSetup,
    exampleAIVillageIntegration,
    exampleBetaNetIntegration,
    exampleConstitutionalValidation,
    exampleErrorRecovery,
    examplePerformanceMonitoring
  ];

  for (let i = 0; i < examples.length; i++) {
    try {
      await examples[i]();
      console.log(`\nExample ${i + 1} completed successfully\n`);
    } catch (error) {
      console.error(`\nExample ${i + 1} failed:`, error, '\n');
    }

    // Wait between examples
    if (i < examples.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  console.log('All examples completed!');
}

// Export functions for individual use
export {
  exampleBasicSetup,
  exampleAIVillageIntegration,
  exampleBetaNetIntegration,
  exampleConstitutionalValidation,
  exampleErrorRecovery,
  examplePerformanceMonitoring,
  runAllExamples
};

// Run examples if this file is executed directly
if (require.main === module) {
  runAllExamples().catch(console.error);
}
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import SystemMetricsPanel from './SystemMetricsPanel';

describe('SystemMetricsPanel', () => {
  const mockMetrics = {
    cpuUsage: 45,
    memoryUsage: 67,
    networkTraffic: 35,
    activeConnections: 25,
    messagesThroughput: 150,
    errorRate: 0.5
  };

  const mockProps = {
    metrics: mockMetrics,
    isMonitoring: true,
    onToggleMonitoring: jest.fn()
  };

  test('renders system metrics', () => {
    render(<SystemMetricsPanel {...mockProps} />);

    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    expect(screen.getByText('45.0%')).toBeInTheDocument(); // CPU usage
    expect(screen.getByText('67.0%')).toBeInTheDocument(); // Memory usage
    expect(screen.getByText('35.0%')).toBeInTheDocument(); // Network traffic
  });

  test('displays correct metric labels', () => {
    render(<SystemMetricsPanel {...mockProps} />);

    expect(screen.getByText('CPU Usage')).toBeInTheDocument();
    expect(screen.getByText('Memory')).toBeInTheDocument();
    expect(screen.getByText('Network')).toBeInTheDocument();
    expect(screen.getByText('Error Rate')).toBeInTheDocument();
  });

  test('handles missing metrics gracefully', () => {
    const emptyMetrics = {
      cpuUsage: 0,
      memoryUsage: 0,
      networkTraffic: 0,
      activeConnections: 0,
      messagesThroughput: 0,
      errorRate: 0
    };

    render(<SystemMetricsPanel metrics={emptyMetrics} isMonitoring={false} onToggleMonitoring={jest.fn()} />);

    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    // Should not crash with empty metrics
  });
});

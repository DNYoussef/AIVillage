import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import SystemMetricsPanel from './SystemMetricsPanel';

describe('SystemMetricsPanel', () => {
  const mockMetrics = {
    cpuUsage: 45,
    memoryUsage: 67,
    networkLatency: 120,
    activeConnections: 25,
    uptime: '2 days, 5 hours'
  };

  test('renders system metrics', () => {
    render(<SystemMetricsPanel metrics={mockMetrics} />);

    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    expect(screen.getByText('45%')).toBeInTheDocument(); // CPU usage
    expect(screen.getByText('67%')).toBeInTheDocument(); // Memory usage
    expect(screen.getByText('120ms')).toBeInTheDocument(); // Network latency
  });

  test('displays correct metric labels', () => {
    render(<SystemMetricsPanel metrics={mockMetrics} />);

    expect(screen.getByText('CPU Usage')).toBeInTheDocument();
    expect(screen.getByText('Memory Usage')).toBeInTheDocument();
    expect(screen.getByText('Network Latency')).toBeInTheDocument();
    expect(screen.getByText('Active Connections')).toBeInTheDocument();
  });

  test('handles missing metrics gracefully', () => {
    render(<SystemMetricsPanel metrics={{}} />);

    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    // Should not crash with empty metrics
  });
});

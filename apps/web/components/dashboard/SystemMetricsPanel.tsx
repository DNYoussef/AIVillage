// System Metrics Panel - Real-time system performance monitoring
import React, { useRef, useEffect } from 'react';
import './SystemMetricsPanel.css';

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkTraffic: number;
  activeConnections: number;
  messagesThroughput: number;
  errorRate: number;
}

interface SystemMetricsPanelProps {
  metrics: SystemMetrics;
  isMonitoring: boolean;
  onToggleMonitoring: () => void;
  className?: string;
}

export const SystemMetricsPanel: React.FC<SystemMetricsPanelProps> = ({
  metrics,
  isMonitoring,
  onToggleMonitoring,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const getMetricColor = (value: number, thresholds: { good: number; fair: number }) => {
    if (value <= thresholds.good) return '#10b981'; // green
    if (value <= thresholds.fair) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  const getMetricStatus = (value: number, thresholds: { good: number; fair: number }) => {
    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.fair) return 'fair';
    return 'poor';
  };

  // Simple chart rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isMonitoring) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = (height / 10) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw metrics as bars
    const barWidth = width / 4;
    const metricsData = [
      { value: metrics.cpuUsage, color: getMetricColor(metrics.cpuUsage, { good: 50, fair: 80 }), label: 'CPU' },
      { value: metrics.memoryUsage, color: getMetricColor(metrics.memoryUsage, { good: 60, fair: 85 }), label: 'Memory' },
      { value: metrics.networkTraffic, color: getMetricColor(metrics.networkTraffic, { good: 70, fair: 90 }), label: 'Network' },
      { value: metrics.errorRate * 20, color: getMetricColor(metrics.errorRate, { good: 1, fair: 3 }), label: 'Errors' }
    ];

    metricsData.forEach((metric, index) => {
      const x = index * barWidth;
      const barHeight = (metric.value / 100) * height;

      ctx.fillStyle = metric.color;
      ctx.fillRect(x + 10, height - barHeight, barWidth - 20, barHeight);

      // Draw labels
      ctx.fillStyle = '#ffffff';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(metric.label, x + barWidth / 2, height - 5);
      ctx.fillText(`${metric.value.toFixed(1)}%`, x + barWidth / 2, height - barHeight - 5);
    });
  }, [metrics, isMonitoring]);

  return (
    <div className={`system-metrics-panel ${className}`}>
      <div className="panel-header">
        <h3>System Metrics</h3>
        <div className="monitoring-controls">
          <button
            className={`monitoring-toggle ${isMonitoring ? 'active' : ''}`}
            onClick={onToggleMonitoring}
            title={isMonitoring ? 'Stop monitoring' : 'Start monitoring'}
          >
            {isMonitoring ? '⏸️' : '▶️'}
          </button>
          <span className="monitoring-status">
            {isMonitoring ? 'Live' : 'Paused'}
          </span>
        </div>
      </div>

      <div className="metrics-grid">
        <div className={`metric-card ${getMetricStatus(metrics.cpuUsage, { good: 50, fair: 80 })}`}>
          <div className="metric-header">
            <span className="metric-icon">🖥️</span>
            <span className="metric-name">CPU Usage</span>
          </div>
          <div className="metric-value">
            {metrics.cpuUsage.toFixed(1)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-fill"
              style={{
                width: `${Math.min(metrics.cpuUsage, 100)}%`,
                backgroundColor: getMetricColor(metrics.cpuUsage, { good: 50, fair: 80 })
              }}
            ></div>
          </div>
        </div>

        <div className={`metric-card ${getMetricStatus(metrics.memoryUsage, { good: 60, fair: 85 })}`}>
          <div className="metric-header">
            <span className="metric-icon">🧠</span>
            <span className="metric-name">Memory</span>
          </div>
          <div className="metric-value">
            {metrics.memoryUsage.toFixed(1)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-fill"
              style={{
                width: `${Math.min(metrics.memoryUsage, 100)}%`,
                backgroundColor: getMetricColor(metrics.memoryUsage, { good: 60, fair: 85 })
              }}
            ></div>
          </div>
        </div>

        <div className={`metric-card ${getMetricStatus(metrics.networkTraffic, { good: 70, fair: 90 })}`}>
          <div className="metric-header">
            <span className="metric-icon">🌐</span>
            <span className="metric-name">Network</span>
          </div>
          <div className="metric-value">
            {metrics.networkTraffic.toFixed(1)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-fill"
              style={{
                width: `${Math.min(metrics.networkTraffic, 100)}%`,
                backgroundColor: getMetricColor(metrics.networkTraffic, { good: 70, fair: 90 })
              }}
            ></div>
          </div>
        </div>

        <div className={`metric-card ${getMetricStatus(metrics.errorRate, { good: 1, fair: 3 })}`}>
          <div className="metric-header">
            <span className="metric-icon">⚠️</span>
            <span className="metric-name">Error Rate</span>
          </div>
          <div className="metric-value">
            {metrics.errorRate.toFixed(2)}%
          </div>
          <div className="metric-bar">
            <div
              className="metric-fill"
              style={{
                width: `${Math.min(metrics.errorRate * 10, 100)}%`,
                backgroundColor: getMetricColor(metrics.errorRate, { good: 1, fair: 3 })
              }}
            ></div>
          </div>
        </div>
      </div>

      <div className="additional-metrics">
        <div className="metric-item">
          <span className="metric-label">Active Connections:</span>
          <span className="metric-data">{metrics.activeConnections}</span>
        </div>
        <div className="metric-item">
          <span className="metric-label">Messages/sec:</span>
          <span className="metric-data">{metrics.messagesThroughput}</span>
        </div>
      </div>

      {isMonitoring && (
        <div className="metrics-chart">
          <canvas
            ref={canvasRef}
            width={300}
            height={100}
            className="metrics-canvas"
          />
        </div>
      )}
    </div>
  );
};

export default SystemMetricsPanel;

// Utility formatters for BitChat and AIVillage UI components

/**
 * Format bytes to human readable format
 */
export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Format duration in milliseconds to human readable format
 */
export const formatDuration = (ms: number): string => {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
};

/**
 * Format timestamp to relative time (e.g., "2 hours ago")
 */
export const formatRelativeTime = (date: Date): string => {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
};

/**
 * Format number with appropriate suffixes (K, M, B)
 */
export const formatNumber = (num: number): string => {
  if (num >= 1000000000) {
    return (num / 1000000000).toFixed(1) + 'B';
  }
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
};

/**
 * Format percentage with appropriate precision
 */
export const formatPercentage = (value: number, decimals = 1): string => {
  return `${value.toFixed(decimals)}%`;
};

/**
 * Format compute credits with currency-like formatting
 */
export const formatCredits = (credits: number): string => {
  if (credits >= 1000000) {
    return (credits / 1000000).toFixed(2) + 'M';
  }
  if (credits >= 1000) {
    return (credits / 1000).toFixed(1) + 'K';
  }
  return credits.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

/**
 * Format network latency with appropriate units
 */
export const formatLatency = (ms: number): string => {
  if (ms < 1) {
    return `${(ms * 1000).toFixed(0)}μs`;
  }
  if (ms < 1000) {
    return `${ms.toFixed(0)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
};

/**
 * Format hash/ID for display (truncate with ellipsis)
 */
export const formatHash = (hash: string, length = 8): string => {
  if (hash.length <= length * 2) return hash;
  return `${hash.slice(0, length)}...${hash.slice(-length)}`;
};

/**
 * Format peer address for display
 */
export const formatPeerAddress = (address: string): string => {
  // Handle various address formats
  if (address.includes(':')) {
    const parts = address.split(':');
    if (parts.length === 2) {
      return `${parts[0]}:${parts[1]}`;
    }
  }
  return formatHash(address, 6);
};

/**
 * Format file size with appropriate units and colors
 */
export const formatFileSize = (bytes: number): { text: string; color: string } => {
  const formatted = formatBytes(bytes);
  let color = '#10b981'; // green

  if (bytes > 50 * 1024 * 1024) { // > 50MB
    color = '#ef4444'; // red
  } else if (bytes > 10 * 1024 * 1024) { // > 10MB
    color = '#f59e0b'; // yellow
  }

  return { text: formatted, color };
};

/**
 * Format connection quality based on various metrics
 */
export const formatConnectionQuality = (
  latency: number,
  packetLoss: number,
  stability: number
): { quality: string; color: string; score: number } => {
  // Calculate composite score
  const latencyScore = Math.max(0, 100 - latency); // Lower latency = higher score
  const lossScore = Math.max(0, 100 - (packetLoss * 20)); // Lower loss = higher score
  const stabilityScore = stability;

  const score = (latencyScore + lossScore + stabilityScore) / 3;

  let quality = 'Poor';
  let color = '#ef4444'; // red

  if (score >= 80) {
    quality = 'Excellent';
    color = '#10b981'; // green
  } else if (score >= 60) {
    quality = 'Good';
    color = '#22c55e'; // light green
  } else if (score >= 40) {
    quality = 'Fair';
    color = '#f59e0b'; // yellow
  }

  return { quality, color, score: Math.round(score) };
};

/**
 * Format bandwidth with appropriate units
 */
export const formatBandwidth = (bytesPerSecond: number): string => {
  const bits = bytesPerSecond * 8;

  if (bits >= 1000000000) {
    return `${(bits / 1000000000).toFixed(1)} Gbps`;
  }
  if (bits >= 1000000) {
    return `${(bits / 1000000).toFixed(1)} Mbps`;
  }
  if (bits >= 1000) {
    return `${(bits / 1000).toFixed(0)} Kbps`;
  }
  return `${bits.toFixed(0)} bps`;
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
};

/**
 * Format device type from user agent or device info
 */
export const formatDeviceType = (userAgent: string): { type: string; icon: string } => {
  const ua = userAgent.toLowerCase();

  if (ua.includes('mobile') || ua.includes('android') || ua.includes('iphone')) {
    return { type: 'Mobile', icon: '📱' };
  }
  if (ua.includes('tablet') || ua.includes('ipad')) {
    return { type: 'Tablet', icon: '📱' };
  }
  if (ua.includes('mac')) {
    return { type: 'Mac', icon: '💻' };
  }
  if (ua.includes('windows')) {
    return { type: 'Windows', icon: '🖥️' };
  }
  if (ua.includes('linux')) {
    return { type: 'Linux', icon: '🐧' };
  }

  return { type: 'Unknown', icon: '🖥️' };
};

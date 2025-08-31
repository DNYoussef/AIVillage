# C4 Architecture Diagrams for SecureAdminServer

## C4 Model Overview

The C4 model provides a hierarchical approach to architecture documentation with four levels:
1. **System Context** - How the system fits into the world
2. **Container** - High-level technology choices and communication
3. **Component** - Components within containers and their interactions
4. **Code** - Classes and interfaces (covered in other documents)

## Level 1: System Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Village System                        │
│                                                                 │
│  ┌──────────────────┐                                          │
│  │   Admin User     │                                          │
│  │  (Super Admin,   │                                          │
│  │   System Admin,  │                                          │
│  │    Operator)     │                                          │
│  └─────────┬────────┘                                          │
│            │                                                   │
│            │ HTTPS (localhost:3006)                            │
│            │ Multi-factor Auth                                 │
│            │                                                   │
│  ┌─────────▼────────┐      ┌──────────────────┐               │
│  │                  │      │                  │               │
│  │ SecureAdminServer│◄─────┤ Authentication   │               │
│  │                  │      │     Service      │               │
│  │ • System Status  │      │                  │               │
│  │ • User Management│      └──────────────────┘               │
│  │ • Audit Logs     │                                         │
│  │ • Security Scans │      ┌──────────────────┐               │
│  │ • Emergency Ops  │◄─────┤    Database      │               │
│  │                  │      │   (Sessions,     │               │
│  └─────────┬────────┘      │   Users, Audit)  │               │
│            │               └──────────────────┘               │
│            │                                                   │
│            │ API Calls                                         │
│            │                                                   │
│  ┌─────────▼────────┐      ┌──────────────────┐               │
│  │                  │      │                  │               │
│  │   Core Services  │      │   P2P Network    │               │
│  │  (AI Agents,     │      │   (Federated     │               │
│  │   Fog Computing, │      │    Learning,     │               │
│  │   Marketplace)   │      │   Blockchain)    │               │
│  │                  │      │                  │               │
│  └──────────────────┘      └──────────────────┘               │
└─────────────────────────────────────────────────────────────────┘

Key:
- Admin users access the system through secure localhost-only interface
- Multi-factor authentication required for all admin operations
- System integrates with core AI Village services for monitoring and control
- Database stores sessions, user data, and audit logs
- P2P network provides federated learning and blockchain capabilities
```

## Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SecureAdminServer System                         │
│                                                                         │
│  ┌──────────────┐    HTTPS/REST     ┌─────────────────────────────────┐ │
│  │   Admin      │◄─────────────────►│                                 │ │
│  │   Client     │   (localhost:3006) │      FastAPI Application        │ │
│  │              │                    │                                 │ │
│  └──────────────┘                    │  ┌─────────────────────────┐   │ │
│                                       │  │   Security Middleware   │   │ │
│                                       │  │  • Security Headers     │   │ │
│                                       │  │  • Localhost Guard      │   │ │
│                                       │  │  • Audit Logging        │   │ │
│                                       │  │  • Rate Limiting        │   │ │
│                                       │  └─────────────────────────┘   │ │
│  ┌──────────────┐                    │                                 │ │
│  │   Session    │◄───────────────────┤  ┌─────────────────────────┐   │ │
│  │   Storage    │     In-Memory      │  │   Request Handlers      │   │ │
│  │  (Redis/     │     Sessions       │  │  • Auth Endpoints       │   │ │
│  │  In-Memory)  │                    │  │  • Admin Endpoints      │   │ │
│  └──────────────┘                    │  │  • Emergency Endpoints  │   │ │
│                                       │  └─────────────────────────┘   │ │
│  ┌──────────────┐                    │                                 │ │
│  │  Audit Log   │◄───────────────────┤  ┌─────────────────────────┐   │ │
│  │   Storage    │     File System    │  │   Security Services     │   │ │
│  │  (Files/     │     Logging        │  │  • Authentication       │   │ │
│  │  Database)   │                    │  │  • Authorization        │   │ │
│  └──────────────┘                    │  │  • MFA Verification     │   │ │
│                                       │  │  • Policy Enforcement   │   │ │
│  ┌──────────────┐                    │  └─────────────────────────┘   │ │
│  │   External   │◄───────────────────┤                                 │ │
│  │  Services    │     HTTPS API      │  ┌─────────────────────────┐   │ │
│  │  (Core AI    │     Calls          │  │   Configuration         │   │ │
│  │  Village,    │                    │  │  • Security Settings    │   │ │
│  │  Monitoring) │                    │  │  • Service Endpoints    │   │ │
│  └──────────────┘                    │  │  • Logging Config       │   │ │
│                                       │  └─────────────────────────┘   │ │
│                                       └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

Container Details:
- FastAPI Application: Main web application container
- Session Storage: In-memory or Redis for session management
- Audit Log Storage: File system or database for audit trails
- External Services: Integration points with core AI Village services
```

## Level 3: Component Diagram

### Authentication Module Components

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Authentication Module                              │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │   ISessionManager │    │IAuthService    │    │   IMFAService   │       │
│  │   Interface      │    │   Interface     │    │    Interface    │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │  SessionManager │    │CredentialValidator│   │  TOTPMFAService │       │
│  │                 │    │                 │    │                 │       │
│  │ • create_session│    │ • authenticate  │    │ • generate_code │       │
│  │ • validate      │    │ • get_user_roles│    │ • verify_token  │       │
│  │ • destroy       │    │ • hash_password │    │ • backup_codes  │       │
│  │ • mfa_required  │    │                 │    │                 │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │  SessionModel   │    │   UserModel     │    │   MFAModel      │       │
│  │                 │    │                 │    │                 │       │
│  │ • session_id    │    │ • user_id       │    │ • challenge_id  │       │
│  │ • user_id       │    │ • username      │    │ • token_hash    │       │
│  │ • roles         │    │ • roles         │    │ • expires_at    │       │
│  │ • permissions   │    │ • permissions   │    │ • verified      │       │
│  │ • client_ip     │    │ • password_hash │    │                 │       │
│  │ • mfa_verified  │    │ • mfa_enabled   │    │                 │       │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Security Module Components

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            Security Module                                │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │ISecurityContext │    │ IPolicyEngine   │    │IAccessController│       │
│  │   Interface     │    │   Interface     │    │   Interface     │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │SecurityContext  │    │AdminPolicyEngine│    │RoleBasedAccess  │       │
│  │   Service       │    │                 │    │   Controller    │       │
│  │                 │    │ • evaluate_role │    │                 │       │
│  │ • create_context│    │ • check_permission│  │ • check_access  │       │
│  │ • validate      │    │ • assess_risk   │    │ • enforce_policy│       │
│  │ • enrich        │    │ • audit_decision│    │ • log_decision  │       │
│  │ • extract       │    │                 │    │                 │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │SecurityContext  │    │ PolicyDecision  │    │ AccessDecision  │       │
│  │     Model       │    │     Model       │    │     Model       │       │
│  │                 │    │                 │    │                 │       │
│  │ • user_id       │    │ • decision      │    │ • allowed       │       │
│  │ • session_id    │    │ • resource      │    │ • reason        │       │
│  │ • roles         │    │ • action        │    │ • conditions    │       │
│  │ • permissions   │    │ • reasoning     │    │ • audit_info    │       │
│  │ • risk_score    │    │ • conditions    │    │                 │       │
│  │ • client_ip     │    │ • expires_at    │    │                 │       │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Middleware Module Components

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Middleware Module                                │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │ISecurityMiddleware│   │IMiddlewareFactory│   │  IAuditLogger   │       │
│  │   Interface     │    │   Interface     │    │   Interface     │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │SecurityHeaders  │    │ MiddlewareFactory│    │ AuditLogger     │       │
│  │   Middleware    │    │                 │    │   Service       │       │
│  │                 │    │ • create_headers│    │                 │       │
│  │ • add_headers   │    │ • create_guard  │    │ • log_request   │       │
│  │ • sanitize_resp │    │ • create_audit  │    │ • log_response  │       │
│  └─────────────────┘    │ • create_rate   │    │ • log_error     │       │
│                         └─────────────────┘    │ • format_entry  │       │
│  ┌─────────────────┐                           └─────────────────┘       │
│  │ LocalhostGuard  │    ┌─────────────────┐                              │
│  │   Middleware    │    │  RateLimiter    │    ┌─────────────────┐       │
│  │                 │    │   Middleware    │    │  AuditLogEntry  │       │
│  │ • check_ip      │    │                 │    │     Model       │       │
│  │ • block_external│    │ • check_limit   │    │                 │       │
│  │ • rate_limit    │    │ • update_count  │    │ • timestamp     │       │
│  │ • log_attempts  │    │ • block_abuse   │    │ • request_id    │       │
│  └─────────────────┘    └─────────────────┘    │ • user_id       │       │
│                                                │ • event_type    │       │
│                                                │ • outcome       │       │
│                                                │ • metadata      │       │
│                                                └─────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Handlers Module Components

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Handlers Module                                 │
│                                                                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │ IRequestHandler │    │ IHandlerFactory │    │ IResponseBuilder│       │
│  │   Interface     │    │   Interface     │    │   Interface     │       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘       │
│            │                      │                      │               │
│            ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │  BaseHandler    │    │ HandlerFactory  │    │ ResponseBuilder │       │
│  │                 │    │                 │    │                 │       │
│  │ • validate_req  │    │ • create_login  │    │ • success       │       │
│  │ • check_perms   │    │ • create_mfa    │    │ • error         │       │
│  │ • build_response│    │ • create_system │    │ • paginated     │       │
│  │ • handle_error  │    │ • create_audit  │    │ • streaming     │       │
│  └─────────┬───────┘    │ • create_emerg  │    └─────────────────┘       │
│            │            └─────────────────┘                              │
│            ▼                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│  │  LoginHandler   │    │ SystemHandler   │    │ EmergencyHandler│       │
│  │                 │    │                 │    │                 │       │
│  │ • handle_login  │    │ • get_status    │    │ • shutdown      │       │
│  │ • validate_creds│    │ • get_metrics   │    │ • validate_super│       │
│  └─────────────────┘    │ • health_check  │    │ • confirm_action│       │
│                         └─────────────────┘    └─────────────────┘       │
│  ┌─────────────────┐                                                     │
│  │   MFAHandler    │    ┌─────────────────┐    ┌─────────────────┐       │
│  │                 │    │  AuditHandler   │    │  UserHandler    │       │
│  │ • handle_mfa    │    │                 │    │                 │       │
│  │ • verify_token  │    │ • get_logs      │    │ • get_users     │       │
│  │ • generate_code │    │ • search_logs   │    │ • create_user   │       │
│  └─────────────────┘    │ • export_logs   │    │ • update_roles  │       │
│                         └─────────────────┘    └─────────────────┘       │
└───────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### Authentication Flow

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Admin User  │────►│  LoginHandler   │────►│SessionManager│
│              │     │                 │     │              │
└──────────────┘     └─────────┬───────┘     └──────┬───────┘
                              │                    │
                              ▼                    ▼
                     ┌─────────────────┐     ┌──────────────┐
                     │CredentialValidator│   │ SessionModel │
                     │                 │     │              │
                     └─────────┬───────┘     └──────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  MFAService     │
                     │                 │
                     └─────────┬───────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ SecurityContext │
                     │   Service       │
                     └─────────────────┘

Flow Steps:
1. User submits credentials to LoginHandler
2. LoginHandler calls CredentialValidator
3. On success, SessionManager creates session
4. MFAService generates challenge
5. SecurityContext created for subsequent requests
```

### Request Processing Flow

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│HTTP Request  │────►│Security Headers │────►│Localhost Guard│
│              │     │   Middleware    │     │  Middleware   │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                    │
                                                    ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Response   │◄────│ Request Handler │◄────│ Audit Logging│
│              │     │                 │     │  Middleware  │
└──────────────┘     └─────────┬───────┘     └──────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Security Context│
                     │    Validation   │
                     └─────────┬───────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Policy Engine   │
                     │   Evaluation    │
                     └─────────────────┘

Processing Steps:
1. Security headers added to request
2. Localhost guard validates IP address
3. Audit logging middleware logs request
4. Security context extracted and validated
5. Policy engine evaluates access permissions
6. Request handler processes business logic
7. Response returned with security headers
```

### Emergency Shutdown Flow

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│ Super Admin  │────►│EmergencyHandler │────►│SecurityContext│
│              │     │                 │     │  Validation  │
└──────────────┘     └─────────┬───────┘     └──────┬───────┘
                              │                    │
                              ▼                    ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Shutdown   │◄────│  Confirmation   │     │ Super Admin  │
│   Complete   │     │   Validation    │     │    Check     │
└──────────────┘     └─────────┬───────┘     └──────┬───────┘
                              │                    │
                              ▼                    ▼
                     ┌─────────────────┐     ┌──────────────┐
                     │   Audit Log     │     │ Policy Engine│
                     │    Entry        │     │  Override    │
                     └─────────────────┘     └──────────────┘

Emergency Steps:
1. Super admin triggers emergency endpoint
2. SecurityContext validates super admin role
3. Policy engine allows emergency override
4. Confirmation token validated
5. Emergency shutdown initiated
6. All operations audited
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Production Environment                           │
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │   Load Balancer │                                                    │
│  │   (localhost    │                                                    │
│  │    only)        │                                                    │
│  └─────────┬───────┘                                                    │
│            │                                                            │
│            ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Application Tier                             │   │
│  │                                                                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                   │   │
│  │  │SecureAdminServer│    │SecureAdminServer│                   │   │
│  │  │   Instance 1    │    │   Instance 2    │                   │   │
│  │  │  (Active)       │    │  (Standby)      │                   │   │
│  │  └─────────────────┘    └─────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Data Tier                                  │   │
│  │                                                                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐  ┌───────────────┐ │   │
│  │  │     Redis       │    │   PostgreSQL    │  │ File System   │ │   │
│  │  │   (Sessions)    │    │  (Users, Audit) │  │ (Audit Logs)  │ │   │
│  │  └─────────────────┘    └─────────────────┘  └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Monitoring Tier                               │   │
│  │                                                                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐  ┌───────────────┐ │   │
│  │  │   Prometheus    │    │    Grafana      │  │  ELK Stack    │ │   │
│  │  │   (Metrics)     │    │  (Dashboards)   │  │ (Log Analysis)│ │   │
│  │  └─────────────────┘    └─────────────────┘  └───────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Deployment Characteristics:
- High availability with active/standby instances
- Localhost-only binding (127.0.0.1) for security
- Distributed session storage with Redis
- Comprehensive monitoring and observability
- Audit logs stored in multiple locations for compliance
```

These C4 diagrams provide a comprehensive view of the SecureAdminServer architecture at multiple levels of abstraction, showing how the modular design promotes security, maintainability, and scalability while adhering to architectural best practices.
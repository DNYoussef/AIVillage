# AIVillage Security Architecture Design

## Executive Summary

This document provides a comprehensive security architecture design for AIVillage, addressing critical security gaps identified in the current system. The architecture emphasizes secure boundaries, localhost-only admin interfaces, comprehensive threat modeling, and SBOM/artifact signing integration.

## Current Security Analysis

### Identified Critical Security Issues

1. **Admin Interface Exposure**: Current admin servers bind to `0.0.0.0` (all interfaces) creating network exposure risks
2. **Missing Security Boundaries**: Weak separation between security domains and modules  
3. **Threat Model Integration Gap**: No automated threat modeling in development workflows
4. **SBOM/Signing Gaps**: Limited software bill of materials and artifact integrity verification
5. **Connascence Violations**: Strong security coupling across module boundaries

### Security Posture Assessment

**Current State:**
- Basic RBAC configuration exists but lacks enforcement depth
- CORS allows `*` origins in admin interfaces (high risk)  
- Security gates exist but need integration improvements
- P2P network security has good foundation but needs boundary enforcement

## Security Architecture Framework

### 1. Security Boundary Architecture

#### Core Principle: Defense in Depth with Clear Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                    🛡️ Security Perimeter                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Public    │───▶│  Gateway    │───▶│  Internal   │         │
│  │   Internet  │    │  Security   │    │  Services   │         │
│  │             │    │  Boundary   │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                            │                                    │
│         ┌─────────────────┬─┴─────────────────┐                 │
│         │                 │                   │                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Admin     │   │    P2P      │   │    Agent    │           │
│  │  Interface  │   │  Network    │   │   Forge     │           │
│  │ (localhost) │   │  Security   │   │  Security   │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Security Domain Boundaries

**Domain 1: Public Gateway (Internet-Facing)**
- **Boundary**: Network firewall + rate limiting + DDoS protection
- **Authentication**: Multi-factor authentication for admin access
- **Authorization**: JWT with short expiry + refresh tokens
- **Data Classification**: Public/Internal only, no sensitive data

**Domain 2: Internal Services (Private Network)**  
- **Boundary**: Internal service mesh with mTLS
- **Authentication**: Service-to-service certificates
- **Authorization**: RBAC with least privilege
- **Data Classification**: All sensitivity levels allowed

**Domain 3: Admin Interfaces (Localhost Only)**
- **Boundary**: Localhost binding (127.0.0.1) with IP validation
- **Authentication**: Multi-factor + hardware tokens
- **Authorization**: Super admin roles with approval workflows  
- **Data Classification**: Full access with audit logging

**Domain 4: P2P Network (Distributed)**
- **Boundary**: Encrypted mesh with node authentication
- **Authentication**: Certificate-based node identity
- **Authorization**: Reputation-based trust scores
- **Data Classification**: Encrypted payloads only

### 2. Secure Module Architecture

#### Connascence-Based Security Boundaries

Following connascence principles, security coupling must remain **local and weak**:

**Security Interface Contracts:**

```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from enum import Enum

class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal" 
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"

class SecurityContext(Protocol):
    """Security context contract - weak connascence interface"""
    user_id: str
    roles: list[str]
    permissions: set[str]
    security_level: SecurityLevel
    session_id: str
    
T = TypeVar('T')

class SecureBoundary(Generic[T], ABC):
    """Abstract security boundary with dependency injection"""
    
    def __init__(self, 
                 auth_service: 'AuthenticationService',
                 authz_service: 'AuthorizationService', 
                 audit_service: 'AuditService'):
        self._auth = auth_service
        self._authz = authz_service  
        self._audit = audit_service
    
    @abstractmethod
    async def validate_access(self, context: SecurityContext) -> bool:
        """Validate access - implementations stay local"""
        pass
    
    @abstractmethod
    async def execute_secured(self, context: SecurityContext, operation: callable) -> T:
        """Execute with security wrapper"""
        pass

class AdminBoundary(SecureBoundary[dict]):
    """Admin interface security boundary - localhost only"""
    
    async def validate_access(self, context: SecurityContext) -> bool:
        # Strong coupling ONLY within this class
        if not self._is_localhost_request(context):
            await self._audit.log_security_violation("non_localhost_admin_access", context)
            return False
            
        if not await self._auth.validate_mfa(context):
            await self._audit.log_security_violation("mfa_failure", context) 
            return False
            
        return await self._authz.check_admin_permission(context)
    
    def _is_localhost_request(self, context: SecurityContext) -> bool:
        """Local method - strong connascence contained"""
        return context.source_ip in ['127.0.0.1', '::1', 'localhost']

class P2PBoundary(SecureBoundary[bytes]):
    """P2P network security boundary"""
    
    async def validate_access(self, context: SecurityContext) -> bool:
        node_cert = await self._auth.get_node_certificate(context.node_id)
        trust_score = await self._get_trust_score(context.node_id)
        
        return (node_cert.is_valid() and 
                trust_score > 0.7 and 
                await self._authz.check_p2p_permission(context))
```

#### Security Service Architecture with Dependency Injection

**Core Security Services:**

```python
class SecurityServiceContainer:
    """Dependency injection container for security services"""
    
    def __init__(self):
        self._services = {}
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize security services with proper dependencies"""
        
        # Core services (no dependencies)
        self._services['crypto'] = CryptographyService()
        self._services['config'] = SecurityConfigService()
        
        # Authentication service
        self._services['auth'] = AuthenticationService(
            crypto_service=self._services['crypto'],
            config_service=self._services['config']
        )
        
        # Authorization service  
        self._services['authz'] = AuthorizationService(
            auth_service=self._services['auth'],
            config_service=self._services['config']
        )
        
        # Audit service
        self._services['audit'] = AuditService(
            crypto_service=self._services['crypto'],
            config_service=self._services['config']
        )
        
        # Threat detection service
        self._services['threat'] = ThreatDetectionService(
            auth_service=self._services['auth'],
            audit_service=self._services['audit']
        )
    
    def get_service(self, service_type: str):
        """Get service instance - dependency injection pattern"""
        return self._services.get(service_type)

    def create_boundary(self, boundary_type: str) -> SecureBoundary:
        """Create security boundary with injected dependencies"""
        auth = self.get_service('auth')
        authz = self.get_service('authz') 
        audit = self.get_service('audit')
        
        if boundary_type == 'admin':
            return AdminBoundary(auth, authz, audit)
        elif boundary_type == 'p2p':
            return P2PBoundary(auth, authz, audit)
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
```

### 3. Localhost-Only Admin Interface Implementation

#### Secure Admin Server Architecture

```python
class SecureAdminServer:
    """Localhost-only admin server with enhanced security"""
    
    def __init__(self, 
                 security_container: SecurityServiceContainer,
                 bind_interface: str = "127.0.0.1",
                 port: int = 3006):
        
        # CRITICAL: Never bind to 0.0.0.0 
        if bind_interface == "0.0.0.0":
            raise SecurityException("Admin interfaces must not bind to all interfaces")
            
        self.bind_interface = bind_interface
        self.port = port
        self.security = security_container
        self.app = self._create_secure_app()
    
    def _create_secure_app(self) -> FastAPI:
        """Create FastAPI app with security middleware"""
        
        app = FastAPI(
            title="AIVillage Secure Admin",
            description="Localhost-only admin interface",
            version="1.0.0",
            # Disable docs in production
            openapi_url="/openapi.json" if settings.DEBUG else None,
            docs_url="/docs" if settings.DEBUG else None
        )
        
        # Security middleware stack
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(LocalhostOnlyMiddleware, allowed_ips=['127.0.0.1', '::1'])
        app.add_middleware(AuditLoggingMiddleware, audit_service=self.security.get_service('audit'))
        app.add_middleware(ThreatDetectionMiddleware, threat_service=self.security.get_service('threat'))
        
        # CORS - NEVER allow wildcard for admin
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:3000", "https://localhost:3000"],  # Explicit only
            allow_credentials=True,
            allow_methods=["GET", "POST"], 
            allow_headers=["Content-Type", "Authorization"]
        )
        
        self._setup_routes(app)
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup secure admin routes"""
        
        admin_boundary = self.security.create_boundary('admin')
        
        @app.middleware("http")
        async def security_boundary_middleware(request: Request, call_next):
            """Apply security boundary to all requests"""
            
            context = await self._extract_security_context(request)
            
            if not await admin_boundary.validate_access(context):
                raise HTTPException(
                    status_code=403, 
                    detail="Access denied - admin privileges required"
                )
            
            response = await call_next(request)
            return response
        
        @app.get("/admin/system/health")
        async def admin_health_check(request: Request):
            """Secure health check with audit logging"""
            context = await self._extract_security_context(request)
            
            return await admin_boundary.execute_secured(
                context,
                lambda: {"status": "healthy", "timestamp": datetime.utcnow()}
            )
        
        @app.get("/admin/security/audit")
        async def get_audit_logs(request: Request, limit: int = 100):
            """Get audit logs - admin only"""
            context = await self._extract_security_context(request)
            
            audit_service = self.security.get_service('audit')
            return await admin_boundary.execute_secured(
                context,
                lambda: audit_service.get_recent_logs(limit, context.user_id)
            )
        
        @app.post("/admin/security/threat-model/scan")
        async def trigger_threat_scan(request: Request):
            """Trigger threat model analysis"""
            context = await self._extract_security_context(request)
            
            threat_service = self.security.get_service('threat')
            return await admin_boundary.execute_secured(
                context,
                lambda: threat_service.trigger_comprehensive_scan(context.user_id)
            )

class LocalhostOnlyMiddleware:
    """Middleware to enforce localhost-only access"""
    
    def __init__(self, app, allowed_ips: list[str]):
        self.app = app
        self.allowed_ips = set(allowed_ips)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            client_ip = scope.get("client", [None, None])[0]
            
            if client_ip not in self.allowed_ips:
                # Log security violation
                logger.warning(f"Blocked non-localhost admin access from {client_ip}")
                
                response = PlainTextResponse(
                    "Admin interface only accessible from localhost",
                    status_code=403
                )
                await response(scope, receive, send)
                return
        
        await self.app(scope, receive, send)
```

### 4. Threat Model Integration Architecture

#### Development Workflow Integration

```python
class ThreatModelIntegration:
    """Integrate threat modeling with development workflows"""
    
    def __init__(self, security_container: SecurityServiceContainer):
        self.security = security_container
        self.threat_db = ThreatModelDatabase()
    
    async def analyze_pr_security_impact(self, pr_data: dict) -> dict:
        """Analyze security impact of pull request"""
        
        analysis = {
            "pr_id": pr_data["id"],
            "security_impact": "low",  # default
            "threats_introduced": [],
            "threats_mitigated": [],
            "recommendations": []
        }
        
        # Analyze changed files
        changed_files = pr_data.get("files_changed", [])
        
        for file_path in changed_files:
            file_analysis = await self._analyze_file_security_impact(file_path)
            
            if file_analysis["introduces_admin_interface"]:
                analysis["threats_introduced"].append({
                    "id": "ADM-001",
                    "description": f"New admin interface in {file_path}",
                    "severity": "high",
                    "mitigation": "Ensure localhost-only binding and MFA requirement"
                })
            
            if file_analysis["modifies_p2p_protocol"]:
                analysis["threats_introduced"].append({
                    "id": "P2P-001", 
                    "description": f"P2P protocol change in {file_path}",
                    "severity": "medium",
                    "mitigation": "Review cryptographic implementation and node authentication"
                })
            
            if file_analysis["adds_external_dependency"]:
                analysis["threats_introduced"].append({
                    "id": "DEP-001",
                    "description": f"New external dependency in {file_path}", 
                    "severity": "medium",
                    "mitigation": "SBOM update and vulnerability scan required"
                })
        
        # Determine overall security impact
        high_severity_count = len([t for t in analysis["threats_introduced"] if t["severity"] == "high"])
        if high_severity_count > 0:
            analysis["security_impact"] = "high"
        elif len(analysis["threats_introduced"]) > 0:
            analysis["security_impact"] = "medium"
            
        return analysis
    
    async def generate_threat_model_for_component(self, component_path: str) -> dict:
        """Generate STRIDE threat model for component"""
        
        component_analysis = await self._analyze_component_architecture(component_path)
        
        threats = {
            "spoofing": [],
            "tampering": [], 
            "repudiation": [],
            "information_disclosure": [],
            "denial_of_service": [],
            "elevation_of_privilege": []
        }
        
        # Spoofing threats
        if component_analysis["has_authentication"]:
            threats["spoofing"].append({
                "id": f"S-{component_path}-001",
                "description": "Authentication bypass or credential theft",
                "impact": "high",
                "likelihood": "medium",
                "mitigation": "Implement multi-factor authentication and secure credential storage"
            })
        
        # Information disclosure
        if component_analysis["processes_pii"]:
            threats["information_disclosure"].append({
                "id": f"I-{component_path}-001", 
                "description": "PII leakage through logs or error messages",
                "impact": "high",
                "likelihood": "high",
                "mitigation": "Implement PII detection and redaction in logs"
            })
        
        # Elevation of privilege  
        if component_analysis["has_admin_functions"]:
            threats["elevation_of_privilege"].append({
                "id": f"E-{component_path}-001",
                "description": "Privilege escalation through admin interface",
                "impact": "critical", 
                "likelihood": "low",
                "mitigation": "Localhost-only binding and comprehensive authorization checks"
            })
        
        return {
            "component": component_path,
            "threats": threats,
            "total_threats": sum(len(v) for v in threats.values()),
            "risk_score": self._calculate_risk_score(threats)
        }

# GitHub Integration
class GitHubThreatModelBot:
    """GitHub bot for automated threat model integration"""
    
    def __init__(self, threat_integration: ThreatModelIntegration):
        self.threat = threat_integration
    
    async def handle_pull_request(self, pr_event: dict):
        """Handle pull request events"""
        
        if pr_event["action"] in ["opened", "synchronize"]:
            # Analyze security impact
            analysis = await self.threat.analyze_pr_security_impact(pr_event["pull_request"])
            
            # Post comment with threat analysis
            comment = self._format_threat_analysis_comment(analysis)
            await self._post_pr_comment(pr_event["pull_request"]["id"], comment)
            
            # Add labels based on security impact
            labels = []
            if analysis["security_impact"] == "high":
                labels.append("security:high-impact")
            elif analysis["security_impact"] == "medium":
                labels.append("security:medium-impact") 
            
            if labels:
                await self._add_pr_labels(pr_event["pull_request"]["id"], labels)
    
    def _format_threat_analysis_comment(self, analysis: dict) -> str:
        """Format threat analysis as GitHub comment"""
        
        comment = "## 🛡️ Security Impact Analysis\n\n"
        comment += f"**Overall Impact:** {analysis['security_impact'].upper()}\n\n"
        
        if analysis["threats_introduced"]:
            comment += "### ⚠️ Potential Threats Introduced\n\n"
            for threat in analysis["threats_introduced"]:
                comment += f"- **{threat['id']}** ({threat['severity']}): {threat['description']}\n"
                comment += f"  - *Mitigation:* {threat['mitigation']}\n\n"
        
        if analysis["recommendations"]:
            comment += "### 💡 Security Recommendations\n\n"
            for rec in analysis["recommendations"]:
                comment += f"- {rec}\n"
        
        comment += "\n*This analysis was generated automatically by the AIVillage security system.*"
        return comment
```

### 5. SBOM and Artifact Signing Architecture

#### Comprehensive SBOM Generation

```python
class EnhancedSBOMGenerator:
    """Enhanced SBOM generation with security metadata"""
    
    def __init__(self, security_container: SecurityServiceContainer):
        self.security = security_container
        self.crypto = security_container.get_service('crypto')
    
    async def generate_comprehensive_sbom(self, project_path: str) -> dict:
        """Generate comprehensive SBOM with security analysis"""
        
        # Base SBOM from existing generator
        base_sbom = await self._run_base_sbom_generation(project_path)
        
        # Enhance with security analysis
        enhanced_sbom = {
            **base_sbom,
            "security_analysis": {
                "vulnerability_scan": await self._scan_vulnerabilities(base_sbom["components"]),
                "license_compliance": await self._analyze_license_compliance(base_sbom["components"]),
                "supply_chain_risk": await self._assess_supply_chain_risk(base_sbom["components"]),
                "outdated_components": await self._identify_outdated_components(base_sbom["components"])
            },
            "attestation": await self._generate_attestation(base_sbom),
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": "aivillage-security-system"
        }
        
        return enhanced_sbom
    
    async def _scan_vulnerabilities(self, components: list) -> dict:
        """Scan components for known vulnerabilities"""
        
        vulnerability_results = {
            "critical": [],
            "high": [],  
            "medium": [],
            "low": [],
            "total_vulnerabilities": 0
        }
        
        for component in components:
            # Query vulnerability databases (NVD, OSV, etc.)
            vulnerabilities = await self._query_vulnerability_db(
                component["name"], 
                component["version"]
            )
            
            for vuln in vulnerabilities:
                severity = vuln["severity"].lower()
                if severity in vulnerability_results:
                    vulnerability_results[severity].append({
                        "component": component["name"],
                        "version": component["version"],
                        "cve_id": vuln["cve_id"],
                        "description": vuln["description"],
                        "fix_version": vuln.get("fix_version")
                    })
                    vulnerability_results["total_vulnerabilities"] += 1
        
        return vulnerability_results
    
    async def _generate_attestation(self, sbom: dict) -> dict:
        """Generate cryptographic attestation for SBOM"""
        
        sbom_hash = self.crypto.hash_data(json.dumps(sbom, sort_keys=True))
        
        attestation = {
            "sbom_hash": sbom_hash,
            "attestation_type": "in-toto",
            "predicate": {
                "buildType": "aivillage-build-system",
                "builder": {"id": "aivillage-ci"},
                "materials": sbom["components"],
                "build_config": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": "secure-build-environment"
                }
            },
            "signature": await self.crypto.sign_data(sbom_hash)
        }
        
        return attestation

class ArtifactSigningService:
    """Service for signing and verifying build artifacts"""
    
    def __init__(self, security_container: SecurityServiceContainer):
        self.security = security_container
        self.crypto = security_container.get_service('crypto')
    
    async def sign_artifact(self, artifact_path: str, metadata: dict = None) -> dict:
        """Sign build artifact with metadata"""
        
        # Calculate artifact hash
        artifact_hash = await self._calculate_artifact_hash(artifact_path)
        
        # Create signing manifest
        manifest = {
            "artifact_path": artifact_path,
            "artifact_hash": artifact_hash,
            "hash_algorithm": "sha256",
            "metadata": metadata or {},
            "signed_at": datetime.utcnow().isoformat(),
            "signer": "aivillage-security-system"
        }
        
        # Sign the manifest
        manifest_json = json.dumps(manifest, sort_keys=True)
        signature = await self.crypto.sign_data(manifest_json)
        
        signed_manifest = {
            **manifest,
            "signature": signature,
            "public_key": await self.crypto.get_public_key()
        }
        
        # Write signature file
        signature_path = f"{artifact_path}.sig"
        with open(signature_path, 'w') as f:
            json.dump(signed_manifest, f, indent=2)
        
        return signed_manifest
    
    async def verify_artifact(self, artifact_path: str) -> dict:
        """Verify artifact signature and integrity"""
        
        signature_path = f"{artifact_path}.sig"
        
        if not Path(signature_path).exists():
            return {"verified": False, "reason": "No signature file found"}
        
        # Load signature manifest
        with open(signature_path) as f:
            signed_manifest = json.load(f)
        
        # Verify signature
        manifest_without_signature = {k: v for k, v in signed_manifest.items() 
                                     if k not in ["signature", "public_key"]}
        manifest_json = json.dumps(manifest_without_signature, sort_keys=True)
        
        signature_valid = await self.crypto.verify_signature(
            manifest_json,
            signed_manifest["signature"],
            signed_manifest["public_key"]
        )
        
        if not signature_valid:
            return {"verified": False, "reason": "Invalid signature"}
        
        # Verify artifact integrity
        current_hash = await self._calculate_artifact_hash(artifact_path)
        if current_hash != signed_manifest["artifact_hash"]:
            return {"verified": False, "reason": "Artifact hash mismatch"}
        
        return {
            "verified": True,
            "signer": signed_manifest["signer"],
            "signed_at": signed_manifest["signed_at"],
            "metadata": signed_manifest["metadata"]
        }
```

## Implementation Roadmap

### Phase 1: Security Boundaries (Week 1-2)
- Implement secure module boundary contracts
- Create localhost-only admin interface patterns
- Deploy security service dependency injection
- Fix all `0.0.0.0` bindings to localhost for admin interfaces

### Phase 2: Threat Model Integration (Week 3-4) 
- Build GitHub bot for automated threat analysis
- Integrate STRIDE analysis with PR workflows
- Create threat model database and tracking
- Deploy security impact assessment pipeline

### Phase 3: SBOM & Signing (Week 5-6)
- Enhance SBOM generator with vulnerability scanning
- Implement artifact signing service
- Create supply chain risk assessment
- Deploy attestation and verification pipeline

### Phase 4: Connascence Enforcement (Week 7-8)
- Implement architectural fitness functions for security
- Create connascence analysis for security boundaries
- Deploy automated coupling metrics for security services
- Add security architectural decision recording

## Security Metrics and Monitoring

### Key Security Indicators
- Admin interface access attempts from non-localhost IPs
- Security boundary violations per module
- Threat model coverage percentage
- SBOM freshness and vulnerability counts
- Artifact signature verification success rate
- Security coupling strength metrics

### Alerting Thresholds
- **Critical**: Non-localhost admin access attempts
- **High**: New critical vulnerabilities in SBOM
- **Medium**: Security boundary coupling violations
- **Low**: SBOM outdated by >7 days

This architecture provides a comprehensive security foundation that addresses the identified gaps while maintaining system performance and developer productivity.
#!/usr/bin/env python3
"""
Threat Model Integration for Development Workflows
Integrates STRIDE analysis with GitHub PR/issue workflows and development processes.
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import yaml

from ..boundaries.secure_boundary_contracts import SecurityContext, SecurityLevel

logger = logging.getLogger(__name__)

class ThreatCategory(Enum):
    """STRIDE threat categories"""
    SPOOFING = "spoofing"
    TAMPERING = "tampering" 
    REPUDIATION = "repudiation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    ELEVATION_OF_PRIVILEGE = "elevation_of_privilege"

class RiskLevel(Enum):
    """Risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComponentType(Enum):
    """Component types for analysis"""
    API_ENDPOINT = "api_endpoint"
    ADMIN_INTERFACE = "admin_interface"
    P2P_PROTOCOL = "p2p_protocol"
    DATA_STORAGE = "data_storage"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    EXTERNAL_DEPENDENCY = "external_dependency"
    AGENT_SYSTEM = "agent_system"
    RAG_SYSTEM = "rag_system"

@dataclass
class Threat:
    """Individual threat definition"""
    id: str
    category: ThreatCategory
    title: str
    description: str
    impact: RiskLevel
    likelihood: RiskLevel
    cvss_score: float
    affected_components: List[str]
    mitigation: str
    validation_steps: List[str]
    references: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class SecurityAnalysisResult:
    """Result of security analysis"""
    pr_id: Optional[str]
    component_path: str
    overall_risk: RiskLevel
    threats_introduced: List[Threat]
    threats_mitigated: List[str]
    security_recommendations: List[str]
    requires_security_review: bool
    compliance_impact: List[str]
    analysis_metadata: Dict[str, Any]

class ThreatDatabase:
    """Database of threat patterns and historical analysis"""
    
    def __init__(self, db_path: str = "data/threat_models"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._threat_patterns = self._load_threat_patterns()
        self._component_mappings = self._load_component_mappings()
        
    def _load_threat_patterns(self) -> Dict[str, Dict]:
        """Load threat patterns from database"""
        patterns_file = self.db_path / "threat_patterns.yaml"
        
        if not patterns_file.exists():
            return self._create_default_threat_patterns()
        
        with open(patterns_file, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_component_mappings(self) -> Dict[str, ComponentType]:
        """Load component type mappings"""
        mappings_file = self.db_path / "component_mappings.yaml"
        
        default_mappings = {
            "admin_server.py": ComponentType.ADMIN_INTERFACE,
            "admin_interface": ComponentType.ADMIN_INTERFACE,
            "gateway": ComponentType.API_ENDPOINT,
            "api": ComponentType.API_ENDPOINT,
            "p2p": ComponentType.P2P_PROTOCOL,
            "auth": ComponentType.AUTHENTICATION,
            "rbac": ComponentType.AUTHORIZATION,
            "rag": ComponentType.RAG_SYSTEM,
            "agent_forge": ComponentType.AGENT_SYSTEM,
            "requirements.txt": ComponentType.EXTERNAL_DEPENDENCY,
            "package.json": ComponentType.EXTERNAL_DEPENDENCY,
            "Cargo.toml": ComponentType.EXTERNAL_DEPENDENCY,
        }
        
        if not mappings_file.exists():
            with open(mappings_file, 'w') as f:
                yaml.dump({k: v.value for k, v in default_mappings.items()}, f)
            return default_mappings
        
        with open(mappings_file, 'r') as f:
            loaded = yaml.safe_load(f) or {}
            return {k: ComponentType(v) for k, v in loaded.items()}
    
    def _create_default_threat_patterns(self) -> Dict[str, Dict]:
        """Create default threat patterns"""
        patterns = {
            "admin_interface_exposure": {
                "category": ThreatCategory.ELEVATION_OF_PRIVILEGE.value,
                "indicators": ["0.0.0.0", "bind.*0.0.0.0", "host.*0.0.0.0"],  # nosec B104 - Security pattern detection
                "description": "Admin interface exposed to all network interfaces",
                "impact": RiskLevel.CRITICAL.value,
                "likelihood": RiskLevel.HIGH.value,
                "mitigation": "Bind admin interfaces to localhost only (127.0.0.1)"
            },
            "cors_wildcard": {
                "category": ThreatCategory.TAMPERING.value,
                "indicators": ["allow_origins.*\\*", "cors.*\\*"],
                "description": "CORS configuration allows all origins",
                "impact": RiskLevel.HIGH.value,
                "likelihood": RiskLevel.MEDIUM.value,
                "mitigation": "Specify explicit allowed origins"
            },
            "hardcoded_secrets": {
                "category": ThreatCategory.INFORMATION_DISCLOSURE.value,
                "indicators": ["password.*=", "secret.*=", "token.*=", "key.*="],
                "description": "Potential hardcoded credentials in code",
                "impact": RiskLevel.CRITICAL.value,
                "likelihood": RiskLevel.MEDIUM.value,
                "mitigation": "Use environment variables or secure credential management"
            },
            "sql_injection": {
                "category": ThreatCategory.TAMPERING.value,
                "indicators": ["query.*%", "execute.*format", "f\".*SELECT"],
                "description": "Potential SQL injection vulnerability",
                "impact": RiskLevel.HIGH.value,
                "likelihood": RiskLevel.MEDIUM.value,
                "mitigation": "Use parameterized queries and input validation"
            },
            "weak_crypto": {
                "category": ThreatCategory.INFORMATION_DISCLOSURE.value,
                "indicators": ["md5", "sha1", "DES", "RC4"],
                "description": "Use of weak cryptographic algorithms",
                "impact": RiskLevel.MEDIUM.value,
                "likelihood": RiskLevel.LOW.value,
                "mitigation": "Use strong cryptographic algorithms (AES-256, SHA-256+)"
            },
            "unvalidated_input": {
                "category": ThreatCategory.TAMPERING.value,
                "indicators": ["request.form", "request.args", "input()", "raw_input()"],
                "description": "Unvalidated user input processing",
                "impact": RiskLevel.HIGH.value,
                "likelihood": RiskLevel.HIGH.value,
                "mitigation": "Implement comprehensive input validation"
            },
            "p2p_unauthenticated": {
                "category": ThreatCategory.SPOOFING.value,
                "indicators": ["peer.connect", "node.accept", "mesh.join"],
                "description": "P2P connections without authentication",
                "impact": RiskLevel.HIGH.value,
                "likelihood": RiskLevel.MEDIUM.value,
                "mitigation": "Implement node certificate-based authentication"
            }
        }
        
        patterns_file = self.db_path / "threat_patterns.yaml"
        with open(patterns_file, 'w') as f:
            yaml.dump(patterns, f)
        
        return patterns
    
    def identify_component_type(self, file_path: str) -> ComponentType:
        """Identify component type from file path"""
        path_lower = file_path.lower()
        
        # Check exact matches first
        for pattern, component_type in self._component_mappings.items():
            if pattern in path_lower:
                return component_type
        
        # Default to API endpoint for Python files
        if path_lower.endswith('.py'):
            return ComponentType.API_ENDPOINT
        
        return ComponentType.EXTERNAL_DEPENDENCY

class ThreatAnalyzer:
    """Analyzes code changes for security threats"""
    
    def __init__(self, threat_db: ThreatDatabase):
        self.threat_db = threat_db
    
    async def analyze_file_changes(self, 
                                  file_path: str, 
                                  content: str, 
                                  diff_lines: List[str] = None) -> SecurityAnalysisResult:
        """Analyze security impact of file changes"""
        
        component_type = self.threat_db.identify_component_type(file_path)
        threats_found = []
        recommendations = []
        
        # Analyze content for threat patterns
        for pattern_name, pattern_data in self.threat_db._threat_patterns.items():
            indicators = pattern_data["indicators"]
            
            for indicator in indicators:
                if re.search(indicator, content, re.IGNORECASE):
                    threat = self._create_threat_from_pattern(
                        pattern_name,
                        pattern_data,
                        file_path,
                        indicator
                    )
                    threats_found.append(threat)
        
        # Component-specific analysis
        if component_type == ComponentType.ADMIN_INTERFACE:
            admin_threats = await self._analyze_admin_interface(file_path, content)
            threats_found.extend(admin_threats)
        elif component_type == ComponentType.P2P_PROTOCOL:
            p2p_threats = await self._analyze_p2p_protocol(file_path, content)
            threats_found.extend(p2p_threats)
        elif component_type == ComponentType.EXTERNAL_DEPENDENCY:
            dep_threats = await self._analyze_dependencies(file_path, content)
            threats_found.extend(dep_threats)
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(threats_found)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(threats_found, component_type)
        
        return SecurityAnalysisResult(
            pr_id=None,
            component_path=file_path,
            overall_risk=overall_risk,
            threats_introduced=threats_found,
            threats_mitigated=[],
            security_recommendations=recommendations,
            requires_security_review=overall_risk in [RiskLevel.CRITICAL, RiskLevel.HIGH],
            compliance_impact=self._assess_compliance_impact(threats_found),
            analysis_metadata={
                "component_type": component_type.value,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "analyzer_version": "1.0.0",
                "patterns_checked": len(self.threat_db._threat_patterns)
            }
        )
    
    async def _analyze_admin_interface(self, file_path: str, content: str) -> List[Threat]:
        """Analyze admin interface specific threats"""
        threats = []
        
        # Check for network binding issues
        if re.search(r'host.*["\']0\.0\.0\.0["\']', content):
            threat = Threat(
                id=f"ADM-{hashlib.sha256(file_path.encode()).hexdigest()[:8]}-001",  # Use SHA256 for security
                category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
                title="Admin Interface Network Exposure",
                description=f"Admin interface in {file_path} binds to all network interfaces (0.0.0.0)",
                impact=RiskLevel.CRITICAL,
                likelihood=RiskLevel.HIGH,
                cvss_score=9.1,
                affected_components=[file_path],
                mitigation="Change host binding to '127.0.0.1' for localhost-only access",
                validation_steps=[
                    "Verify host parameter is set to '127.0.0.1'",
                    "Test that admin interface is not accessible from external IPs",
                    "Confirm firewall rules block external access to admin ports"
                ],
                references=[
                    "OWASP Top 10 - Broken Access Control",
                    "CWE-284: Improper Access Control"
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            threats.append(threat)
        
        # Check for missing authentication
        if "admin" in content.lower() and not re.search(r'auth|authenticate|login', content, re.IGNORECASE):
            threat = Threat(
                id=f"ADM-{hashlib.sha256(file_path.encode()).hexdigest()[:8]}-002",  # Use SHA256 for security
                category=ThreatCategory.SPOOFING,
                title="Admin Interface Missing Authentication",
                description=f"Admin interface in {file_path} may lack proper authentication",
                impact=RiskLevel.HIGH,
                likelihood=RiskLevel.MEDIUM,
                cvss_score=8.1,
                affected_components=[file_path],
                mitigation="Implement multi-factor authentication for admin access",
                validation_steps=[
                    "Verify authentication middleware is applied",
                    "Test that unauthenticated requests are rejected",
                    "Confirm MFA is required for admin functions"
                ],
                references=[
                    "NIST SP 800-63B - Authentication Guidelines",
                    "OWASP Authentication Cheat Sheet"
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            threats.append(threat)
        
        return threats
    
    async def _analyze_p2p_protocol(self, file_path: str, content: str) -> List[Threat]:
        """Analyze P2P protocol specific threats"""
        threats = []
        
        # Check for unencrypted communications
        if re.search(r'socket|tcp|udp', content, re.IGNORECASE) and not re.search(r'tls|ssl|encrypt', content, re.IGNORECASE):
            threat = Threat(
                id=f"P2P-{hashlib.sha256(file_path.encode()).hexdigest()[:8]}-001",  # Use SHA256 for security
                category=ThreatCategory.INFORMATION_DISCLOSURE,
                title="P2P Unencrypted Communications",
                description=f"P2P protocol in {file_path} may use unencrypted communications",
                impact=RiskLevel.HIGH,
                likelihood=RiskLevel.MEDIUM,
                cvss_score=7.5,
                affected_components=[file_path],
                mitigation="Implement TLS encryption for all P2P communications",
                validation_steps=[
                    "Verify TLS is used for peer connections",
                    "Test that unencrypted connections are rejected",
                    "Confirm certificate validation is implemented"
                ],
                references=[
                    "RFC 8446 - Transport Layer Security (TLS) Protocol Version 1.3",
                    "NIST SP 800-52 Rev. 2 - TLS Guidelines"
                ],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            threats.append(threat)
        
        return threats
    
    async def _analyze_dependencies(self, file_path: str, content: str) -> List[Threat]:
        """Analyze external dependencies for threats"""
        threats = []
        
        # This would integrate with vulnerability databases
        # For now, we'll do basic pattern matching
        
        if "requirements.txt" in file_path:
            lines = content.split('\n')
            for line in lines:
                if '==' in line:
                    package, version = line.split('==')
                    package = package.strip()
                    version = version.strip()
                    
                    # Check for known vulnerable packages (simplified)
                    vulnerable_packages = {
                        "flask": ["0.12.0", "0.12.1", "0.12.2"],
                        "requests": ["2.19.0", "2.19.1"],
                        "pillow": ["5.2.0", "6.2.0"]
                    }
                    
                    if package.lower() in vulnerable_packages:
                        if version in vulnerable_packages[package.lower()]:
                            threat = Threat(
                                id=f"DEP-{hashlib.sha256(f'{package}-{version}'.encode()).hexdigest()[:8]}",  # Use SHA256 for security
                                category=ThreatCategory.TAMPERING,
                                title=f"Vulnerable Dependency: {package}",
                                description=f"Package {package} version {version} has known vulnerabilities",
                                impact=RiskLevel.HIGH,
                                likelihood=RiskLevel.HIGH,
                                cvss_score=8.0,
                                affected_components=[file_path],
                                mitigation=f"Update {package} to latest secure version",
                                validation_steps=[
                                    f"Update {package} in requirements.txt",
                                    "Run vulnerability scan after update",
                                    "Test application functionality after update"
                                ],
                                references=[
                                    "National Vulnerability Database",
                                    "GitHub Security Advisories"
                                ],
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow()
                            )
                            threats.append(threat)
        
        return threats
    
    def _create_threat_from_pattern(self, 
                                   pattern_name: str, 
                                   pattern_data: Dict, 
                                   file_path: str,
                                   matched_indicator: str) -> Threat:
        """Create threat object from pattern match"""
        
        return Threat(
            id=f"PAT-{hashlib.sha256(f'{pattern_name}-{file_path}'.encode()).hexdigest()[:8]}",  # Use SHA256 for security
            category=ThreatCategory(pattern_data["category"]),
            title=pattern_name.replace('_', ' ').title(),
            description=f"{pattern_data['description']} (matched: {matched_indicator})",
            impact=RiskLevel(pattern_data["impact"]),
            likelihood=RiskLevel(pattern_data["likelihood"]),
            cvss_score=self._calculate_cvss(
                RiskLevel(pattern_data["impact"]),
                RiskLevel(pattern_data["likelihood"])
            ),
            affected_components=[file_path],
            mitigation=pattern_data["mitigation"],
            validation_steps=[
                f"Review {file_path} for {matched_indicator}",
                "Apply recommended mitigation",
                "Verify fix resolves the threat"
            ],
            references=["AIVillage Threat Database"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def _calculate_cvss(self, impact: RiskLevel, likelihood: RiskLevel) -> float:
        """Calculate CVSS score from impact and likelihood"""
        impact_scores = {
            RiskLevel.CRITICAL: 10.0,
            RiskLevel.HIGH: 8.0,
            RiskLevel.MEDIUM: 6.0,
            RiskLevel.LOW: 4.0,
            RiskLevel.INFO: 1.0
        }
        
        likelihood_multipliers = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.9,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.LOW: 0.5,
            RiskLevel.INFO: 0.1
        }
        
        base_score = impact_scores[impact] * likelihood_multipliers[likelihood]
        return round(min(10.0, base_score), 1)
    
    def _calculate_overall_risk(self, threats: List[Threat]) -> RiskLevel:
        """Calculate overall risk level from threats"""
        if not threats:
            return RiskLevel.INFO
        
        risk_weights = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1,
            RiskLevel.INFO: 0
        }
        
        max_weight = max(risk_weights[threat.impact] for threat in threats)
        
        if max_weight >= 4:
            return RiskLevel.CRITICAL
        elif max_weight >= 3:
            return RiskLevel.HIGH
        elif max_weight >= 2:
            return RiskLevel.MEDIUM
        elif max_weight >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO
    
    def _generate_recommendations(self, 
                                threats: List[Threat], 
                                component_type: ComponentType) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Generic recommendations based on threats
        threat_categories = {threat.category for threat in threats}
        
        if ThreatCategory.ELEVATION_OF_PRIVILEGE in threat_categories:
            recommendations.append("Implement least privilege principle and proper authorization checks")
        
        if ThreatCategory.INFORMATION_DISCLOSURE in threat_categories:
            recommendations.append("Add data classification and encryption for sensitive information")
        
        if ThreatCategory.TAMPERING in threat_categories:
            recommendations.append("Implement input validation and integrity checks")
        
        # Component-specific recommendations
        if component_type == ComponentType.ADMIN_INTERFACE:
            recommendations.extend([
                "Ensure admin interfaces bind to localhost only",
                "Implement multi-factor authentication",
                "Add comprehensive audit logging"
            ])
        elif component_type == ComponentType.P2P_PROTOCOL:
            recommendations.extend([
                "Use TLS encryption for all peer communications",
                "Implement node certificate-based authentication",
                "Add reputation-based trust scoring"
            ])
        elif component_type == ComponentType.EXTERNAL_DEPENDENCY:
            recommendations.extend([
                "Keep dependencies updated to latest secure versions",
                "Implement automated vulnerability scanning",
                "Use software bill of materials (SBOM)"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _assess_compliance_impact(self, threats: List[Threat]) -> List[str]:
        """Assess compliance impact of threats"""
        compliance_issues = []
        
        for threat in threats:
            if threat.category == ThreatCategory.INFORMATION_DISCLOSURE:
                compliance_issues.append("GDPR - Data Protection")
                compliance_issues.append("CCPA - Consumer Privacy")
            elif threat.category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
                compliance_issues.append("SOC 2 - Access Controls")
                compliance_issues.append("ISO 27001 - Access Management")
            elif threat.impact == RiskLevel.CRITICAL:
                compliance_issues.append("PCI DSS - Security Requirements")
        
        return list(set(compliance_issues))

class GitHubIntegration:
    """Integration with GitHub for automated threat modeling"""
    
    def __init__(self, threat_analyzer: ThreatAnalyzer):
        self.analyzer = threat_analyzer
        self.webhook_handlers = {}
    
    async def analyze_pull_request(self, pr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pull request for security impact"""
        
        pr_id = pr_data.get("number")
        files_changed = pr_data.get("changed_files", [])
        
        analysis_results = []
        overall_risk = RiskLevel.INFO
        security_review_required = False
        
        # Analyze each changed file
        for file_data in files_changed:
            file_path = file_data.get("filename", "")
            file_content = file_data.get("content", "")
            diff_lines = file_data.get("diff", "").split('\n')
            
            if self._should_analyze_file(file_path):
                result = await self.analyzer.analyze_file_changes(
                    file_path, file_content, diff_lines
                )
                result.pr_id = str(pr_id)
                analysis_results.append(result)
                
                # Update overall assessment
                if result.overall_risk.value > overall_risk.value:
                    overall_risk = result.overall_risk
                
                if result.requires_security_review:
                    security_review_required = True
        
        # Compile PR analysis summary
        all_threats = []
        all_recommendations = []
        
        for result in analysis_results:
            all_threats.extend(result.threats_introduced)
            all_recommendations.extend(result.security_recommendations)
        
        pr_analysis = {
            "pr_id": pr_id,
            "overall_risk": overall_risk.value,
            "security_review_required": security_review_required,
            "total_threats": len(all_threats),
            "critical_threats": len([t for t in all_threats if t.impact == RiskLevel.CRITICAL]),
            "high_threats": len([t for t in all_threats if t.impact == RiskLevel.HIGH]),
            "threats_by_category": self._categorize_threats(all_threats),
            "recommendations": list(set(all_recommendations)),
            "files_analyzed": len(analysis_results),
            "analysis_results": [asdict(result) for result in analysis_results],
            "compliance_impact": list(set(sum([r.compliance_impact for r in analysis_results], []))),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        return pr_analysis
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Determine if file should be analyzed for security"""
        
        # Skip certain file types
        skip_extensions = {'.md', '.txt', '.json', '.yaml', '.yml', '.gitignore', '.png', '.jpg', '.gif'}
        skip_directories = {'node_modules', '.git', '__pycache__', '.pytest_cache', 'venv', '.venv'}
        
        path_obj = Path(file_path)
        
        # Skip files with excluded extensions
        if path_obj.suffix.lower() in skip_extensions:
            return False
        
        # Skip files in excluded directories
        for part in path_obj.parts:
            if part in skip_directories:
                return False
        
        return True
    
    def _categorize_threats(self, threats: List[Threat]) -> Dict[str, int]:
        """Categorize threats by STRIDE category"""
        categories = {}
        
        for threat in threats:
            category = threat.category.value
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def format_github_comment(self, pr_analysis: Dict[str, Any]) -> str:
        """Format threat analysis as GitHub PR comment"""
        
        overall_risk = pr_analysis["overall_risk"].upper()
        risk_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠", 
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "INFO": "ℹ️"
        }
        
        comment = f"## {risk_emoji.get(overall_risk, '🔍')} Security Impact Analysis\n\n"
        comment += f"**Overall Risk Level:** {overall_risk}\n"
        comment += f"**Files Analyzed:** {pr_analysis['files_analyzed']}\n"
        comment += f"**Total Threats Found:** {pr_analysis['total_threats']}\n\n"
        
        if pr_analysis["security_review_required"]:
            comment += "⚠️ **This PR requires security review before merging.**\n\n"
        
        # Threat summary
        if pr_analysis["total_threats"] > 0:
            comment += "### 📊 Threat Summary\n\n"
            comment += f"- 🔴 Critical: {pr_analysis['critical_threats']}\n"
            comment += f"- 🟠 High: {pr_analysis['high_threats']}\n"
            comment += f"- 🟡 Medium: {pr_analysis['total_threats'] - pr_analysis['critical_threats'] - pr_analysis['high_threats']}\n\n"
            
            # STRIDE category breakdown
            if pr_analysis["threats_by_category"]:
                comment += "### 🎯 STRIDE Category Breakdown\n\n"
                for category, count in pr_analysis["threats_by_category"].items():
                    comment += f"- **{category.replace('_', ' ').title()}:** {count}\n"
                comment += "\n"
        
        # Recommendations
        if pr_analysis["recommendations"]:
            comment += "### 💡 Security Recommendations\n\n"
            for rec in pr_analysis["recommendations"][:5]:  # Limit to top 5
                comment += f"- {rec}\n"
            comment += "\n"
        
        # Compliance impact
        if pr_analysis["compliance_impact"]:
            comment += "### 📋 Compliance Impact\n\n"
            comment += "This PR may affect compliance with:\n"
            for compliance in pr_analysis["compliance_impact"]:
                comment += f"- {compliance}\n"
            comment += "\n"
        
        comment += "---\n"
        comment += "*This analysis was generated automatically by AIVillage Security System.*\n"
        comment += f"*Analysis timestamp: {pr_analysis['analysis_timestamp']}*"
        
        return comment

# Example usage and testing
if __name__ == "__main__":
    async def example_analysis():
        """Example of threat model integration"""
        
        # Initialize components
        threat_db = ThreatDatabase()
        analyzer = ThreatAnalyzer(threat_db)
        github_integration = GitHubIntegration(analyzer)
        
        # Example file content with security issues
        admin_server_content = '''
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/admin/users")
def get_users():
    return {"users": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3006)  # Security issue!
'''
        
        # Analyze file
        result = await analyzer.analyze_file_changes(
            "infrastructure/gateway/admin_server.py",
            admin_server_content
        )
        
        print("Analysis Result:")
        print(f"Overall Risk: {result.overall_risk}")
        print(f"Threats Found: {len(result.threats_introduced)}")
        print(f"Security Review Required: {result.requires_security_review}")
        
        for threat in result.threats_introduced:
            print(f"\nThreat: {threat.title}")
            print(f"Category: {threat.category}")
            print(f"Impact: {threat.impact}")
            print(f"Description: {threat.description}")
            print(f"Mitigation: {threat.mitigation}")
        
        # Example PR analysis
        pr_data = {
            "number": 123,
            "changed_files": [
                {
                    "filename": "infrastructure/gateway/admin_server.py",
                    "content": admin_server_content,
                    "diff": "+    uvicorn.run(app, host=\"0.0.0.0\", port=3006)"
                }
            ]
        }
        
        pr_analysis = await github_integration.analyze_pull_request(pr_data)
        comment = github_integration.format_github_comment(pr_analysis)
        
        print("\n" + "="*50)
        print("GitHub PR Comment:")
        print("="*50)
        print(comment)
    
    # Run example
    asyncio.run(example_analysis())
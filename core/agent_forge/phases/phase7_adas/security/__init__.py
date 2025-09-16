"""
Phase 7 ADAS Security Module
Automotive-grade cybersecurity implementation for autonomous driving systems
Compliant with UN R155, ISO/SAE 21434, and automotive cybersecurity standards
"""

from .secure_communication import (
    SecureCommunicationManager,
    V2XEncryption,
    SecureCANBus,
    AuthenticationManager,
    AntiTamperingSystem,
    V2XMessage,
    CANFrame,
    V2XMessageType,
    SecurityLevel
)

from .intrusion_detection import (
    AutomotiveIntrusionDetectionSystem,
    NetworkIntrusionDetector,
    SensorAnomalyDetector,
    SystemIntegrityMonitor,
    SecurityEventLogger,
    SecurityEvent,
    SensorData,
    CANMessage,
    ThreatLevel,
    AttackType,
    EventType
)

from .secure_boot import (
    SecureBootManager,
    HardwareSecurityModule,
    SecureKeyStorage,
    FirmwareVerificationEngine,
    FirmwareImage,
    SecureBootEvent,
    TrustedKey,
    BootStage,
    VerificationResult
)

from .compliance_validation import (
    AutomotiveComplianceValidator,
    UN_R155_Validator,
    ISO_21434_Validator,
    VulnerabilityManager,
    SecurityControlValidator,
    ComplianceReportGenerator,
    ComplianceRequirement,
    ComplianceFinding,
    VulnerabilityAssessment,
    SecurityControl,
    ComplianceStandard,
    ComplianceLevel,
    SeverityLevel,
    ThreatCategory
)

__version__ = "1.0.0"
__author__ = "ADAS Security Team"
__description__ = "Automotive cybersecurity implementation for Phase 7 ADAS systems"

# Security system initialization
def initialize_adas_security(vehicle_id: str, config: dict = None):
    """
    Initialize comprehensive ADAS security system

    Args:
        vehicle_id: Unique vehicle identifier
        config: Security configuration parameters

    Returns:
        Dictionary containing initialized security components
    """
    if config is None:
        config = get_default_security_config()

    # Initialize secure communication
    comm_manager = SecureCommunicationManager(vehicle_id)

    # Initialize intrusion detection
    ids_system = AutomotiveIntrusionDetectionSystem(vehicle_id)
    ids_system.start()

    # Initialize secure boot
    boot_manager = SecureBootManager(vehicle_id)

    # Initialize compliance validation
    compliance_validator = AutomotiveComplianceValidator(vehicle_id)

    return {
        'communication': comm_manager,
        'intrusion_detection': ids_system,
        'secure_boot': boot_manager,
        'compliance': compliance_validator,
        'vehicle_id': vehicle_id,
        'initialized': True
    }

def get_default_security_config():
    """Get default security configuration"""
    return {
        'encryption': {
            'v2x_algorithm': 'AES-256-CBC',
            'can_authentication': True,
            'key_rotation_interval': 3600  # 1 hour
        },
        'intrusion_detection': {
            'monitoring_interval': 1.0,  # seconds
            'anomaly_threshold': 2.5,
            'alert_threshold': 'MEDIUM'
        },
        'secure_boot': {
            'signature_algorithm': 'RSA-2048',
            'hash_algorithm': 'SHA-256',
            'rollback_protection': True
        },
        'compliance': {
            'standards': ['UN_R155', 'ISO_SAE_21434'],
            'assessment_frequency': 'quarterly',
            'auto_reporting': True
        }
    }

def validate_security_posture(security_systems: dict):
    """
    Validate overall security posture

    Args:
        security_systems: Dictionary of initialized security systems

    Returns:
        Security posture assessment report
    """
    if not security_systems.get('initialized', False):
        return {'status': 'ERROR', 'message': 'Security systems not initialized'}

    comm_status = security_systems['communication'].get_security_status()
    ids_status = security_systems['intrusion_detection'].get_system_status()
    boot_status = security_systems['secure_boot'].get_security_status()

    # Calculate overall security score
    security_score = calculate_security_score(comm_status, ids_status, boot_status)

    return {
        'status': 'OK',
        'vehicle_id': security_systems['vehicle_id'],
        'overall_score': security_score,
        'communication': comm_status,
        'intrusion_detection': ids_status,
        'secure_boot': boot_status,
        'timestamp': time.time(),
        'recommendations': generate_security_recommendations(security_score, comm_status, ids_status, boot_status)
    }

def calculate_security_score(comm_status, ids_status, boot_status):
    """Calculate overall security score (0-100)"""
    import time

    score = 100

    # Communication security penalties
    if comm_status.get('integrity_status') == 'COMPROMISED':
        score -= 30

    # IDS penalties
    recent_critical = ids_status.get('event_summary', {}).get('recent_critical', 0)
    score -= min(recent_critical * 10, 40)

    # Boot security penalties
    if boot_status.get('current_stage') != 'operational':
        score -= 20

    return max(score, 0)

def generate_security_recommendations(score, comm_status, ids_status, boot_status):
    """Generate security recommendations based on status"""
    recommendations = []

    if score < 60:
        recommendations.append("URGENT: Critical security issues detected - immediate attention required")

    if comm_status.get('integrity_status') == 'COMPROMISED':
        recommendations.append("Investigate communication integrity violations")

    if ids_status.get('event_summary', {}).get('recent_critical', 0) > 0:
        recommendations.append("Review recent critical security events")

    if boot_status.get('current_stage') != 'operational':
        recommendations.append("Complete secure boot process")

    if not recommendations:
        recommendations.append("Security posture is good - continue monitoring")

    return recommendations

# Export main components
__all__ = [
    # Main managers
    'SecureCommunicationManager',
    'AutomotiveIntrusionDetectionSystem',
    'SecureBootManager',
    'AutomotiveComplianceValidator',

    # Communication security
    'V2XEncryption',
    'SecureCANBus',
    'AuthenticationManager',
    'AntiTamperingSystem',

    # Intrusion detection
    'NetworkIntrusionDetector',
    'SensorAnomalyDetector',
    'SystemIntegrityMonitor',
    'SecurityEventLogger',

    # Secure boot
    'HardwareSecurityModule',
    'SecureKeyStorage',
    'FirmwareVerificationEngine',

    # Compliance validation
    'UN_R155_Validator',
    'ISO_21434_Validator',
    'VulnerabilityManager',
    'SecurityControlValidator',
    'ComplianceReportGenerator',

    # Data structures
    'V2XMessage',
    'CANFrame',
    'SecurityEvent',
    'SensorData',
    'FirmwareImage',
    'SecureBootEvent',
    'TrustedKey',
    'ComplianceRequirement',
    'ComplianceFinding',
    'VulnerabilityAssessment',
    'SecurityControl',

    # Enums
    'V2XMessageType',
    'SecurityLevel',
    'ThreatLevel',
    'AttackType',
    'EventType',
    'BootStage',
    'VerificationResult',
    'ComplianceStandard',
    'ComplianceLevel',
    'SeverityLevel',
    'ThreatCategory',

    # Utility functions
    'initialize_adas_security',
    'get_default_security_config',
    'validate_security_posture',
    'calculate_security_score',
    'generate_security_recommendations'
]
"""
Attestation Service

Provides comprehensive remote attestation capabilities for TEE environments:
- Remote attestation protocol implementation
- Measurement database and verification
- Certificate management and validation
- Quote verification for different TEE types
"""

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
import secrets
import time
from typing import Any

from cryptography import x509

from .tee_types import (
    AttestationReport,
    AttestationStatus,
    EnclaveContext,
    Measurement,
    TEEType,
)

logger = logging.getLogger(__name__)


class AttestationError(Exception):
    """Base exception for attestation errors"""

    pass


class MeasurementDatabase:
    """Database for storing and validating TEE measurements"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or Path.home() / ".tee_runtime" / "measurements.json"
        self.measurements: dict[str, dict[str, Any]] = {}
        self.load_database()

    def load_database(self) -> None:
        """Load measurement database from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    self.measurements = json.load(f)
                logger.debug(f"Loaded {len(self.measurements)} measurement entries")
            except Exception as e:
                logger.warning(f"Failed to load measurement database: {e}")
                self.measurements = {}
        else:
            # Create directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_database(self) -> None:
        """Save measurement database to disk"""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.measurements, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save measurement database: {e}")

    def add_trusted_measurement(self, measurement: Measurement, description: str = "", source: str = "manual") -> None:
        """Add a trusted measurement to the database"""
        key = f"{measurement.measurement_type}:{measurement.index}"
        self.measurements[key] = {
            "value": measurement.value,
            "algorithm": measurement.algorithm,
            "description": description or measurement.description,
            "source": source,
            "added_at": datetime.now(UTC).isoformat(),
        }
        self.save_database()
        logger.info(f"Added trusted measurement {key}: {measurement.value[:16]}...")

    def is_measurement_trusted(self, measurement: Measurement) -> bool:
        """Check if a measurement is in the trusted database"""
        key = f"{measurement.measurement_type}:{measurement.index}"
        entry = self.measurements.get(key)

        if not entry:
            return False

        return entry["value"] == measurement.value and entry["algorithm"] == measurement.algorithm

    def get_trusted_measurements(self) -> list[Measurement]:
        """Get all trusted measurements"""
        measurements = []
        for key, entry in self.measurements.items():
            measurement_type, index_str = key.split(":", 1)
            try:
                index = int(index_str)
                measurement = Measurement(
                    measurement_type=measurement_type,
                    index=index,
                    value=entry["value"],
                    algorithm=entry["algorithm"],
                    description=entry["description"],
                )
                measurements.append(measurement)
            except ValueError:
                logger.warning(f"Invalid measurement key in database: {key}")

        return measurements


class CertificateManager:
    """Manages attestation certificates and validation"""

    def __init__(self, cert_store_path: Path | None = None):
        self.cert_store_path = cert_store_path or Path.home() / ".tee_runtime" / "certs"
        self.cert_store_path.mkdir(parents=True, exist_ok=True)

        # Certificate stores for different vendors
        self.vendor_stores = {
            "amd": self.cert_store_path / "amd",
            "intel": self.cert_store_path / "intel",
            "arm": self.cert_store_path / "arm",
        }

        for store_path in self.vendor_stores.values():
            store_path.mkdir(exist_ok=True)

        # Root certificates
        self.root_certificates: dict[str, x509.Certificate] = {}
        self.load_root_certificates()

    def load_root_certificates(self) -> None:
        """Load vendor root certificates"""
        for vendor, store_path in self.vendor_stores.items():
            for cert_file in store_path.glob("*.pem"):
                try:
                    with open(cert_file, "rb") as f:
                        cert_data = f.read()
                        certificate = x509.load_pem_x509_certificate(cert_data)
                        key = f"{vendor}_{cert_file.stem}"
                        self.root_certificates[key] = certificate
                        logger.debug(f"Loaded root certificate {key}")
                except Exception as e:
                    logger.warning(f"Failed to load certificate {cert_file}: {e}")

    def verify_certificate_chain(self, chain: list[bytes], vendor: str) -> bool:
        """Verify a certificate chain"""
        if not chain:
            return False

        try:
            # Load certificates from chain
            certificates = []
            for cert_bytes in chain:
                cert = x509.load_der_x509_certificate(cert_bytes)
                certificates.append(cert)

            # For now, implement basic validation
            # In production, this would do full chain validation
            if certificates:
                leaf_cert = certificates[0]

                # Basic validity checks
                now = datetime.now(UTC)
                if now < leaf_cert.not_valid_before or now > leaf_cert.not_valid_after:
                    logger.warning("Certificate validity period check failed")
                    return False

                logger.debug(f"Certificate chain validation passed for {vendor}")
                return True

        except Exception as e:
            logger.error(f"Certificate chain verification failed: {e}")

        return False

    def add_root_certificate(self, vendor: str, certificate: bytes) -> bool:
        """Add a new root certificate for a vendor"""
        try:
            cert = x509.load_pem_x509_certificate(certificate)
            cert_path = self.vendor_stores[vendor] / f"root_{int(time.time())}.pem"

            with open(cert_path, "wb") as f:
                f.write(certificate)

            key = f"{vendor}_{cert_path.stem}"
            self.root_certificates[key] = cert

            logger.info(f"Added root certificate for {vendor}")
            return True

        except Exception as e:
            logger.error(f"Failed to add root certificate for {vendor}: {e}")
            return False


class QuoteVerifier:
    """Verifies hardware attestation quotes"""

    def __init__(self, certificate_manager: CertificateManager):
        self.certificate_manager = certificate_manager

    async def verify_amd_sev_quote(self, quote: bytes, report_data: bytes) -> bool:
        """Verify AMD SEV-SNP attestation quote"""
        try:
            # AMD SEV-SNP quote verification
            # This is a simplified implementation
            # Production would use AMD's attestation libraries

            if len(quote) < 1184:  # Minimum SEV-SNP report size
                return False

            # Extract report from quote
            quote[:1184]
            quote[1184:]

            # Verify report structure and signature
            # In production, this would:
            # 1. Parse the attestation report structure
            # 2. Verify VCEK certificate chain
            # 3. Verify report signature
            # 4. Check report data matches expected values

            logger.debug("AMD SEV-SNP quote verification (simplified)")
            return True

        except Exception as e:
            logger.error(f"AMD SEV-SNP quote verification failed: {e}")
            return False

    async def verify_intel_tdx_quote(self, quote: bytes, report_data: bytes) -> bool:
        """Verify Intel TDX attestation quote"""
        try:
            # Intel TDX quote verification
            # This is a simplified implementation
            # Production would use Intel's DCAP libraries

            if len(quote) < 2048:  # Minimum TDX quote size
                return False

            # TDX quote structure verification
            # In production, this would:
            # 1. Parse the TDX quote structure
            # 2. Verify QE identity and PCK certificate chain
            # 3. Verify quote signature
            # 4. Check TD report measurements

            logger.debug("Intel TDX quote verification (simplified)")
            return True

        except Exception as e:
            logger.error(f"Intel TDX quote verification failed: {e}")
            return False

    async def verify_intel_sgx_quote(self, quote: bytes, report_data: bytes) -> bool:
        """Verify Intel SGX attestation quote"""
        try:
            # Intel SGX quote verification using DCAP
            # This is a simplified implementation

            if len(quote) < 1020:  # Minimum SGX quote size
                return False

            # SGX quote structure verification
            # In production, this would use Intel DCAP libraries

            logger.debug("Intel SGX quote verification (simplified)")
            return True

        except Exception as e:
            logger.error(f"Intel SGX quote verification failed: {e}")
            return False


class AttestationService:
    """
    Remote Attestation Service

    Provides comprehensive attestation capabilities for TEE environments
    including report generation, verification, and certificate management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.measurement_db = MeasurementDatabase()
        self.certificate_manager = CertificateManager()
        self.quote_verifier = QuoteVerifier(self.certificate_manager)

        # Attestation cache to avoid repeated verifications
        self.attestation_cache: dict[str, tuple[bool, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)

        # Nonce management for replay protection
        self.nonces: dict[str, datetime] = {}
        self.nonce_ttl = timedelta(minutes=10)

        logger.info("Attestation Service initialized")

    async def initialize(self) -> None:
        """Initialize the attestation service"""
        logger.info("Initializing Attestation Service...")

        # Load default trusted measurements if available
        await self._load_default_measurements()

        # Clean up expired nonces and cache entries
        asyncio.create_task(self._cleanup_loop())

        logger.info("Attestation Service initialization complete")

    async def shutdown(self) -> None:
        """Shutdown the attestation service"""
        logger.info("Shutting down Attestation Service...")

        # Save measurement database
        self.measurement_db.save_database()

        logger.info("Attestation Service shutdown complete")

    async def generate_attestation_report(self, context: EnclaveContext) -> AttestationReport:
        """Generate an attestation report for an enclave"""
        logger.info(f"Generating attestation report for enclave {context.spec.enclave_id}")

        # Create new report
        report = AttestationReport(
            enclave_id=context.spec.enclave_id, tee_type=context.tee_type, nonce=secrets.token_bytes(32)
        )

        try:
            # Generate measurements based on TEE type
            if context.tee_type == TEEType.AMD_SEV_SNP:
                await self._generate_sev_measurements(report, context)
            elif context.tee_type == TEEType.INTEL_TDX:
                await self._generate_tdx_measurements(report, context)
            elif context.tee_type == TEEType.INTEL_SGX:
                await self._generate_sgx_measurements(report, context)
            else:
                await self._generate_software_measurements(report, context)

            # Generate hardware quote if supported
            if context.tee_type != TEEType.SOFTWARE_ISOLATION:
                quote = await self._generate_hardware_quote(context, report.nonce)
                if quote:
                    report.quote = quote

            # Set expiration
            report.expires_at = datetime.now(UTC) + timedelta(seconds=context.spec.config.attestation_timeout_seconds)

            logger.info(f"Generated attestation report {report.report_id}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate attestation report: {e}")
            report.status = AttestationStatus.FAILED
            report.verification_errors.append(str(e))
            raise AttestationError(f"Report generation failed: {e}")

    async def verify_attestation(
        self, report: AttestationReport, expected_measurements: list[Measurement] = None
    ) -> bool:
        """Verify an attestation report"""
        logger.info(f"Verifying attestation report {report.report_id}")

        # Check cache first
        cache_key = self._get_cache_key(report)
        if cache_key in self.attestation_cache:
            result, cached_at = self.attestation_cache[cache_key]
            if datetime.now(UTC) - cached_at < self.cache_ttl:
                logger.debug(f"Using cached attestation result: {result}")
                return result

        try:
            # Verify report hasn't expired
            if report.expires_at and datetime.now(UTC) > report.expires_at:
                report.status = AttestationStatus.EXPIRED
                report.verification_errors.append("Report has expired")
                return False

            # Verify nonce freshness
            if not self._verify_nonce(report.nonce):
                report.verification_errors.append("Nonce verification failed")
                return False

            # Verify measurements
            measurement_result = await self._verify_measurements(report, expected_measurements)
            if not measurement_result:
                report.verification_errors.append("Measurement verification failed")
                return False

            # Verify hardware quote if present
            if report.quote:
                quote_result = await self._verify_hardware_quote(report)
                if not quote_result:
                    report.verification_errors.append("Quote verification failed")
                    return False

            # Verify certificate chain
            if report.certificate_chain:
                vendor = self._get_vendor_from_tee_type(report.tee_type)
                cert_result = self.certificate_manager.verify_certificate_chain(report.certificate_chain, vendor)
                if not cert_result:
                    report.verification_errors.append("Certificate chain verification failed")
                    return False

            # All verifications passed
            report.status = AttestationStatus.VERIFIED
            report.verified_at = datetime.now(UTC)

            # Cache result
            self.attestation_cache[cache_key] = (True, datetime.now(UTC))

            logger.info(f"Attestation report {report.report_id} verification successful")
            return True

        except Exception as e:
            logger.error(f"Attestation verification failed: {e}")
            report.status = AttestationStatus.FAILED
            report.verification_errors.append(str(e))

            # Cache negative result too (with shorter TTL)
            self.attestation_cache[cache_key] = (False, datetime.now(UTC))
            return False

    async def _generate_sev_measurements(self, report: AttestationReport, context: EnclaveContext) -> None:
        """Generate measurements for AMD SEV-SNP"""
        # Simulate SEV-SNP measurements
        # In production, these would come from actual hardware

        # Launch measurement (hash of initial VM state)
        launch_measure = hashlib.sha384(
            context.spec.code_hash.encode() + str(context.spec.memory_mb).encode()
        ).hexdigest()

        report.add_measurement("launch_digest", 0, launch_measure, "sha384", "VM launch measurement")

        # Guest policy
        policy_value = hashlib.sha256(b"sev_snp_policy").hexdigest()
        report.add_measurement("guest_policy", 1, policy_value, "sha256", "Guest policy measurement")

    async def _generate_tdx_measurements(self, report: AttestationReport, context: EnclaveContext) -> None:
        """Generate measurements for Intel TDX"""
        # Simulate TDX measurements
        # In production, these would come from TDX hardware

        # MRTD (Measurement of Trust Domain)
        mrtd = hashlib.sha384(context.spec.code_hash.encode() + b"tdx_configuration").hexdigest()

        report.add_measurement("mrtd", 0, mrtd, "sha384", "Trust Domain measurement")

        # Runtime measurements
        for i in range(4):
            rtmr = hashlib.sha384(f"rtmr_{i}_{context.spec.enclave_id}".encode()).hexdigest()
            report.add_measurement("rtmr", i, rtmr, "sha384", f"Runtime measurement {i}")

    async def _generate_sgx_measurements(self, report: AttestationReport, context: EnclaveContext) -> None:
        """Generate measurements for Intel SGX"""
        # Simulate SGX measurements

        # MRENCLAVE (enclave code measurement)
        mrenclave = hashlib.sha256(context.spec.code_hash.encode()).hexdigest()
        report.add_measurement("mrenclave", 0, mrenclave, "sha256", "Enclave code measurement")

        # MRSIGNER (enclave signer measurement)
        mrsigner = hashlib.sha256(b"test_signer_key").hexdigest()
        report.add_measurement("mrsigner", 0, mrsigner, "sha256", "Enclave signer measurement")

    async def _generate_software_measurements(self, report: AttestationReport, context: EnclaveContext) -> None:
        """Generate measurements for software isolation"""
        # For software TEE, create measurements of container/VM state

        # Code measurement
        code_measure = hashlib.sha256(context.spec.code_hash.encode()).hexdigest()
        report.add_measurement("code_hash", 0, code_measure, "sha256", "Application code measurement")

        # Configuration measurement
        config_data = json.dumps(context.spec.environment_variables, sort_keys=True)
        config_measure = hashlib.sha256(config_data.encode()).hexdigest()
        report.add_measurement("config_hash", 1, config_measure, "sha256", "Configuration measurement")

    async def _generate_hardware_quote(self, context: EnclaveContext, nonce: bytes) -> bytes | None:
        """Generate hardware attestation quote"""
        try:
            if context.tee_type == TEEType.AMD_SEV_SNP:
                # Simulate SEV-SNP quote generation
                # In production, this would use SEV-SNP APIs
                quote_data = b"sev_snp_quote" + nonce + context.spec.enclave_id.encode()
                return hashlib.sha256(quote_data).digest() + b"signature_placeholder"

            elif context.tee_type == TEEType.INTEL_TDX:
                # Simulate TDX quote generation
                quote_data = b"tdx_quote" + nonce + context.spec.enclave_id.encode()
                return hashlib.sha384(quote_data).digest() + b"tdx_signature_placeholder"

            elif context.tee_type == TEEType.INTEL_SGX:
                # Simulate SGX quote generation
                quote_data = b"sgx_quote" + nonce + context.spec.enclave_id.encode()
                return hashlib.sha256(quote_data).digest() + b"sgx_signature_placeholder"

        except Exception as e:
            logger.error(f"Hardware quote generation failed: {e}")

        return None

    async def _verify_measurements(
        self, report: AttestationReport, expected_measurements: list[Measurement] = None
    ) -> bool:
        """Verify measurements in attestation report"""
        if not report.measurements:
            logger.warning("No measurements in attestation report")
            return False

        # If expected measurements provided, verify against them
        if expected_measurements:
            for expected in expected_measurements:
                actual = report.get_measurement(expected.measurement_type, expected.index)
                if not actual or actual.value != expected.value:
                    logger.warning(f"Measurement mismatch: {expected.measurement_type}:{expected.index}")
                    return False
        else:
            # Verify against trusted measurement database
            for measurement in report.measurements:
                if not self.measurement_db.is_measurement_trusted(measurement):
                    logger.debug(
                        f"Measurement not in trusted database: {measurement.measurement_type}:{measurement.index}"
                    )
                    # For now, don't fail on unknown measurements in software mode
                    if report.tee_type != TEEType.SOFTWARE_ISOLATION:
                        return False

        return True

    async def _verify_hardware_quote(self, report: AttestationReport) -> bool:
        """Verify hardware attestation quote"""
        try:
            if report.tee_type == TEEType.AMD_SEV_SNP:
                return await self.quote_verifier.verify_amd_sev_quote(report.quote, report.nonce)
            elif report.tee_type == TEEType.INTEL_TDX:
                return await self.quote_verifier.verify_intel_tdx_quote(report.quote, report.nonce)
            elif report.tee_type == TEEType.INTEL_SGX:
                return await self.quote_verifier.verify_intel_sgx_quote(report.quote, report.nonce)
        except Exception as e:
            logger.error(f"Quote verification failed: {e}")
            return False

        return True

    def _verify_nonce(self, nonce: bytes) -> bool:
        """Verify nonce for replay protection"""
        nonce_hex = nonce.hex()

        # Check if nonce was already used
        if nonce_hex in self.nonces:
            logger.warning("Nonce replay detected")
            return False

        # Store nonce
        self.nonces[nonce_hex] = datetime.now(UTC)
        return True

    def _get_cache_key(self, report: AttestationReport) -> str:
        """Generate cache key for attestation report"""
        key_data = f"{report.enclave_id}_{report.tee_type.value}_{report.nonce.hex()}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_vendor_from_tee_type(self, tee_type: TEEType) -> str:
        """Get vendor name from TEE type"""
        if tee_type == TEEType.AMD_SEV_SNP:
            return "amd"
        elif tee_type in [TEEType.INTEL_TDX, TEEType.INTEL_SGX]:
            return "intel"
        elif tee_type == TEEType.ARM_TRUSTZONE:
            return "arm"
        return "generic"

    async def _load_default_measurements(self) -> None:
        """Load default trusted measurements"""
        # Add some default measurements for testing
        default_measurements = [
            Measurement("code_hash", 0, "a" * 64, "sha256", "Test application hash"),
            Measurement("config_hash", 1, "b" * 64, "sha256", "Test configuration hash"),
        ]

        for measurement in default_measurements:
            if not self.measurement_db.is_measurement_trusted(measurement):
                self.measurement_db.add_trusted_measurement(measurement, "Default measurement", "system")

    async def _cleanup_loop(self) -> None:
        """Clean up expired nonces and cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                now = datetime.now(UTC)

                # Clean up expired nonces
                expired_nonces = [nonce for nonce, timestamp in self.nonces.items() if now - timestamp > self.nonce_ttl]
                for nonce in expired_nonces:
                    del self.nonces[nonce]

                # Clean up expired cache entries
                expired_cache = [
                    key for key, (_, timestamp) in self.attestation_cache.items() if now - timestamp > self.cache_ttl
                ]
                for key in expired_cache:
                    del self.attestation_cache[key]

                if expired_nonces or expired_cache:
                    logger.debug(f"Cleaned up {len(expired_nonces)} nonces, " f"{len(expired_cache)} cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in attestation cleanup: {e}")
                await asyncio.sleep(60)

    def add_trusted_measurement(
        self, measurement_type: str, index: int, value: str, algorithm: str = "sha256", description: str = ""
    ) -> None:
        """Add a trusted measurement to the database"""
        measurement = Measurement(measurement_type, index, value, algorithm, description)
        self.measurement_db.add_trusted_measurement(measurement, description, "manual")

    def get_trusted_measurements(self) -> list[Measurement]:
        """Get all trusted measurements"""
        return self.measurement_db.get_trusted_measurements()

    def get_service_status(self) -> dict[str, Any]:
        """Get attestation service status"""
        return {
            "measurement_database": {
                "path": str(self.measurement_db.db_path),
                "entries": len(self.measurement_db.measurements),
            },
            "certificate_manager": {
                "cert_store_path": str(self.certificate_manager.cert_store_path),
                "root_certificates": len(self.certificate_manager.root_certificates),
            },
            "cache": {"entries": len(self.attestation_cache), "nonces": len(self.nonces)},
        }

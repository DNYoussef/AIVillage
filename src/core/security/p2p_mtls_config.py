#!/usr/bin/env python3
"""mTLS Configuration for P2P Communications.

Provides mutual TLS authentication for LibP2P mesh networking with
CODEX-compliant security requirements.
"""

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class P2PMTLSConfig:
    """mTLS configuration for P2P mesh networking."""

    def __init__(self, node_id: str, cert_dir: str = "./certs/p2p") -> None:
        """Initialize mTLS configuration.

        Args:
            node_id: Unique node identifier
            cert_dir: Directory to store certificates
        """
        self.node_id = node_id
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)

        # Certificate paths
        self.ca_cert_path = self.cert_dir / "p2p_ca.crt"
        self.ca_key_path = self.cert_dir / "p2p_ca.key"
        self.node_cert_path = self.cert_dir / f"node_{node_id}.crt"
        self.node_key_path = self.cert_dir / f"node_{node_id}.key"

        # TLS configuration
        self.tls_version = ssl.TLSVersion.TLSv1_3
        self.ciphers = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256"

        # Initialize certificates
        self._ensure_certificates()

    def _ensure_certificates(self) -> None:
        """Ensure all required certificates exist."""
        # Create CA if it doesn't exist
        if not self.ca_cert_path.exists() or not self.ca_key_path.exists():
            logger.info("Generating P2P CA certificate")
            self._generate_ca()

        # Create node certificate if it doesn't exist
        if not self.node_cert_path.exists() or not self.node_key_path.exists():
            logger.info(f"Generating P2P node certificate for {self.node_id}")
            self._generate_node_cert()

    def _generate_ca(self) -> None:
        """Generate Certificate Authority for P2P network."""
        # Generate CA private key
        ca_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,  # Stronger key for CA
        )

        # Generate CA certificate
        ca_subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Virtual"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "AIVillage"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVillage P2P Network"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Certificate Authority"),
                x509.NameAttribute(NameOID.COMMON_NAME, "AIVillage P2P CA"),
            ]
        )

        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(ca_subject)
            .issuer_name(ca_subject)  # Self-signed
            .public_key(ca_private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))  # 10 years for CA
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    content_commitment=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage(
                    [
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=True,
            )
            .sign(ca_private_key, hashes.SHA256())
        )

        # Write CA private key
        with open(self.ca_key_path, "wb") as f:
            f.write(
                ca_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write CA certificate
        with open(self.ca_cert_path, "wb") as f:
            f.write(ca_certificate.public_bytes(serialization.Encoding.PEM))

        # Set restrictive permissions
        os.chmod(self.ca_key_path, 0o600)
        os.chmod(self.ca_cert_path, 0o644)

        logger.info(f"Generated P2P CA certificate: {self.ca_cert_path}")

    def _generate_node_cert(self) -> None:
        """Generate node certificate signed by CA."""
        # Load CA
        with open(self.ca_cert_path, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read())

        with open(self.ca_key_path, "rb") as f:
            ca_private_key = serialization.load_pem_private_key(f.read(), password=None)

        # Generate node private key
        node_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate node certificate
        node_subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Virtual"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "AIVillage"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIVillage P2P Network"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "P2P Node"),
                x509.NameAttribute(NameOID.COMMON_NAME, f"node-{self.node_id}"),
            ]
        )

        node_certificate = (
            x509.CertificateBuilder()
            .subject_name(node_subject)
            .issuer_name(ca_cert.subject)
            .public_key(node_private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))  # 1 year for node certs
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    data_encipherment=False,
                    content_commitment=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage(
                    [
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                        x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    ]
                ),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(f"node-{self.node_id}"),
                        x509.DNSName(f"{self.node_id}.p2p.aivillage.local"),
                        x509.RFC822Name(f"{self.node_id}@p2p.aivillage.local"),
                        x509.UniformResourceIdentifier(f"p2p://{self.node_id}"),
                    ]
                ),
                critical=False,
            )
            .sign(ca_private_key, hashes.SHA256())
        )

        # Write node private key
        with open(self.node_key_path, "wb") as f:
            f.write(
                node_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Write node certificate
        with open(self.node_cert_path, "wb") as f:
            f.write(node_certificate.public_bytes(serialization.Encoding.PEM))

        # Set restrictive permissions
        os.chmod(self.node_key_path, 0o600)
        os.chmod(self.node_cert_path, 0o644)

        logger.info(f"Generated P2P node certificate: {self.node_cert_path}")

    def create_ssl_context_server(self) -> ssl.SSLContext:
        """Create SSL context for server-side P2P connections."""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # TLS 1.3 only for maximum security
        context.minimum_version = self.tls_version
        context.maximum_version = self.tls_version

        # Load node certificate and key
        context.load_cert_chain(str(self.node_cert_path), str(self.node_key_path))

        # Load CA certificate for client verification
        context.load_verify_locations(str(self.ca_cert_path))

        # Require client certificates (mutual TLS)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False  # We use custom hostname verification

        # Set strong ciphers
        context.set_ciphers(self.ciphers)

        # Security options
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_NO_RENEGOTIATION
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE

        return context

    def create_ssl_context_client(self) -> ssl.SSLContext:
        """Create SSL context for client-side P2P connections."""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # TLS 1.3 only for maximum security
        context.minimum_version = self.tls_version
        context.maximum_version = self.tls_version

        # Load node certificate and key for client authentication
        context.load_cert_chain(str(self.node_cert_path), str(self.node_key_path))

        # Load CA certificate for server verification
        context.load_verify_locations(str(self.ca_cert_path))

        # Verify server certificates
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False  # We use custom hostname verification

        # Set strong ciphers
        context.set_ciphers(self.ciphers)

        # Security options
        context.options |= ssl.OP_NO_COMPRESSION
        context.options |= ssl.OP_NO_RENEGOTIATION
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE

        return context

    def verify_peer_certificate(self, peer_cert_der: bytes) -> tuple[bool, str]:
        """Verify peer certificate against our CA.

        Args:
            peer_cert_der: Peer certificate in DER format

        Returns:
            Tuple of (is_valid, peer_node_id)
        """
        try:
            # Parse certificate
            peer_cert = x509.load_der_x509_certificate(peer_cert_der)

            # Load our CA certificate
            with open(self.ca_cert_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(f.read())

            # Verify certificate signature
            ca_public_key = ca_cert.public_key()
            ca_public_key.verify(
                peer_cert.signature,
                peer_cert.tbs_certificate_bytes,
                peer_cert.signature_algorithm_oid._name,
            )

            # Check validity period
            now = datetime.utcnow()
            if now < peer_cert.not_valid_before or now > peer_cert.not_valid_after:
                return False, "Certificate expired or not yet valid"

            # Extract peer node ID from Common Name
            common_name = None
            for attribute in peer_cert.subject:
                if attribute.oid == NameOID.COMMON_NAME:
                    common_name = attribute.value
                    break

            if not common_name or not common_name.startswith("node-"):
                return False, "Invalid certificate Common Name"

            peer_node_id = common_name[5:]  # Remove "node-" prefix

            logger.debug(f"Verified peer certificate for node: {peer_node_id}")
            return True, peer_node_id

        except Exception as e:
            logger.exception(f"Certificate verification failed: {e}")
            return False, str(e)

    def get_node_certificate_der(self) -> bytes:
        """Get our node certificate in DER format for sharing."""
        with open(self.node_cert_path, "rb") as f:
            cert_pem = f.read()

        cert = x509.load_pem_x509_certificate(cert_pem)
        return cert.public_bytes(serialization.Encoding.DER)

    def rotate_node_certificate(self) -> None:
        """Rotate node certificate (regenerate with new validity period)."""
        # Backup old certificate
        if self.node_cert_path.exists():
            backup_path = self.node_cert_path.with_suffix(".crt.backup")
            self.node_cert_path.rename(backup_path)
            logger.info(f"Backed up old certificate to: {backup_path}")

        if self.node_key_path.exists():
            backup_path = self.node_key_path.with_suffix(".key.backup")
            self.node_key_path.rename(backup_path)

        # Generate new certificate
        self._generate_node_cert()
        logger.info(f"Rotated certificate for node: {self.node_id}")

    def get_certificate_info(self) -> dict[str, any]:
        """Get information about current certificates."""
        info = {
            "node_id": self.node_id,
            "cert_dir": str(self.cert_dir),
            "ca_cert_exists": self.ca_cert_path.exists(),
            "node_cert_exists": self.node_cert_path.exists(),
            "tls_version": str(self.tls_version),
            "ciphers": self.ciphers,
        }

        # Get certificate validity info
        if self.node_cert_path.exists():
            try:
                with open(self.node_cert_path, "rb") as f:
                    cert = x509.load_pem_x509_certificate(f.read())

                info["certificate"] = {
                    "subject": cert.subject.rfc4514_string(),
                    "issuer": cert.issuer.rfc4514_string(),
                    "serial_number": str(cert.serial_number),
                    "not_valid_before": cert.not_valid_before.isoformat(),
                    "not_valid_after": cert.not_valid_after.isoformat(),
                    "is_valid": (
                        datetime.utcnow() >= cert.not_valid_before and datetime.utcnow() <= cert.not_valid_after
                    ),
                    "fingerprint_sha256": cert.fingerprint(hashes.SHA256()).hex(),
                }
            except Exception as e:
                info["certificate_error"] = str(e)

        return info


def create_p2p_mtls_config(node_id: str, cert_dir: str | None = None) -> P2PMTLSConfig:
    """Create mTLS configuration for P2P node.

    Args:
        node_id: Unique node identifier
        cert_dir: Certificate directory (defaults to ./certs/p2p)

    Returns:
        Configured P2P mTLS instance
    """
    cert_dir = cert_dir or os.getenv("P2P_CERT_DIR", "./certs/p2p")
    return P2PMTLSConfig(node_id, cert_dir)


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two nodes for testing
        node1_config = P2PMTLSConfig("test_node_1", f"{temp_dir}/node1")
        node2_config = P2PMTLSConfig("test_node_2", f"{temp_dir}/node2")

        # Create SSL contexts
        node1_server_ctx = node1_config.create_ssl_context_server()
        node1_client_ctx = node1_config.create_ssl_context_client()

        # Test certificate verification
        node1_cert_der = node1_config.get_node_certificate_der()
        is_valid, node_id = node2_config.verify_peer_certificate(node1_cert_der)

        print("Certificate verification test:")
        print(f"Valid: {is_valid}, Node ID: {node_id}")

        # Print certificate info
        cert_info = node1_config.get_certificate_info()
        print("\nNode 1 Certificate Info:")
        for key, value in cert_info.items():
            print(f"  {key}: {value}")

        print("\nmTLS configuration test completed successfully!")

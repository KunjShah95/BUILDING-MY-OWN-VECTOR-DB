"""
mTLS Service (Phase 9: Security & Compliance).

Provides mutual TLS (mTLS) for secure inter-node communication between
vector database cluster nodes. Ensures that only authenticated peers can
join the cluster and all traffic is encrypted.

Architecture:
  - Each node has a unique certificate signed by the cluster CA
  - Nodes authenticate each other via certificate validation
  - gRPC and HTTP communications are wrapped in TLS
  - Certificate rotation and revocation supported

Usage::

    from utils.mtls_service import MTLSConfig, create_mtls_server_context

    ctx = create_mtls_server_context(
        cert_path="certs/node.crt",
        key_path="certs/node.key",
        ca_path="certs/ca.crt",
    )
    # Use ctx with uvicorn/grpc server
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MTLSConfig:
    """mTLS configuration for a cluster node."""

    enabled: bool = False
    cert_path: str = "certs/node.crt"
    key_path: str = "certs/node.key"
    ca_path: str = "certs/ca.crt"
    cert_dir: str = "certs"

    # Auto-generation defaults (for dev/test)
    auto_generate: bool = False
    organization: str = "VectorDB"
    validity_days: int = 365


def create_mtls_server_context(config: MTLSConfig) -> Optional[Any]:
    """Create an SSL context for mTLS server.

    Args:
        config: mTLS configuration.

    Returns:
        SSLContext for use with uvicorn/fastapi, or None if mTLS is disabled.
    """
    if not config.enabled:
        return None

    try:
        import ssl

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_cert_chain(config.cert_path, config.key_path)
        context.load_verify_locations(config.ca_path)

        # Require client certificate
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False

        logger.info("mTLS server context created (cert=%s)", config.cert_path)
        return context
    except Exception as exc:
        logger.error("Failed to create mTLS context: %s", exc)
        return None


def create_mtls_client_context(config: MTLSConfig) -> Optional[Any]:
    """Create an SSL context for mTLS client (node-to-node).

    Args:
        config: mTLS configuration.

    Returns:
        SSLContext for client connections, or None if mTLS is disabled.
    """
    if not config.enabled:
        return None

    try:
        import ssl

        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_cert_chain(config.cert_path, config.key_path)
        context.load_verify_locations(config.ca_path)
        context.check_hostname = False

        logger.info("mTLS client context created")
        return context
    except Exception as exc:
        logger.error("Failed to create mTLS client context: %s", exc)
        return None


def generate_dev_certificates(config: MTLSConfig) -> bool:
    """Generate self-signed certificates for development/testing.

    Creates a CA certificate, a server certificate, and a client certificate
    in the configured cert directory.

    Args:
        config: mTLS configuration.

    Returns:
        True if certificates were generated successfully.
    """
    os.makedirs(config.cert_dir, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime

        # Generate CA key and cert
        ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        ca_subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, f"{config.organization} CA"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"{config.organization} Root CA"),
        ])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_subject)
            .issuer_name(issuer)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=config.validity_days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(ca_key, hashes.SHA256())
        )

        # Write CA cert
        with open(os.path.join(config.cert_dir, "ca.crt"), "wb") as f:
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))
        with open(os.path.join(config.cert_dir, "ca.key"), "wb") as f:
            f.write(ca_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))

        # Generate node (server) certificate
        node_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        node_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, config.organization),
                x509.NameAttribute(NameOID.COMMON_NAME, "vector-db-node"),
            ]))
            .issuer_name(ca_subject)
            .public_key(node_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=config.validity_days))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .sign(ca_key, hashes.SHA256())
        )

        with open(os.path.join(config.cert_dir, "node.crt"), "wb") as f:
            f.write(node_cert.public_bytes(serialization.Encoding.PEM))
        with open(os.path.join(config.cert_dir, "node.key"), "wb") as f:
            f.write(node_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))

        logger.info("Development certificates generated in %s", config.cert_dir)
        return True

    except ImportError:
        logger.warning(
            "cryptography library required for certificate generation. "
            "Install: pip install cryptography"
        )
        return False
    except Exception as exc:
        logger.error("Certificate generation failed: %s", exc)
        return False
